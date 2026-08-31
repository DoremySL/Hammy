"""llama 模型扫描缓存（llama_scan_cache.json）测试。

scan_models 按模型目录指纹（*.gguf 的路径+大小+mtime_ns）持久缓存：
目录未变化时直接复用上次扫描结果，不重复读取 GGUF 头部元数据；
任一 gguf 增删改即失效重扫；缓存文件缺失/损坏时回退全量扫描。
"""
import json
import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gui_app import llama_cpp


def _min_gguf() -> bytes:
    """最小合法 GGUF v3（0 个 KV 项）：无元数据 → 扫描信息全默认 0。"""
    return b"GGUF" + struct.pack("<IQ", 3, 0)


class _CacheTestCase(unittest.TestCase):
    """统一把扫描缓存文件重定向到临时目录，避免污染真实 workspace。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_file = Path(self._tmp.name) / "llama_scan_cache.json"
        self._p = mock.patch("gui_app.workspace_paths.LLAMA_SCAN_CACHE_FILE",
                             new=self.cache_file)
        self._p.start()
        self.addCleanup(self._p.stop)
        self.addCleanup(self._tmp.cleanup)
        # 指纹 TTL 内存缓存是独立于失效语义的性能层（轮询专用，显式扫描走
        # force=True）：固定为 0，让测试直接验证「目录变化 → 指纹变化 → 重扫」
        self._ttl_p = mock.patch.object(llama_cpp, "_FP_TTL_SEC", 0.0)
        self._ttl_p.start()
        self.addCleanup(self._ttl_p.stop)

    def _models_dir(self) -> Path:
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        return d


class TestScanCacheHit(_CacheTestCase):
    def test_second_scan_reuses_cache(self):
        d = self._models_dir()
        (d / "m.gguf").write_bytes(_min_gguf())
        real = llama_cpp._scan_full
        with mock.patch("gui_app.llama_cpp._scan_full",
                        side_effect=real) as m:
            r1 = llama_cpp.scan_models(d)
            r2 = llama_cpp.scan_models(d)
        self.assertEqual(m.call_count, 1)  # 第二次未重扫
        self.assertEqual(r1, r2)
        self.assertEqual([x["name"] for x in r1], ["m.gguf"])

    def test_scan_result_persisted(self):
        d = self._models_dir()
        (d / "m.gguf").write_bytes(_min_gguf())
        llama_cpp.scan_models(d)
        data = json.loads(self.cache_file.read_text(encoding="utf-8"))
        entry = data["dirs"][str(d)]
        self.assertTrue(entry["fingerprint"])
        self.assertEqual([x["name"] for x in entry["models"]], ["m.gguf"])

    def test_cache_reused_after_new_process(self):
        # 缓存持久化：重新构造 scan_models 环境（模拟重启）仍命中
        d = self._models_dir()
        (d / "m.gguf").write_bytes(_min_gguf())
        llama_cpp.scan_models(d)
        real = llama_cpp._scan_full
        with mock.patch("gui_app.llama_cpp._scan_full",
                        side_effect=real) as m:
            r = llama_cpp.scan_models(d)
        self.assertEqual(m.call_count, 0)  # 直接读缓存，零重扫
        self.assertEqual([x["name"] for x in r], ["m.gguf"])


class TestScanCacheInvalidation(_CacheTestCase):
    def test_new_model_invalidates(self):
        d = self._models_dir()
        (d / "a.gguf").write_bytes(_min_gguf())
        real = llama_cpp._scan_full
        with mock.patch("gui_app.llama_cpp._scan_full",
                        side_effect=real) as m:
            llama_cpp.scan_models(d)
            (d / "b.gguf").write_bytes(_min_gguf())
            r = llama_cpp.scan_models(d)
        self.assertEqual(m.call_count, 2)
        self.assertEqual(len(r), 2)

    def test_mtime_change_invalidates(self):
        d = self._models_dir()
        p = d / "m.gguf"
        p.write_bytes(_min_gguf())
        real = llama_cpp._scan_full
        with mock.patch("gui_app.llama_cpp._scan_full",
                        side_effect=real) as m:
            llama_cpp.scan_models(d)
            os.utime(p, (p.stat().st_atime + 10, p.stat().st_mtime + 10))
            llama_cpp.scan_models(d)
        self.assertEqual(m.call_count, 2)

    def test_size_change_invalidates(self):
        d = self._models_dir()
        p = d / "m.gguf"
        p.write_bytes(_min_gguf())
        real = llama_cpp._scan_full
        with mock.patch("gui_app.llama_cpp._scan_full",
                        side_effect=real) as m:
            llama_cpp.scan_models(d)
            p.write_bytes(_min_gguf() + b"\x00" * 16)  # 大小变化
            llama_cpp.scan_models(d)
        self.assertEqual(m.call_count, 2)

    def test_mmproj_change_invalidates(self):
        # mmproj 也是 *.gguf，增删应触发重扫（影响扫描结果中的 mmprojs 列表）
        d = self._models_dir()
        (d / "m.gguf").write_bytes(_min_gguf())
        real = llama_cpp._scan_full
        with mock.patch("gui_app.llama_cpp._scan_full",
                        side_effect=real) as m:
            llama_cpp.scan_models(d)
            (d / "mmproj-BF16.gguf").write_bytes(_min_gguf())
            r = llama_cpp.scan_models(d)
        self.assertEqual(m.call_count, 2)
        self.assertEqual(r[0]["mmprojs"], [str(d / "mmproj-BF16.gguf")])


class TestScanCacheIsolation(_CacheTestCase):
    def test_cache_per_models_dir(self):
        d1, d2 = self._models_dir(), self._models_dir()
        (d1 / "a.gguf").write_bytes(_min_gguf())
        (d2 / "b.gguf").write_bytes(_min_gguf())
        real = llama_cpp._scan_full
        with mock.patch("gui_app.llama_cpp._scan_full",
                        side_effect=real) as m:
            llama_cpp.scan_models(d1)   # 全量
            llama_cpp.scan_models(d2)   # 全量（目录不同）
            r1 = llama_cpp.scan_models(d1)  # 命中 d1 缓存
            r2 = llama_cpp.scan_models(d2)  # 命中 d2 缓存
        self.assertEqual(m.call_count, 2)
        self.assertEqual([x["name"] for x in r1], ["a.gguf"])
        self.assertEqual([x["name"] for x in r2], ["b.gguf"])


class TestScanCacheFallback(_CacheTestCase):
    def test_corrupt_cache_falls_back(self):
        d = self._models_dir()
        (d / "m.gguf").write_bytes(_min_gguf())
        self.cache_file.write_text("{ not json !!", encoding="utf-8")
        r = llama_cpp.scan_models(d)  # 不抛异常
        self.assertEqual([x["name"] for x in r], ["m.gguf"])

    def test_cache_wrong_shape_falls_back(self):
        d = self._models_dir()
        (d / "m.gguf").write_bytes(_min_gguf())
        self.cache_file.write_text("[1, 2, 3]", encoding="utf-8")  # 非 dict
        r = llama_cpp.scan_models(d)
        self.assertEqual([x["name"] for x in r], ["m.gguf"])

    def test_missing_dir_no_cache_touch(self):
        # 目录不存在：直接返回 []，不读写缓存
        d = self._models_dir() / "not-exist"
        r = llama_cpp.scan_models(d)
        self.assertEqual(r, [])
        self.assertFalse(self.cache_file.exists())


if __name__ == "__main__":
    unittest.main()
