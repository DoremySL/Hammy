"""GGUF 元数据解析（MoE 判定）与 --n-cpu-moe 参数组装测试。"""
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app import llama_cpp


# ── 最小 GGUF 文件构造 ──

def _str(v: str) -> bytes:
    b = v.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _u32(v: int) -> bytes:
    return struct.pack("<I", v)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def _gguf_bytes(pairs) -> bytes:
    """构造最小 GGUF 文件字节。

    pairs: [(key, value_type_id, payload_bytes)]，按顺序写入 KV 区。
    """
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)            # version
    out += struct.pack("<Q", 0)            # tensor_count
    out += struct.pack("<Q", len(pairs))   # metadata_kv_count
    for key, vt, payload in pairs:
        kb = key.encode("utf-8")
        out += struct.pack("<Q", len(kb)) + kb
        out += struct.pack("<I", vt)
        out += payload
    return bytes(out)


class TestReadGgufMeta(unittest.TestCase):
    def test_parse_expert_keys(self):
        data = _gguf_bytes([
            ("general.name", 8, _str("TestMoE")),
            ("llama.expert_count", 4, _u32(8)),
            ("llama.expert_used_count", 4, _u32(2)),
        ])
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "moe.gguf"
            p.write_bytes(data)
            meta = llama_cpp._read_gguf_meta(p)
        self.assertEqual(meta["llama.expert_count"], 8)
        self.assertEqual(meta["llama.expert_used_count"], 2)
        self.assertEqual(meta["general.name"], "TestMoE")

    def test_expert_count_uint64_type(self):
        # 部分模型用 uint64 存储 expert_count
        data = _gguf_bytes([("llama.expert_count", 10, _u64(8))])
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "moe.gguf"
            p.write_bytes(data)
            info = llama_cpp._extract_model_info(llama_cpp._read_gguf_meta(p))
        self.assertTrue(info["moe"])

    def test_string_kv_then_array_kv(self):
        # 字符串与数组 KV 都能正确跳过，后续 int 键仍能解析
        data = _gguf_bytes([
            ("general.name", 8, _str("x")),
            ("test.array", 9, _u32(4) + _u64(2) + _u32(1) + _u32(2)),
            ("llama.expert_count", 4, _u32(4)),
        ])
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "moe.gguf"
            p.write_bytes(data)
            meta = llama_cpp._read_gguf_meta(p)
        self.assertEqual(meta["llama.expert_count"], 4)
        self.assertEqual(meta["test.array"], [1, 2])

    def test_non_gguf_magic(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.gguf"
            p.write_bytes(b"GGUX" + b"\x00" * 64)
            self.assertEqual(llama_cpp._read_gguf_meta(p), {})

    def test_truncated_kv_region(self):
        # 头部声明 5 个 KV 但实际只有 1 个 → 返回已解析部分，不抛异常
        data = _gguf_bytes([("general.name", 8, _str("x"))])
        head = bytearray(data)
        # 把 kv_count 改为 5（越界），模拟截断
        struct.pack_into("<Q", head, 16, 5)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.gguf"
            p.write_bytes(bytes(head))
            meta = llama_cpp._read_gguf_meta(p)
        self.assertEqual(meta.get("general.name"), "x")

    def test_missing_file(self):
        self.assertEqual(llama_cpp._read_gguf_meta(Path("N:/不存在的文件.gguf")), {})
        info = llama_cpp._extract_model_info(llama_cpp._read_gguf_meta(Path("N:/不存在的文件.gguf")))
        self.assertFalse(info["moe"])


class TestScanModelsMoeFlag(unittest.TestCase):
    def setUp(self):
        # 扫描缓存命中/落盘由 test_llama_scan_cache.py 专门覆盖；此处禁写防污染
        p = mock.patch("gui_app.llama_cpp._save_scan_cache")
        p.start()
        self.addCleanup(p.stop)

    def test_moe_flags_in_scan(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "moe.gguf").write_bytes(
                _gguf_bytes([("llama.expert_count", 4, _u32(8))]))
            (Path(td) / "plain.gguf").write_bytes(
                _gguf_bytes([("general.name", 8, _str("Plain"))]))
            # 投影文件应被排除在模型列表外
            (Path(td) / "vision.mmproj-Q4_K_M.gguf").write_bytes(
                _gguf_bytes([("general.name", 8, _str("Proj"))]))
            models = llama_cpp.scan_models(Path(td))
        by_name = {m["name"]: m for m in models}
        self.assertEqual(set(by_name), {"moe.gguf", "plain.gguf"})
        self.assertTrue(by_name["moe.gguf"]["moe"])
        self.assertEqual(by_name["moe.gguf"]["expert_count"], 8)
        self.assertFalse(by_name["plain.gguf"]["moe"])
        self.assertEqual(by_name["plain.gguf"]["expert_count"], 0)


class TestBuildArgsNcpumoe(unittest.TestCase):
    def test_zero_omitted(self):
        args = llama_cpp._build_args(Path("m.gguf"), {"n_cpu_moe": 0})
        self.assertNotIn("--n-cpu-moe", args)

    def test_missing_defaults_to_zero(self):
        args = llama_cpp._build_args(Path("m.gguf"), {})
        self.assertNotIn("--n-cpu-moe", args)

    def test_positive_injected(self):
        args = llama_cpp._build_args(Path("m.gguf"), {"n_cpu_moe": 16})
        i = args.index("--n-cpu-moe")
        self.assertEqual(args[i + 1], "16")

    def test_defaults_contains_ncpumoe(self):
        self.assertEqual(llama_cpp.DEFAULTS["n_cpu_moe"], 0)


class TestBuildArgsKvQuant(unittest.TestCase):
    """_build_args：KV 量化小写传参；空 = 不启用（不传入参数）。"""

    def test_default_q8(self):
        args = llama_cpp._build_args(Path("m.gguf"), {})
        i = args.index("--cache-type-k")
        self.assertEqual(args[i + 1], "q8_0")  # 默认 q8_0，小写
        self.assertEqual(args[i + 2], "--cache-type-v")
        self.assertEqual(args[i + 3], "q8_0")

    def test_empty_omitted(self):
        args = llama_cpp._build_args(Path("m.gguf"), {"kv_quant": ""})
        self.assertNotIn("--cache-type-k", args)  # 不启用 → 不传入

    def test_uppercase_normalized(self):
        args = llama_cpp._build_args(Path("m.gguf"), {"kv_quant": "Q8_0"})
        i = args.index("--cache-type-k")
        self.assertEqual(args[i + 1], "q8_0")  # 大写配置小写化后传参

    def test_other_values_lowercased(self):
        args = llama_cpp._build_args(Path("m.gguf"), {"kv_quant": "IQ4_NL"})
        i = args.index("--cache-type-k")
        self.assertEqual(args[i + 1], "iq4_nl")

    def test_kv_preset_off_by_default(self):
        # 默认（空预设）→ 不传 --kv-unified
        args = llama_cpp._build_args(Path("m.gguf"), {})
        self.assertNotIn("--kv-unified", args)
        # 预设 kv_unified → 传 --kv-unified
        args = llama_cpp._build_args(Path("m.gguf"), {"kv_preset": "kv_unified"})
        self.assertIn("--kv-unified", args)

    def test_load_mode_default_none(self):
        # 默认 none（DEFAULTS）→ 传 --load-mode none
        args = llama_cpp._build_args(Path("m.gguf"), {})
        i = args.index("--load-mode")
        self.assertEqual(args[i + 1], "none")
        # 空（自动）→ 不传
        args = llama_cpp._build_args(Path("m.gguf"), {"load_mode": ""})
        self.assertNotIn("--load-mode", args)
        # 大写归一 + 组合模式 + 非法值忽略
        args = llama_cpp._build_args(Path("m.gguf"), {"load_mode": "MLOCK"})
        i = args.index("--load-mode")
        self.assertEqual(args[i + 1], "mlock")
        args = llama_cpp._build_args(Path("m.gguf"), {"load_mode": "mmap+mlock"})
        i = args.index("--load-mode")
        self.assertEqual(args[i + 1], "mmap+mlock")
        args = llama_cpp._build_args(Path("m.gguf"), {"load_mode": "bogus"})
        self.assertNotIn("--load-mode", args)

    def test_image_tokens(self):
        # 默认自动（空）→ 不传
        args = llama_cpp._build_args(Path("m.gguf"), {})
        self.assertNotIn("--image-min-tokens", args)
        self.assertNotIn("--image-max-tokens", args)
        # 具体值 → 传参
        args = llama_cpp._build_args(Path("m.gguf"),
                                     {"image_min_tokens": "70", "image_max_tokens": "140"})
        i = args.index("--image-min-tokens")
        self.assertEqual(args[i + 1], "70")
        i = args.index("--image-max-tokens")
        self.assertEqual(args[i + 1], "140")
        # 非数字（自动之外的非法值）忽略
        args = llama_cpp._build_args(Path("m.gguf"),
                                     {"image_min_tokens": "abc", "image_max_tokens": "1e3"})
        self.assertNotIn("--image-min-tokens", args)
        self.assertNotIn("--image-max-tokens", args)

    def test_extra_args_flat_and_nested(self):
        # 平铺列表 → 原样逐项传
        args = llama_cpp._build_args(Path("m.gguf"), {"extra_args": ["-ts", "1,2"]})
        self.assertIn("-ts", args)
        self.assertIn("1,2", args)
        # 嵌套列表（前端分组存储）→ 展平后逐项传
        args = llama_cpp._build_args(Path("m.gguf"),
                                     {"extra_args": [["-ts", "1,2"], ["--temp", "0.7"]]})
        i = args.index("-ts"); self.assertEqual(args[i + 1], "1,2")
        i = args.index("--temp"); self.assertEqual(args[i + 1], "0.7")
        # 字符串 → 按空格拆分
        args = llama_cpp._build_args(Path("m.gguf"), {"extra_args": "-ts 1,2"})
        self.assertIn("-ts", args)
        self.assertIn("1,2", args)

    def test_mmproj_cpu_offload(self):
        # 无 mmproj → 不传 --mmproj 与 --no-mmproj-offload
        args = llama_cpp._build_args(Path("m.gguf"), {"no_mmproj_offload": True})
        self.assertNotIn("--mmproj", args)
        self.assertNotIn("--no-mmproj-offload", args)
        # 有 mmproj + 默认关闭 → 只传 --mmproj
        args = llama_cpp._build_args(Path("m.gguf"), {"mmproj": "C:/p.gguf"})
        self.assertIn("--mmproj", args)
        self.assertNotIn("--no-mmproj-offload", args)
        # 有 mmproj + 开启 → 附加 --no-mmproj-offload
        args = llama_cpp._build_args(Path("m.gguf"),
                                     {"mmproj": "C:/p.gguf", "no_mmproj_offload": True})
        self.assertIn("--mmproj", args)
        self.assertIn("--no-mmproj-offload", args)

    def test_flash_attn_no_longer_passed(self):
        # 新版 llama.cpp Flash Attention 默认开启：任何情况都不显式传 --flash-attn
        args = llama_cpp._build_args(Path("m.gguf"), {})
        self.assertNotIn("--flash-attn", args)
        args = llama_cpp._build_args(Path("m.gguf"), {"kv_quant": "q8_0"})
        self.assertIn("--cache-type-k", args)  # 量化参数与 FA 无关，照常传入


class TestLaunchMmprojAuto(unittest.TestCase):
    """launch()：多模态三态——mmproj 未显式指定时按 mmproj_auto（默认开）自动检测。"""

    def setUp(self):
        self.proc = mock.MagicMock()
        self.proc.pid = 4321
        self.proc.poll.return_value = None
        self._patches = [
            mock.patch("pathlib.Path.is_file", return_value=True),
            mock.patch("gui_app.llama_cpp.subprocess.Popen", return_value=self.proc),
            mock.patch("gui_app.llama_cpp.register_subprocess"),
            mock.patch("gui_app.llama_cpp.unregister_subprocess"),
            mock.patch("gui_app.llama_cpp._wait_for_health", return_value=True),
            mock.patch("gui_app.llama_cpp._reader_thread"),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._stop_patches)

    def _stop_patches(self):
        for p in self._patches:
            p.stop()
        llama_cpp._llama_proc = None  # 重置启动进程，避免影响后续测试

    def _launch_cmd(self, params):
        r = llama_cpp.launch("C:/models/llama.gguf", params, log_fn=lambda m: None)
        self.assertTrue(r["ok"], r.get("error"))
        return llama_cpp.subprocess.Popen.call_args[0][0]

    def test_auto_detect_when_unset(self):
        # 未显式指定 mmproj（mmproj_auto 默认开）→ 自动检测并传入 --mmproj
        with mock.patch("gui_app.llama_cpp.scan_mmprojs",
                        return_value=["C:/models/mmproj-qwen2.5vl.gguf"]) as sc, \
             mock.patch("gui_app.llama_cpp._pick_mmproj",
                        return_value="C:/models/mmproj-qwen2.5vl.gguf") as pk:
            cmd = self._launch_cmd({})
        sc.assert_called_once()
        pk.assert_called_once()
        i = cmd.index("--mmproj")
        self.assertEqual(cmd[i + 1], "C:/models/mmproj-qwen2.5vl.gguf")

    def test_no_auto_detect_when_disabled(self):
        # mmproj_auto=False（前端「不使用」）→ 不自动检测，不传 --mmproj
        with mock.patch("gui_app.llama_cpp.scan_mmprojs") as sc, \
             mock.patch("gui_app.llama_cpp._pick_mmproj") as pk:
            cmd = self._launch_cmd({"mmproj_auto": False})
        sc.assert_not_called()
        pk.assert_not_called()
        self.assertNotIn("--mmproj", cmd)

    def test_explicit_mmproj_wins(self):
        # 显式指定投影文件 → 跳过自动检测，用显式路径
        with mock.patch("gui_app.llama_cpp.scan_mmprojs") as sc, \
             mock.patch("gui_app.llama_cpp._pick_mmproj") as pk:
            cmd = self._launch_cmd({"mmproj": "C:/other.gguf", "mmproj_auto": True})
        sc.assert_not_called()
        pk.assert_not_called()
        i = cmd.index("--mmproj")
        self.assertEqual(cmd[i + 1], "C:/other.gguf")

    def test_no_mmproj_found_omits_arg(self):
        # 自动检测但同目录无投影文件 → 不传 --mmproj
        with mock.patch("gui_app.llama_cpp.scan_mmprojs", return_value=[]) as sc:
            cmd = self._launch_cmd({})
        sc.assert_called_once()
        self.assertNotIn("--mmproj", cmd)


if __name__ == "__main__":
    unittest.main()
