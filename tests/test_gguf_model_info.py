"""GGUF 头部模型信息提取（架构前缀键 / MoE>1 / 大端 / 分片过滤）。

覆盖：_extract_model_info（架构前缀优先、回退 llama.*、MoE 判定专家数 > 1）、
_read_gguf_meta（读全部 KV 与大端支持）、scan_models 分片过滤。
"""
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app import llama_cpp


# ── 最小 GGUF 文件构造（小端 / 大端）──

def _str(v: str, e: str = "<") -> bytes:
    b = v.encode("utf-8")
    return struct.pack(e + "Q", len(b)) + b


def _u32(v: int, e: str = "<") -> bytes:
    return struct.pack(e + "I", v)


def _u64(v: int, e: str = "<") -> bytes:
    return struct.pack(e + "Q", v)


def _gguf_bytes(pairs, version: int = 3, e: str = "<") -> bytes:
    """构造最小 GGUF 文件字节。

    pairs: [(key, value_type_id, payload_bytes)]，按顺序写入 KV 区。
    """
    out = bytearray(b"GGUF")
    out += struct.pack(e + "I", version)
    out += struct.pack(e + "Q", 0)            # tensor_count
    out += struct.pack(e + "Q", len(pairs))   # metadata_kv_count
    for key, vt, payload in pairs:
        kb = key.encode("utf-8")
        out += struct.pack(e + "Q", len(kb)) + kb
        out += struct.pack(e + "I", vt)
        out += payload
    return bytes(out)


def _gguf_bytes_be(pairs) -> bytes:
    """大端 GGUF（规范版本 1/2/3 之外，极少见）。"""
    return _gguf_bytes(pairs, version=5, e=">")


class TestExtractModelInfo(unittest.TestCase):
    """纯函数：架构前缀键优先、回退 llama.*、MoE 判定专家数 > 1。"""

    def test_arch_prefix_keys(self):
        meta = {"general.architecture": "qwen2",
                "qwen2.context_length": 131072, "qwen2.block_count": 80,
                "qwen2.expert_count": 64}
        info = llama_cpp._extract_model_info(meta)
        self.assertEqual(info, {"ctx": 131072, "layers": 80, "experts": 64, "moe": True})

    def test_arch_prefix_fallback_llama(self):
        # 架构前缀键缺失时回退 llama.*
        meta = {"general.architecture": "qwen2", "qwen2.context_length": 32768,
                "llama.context_length": 1024, "llama.block_count": 16}
        info = llama_cpp._extract_model_info(meta)
        self.assertEqual(info["ctx"], 32768)   # 架构键优先
        self.assertEqual(info["layers"], 16)   # 架构键缺失回退 llama

    def test_llama_fallback_keys(self):
        meta = {"general.architecture": "llama",
                "llama.context_length": 8192, "llama.block_count": 32}
        info = llama_cpp._extract_model_info(meta)
        self.assertEqual(info["ctx"], 8192)
        self.assertEqual(info["layers"], 32)
        self.assertFalse(info["moe"])

    def test_no_arch_key(self):
        # 无 general.architecture 的旧文件也按 llama.* 读取
        meta = {"llama.context_length": 4096}
        info = llama_cpp._extract_model_info(meta)
        self.assertEqual(info["ctx"], 4096)

    def test_arch_as_list(self):
        meta = {"general.architecture": ["deepseek2"],
                "deepseek2.context_length": 163840, "deepseek2.block_count": 61,
                "deepseek2.expert_count": 256}
        info = llama_cpp._extract_model_info(meta)
        self.assertEqual(info["ctx"], 163840)
        self.assertTrue(info["moe"])

    def test_one_expert_not_moe(self):
        # 专家数 1 不算 MoE（判定标准为专家数 > 1）
        meta = {"general.architecture": "qwen2", "qwen2.expert_count": 1}
        info = llama_cpp._extract_model_info(meta)
        self.assertFalse(info["moe"])
        self.assertEqual(info["experts"], 1)

    def test_zero_expert_not_moe(self):
        meta = {"llama.expert_count": 0}
        info = llama_cpp._extract_model_info(meta)
        self.assertFalse(info["moe"])

    def test_empty_meta(self):
        info = llama_cpp._extract_model_info({})
        self.assertEqual(info, {"ctx": 0, "layers": 0, "experts": 0, "moe": False})

    def test_unparseable_value(self):
        meta = {"general.architecture": "llama", "llama.context_length": "很多"}
        info = llama_cpp._extract_model_info(meta)
        self.assertEqual(info["ctx"], 0)

    def test_list_value_skipped(self):
        # 异常文件：值为数组时跳过该键而不是报错
        meta = {"general.architecture": "llama", "llama.context_length": [32768]}
        info = llama_cpp._extract_model_info(meta)
        self.assertEqual(info["ctx"], 0)


class TestReadGgufMetaFullKv(unittest.TestCase):
    def test_context_after_expert_keys(self):
        # 回归：旧实现读齐 expert 两个键后提前 break，其后的 context/block 读不到
        data = _gguf_bytes([
            ("llama.expert_count", 4, _u32(8)),
            ("llama.expert_used_count", 4, _u32(2)),
            ("llama.context_length", 4, _u32(32768)),
            ("llama.block_count", 4, _u32(64)),
        ])
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "moe.gguf"
            p.write_bytes(data)
            meta = llama_cpp._read_gguf_meta(p)
        self.assertEqual(meta["llama.context_length"], 32768)
        self.assertEqual(meta["llama.block_count"], 64)
        self.assertEqual(meta["llama.expert_count"], 8)

    def test_big_endian_file(self):
        # 规范版本 1/2/3 之外（极少见）按大端解析，同样读全部 KV
        data = _gguf_bytes_be([
            ("general.architecture", 8, _str("qwen2", ">")),
            ("qwen2.context_length", 4, _u32(131072, ">")),
            ("qwen2.block_count", 4, _u32(80, ">")),
            ("qwen2.expert_count", 4, _u32(64, ">")),
        ])
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "moe-be.gguf"
            p.write_bytes(data)
            meta = llama_cpp._read_gguf_meta(p)
        self.assertEqual(meta["general.architecture"], "qwen2")
        self.assertEqual(meta["qwen2.context_length"], 131072)
        self.assertEqual(meta["qwen2.block_count"], 80)
        self.assertEqual(meta["qwen2.expert_count"], 64)

    def test_big_endian_uint64_expert(self):
        data = _gguf_bytes_be([
            ("llama.expert_count", 10, _u64(8, ">")),
            ("llama.context_length", 4, _u32(32768, ">")),
        ])
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "moe-be64.gguf"
            p.write_bytes(data)
            meta = llama_cpp._read_gguf_meta(p)
        self.assertEqual(meta["llama.expert_count"], 8)
        self.assertEqual(meta["llama.context_length"], 32768)


class TestIsFirstShard(unittest.TestCase):
    def test_non_sharded(self):
        self.assertTrue(llama_cpp._is_first_shard("model.gguf"))
        self.assertTrue(llama_cpp._is_first_shard("model-q4_K_M.gguf"))
        self.assertTrue(llama_cpp._is_first_shard("llama-2-7b.gguf"))

    def test_first_shard_kept(self):
        self.assertTrue(llama_cpp._is_first_shard("model-00001-of-00003.gguf"))
        self.assertTrue(llama_cpp._is_first_shard("model-00001-of-00001.gguf"))

    def test_later_shards_skipped(self):
        self.assertFalse(llama_cpp._is_first_shard("model-00002-of-00003.gguf"))
        self.assertFalse(llama_cpp._is_first_shard("model-00010-of-00010.gguf"))


class TestScanModelsShards(unittest.TestCase):
    def setUp(self):
        # 扫描缓存命中/落盘由 test_llama_scan_cache.py 专门覆盖；此处禁写防污染
        p = mock.patch("gui_app.llama_cpp._save_scan_cache")
        p.start()
        self.addCleanup(p.stop)

    def test_only_first_shard_listed(self):
        with tempfile.TemporaryDirectory() as td:
            # 分片模型：仅第 1 片含元数据
            (Path(td) / "model-00001-of-00002.gguf").write_bytes(_gguf_bytes([
                ("general.architecture", 8, _str("qwen2")),
                ("qwen2.context_length", 4, _u32(32768)),
                ("qwen2.block_count", 4, _u32(64)),
                ("qwen2.expert_count", 4, _u32(8)),
            ]))
            (Path(td) / "model-00002-of-00002.gguf").write_bytes(_gguf_bytes([
                ("general.name", 8, _str("shard2")),
            ]))
            (Path(td) / "plain.gguf").write_bytes(_gguf_bytes([
                ("general.name", 8, _str("x")),
            ]))
            models = llama_cpp.scan_models(Path(td))
        by_name = {m["name"]: m for m in models}
        self.assertEqual(set(by_name), {"model-00001-of-00002.gguf", "plain.gguf"})
        self.assertEqual(by_name["model-00001-of-00002.gguf"]["ctx"], 32768)
        self.assertEqual(by_name["model-00001-of-00002.gguf"]["layers"], 64)
        self.assertTrue(by_name["model-00001-of-00002.gguf"]["moe"])
        self.assertEqual(by_name["model-00001-of-00002.gguf"]["expert_count"], 8)

    def test_missing_keys_default_zero(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "m.gguf").write_bytes(_gguf_bytes([("general.name", 8, _str("x"))]))
            models = llama_cpp.scan_models(Path(td))
        self.assertEqual(models[0]["ctx"], 0)
        self.assertEqual(models[0]["layers"], 0)
        self.assertFalse(models[0]["moe"])

    def test_arch_prefix_keys_in_scan(self):
        # 非 llama 架构的 ctx/layers 也能进 scan 结果
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "deepseek.gguf").write_bytes(_gguf_bytes([
                ("general.architecture", 8, _str("deepseek2")),
                ("deepseek2.context_length", 4, _u32(163840)),
                ("deepseek2.block_count", 4, _u32(61)),
                ("deepseek2.expert_count", 4, _u32(256)),
            ]))
            models = llama_cpp.scan_models(Path(td))
        self.assertEqual(models[0]["ctx"], 163840)
        self.assertEqual(models[0]["layers"], 61)
        self.assertTrue(models[0]["moe"])


if __name__ == "__main__":
    unittest.main()
