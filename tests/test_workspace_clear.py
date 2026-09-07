"""clear_workspace_cache 单测：「清除全部」须一并清掉探针缓存与去重指纹缓存。

路径全部 patch 到临时目录，不触碰真实 _workspace。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app import discovery
from gui_app.workspace_service import clear_workspace_cache


class _ClearTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for target, name in [
            ("gui_app.workspace_service.THUMB_DIR", "thumbnails"),
            ("gui_app.workspace_service.NFO_DIR", "nfo"),
            ("gui_app.workspace_service.SIMILAR_CACHE_FILE", "similar_cache.json"),
            ("gui_app.workspace_store.HISTORY_FILE", "history.json"),
            ("gui_app.discovery.PROBE_CACHE_FILE", "probe_cache.json"),
        ]:
            p = mock.patch(target, self.root / name)
            p.start()
            self.addCleanup(p.stop)
        # discovery 探针缓存全局态复原，避免污染同进程其他测试
        def _reset_probe():
            with discovery._probe_lock:
                discovery._probe_mem.clear()
                discovery._probe_dirty = False
                discovery._probe_loaded = False
        self.addCleanup(_reset_probe)
        self.addCleanup(self._tmp.cleanup)

    def _seed_caches(self):
        self.probe_file = self.root / "probe_cache.json"
        self.similar_file = self.root / "similar_cache.json"
        self.probe_file.write_text(json.dumps({"k": {"mtime": 1, "size": 2}}), encoding="utf-8")
        self.similar_file.write_text(json.dumps({"k": {"mode": "fast"}}), encoding="utf-8")
        # 模拟进程内热缓存：内存有条目且待落盘
        with discovery._probe_lock:
            discovery._probe_mem.clear()
            discovery._probe_mem["k"] = {"mtime": 1, "size": 2}
            discovery._probe_dirty = True
            discovery._probe_loaded = True


class TestClearWorkspaceCache(_ClearTestBase):
    def test_clear_all_clears_probe_and_similar_cache(self):
        self._seed_caches()
        # 与前端「清除全部」按钮的调用形态一致
        r = clear_workspace_cache(clear_probe=True, clear_similar=True)
        self.assertTrue(r["ok"])
        self.assertTrue(r["cleared"]["probe"])
        self.assertTrue(r["cleared"]["similar"])
        self.assertEqual(r["cleared"]["history"], 1)
        self.assertFalse(self.probe_file.exists())
        self.assertFalse(self.similar_file.exists())
        # 内存条目一并清空，且不会在后续 flush 时把旧数据写回磁盘
        with discovery._probe_lock:
            self.assertEqual(discovery._probe_mem, {})
            self.assertFalse(discovery._probe_dirty)
        discovery.flush_probe_cache(force=True)
        self.assertFalse(self.probe_file.exists())
        # 已处理记录已重置
        hist = json.loads((self.root / "history.json").read_text(encoding="utf-8"))
        self.assertEqual(hist.get("entries"), [])

    def test_default_flags_keep_probe_and_similar_cache(self):
        self._seed_caches()
        r = clear_workspace_cache(clear_history=False, clear_thumbs=False, clear_nfo=False)
        self.assertFalse(r["cleared"]["probe"])
        self.assertFalse(r["cleared"]["similar"])
        self.assertTrue(self.probe_file.exists())
        self.assertTrue(self.similar_file.exists())
        with discovery._probe_lock:
            self.assertEqual(discovery._probe_mem.get("k"), {"mtime": 1, "size": 2})


if __name__ == "__main__":
    unittest.main()
