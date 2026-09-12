"""workspace_service.remove_history_for_path 单测：去重移动后按 new_path 清理 history 条目。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gui_app.workspace_store as ws
from gui_app import discovery
from gui_app.workspace_service import remove_history_for_path


class _ServiceTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for target, name in [
            ("gui_app.workspace_service.NFO_DIR", "nfo"),
            ("gui_app.workspace_store.HISTORY_FILE", "history.json"),
        ]:
            p = mock.patch(target, self.root / name)
            p.start()
            self.addCleanup(p.stop)
        self._invalidate = mock.patch.object(discovery, "invalidate_thumbnail")
        self._invalidate.start()
        self.addCleanup(self._invalidate.stop)
        ws._hist_cache = None
        ws._hist_mtime = None
        self.addCleanup(self._tmp.cleanup)

    def _seed(self, **entry):
        data = {"entries": [entry]}
        (self.root / "history.json").write_text(json.dumps(data), encoding="utf-8")


class TestRemoveHistoryForPath(_ServiceTestBase):
    def test_removes_entry_by_new_path(self):
        self._seed(id="vid1", original_path=r"E:\lib\orig.mp4",
                   new_path=r"E:\lib\new.mp4", status="ok")
        nfo = self.root / "nfo" / "vid1.nfo"
        nfo.parent.mkdir()
        nfo.write_text("<movie/>", encoding="utf-8")
        self.assertTrue(remove_history_for_path(r"E:\lib\new.mp4"))
        self.assertEqual(ws.load_history()["entries"], [])
        discovery.invalidate_thumbnail.assert_called_once_with("vid1")
        self.assertFalse(nfo.exists())

    def test_original_path_copy_does_not_delete_entry(self):
        # 去重移除的是 original_path 上的待处理副本
        self._seed(id="vid1", original_path=r"E:\lib\orig.mp4",
                   new_path=r"E:\lib\new.mp4", status="ok")
        self.assertFalse(remove_history_for_path(r"E:\lib\orig.mp4"))
        entries = ws.load_history()["entries"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], "vid1")
        discovery.invalidate_thumbnail.assert_not_called()

    def test_skipped_entry_matched_via_new_path(self):
        # skipped 条目 new_path == original_path
        self._seed(id="vid2", original_path=r"E:\lib\same.mp4",
                   new_path=r"E:\lib\same.mp4", status="skipped")
        self.assertTrue(remove_history_for_path(r"E:\lib\same.mp4"))
        self.assertEqual(ws.load_history()["entries"], [])

    def test_no_match_returns_false(self):
        self._seed(id="vid1", original_path=r"E:\lib\orig.mp4",
                   new_path=r"E:\lib\new.mp4", status="ok")
        self.assertFalse(remove_history_for_path(r"E:\elsewhere\x.mp4"))
        self.assertEqual(len(ws.load_history()["entries"]), 1)


if __name__ == "__main__":
    unittest.main()
