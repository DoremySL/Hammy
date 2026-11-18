import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import tempfile
import unittest
from unittest.mock import patch


class TestWorkspaceStore(unittest.TestCase):
    """测试 workspace_store 的历史去重与批量落盘逻辑。

    通过 patch workspace_paths 中的路径常量，把 I/O 重定向到临时目录。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws_dir = Path(self._tmp.name)
        self.hist_file = self.ws_dir / "history.json"
        self.hist_file.write_text(json.dumps({"entries": []}), encoding="utf-8")

        # patch 路径常量 + 重置模块级缓存
        self._patches = [
            patch("gui_app.workspace_store.HISTORY_FILE", self.hist_file),
        ]
        for p in self._patches:
            p.start()

        # 重置模块级缓存状态
        import gui_app.workspace_store as ws
        ws._hist_cache = None
        ws._hist_mtime = None
        ws._batch_mode = False
        ws._batch_dirty = False
        self.ws = ws

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def _read_disk(self):
        return json.loads(self.hist_file.read_text(encoding="utf-8"))

    def test_append_and_load(self):
        self.ws.append_history_entry({"id": "a1", "original_path": "/v/1.mp4",
                                      "status": "ok"})
        h = self.ws.load_history()
        self.assertEqual(len(h["entries"]), 1)
        self.assertEqual(h["entries"][0]["id"], "a1")

    def test_dedup_same_id_and_path(self):
        self.ws.append_history_entry({"id": "a1", "original_path": "/v/1.mp4",
                                      "status": "ok", "title": "old"})
        self.ws.append_history_entry({"id": "a1", "original_path": "/v/1.mp4",
                                      "status": "ok", "title": "new"})
        h = self.ws.load_history()
        # 同 id+original_path 应覆盖，不重复
        self.assertEqual(len(h["entries"]), 1)
        self.assertEqual(h["entries"][0]["title"], "new")

    def test_different_id_not_deduped(self):
        self.ws.append_history_entry({"id": "a1", "original_path": "/v/1.mp4",
                                      "status": "ok"})
        self.ws.append_history_entry({"id": "a2", "original_path": "/v/1.mp4",
                                      "status": "ok"})
        h = self.ws.load_history()
        self.assertEqual(len(h["entries"]), 2)

    def test_batch_mode_defers_write(self):
        self.ws.begin_batch()
        self.ws.append_history_entry({"id": "b1", "original_path": "/v/2.mp4",
                                      "status": "ok"})
        # 批量模式下磁盘未更新
        disk = self._read_disk()
        self.assertEqual(len(disk["entries"]), 0)
        # 内存中已有
        h = self.ws.load_history()
        self.assertEqual(len(h["entries"]), 1)
        # flush 后落盘
        self.ws.flush_batch()
        disk = self._read_disk()
        self.assertEqual(len(disk["entries"]), 1)

    def test_flush_without_dirty_no_write(self):
        self.ws.begin_batch()
        self.ws.flush_batch()
        disk = self._read_disk()
        self.assertEqual(len(disk["entries"]), 0)

    def test_batch_flush_keep_mode_persists_without_exiting(self):
        # H3 停止检查点语义：keep_mode 落盘后批量模式仍保持，
        # 后续条目继续累积，最终由普通 flush_batch 收尾
        self.ws.begin_batch()
        self.ws.append_history_entry({"id": "k1", "original_path": "/v/1.mp4",
                                      "status": "ok"})
        self.ws.flush_batch(keep_mode=True)
        disk = self._read_disk()
        self.assertEqual(len(disk["entries"]), 1)
        # 仍是批量模式：k2 只进内存，磁盘不动
        self.ws.append_history_entry({"id": "k2", "original_path": "/v/2.mp4",
                                      "status": "ok"})
        disk = self._read_disk()
        self.assertEqual(len(disk["entries"]), 1)
        h = self.ws.load_history()
        self.assertEqual(len(h["entries"]), 2)
        # 最终 flush 退出批量模式并落盘
        self.ws.flush_batch()
        disk = self._read_disk()
        self.assertEqual(len(disk["entries"]), 2)

    def test_remove_history_by_id(self):
        self.ws.append_history_entry({"id": "r1", "original_path": "/v/1.mp4",
                                      "status": "ok"})
        self.ws.remove_history_by_id("r1")
        h = self.ws.load_history()
        self.assertEqual(len(h["entries"]), 0)


if __name__ == "__main__":
    unittest.main()
