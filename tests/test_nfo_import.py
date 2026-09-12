import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

from batch_rename.nfo import _build_nfo_xml, write_nfo
from gui_app.nfo_import import import_from_sources, stable_id


def _make_nfo_xml(status: str = "ok", thumb_time: str = "",
                  original_name: str = "原始.mp4") -> str:
    info = {"duration": 120.0, "size": 1000,
            "video": {"codec": "h264", "width": 1280, "height": 720, "duration": 120.0}}
    if thumb_time:
        info["thumb_time"] = thumb_time
    return _build_nfo_xml("标题", "剧情", ["t1", "t2"], info, original_name, status)


def _write_nfo(tmp: Path, name: str, status: str = "ok",
               thumb_time: str = "", original_name: str = "原始.mp4") -> str:
    xml = _make_nfo_xml(status, thumb_time, original_name)
    (tmp / name).write_text(xml, encoding="utf-8")
    return xml


class TestNfoMarker(unittest.TestCase):
    def test_marker_fields(self):
        root = ET.fromstring(_make_nfo_xml("ok", "00:02:15"))
        hammy = root.find("hammy")
        self.assertIsNotNone(hammy)
        self.assertEqual(hammy.get("version"), "1")
        self.assertEqual(hammy.findtext("status"), "ok")
        float(hammy.findtext("processed_at"))
        self.assertEqual(hammy.findtext("thumb_time"), "00:02:15")

    def test_marker_no_thumb_time(self):
        root = ET.fromstring(_make_nfo_xml("skipped"))
        hammy = root.find("hammy")
        self.assertEqual(hammy.findtext("status"), "skipped")
        self.assertIsNone(hammy.find("thumb_time"))


class TestNfoImport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.src = Path(self._tmp.name) / "src"
        self.src.mkdir()
        self.ws_dir = Path(self._tmp.name) / "ws"
        self.ws_dir.mkdir()
        self.nfo_dir = self.ws_dir / "nfo"
        self.hist_file = self.ws_dir / "history.json"
        self.hist_file.write_text(json.dumps({"entries": []}), encoding="utf-8")
        self._patches = [
            patch("gui_app.workspace_store.HISTORY_FILE", self.hist_file),
            patch("gui_app.nfo_import.NFO_DIR", self.nfo_dir),
        ]
        for p in self._patches:
            p.start()
        import gui_app.workspace_store as ws
        ws._hist_cache = None
        ws._hist_mtime = None
        self.ws = ws

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def _make_video(self, stem: str, ext: str = ".mp4") -> Path:
        v = self.src / (stem + ext)
        v.write_bytes(b"fake")
        return v

    def test_import_roundtrip(self):
        v = self._make_video("video")
        _write_nfo(self.src, "video.nfo", "ok", "00:01:30", "原始.mp4")
        counts = import_from_sources([str(self.src)])
        self.assertEqual(counts["imported"], 1)
        entry = self.ws.load_history()["entries"][0]
        self.assertEqual(entry["id"], stable_id(str(v)))
        self.assertEqual(entry["new_path"], str(v))
        self.assertEqual(entry["original_name"], "原始.mp4")
        self.assertEqual(entry["original_path"], str(self.src / "原始.mp4"))
        self.assertEqual(entry["status"], "ok")
        self.assertEqual(entry["title"], "标题")
        self.assertEqual(entry["tags"], ["t1", "t2"])
        self.assertEqual(entry["info"]["duration"], 120)
        self.assertEqual(entry["info"]["resolution"], "1280x720")
        self.assertEqual(entry["info"]["codec"], "h264")
        self.assertEqual(entry["thumb_time"], "00:01:30")
        self.assertTrue((self.nfo_dir / f"{entry['id']}.nfo").exists())

    def test_reimport_idempotent(self):
        self._make_video("video")
        _write_nfo(self.src, "video.nfo")
        import_from_sources([str(self.src)])
        counts = import_from_sources([str(self.src)])
        self.assertEqual(counts["imported"], 0)
        self.assertEqual(counts["skipped_dup"], 1)
        self.assertEqual(len(self.ws.load_history()["entries"]), 1)

    def test_no_sibling_video(self):
        _write_nfo(self.src, "lonely.nfo")
        counts = import_from_sources([str(self.src)])
        self.assertEqual(counts["skipped_no_video"], 1)
        self.assertEqual(len(self.ws.load_history()["entries"]), 0)

    def test_ambiguous_stem(self):
        self._make_video("video", ".mp4")
        self._make_video("video", ".mkv")
        _write_nfo(self.src, "video.nfo")
        counts = import_from_sources([str(self.src)])
        self.assertEqual(counts["skipped_no_video"], 1)

    def test_plain_nfo_ignored(self):
        self._make_video("video")
        (self.src / "video.nfo").write_text(
            '<?xml version="1.0" encoding="utf-8"?><movie><title>x</title></movie>',
            encoding="utf-8")
        counts = import_from_sources([str(self.src)])
        self.assertEqual(counts["skipped_bad"], 1)
        self.assertEqual(len(self.ws.load_history()["entries"]), 0)

    def test_adhoc_video_sibling_nfo(self):
        v = self._make_video("adhoc")
        _write_nfo(self.src, "adhoc.nfo")
        counts = import_from_sources([str(v)])
        self.assertEqual(counts["imported"], 1)

    def test_existing_new_path_dedup(self):
        v = self._make_video("video")
        self.ws.append_history_entry({
            "id": "other", "original_path": "E:\\elsewhere\\a.mp4",
            "new_path": str(v), "status": "ok",
        })
        _write_nfo(self.src, "video.nfo")
        counts = import_from_sources([str(self.src)])
        self.assertEqual(counts["skipped_dup"], 1)
        self.assertEqual(len(self.ws.load_history()["entries"]), 1)


if __name__ == "__main__":
    unittest.main()
