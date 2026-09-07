import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import tempfile
import unittest
from unittest import mock

from gui_app import prompts
from gui_app.prompts import normalize_priority_items, load_priority_tags, save_priority_tags


class TestNormalizePriorityItems(unittest.TestCase):
    def test_filter_and_strip(self):
        items = [
            {"keyword": " a ", "description": " d "},
            {"keyword": ""},
            "x",
            {"description": "no kw"},
        ]
        self.assertEqual(normalize_priority_items(items),
                         [{"keyword": "a", "description": "d"}])

    def test_non_list(self):
        self.assertEqual(normalize_priority_items("nope"), [])


class TestSaveLoadRoundtrip(unittest.TestCase):
    """save → 磁盘 → load 往返：mode 权威，非法值在写入前钳为 off。"""

    def _roundtrip(self, mode):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pt_file = Path(tmp.name) / "priority_tags.json"
        items = [{"keyword": "海边", "description": "海边的场景"}]
        with mock.patch.object(prompts, "PRIORITY_TAGS_FILE", pt_file):
            save_priority_tags(items, mode)
            disk = json.loads(pt_file.read_text(encoding="utf-8"))
            loaded = load_priority_tags()
        return disk, loaded

    def test_roundtrip_on(self):
        disk, loaded = self._roundtrip("on")
        self.assertEqual(disk["mode"], "on")
        self.assertEqual(loaded["mode"], "on")
        self.assertEqual(loaded["items"],
                         [{"keyword": "海边", "description": "海边的场景"}])

    def test_roundtrip_enhanced(self):
        disk, loaded = self._roundtrip("enhanced")
        self.assertEqual(disk["mode"], "enhanced")
        self.assertEqual(loaded["mode"], "enhanced")

    def test_roundtrip_off(self):
        disk, loaded = self._roundtrip("off")
        self.assertEqual(disk["mode"], "off")
        self.assertEqual(loaded["mode"], "off")

    def test_invalid_mode_saved_as_off(self):
        disk, loaded = self._roundtrip("bogus")
        self.assertEqual(disk["mode"], "off")
        self.assertEqual(loaded["mode"], "off")

    def test_missing_file_defaults_off(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch.object(prompts, "PRIORITY_TAGS_FILE",
                               Path(tmp.name) / "absent.json"):
            self.assertEqual(load_priority_tags(), {"mode": "off", "items": []})

    def test_corrupt_file_defaults_off(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pt_file = Path(tmp.name) / "priority_tags.json"
        pt_file.write_text("not json", encoding="utf-8")
        with mock.patch.object(prompts, "PRIORITY_TAGS_FILE", pt_file):
            self.assertEqual(load_priority_tags(), {"mode": "off", "items": []})


if __name__ == "__main__":
    unittest.main()
