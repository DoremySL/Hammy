import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import tempfile
import unittest
from unittest import mock

from gui_app import prompts
from gui_app.prompts import (normalize_priority_items, load_priority_tags,
                             save_priority_tags, normalize_disabled_groups)


class TestNormalizePriorityItems(unittest.TestCase):
    def test_filter_and_strip(self):
        items = [
            {"keyword": " a ", "description": " d ", "related": " r1 , r2 ", "group": " g "},
            {"keyword": ""},
            "x",
            {"description": "no kw"},
        ]
        self.assertEqual(normalize_priority_items(items),
                         [{"keyword": "a", "description": "d", "related": "r1 , r2", "group": "g"}])

    def test_missing_fields_default_empty(self):
        self.assertEqual(normalize_priority_items([{"keyword": "k"}]),
                         [{"keyword": "k", "description": "", "related": "", "group": ""}])

    def test_non_list(self):
        self.assertEqual(normalize_priority_items("nope"), [])


class TestNormalizeDisabledGroups(unittest.TestCase):
    def test_dedup_and_strip(self):
        self.assertEqual(normalize_disabled_groups([" a ", "a", "", None, "b"]),
                         ["a", "", "b"])

    def test_non_list(self):
        self.assertEqual(normalize_disabled_groups("a"), [])
        self.assertEqual(normalize_disabled_groups(None), [])


class TestDisabledGroupsRoundtrip(unittest.TestCase):
    """黑名单往返：空串代表默认分组；残留组名在保存时清理。"""

    def _save_and_load(self, items, disabled):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pt_file = Path(tmp.name) / "priority_tags.json"
        with mock.patch.object(prompts, "PRIORITY_TAGS_FILE", pt_file):
            save_priority_tags(items, "enhanced", disabled)
            disk = json.loads(pt_file.read_text(encoding="utf-8"))
            loaded = load_priority_tags()
        return disk, loaded

    def test_roundtrip_keeps_live_groups(self):
        items = [{"keyword": "a", "group": "画质"}, {"keyword": "b", "group": ""}]
        disk, loaded = self._save_and_load(items, ["画质", ""])
        self.assertEqual(disk["disabled_groups"], ["画质", ""])
        self.assertEqual(loaded["disabled_groups"], ["画质", ""])

    def test_stale_groups_pruned_on_save(self):
        items = [{"keyword": "a", "group": "画质"}]
        disk, loaded = self._save_and_load(items, ["画质", "已删组", ""])
        self.assertEqual(disk["disabled_groups"], ["画质"])
        self.assertEqual(loaded["disabled_groups"], ["画质"])

    def test_default_param_empty(self):
        disk, _ = self._save_and_load([{"keyword": "a", "group": ""}], "on")
        self.assertEqual(disk["disabled_groups"], [])

    def test_load_legacy_file_without_field(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pt_file = Path(tmp.name) / "priority_tags.json"
        pt_file.write_text(json.dumps({"mode": "on", "items": [{"keyword": "a"}]}),
                           encoding="utf-8")
        with mock.patch.object(prompts, "PRIORITY_TAGS_FILE", pt_file):
            loaded = load_priority_tags()
        self.assertEqual(loaded["disabled_groups"], [])
        self.assertEqual(loaded["items"][0]["keyword"], "a")


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
                         [{"keyword": "海边", "description": "海边的场景",
                           "related": "", "group": ""}])

    def test_roundtrip_new_fields(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pt_file = Path(tmp.name) / "priority_tags.json"
        items = [{"keyword": "4K", "description": "高分辨率",
                  "related": "2160p UHD,超清", "group": "画质"}]
        with mock.patch.object(prompts, "PRIORITY_TAGS_FILE", pt_file):
            save_priority_tags(items, "enhanced")
            loaded = load_priority_tags()
        self.assertEqual(loaded["items"], items)

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
            self.assertEqual(load_priority_tags(),
                             {"mode": "off", "disabled_groups": [], "items": []})

    def test_corrupt_file_defaults_off(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pt_file = Path(tmp.name) / "priority_tags.json"
        pt_file.write_text("not json", encoding="utf-8")
        with mock.patch.object(prompts, "PRIORITY_TAGS_FILE", pt_file):
            self.assertEqual(load_priority_tags(),
                             {"mode": "off", "disabled_groups": [], "items": []})


if __name__ == "__main__":
    unittest.main()
