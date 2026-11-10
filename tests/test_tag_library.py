import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import tempfile
import unittest
from unittest import mock

from gui_app import prompts
from gui_app import tagger
from gui_app.prompts import (normalize_tag_library_items, load_tag_library,
                             save_tag_library, normalize_disabled_groups)


class TestNormalizeTagLibraryItems(unittest.TestCase):
    def test_filter_and_strip(self):
        items = [
            {"keyword": " a ", "description": " d ", "related": " r1 ", "group": " g "},
            {"keyword": ""},
            "x",
            {"description": "no kw"},
        ]
        self.assertEqual(normalize_tag_library_items(items),
                         [{"keyword": "a", "description": "d", "related": "r1", "group": "g"}])

    def test_missing_fields_default_empty(self):
        self.assertEqual(normalize_tag_library_items([{"keyword": "k"}]),
                         [{"keyword": "k", "description": "", "related": "", "group": ""}])

    def test_non_list(self):
        self.assertEqual(normalize_tag_library_items("nope"), [])


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
        lib_file = Path(tmp.name) / "tag_library.json"
        with mock.patch.object(prompts, "TAG_LIBRARY_FILE", lib_file):
            save_tag_library(items, "enhanced", disabled)
            disk = json.loads(lib_file.read_text(encoding="utf-8"))
            loaded = load_tag_library()
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
        lib_file = Path(tmp.name) / "tag_library.json"
        lib_file.write_text(json.dumps({"mode": "on", "items": [{"keyword": "a"}]}),
                            encoding="utf-8")
        with mock.patch.object(prompts, "TAG_LIBRARY_FILE", lib_file):
            loaded = load_tag_library()
        self.assertEqual(loaded["disabled_groups"], [])
        self.assertEqual(loaded["items"][0]["keyword"], "a")


class TestSaveLoadRoundtrip(unittest.TestCase):
    """save → 磁盘 → load 往返：mode 权威，非法值在写入前钳为 off。"""

    def _roundtrip(self, mode):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        lib_file = Path(tmp.name) / "tag_library.json"
        items = [{"keyword": "海边", "description": "海边的场景"}]
        with mock.patch.object(prompts, "TAG_LIBRARY_FILE", lib_file):
            save_tag_library(items, mode)
            disk = json.loads(lib_file.read_text(encoding="utf-8"))
            loaded = load_tag_library()
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
        lib_file = Path(tmp.name) / "tag_library.json"
        items = [{"keyword": "4K", "description": "高分辨率",
                  "related": "2160p UHD", "group": "画质"}]
        with mock.patch.object(prompts, "TAG_LIBRARY_FILE", lib_file):
            save_tag_library(items, "enhanced")
            loaded = load_tag_library()
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
        with mock.patch.object(prompts, "TAG_LIBRARY_FILE",
                               Path(tmp.name) / "absent.json"):
            self.assertEqual(load_tag_library(),
                             {"mode": "off", "disabled_groups": [], "items": []})

    def test_corrupt_file_defaults_off(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        lib_file = Path(tmp.name) / "tag_library.json"
        lib_file.write_text("not json", encoding="utf-8")
        with mock.patch.object(prompts, "TAG_LIBRARY_FILE", lib_file):
            self.assertEqual(load_tag_library(),
                             {"mode": "off", "disabled_groups": [], "items": []})


class TestImportPresetTags(unittest.TestCase):
    """安装尾部的预设导入：按英文名幂等 / keyword 全局撞车跳过 / off 档自动开启。"""

    def _run(self, preset, existing=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name) / "repo"
        repo.mkdir()
        lib_file = Path(tmp.name) / "tag_library.json"
        (repo / "preset_tags.json").write_text(
            json.dumps(preset, ensure_ascii=False), encoding="utf-8")
        if existing is not None:
            lib_file.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
        with mock.patch.object(tagger, "_repo_dir", lambda: repo), \
             mock.patch.object(prompts, "TAG_LIBRARY_FILE", lib_file):
            r = tagger.import_preset_tags()
        disk = json.loads(lib_file.read_text(encoding="utf-8")) if lib_file.is_file() else None
        return r, disk

    def test_off_mode_imports_and_enables(self):
        preset = [{"keyword": "博丽灵梦", "description": "", "related": "hakurei_reimu",
                   "group": "Hammy-tagger"}]
        r, disk = self._run(preset)
        self.assertEqual(r["imported"], 1)
        self.assertEqual(r["mode"], "on")
        self.assertEqual(disk["mode"], "on")
        self.assertEqual(disk["items"][0]["keyword"], "博丽灵梦")
        self.assertEqual(disk["items"][0]["group"], "Hammy-tagger")

    def test_idempotent_and_on_keeps_mode(self):
        preset = [{"keyword": "博丽灵梦", "description": "", "related": "hakurei_reimu",
                   "group": "Hammy-tagger"}]
        existing = {"mode": "on", "disabled_groups": [],
                    "items": [{"keyword": "博丽灵梦", "description": "旧",
                               "related": "hakurei_reimu", "group": "Hammy-tagger"}]}
        r, disk = self._run(preset, existing)
        self.assertEqual(r["imported"], 0)
        self.assertEqual(r["skipped"], 1)
        self.assertEqual(r["mode"], "on")
        self.assertEqual(disk["items"][0]["description"], "旧")  # 已有条目不覆盖

    def test_keyword_conflict_skipped(self):
        preset = [{"keyword": "水手服", "description": "", "related": "sailor_fuku",
                   "group": "Hammy-tagger"}]
        existing = {"mode": "enhanced", "disabled_groups": [],
                    "items": [{"keyword": "水手服", "description": "自建",
                               "related": "", "group": "服装"}]}
        r, disk = self._run(preset, existing)
        self.assertEqual(r["imported"], 0)
        self.assertEqual(r["skipped"], 1)
        self.assertEqual(r["mode"], "enhanced")  # 非关闭档不动
        self.assertEqual(len(disk["items"]), 1)  # 无新增

    def test_off_mode_duplicate_still_enables(self):
        preset = [{"keyword": "博丽灵梦", "description": "", "related": "hakurei_reimu",
                   "group": "Hammy-tagger"}]
        existing = {"mode": "off", "disabled_groups": [],
                    "items": [{"keyword": "博丽灵梦", "description": "d",
                               "related": "hakurei_reimu", "group": "Hammy-tagger"}]}
        r, disk = self._run(preset, existing)
        self.assertEqual(r["imported"], 0)
        self.assertEqual(r["mode"], "on")
        self.assertEqual(disk["mode"], "on")


if __name__ == "__main__":
    unittest.main()
