import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import tempfile
import unittest
from unittest import mock

from gui_app import prompts
from gui_app import tagger
from gui_app.prompts import (normalize_term_library_items, load_term_library,
                             save_term_library, normalize_disabled_groups)


class TestNormalizeTagLibraryItems(unittest.TestCase):
    def test_filter_and_strip(self):
        items = [
            {"keyword": " a ", "description": " d ", "related": " r1 ", "group": " g "},
            {"keyword": ""},
            "x",
            {"description": "no kw"},
        ]
        self.assertEqual(normalize_term_library_items(items),
                         [{"keyword": "a", "description": "d", "related": "r1", "group": "g"}])

    def test_missing_fields_default_empty(self):
        self.assertEqual(normalize_term_library_items([{"keyword": "k"}]),
                         [{"keyword": "k", "description": "", "related": "", "group": ""}])

    def test_non_list(self):
        self.assertEqual(normalize_term_library_items("nope"), [])


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
        lib_file = Path(tmp.name) / "term_library.json"
        with mock.patch.object(prompts, "TERM_LIBRARY_FILE", lib_file):
            save_term_library(items, "enhanced", disabled)
            disk = json.loads(lib_file.read_text(encoding="utf-8"))
            loaded = load_term_library()
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
        lib_file = Path(tmp.name) / "term_library.json"
        lib_file.write_text(json.dumps({"mode": "on", "items": [{"keyword": "a"}]}),
                            encoding="utf-8")
        with mock.patch.object(prompts, "TERM_LIBRARY_FILE", lib_file):
            loaded = load_term_library()
        self.assertEqual(loaded["disabled_groups"], [])
        self.assertEqual(loaded["items"][0]["keyword"], "a")


class TestSaveLoadRoundtrip(unittest.TestCase):
    """save → 磁盘 → load 往返：mode 权威，非法值在写入前钳为 off。"""

    def _roundtrip(self, mode):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        lib_file = Path(tmp.name) / "term_library.json"
        items = [{"keyword": "海边", "description": "海边的场景"}]
        with mock.patch.object(prompts, "TERM_LIBRARY_FILE", lib_file):
            save_term_library(items, mode)
            disk = json.loads(lib_file.read_text(encoding="utf-8"))
            loaded = load_term_library()
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
        lib_file = Path(tmp.name) / "term_library.json"
        items = [{"keyword": "4K", "description": "高分辨率",
                  "related": "2160p UHD", "group": "画质"}]
        with mock.patch.object(prompts, "TERM_LIBRARY_FILE", lib_file):
            save_term_library(items, "enhanced")
            loaded = load_term_library()
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
        with mock.patch.object(prompts, "TERM_LIBRARY_FILE",
                               Path(tmp.name) / "absent.json"):
            self.assertEqual(load_term_library(),
                             {"mode": "off", "disabled_groups": [], "items": []})

    def test_corrupt_file_defaults_off(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        lib_file = Path(tmp.name) / "term_library.json"
        lib_file.write_text("not json", encoding="utf-8")
        with mock.patch.object(prompts, "TERM_LIBRARY_FILE", lib_file):
            self.assertEqual(load_term_library(),
                             {"mode": "off", "disabled_groups": [], "items": []})


class TestImportPresetTags(unittest.TestCase):
    """安装尾部的预设导入：按英文名幂等 / keyword 全局撞车跳过 / off 档自动开启。"""

    def _run(self, preset, existing=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name) / "repo"
        repo.mkdir()
        lib_file = Path(tmp.name) / "term_library.json"
        (repo / "preset_terms.json").write_text(
            json.dumps(preset, ensure_ascii=False), encoding="utf-8")
        if existing is not None:
            lib_file.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
        with mock.patch.object(tagger, "_repo_dir", lambda: repo), \
             mock.patch.object(prompts, "TERM_LIBRARY_FILE", lib_file):
            r = tagger.import_preset_terms()
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


class TestTaggerInjection(unittest.TestCase):
    """注入拆分：Hammy-tagger 组置顶（分数序），G 组走查询词；库外丢弃。"""

    LIB = {"hakurei_reimu": {"zh": "灵梦", "desc": "红白", "group": "Hammy-tagger"},
           "touhou": {"zh": "东方", "desc": "", "group": "Hammy-tagger"},
           "school_uniform": {"zh": "制服", "desc": "", "group": "Hammy-tagger-G"},
           "unknown_tag": {"zh": "库外有词条但输出未含", "desc": "", "group": "Hammy-tagger-G"}}
    ENTRY = {"character_tags": [{"name": "hakurei_reimu", "score": 0.9}],
             "ip_tags": [{"name": "touhou", "score": 0.8}],
             "general_tags": [{"name": "school_uniform", "score": 0.7},
                               {"name": "not_in_lib", "score": 0.6}]}

    def test_injection_split(self):
        hits, q = tagger.tagger_injection(self.ENTRY, self.LIB, 15)
        # A 组（角色/IP）置顶注入：分数降序
        self.assertEqual([h["keyword"] for h in hits], ["灵梦", "东方"])
        # G 组走查询词：英文名+中文名；库外词条丢弃
        self.assertEqual(q, "school_uniform 制服")

    def test_injection_c_truncated_after_filter(self):
        hits, q = tagger.tagger_injection(self.ENTRY, self.LIB, 0)
        self.assertEqual(q, "")
        self.assertEqual(len(hits), 2)  # gen_n 只截 C 段，不影响置顶条目

    def test_injection_lib_none_degrades(self):
        hits, q = tagger.tagger_injection(self.ENTRY, None, 15)
        self.assertEqual(hits, [])
        self.assertIn("hakurei_reimu", q)


if __name__ == "__main__":
    unittest.main()
