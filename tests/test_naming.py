import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import re
import tempfile
import unittest
from pathlib import Path

from batch_rename.naming import (
    sanitize_filename, resolve_collision, extract_date_str, build_new_stem,
    apply_manual_transform, strip_sub_lang, match_subtitle_files,
)
from batch_rename.utils import to_long_path
from batch_rename.config import Config


class TestSanitizeFilename(unittest.TestCase):
    def test_whitespace_to_underscore(self):
        self.assertEqual(sanitize_filename("hello world"), "hello_world")

    def test_illegal_chars_replaced(self):
        self.assertEqual(sanitize_filename("a<b>c"), "a_b_c")
        self.assertEqual(sanitize_filename("a/b:c"), "a_b_c")

    def test_consecutive_underscores_merged(self):
        self.assertEqual(sanitize_filename("a__b"), "a_b")

    def test_strip_edges(self):
        self.assertEqual(sanitize_filename("  spaced  out  "), "spaced_out")

    def test_empty_fallback(self):
        self.assertEqual(sanitize_filename(""), "untitled")
        self.assertEqual(sanitize_filename("///"), "untitled")

    def test_truncate(self):
        self.assertEqual(len(sanitize_filename("a" * 100, 50)), 50)

    def test_normal_unchanged(self):
        self.assertEqual(sanitize_filename("normal-title_123"), "normal-title_123")


class TestResolveCollision(unittest.TestCase):
    def test_ok_when_free(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            target, status = resolve_collision(dest, "stem", ".mp4", "src")
            self.assertEqual(status, "ok")
            self.assertEqual(target, dest / "stem.mp4")

    def test_collision_appends_counter(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            (dest / "stem.mp4").write_text("x")
            target, status = resolve_collision(dest, "stem", ".mp4", "other_src")
            self.assertEqual(status, "ok")
            self.assertEqual(target, dest / "stem_1.mp4")

    def test_skipped_same_path(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            existing = dest / "stem.mp4"
            existing.write_text("x")
            src_long = to_long_path(str(existing))
            target, status = resolve_collision(dest, "stem", ".mp4", src_long)
            self.assertEqual(status, "skipped")


class TestExtractDateStr(unittest.TestCase):
    def test_format_pattern(self):
        s = extract_date_str("nonexistent_video.mp4", "")
        self.assertRegex(s, re.compile(r"^\d{8}-\d{4}$"))

    def test_from_creation_time(self):
        s = extract_date_str("nonexistent_video.mp4", "2023-05-10T12:30:00")
        self.assertRegex(s, re.compile(r"^\d{8}-\d{4}$"))


class TestBuildNewStem(unittest.TestCase):
    def test_title_only(self):
        cfg = Config(include_date=False, include_original=False)
        self.assertEqual(build_new_stem("x.mp4", {}, "My Title", cfg), "My_Title")

    def test_with_date(self):
        cfg = Config(include_date=True, include_original=False)
        stem = build_new_stem("x.mp4", {"creation_time": "2023-05-10T12:30:00"}, "My Title", cfg)
        # 日期前缀 + 标题
        self.assertRegex(stem, r"^\d{8}-\d{4}_My_Title$")

    def test_with_original(self):
        cfg = Config(include_date=False, include_original=True)
        self.assertEqual(build_new_stem("original_name.mp4", {}, "My Title", cfg),
                         "My_Title_original_name")


class TestManualTransform(unittest.TestCase):
    def test_prefix_suffix(self):
        self.assertEqual(apply_manual_transform("abc", "prefix", "pre_"),
                         ("pre_abc", ""))
        self.assertEqual(apply_manual_transform("abc", "suffix", "_x"),
                         ("abc_x", ""))

    def test_remove_replace_literal(self):
        self.assertEqual(apply_manual_transform("a_b_c", "remove", "_"),
                         ("abc", ""))
        self.assertEqual(apply_manual_transform("a_b", "replace", "_", "-"),
                         ("a-b", ""))

    def test_remove_regex(self):
        stem, err = apply_manual_transform("a1b22c", "remove", r"\d+", use_regex=True)
        self.assertEqual((stem, err), ("abc", ""))

    def test_replace_regex_backref(self):
        stem, err = apply_manual_transform("abc_第1集", "replace",
                                           r"第(\d+)集", r"S\1", use_regex=True)
        self.assertEqual((stem, err), ("abc_S1", ""))

    def test_literal_dot_is_literal(self):
        # 字面模式下 "." 是普通字符，正则模式下才匹配任意字符
        self.assertEqual(apply_manual_transform("a.b", "replace", ".", "_"),
                         ("a_b", ""))
        self.assertEqual(apply_manual_transform("a.b", "replace", ".", "x", use_regex=True),
                         ("xxx", ""))

    def test_invalid_regex(self):
        stem, err = apply_manual_transform("abc", "remove", "([", use_regex=True)
        self.assertEqual(stem, "abc")
        self.assertTrue(err.startswith("正则表达式错误"))

    def test_empty_input_rejected(self):
        self.assertEqual(apply_manual_transform("abc", "remove", " "),
                         ("abc", "请输入要删除的文本"))
        # 空查找的 replace 合法（= 不替换，配合大小写变换整体处理）
        self.assertEqual(apply_manual_transform("abc", "replace", " "),
                         ("abc", ""))

    def test_empty_result_rejected(self):
        self.assertEqual(apply_manual_transform("abc", "remove", "abc"),
                         ("abc", "重命名结果为空"))


if __name__ == "__main__":
    unittest.main()

class TestCounterTemplate(unittest.TestCase):
    """${...} 编号占位符展开（apply_manual_transform 的 counter 参数）。"""

    def test_shorthand_n(self):
        stem, err = apply_manual_transform("第1集", "replace", r"第(\d+)集",
                                           r"S1E${n}", True, 1)
        self.assertEqual((stem, err), ("S1E01", ""))

    def test_sequential_counting(self):
        for i, n in enumerate(("S1E01", "S1E02", "S1E03"), start=1):
            stem, _ = apply_manual_transform("第x集", "replace", r"第x集",
                                             r"S1E${n}", True, i)
            self.assertEqual(stem, n)

    def test_padding_start_custom(self):
        stem, _ = apply_manual_transform("第x集", "replace", r"第x集",
                                         r"${padding=3;start=0}", True, 10)
        self.assertEqual(stem, "009")

    def test_partial_keys_default(self):
        stem, _ = apply_manual_transform("第x集", "replace", r"第x集",
                                         r"${start=100}", True, 1)
        self.assertEqual(stem, "100")

    def test_backref_plus_counter(self):
        stem, err = apply_manual_transform("第1集abc", "replace", r"第(\d+)集",
                                           r"S\1E${n} ", True, 7)
        self.assertEqual((stem, err), ("S1E07 abc", ""))

    def test_counter_none_no_expand(self):
        stem, err = apply_manual_transform("第1集", "replace", r"第(\d+)集",
                                           r"S1E${n} ", True)
        self.assertEqual((stem, err), ("S1E${n}", ""))

    def test_invalid_spec_uses_defaults(self):
        stem, _ = apply_manual_transform("第x集", "replace", r"第x集",
                                         r"${padding=ab;start=x}", True, 2)
        self.assertEqual(stem, "02")

class TestCounterTemplate(unittest.TestCase):
    """${...} 编号占位符展开（apply_manual_transform 的 counter 参数）。"""

    def test_shorthand_n(self):
        stem, err = apply_manual_transform("第1集", "replace", r"第(\d+)集",
                                           r"S1E${n}", True, 1)
        self.assertEqual((stem, err), ("S1E01", ""))

    def test_sequential_counting(self):
        for i, n in enumerate(("S1E01", "S1E02", "S1E03"), start=1):
            stem, _ = apply_manual_transform("第x集", "replace", r"第x集",
                                             r"S1E${n}", True, i)
            self.assertEqual(stem, n)

    def test_padding_start_custom(self):
        stem, _ = apply_manual_transform("第x集", "replace", r"第x集",
                                         r"${padding=3;start=0}", True, 10)
        self.assertEqual(stem, "009")

    def test_partial_keys_default(self):
        stem, _ = apply_manual_transform("第x集", "replace", r"第x集",
                                         r"${start=100}", True, 1)
        self.assertEqual(stem, "100")

    def test_backref_plus_counter(self):
        stem, err = apply_manual_transform("第1集abc", "replace", r"第(\d+)集",
                                           r"S\1E${n} ", True, 7)
        self.assertEqual((stem, err), ("S1E07 abc", ""))

    def test_counter_none_no_expand(self):
        stem, err = apply_manual_transform("第1集", "replace", r"第(\d+)集",
                                           r"S1E${n} ", True)
        self.assertEqual((stem, err), ("S1E${n}", ""))

    def test_invalid_spec_uses_defaults(self):
        stem, _ = apply_manual_transform("第x集", "replace", r"第x集",
                                         r"${padding=ab;start=x}", True, 2)
        self.assertEqual(stem, "02")

class TestTransformOptions(unittest.TestCase):
    """increment 键 / match_all / case_mode（对齐 PowerRename 的选项）。"""

    def test_increment_key(self):
        stem, _ = apply_manual_transform("x", "replace", "x", "${increment=2}", True, 2)
        self.assertEqual(stem, "03")

    def test_negative_start_increment(self):
        stem, _ = apply_manual_transform("x", "replace", "x",
                                         "${start=10;increment=-1}", True, 3)
        self.assertEqual(stem, "08")

    def test_match_all_false_regex(self):
        stem, _ = apply_manual_transform("a1b2c", "replace", r"\d", "X", True, None, False)
        self.assertEqual(stem, "aXb2c")

    def test_match_all_true_regex(self):
        stem, _ = apply_manual_transform("a1b2c", "replace", r"\d", "X", True, None, True)
        self.assertEqual(stem, "aXbXc")

    def test_match_all_false_literal(self):
        stem, _ = apply_manual_transform("aaa", "replace", "a", "X", False, None, False)
        self.assertEqual(stem, "Xaa")

    def test_case_upper(self):
        stem, _ = apply_manual_transform("abc", "replace", "abc", "def",
                                         False, None, True, "upper")
        self.assertEqual(stem, "DEF")

    def test_case_lower_empty_lookup(self):
        stem, _ = apply_manual_transform("ABC DEF", "replace", "",
                                         "", False, None, True, "lower")
        self.assertEqual(stem, "abc def")

    def test_case_capitalized(self):
        stem, _ = apply_manual_transform("hello world", "replace", "",
                                         "", False, None, True, "capitalized")
        self.assertEqual(stem, "Hello World")

    def test_case_title(self):
        stem, _ = apply_manual_transform("S1E01 hello world", "replace", "",
                                         "", False, None, True, "title")
        self.assertEqual(stem, "S1e01 Hello World")

    def test_empty_lookup_no_case_no_change(self):
        stem, err = apply_manual_transform("abc", "replace", "", "")
        self.assertEqual((stem, err), ("abc", ""))


class TestStripSubLang(unittest.TestCase):
    def test_common_tags(self):
        self.assertEqual(strip_sub_lang("01.简体"), ("01", "简体"))
        self.assertEqual(strip_sub_lang("01.zh"), ("01", "zh"))
        self.assertEqual(strip_sub_lang("01.big5"), ("01", "big5"))
        self.assertEqual(strip_sub_lang("01.zh-hans"), ("01", "zh-hans"))

    def test_traditional_and_gb_tags(self):
        self.assertEqual(strip_sub_lang("01.繁體"), ("01", "繁體"))
        self.assertEqual(strip_sub_lang("01.簡體"), ("01", "簡體"))
        self.assertEqual(strip_sub_lang("01.GB"), ("01", "GB"))
        self.assertEqual(strip_sub_lang("01.gb2312"), ("01", "gb2312"))
        self.assertEqual(strip_sub_lang("01.简"), ("01", "简"))

    def test_no_tag(self):
        self.assertEqual(strip_sub_lang("01"), ("01", ""))

    def test_tag_without_dot(self):
        self.assertEqual(strip_sub_lang("01简体"), ("01简体", ""))

    def test_multiple_layers(self):
        self.assertEqual(strip_sub_lang("a.zh.zh"), ("a", "zh.zh"))

    def test_long_tag_priority(self):
        # zh-hans 优先于 zh 匹配
        self.assertEqual(strip_sub_lang("a.zh-hans"), ("a", "zh-hans"))


class TestMatchSubtitleFiles(unittest.TestCase):
    def _names(self, pairs, v):
        """返回视频 v 配到的字幕文件名列表（按配到顺序）。"""
        out = []
        for vi, sl in pairs:
            if vi == v:
                out.extend(Path(sp).name for sp, _ in sl)
        return out

    def test_l1_exact_name_any_order(self):
        # L1 同名优先，与传入顺序无关
        vs = ["dir/A.mkv", "dir/B.mkv"]
        subs = ["dir/B.zh.ass", "dir/A.简体.ass"]
        pairs = match_subtitle_files(vs, subs)
        self.assertEqual(self._names(pairs, 0), ["A.简体.ass"])
        self.assertEqual(self._names(pairs, 1), ["B.zh.ass"])

    def test_l2_sequential(self):
        vs = ["dir/01.mkv", "dir/02.mkv", "dir/03.mkv"]
        subs = ["dir/01.ass", "dir/02.ass", "dir/03.zh.ass"]
        pairs = match_subtitle_files(vs, subs)
        self.assertEqual(self._names(pairs, 0), ["01.ass"])
        self.assertEqual(self._names(pairs, 1), ["02.ass"])
        self.assertEqual(self._names(pairs, 2), ["03.zh.ass"])

    def test_natural_order(self):
        # 自然排序：10 在 2 之后，否则 10 会配到 2 的字幕
        vs = ["dir/1.mkv", "dir/2.mkv", "dir/10.mkv"]
        subs = ["dir/1.ass", "dir/10.ass", "dir/2.ass"]
        pairs = match_subtitle_files(vs, subs)
        self.assertEqual(self._names(pairs, 0), ["1.ass"])
        self.assertEqual(self._names(pairs, 1), ["2.ass"])
        self.assertEqual(self._names(pairs, 2), ["10.ass"])

    def test_multi_sub_group(self):
        # 同键多字幕（简+繁）归同一视频
        vs = ["dir/01.mkv", "dir/02.mkv"]
        subs = ["dir/01.简体.ass", "dir/01.繁体.ass", "dir/02.zh.ass"]
        pairs = match_subtitle_files(vs, subs)
        self.assertEqual(self._names(pairs, 0), ["01.简体.ass", "01.繁体.ass"])
        self.assertEqual(self._names(pairs, 1), ["02.zh.ass"])

    def test_ova_l1_rescue(self):
        # OVA 不按数字顺序，靠 L1 同名兜底
        vs = ["dir/01.mkv", "dir/OVA.mkv"]
        subs = ["dir/OVA.zh.ass", "dir/01.ass"]
        pairs = match_subtitle_files(vs, subs)
        self.assertEqual(self._names(pairs, 0), ["01.ass"])
        self.assertEqual(self._names(pairs, 1), ["OVA.zh.ass"])

    def test_unmatched_sub_untouched(self):
        vs = ["dir/01.mkv"]
        subs = ["dir/01.ass", "dir/stray.ass"]
        pairs = match_subtitle_files(vs, subs)
        self.assertEqual(self._names(pairs, 0), ["01.ass"])
        self.assertEqual(len(pairs), 1)

    def test_missing_sub_shifts(self):
        # 缺字幕时 L2 顺序错位（固有行为，固化文档化）；同名会被 L1 认领不漂移
        vs = ["dir/第01话.mkv", "dir/第02话.mkv", "dir/第03话.mkv"]
        subs = ["dir/01.ass", "dir/03.ass"]
        pairs = match_subtitle_files(vs, subs)
        self.assertEqual(self._names(pairs, 0), ["01.ass"])
        self.assertEqual(self._names(pairs, 1), ["03.ass"])
        self.assertEqual(self._names(pairs, 2), [])

    def test_video_idx_preserves_input_order(self):
        # 返回的视频下标对应 video_paths 原顺序
        vs = ["dir/B.mkv", "dir/A.mkv"]
        subs = ["dir/A.ass", "dir/B.ass"]
        pairs = match_subtitle_files(vs, subs)
        self.assertEqual(self._names(pairs, 0), ["B.ass"])
        self.assertEqual(self._names(pairs, 1), ["A.ass"])
