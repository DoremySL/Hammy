import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from batch_rename.ai import (
    _extract_first_json, _clean_tags, _parse_ai_response, _get_ci, AIFormatError,
)


class TestExtractFirstJson(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(
            _extract_first_json('prefix {"title": "x", "tags": []} suffix'),
            {"title": "x", "tags": []})

    def test_require_key_filter(self):
        self.assertEqual(
            _extract_first_json('{"foo":1} {"title":"x"}', require_key="title"),
            {"title": "x"})

    def test_no_json(self):
        self.assertIsNone(_extract_first_json("no json here"))

    def test_require_key_none(self):
        self.assertEqual(_extract_first_json('{"a":1}', require_key=None), {"a": 1})

    def test_require_key_none_invalid_skipped(self):
        self.assertIsNone(_extract_first_json('{a: 1}', require_key=None))

    def test_nested(self):
        self.assertEqual(
            _extract_first_json('{"title":"x","inner":{"a":1}}'),
            {"title": "x", "inner": {"a": 1}})

    def test_unmatched_brace_skipped(self):
        # 首个 { 未配对闭合（正文中的孤立字符），应继续搜索后面的合法 JSON
        self.assertEqual(
            _extract_first_json('broken { text {"title": "x"}'),
            {"title": "x"})

    def test_unmatched_brace_only(self):
        self.assertIsNone(_extract_first_json('stray { only'))

    def test_trailing_commas(self):
        self.assertEqual(
            _extract_first_json('{"title": "x", "tags": ["a", ], }'),
            {"title": "x", "tags": ["a"]})

    def test_trailing_comma_inside_string_kept(self):
        # 字符串内部的 ",}" 不能被尾随逗号清理误伤
        self.assertEqual(
            _extract_first_json('{"title": "a,}", "tags": [],}'),
            {"title": "a,}", "tags": []})

    def test_control_char_in_string(self):
        # 字符串值内未转义的换行（strict=False 容忍）
        raw = '{"title": "x", "plot": "a\nb"}'
        self.assertEqual(_extract_first_json(raw), {"title": "x", "plot": "a\nb"})

    def test_title_key_case_insensitive(self):
        # 模型偶尔输出 Title / TITLE 变体，提取时应接受
        self.assertEqual(
            _extract_first_json('{"Title": "x", "tags": []}'),
            {"Title": "x", "tags": []})


class TestGetCi(unittest.TestCase):
    def test_case_insensitive(self):
        data = {"Title": "x", "plot": "p"}
        self.assertEqual(_get_ci(data, "title"), "x")
        self.assertEqual(_get_ci(data, "PLOT"), "p")

    def test_missing(self):
        self.assertIsNone(_get_ci({"a": 1}, "title"))


class TestCleanTags(unittest.TestCase):
    def test_dedup(self):
        self.assertEqual(_clean_tags(["a", "b", "a"]), ["a", "b"])

    def test_case_insensitive(self):
        self.assertEqual(_clean_tags(["Cat", "cat", "CAT"]), ["Cat"])

    def test_strip_and_skip_empty(self):
        self.assertEqual(_clean_tags(["  x  ", "", None, "y"]), ["x", "y"])

    def test_non_list(self):
        self.assertEqual(_clean_tags(42), [])
        self.assertEqual(_clean_tags(None), [])

    def test_tags_string(self):
        self.assertEqual(_clean_tags("a, b、c，d"), ["a", "b", "c", "d"])

    def test_tags_string_no_comma(self):
        self.assertEqual(_clean_tags("solo"), ["solo"])


class TestParseAiResponse(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(_parse_ai_response('{"title":"x"}'), {"title": "x"})

    def test_invalid(self):
        with self.assertRaises(AIFormatError):
            _parse_ai_response("not json")

    def test_prose(self):
        raw = 'here you go {"title":"x","tags":[]} done'
        self.assertEqual(_parse_ai_response(raw), {"title": "x", "tags": []})

    def test_no_title(self):
        with self.assertRaises(AIFormatError):
            _parse_ai_response('{"foo":1} only')

    def test_bom_prefix(self):
        self.assertEqual(_parse_ai_response('\ufeff{"title":"x"}'), {"title": "x"})

    def test_markdown_fence(self):
        raw = '```json\n{"title":"x"}\n```'
        self.assertEqual(_parse_ai_response(raw), {"title": "x"})

    def test_array_wrapped(self):
        self.assertEqual(_parse_ai_response('[{"title":"x"}]'), {"title": "x"})

    def test_trailing_comma(self):
        self.assertEqual(_parse_ai_response('{"title":"x",}'), {"title": "x"})

    def test_control_char_in_string(self):
        raw = '{"title":"x","plot":"a\nb"}'
        self.assertEqual(_parse_ai_response(raw), {"title": "x", "plot": "a\nb"})

    def test_title_key_case_insensitive(self):
        self.assertEqual(_parse_ai_response('{"TITLE":"x"}'), {"TITLE": "x"})


if __name__ == "__main__":
    unittest.main()
