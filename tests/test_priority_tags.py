import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from gui_app.prompts import normalize_priority_items, build_priority_tags_section


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


class TestBuildPriorityTagsSection(unittest.TestCase):
    def test_disabled(self):
        self.assertEqual(build_priority_tags_section(False, [{"keyword": "a"}]), "")

    def test_enabled_no_items(self):
        self.assertEqual(build_priority_tags_section(True, []), "")

    def test_enabled_with_description(self):
        s = build_priority_tags_section(True, [{"keyword": "cat", "description": "feline"}])
        self.assertIn("\u3010\u6807\u7b7e\u68c0\u7d22\u3011", s)
        self.assertIn("- cat\uff1afeline", s)

    def test_enabled_without_description(self):
        s = build_priority_tags_section(True, [{"keyword": "cat"}])
        self.assertIn("- cat", s)


if __name__ == "__main__":
    unittest.main()
