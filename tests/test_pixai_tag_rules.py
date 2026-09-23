"""apply_char_tag_rules：角色阈值规则（v1.0 标签获取的后处理规格）。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app.pixai_tagger import apply_char_tag_rules


class TestApplyCharTagRules(unittest.TestCase):
    """>0.9 全保留；无则取 >0.65 中最高 1 个兜底；仍无为空（宁缺毋滥）。"""

    def test_above_threshold_all_kept(self):
        merged = {"a": 0.95, "b": 0.91, "c": 0.7}
        self.assertEqual(apply_char_tag_rules(merged), {"a": 0.95, "b": 0.91})

    def test_fallback_single_highest(self):
        merged = {"a": 0.8, "b": 0.7, "c": 0.5}
        self.assertEqual(apply_char_tag_rules(merged), {"a": 0.8})

    def test_nothing_above_fallback_empty(self):
        self.assertEqual(apply_char_tag_rules({"a": 0.5, "b": 0.65}), {})

    def test_empty_input(self):
        self.assertEqual(apply_char_tag_rules({}), {})

    def test_boundary_strict(self):
        self.assertEqual(apply_char_tag_rules({"a": 0.65}), {})
        self.assertEqual(apply_char_tag_rules({"a": 0.9}), {"a": 0.9})

    def test_custom_thresholds(self):
        merged = {"a": 0.8, "b": 0.75}
        self.assertEqual(apply_char_tag_rules(merged, 0.7, 0.6),
                         {"a": 0.8, "b": 0.75})


if __name__ == "__main__":
    unittest.main()
