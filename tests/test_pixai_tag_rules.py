"""apply_char_tag_rules：角色标签规则（v1.0 标签获取的后处理规格）。

char_stats: tag → (初筛内出现次数, 最高置信度)；得分 = 最高置信度 + (次数-1)*0.1，
总分 > 阈值全部保留（返回值为最高置信度，即显示分数），否则为空。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app.pixai_tagger import apply_char_tag_rules


class TestApplyCharTagRules(unittest.TestCase):
    """得分 >阈值 全保留，否则为空（无兜底）。用例显式传 0.9，与默认值解耦。"""

    def test_above_threshold_all_kept(self):
        stats = {"a": (1, 0.95), "b": (3, 0.91), "c": (2, 0.7)}
        self.assertEqual(apply_char_tag_rules(stats, 0.9), {"a": 0.95, "b": 0.91})

    def test_freq_weight_outruns_threshold(self):
        # 最高仅 0.31，出现 7 次：0.31 + 6*0.1 = 0.91 > 0.9 → 输出，显示 0.31
        self.assertEqual(apply_char_tag_rules({"x": (7, 0.31)}, 0.9), {"x": 0.31})

    def test_single_weak_occurrence_dropped(self):
        self.assertEqual(apply_char_tag_rules({"x": (1, 0.31)}, 0.9), {})

    def test_no_fallback(self):
        # 都不达标时直接为空，不再取最高 1 个兜底
        self.assertEqual(apply_char_tag_rules({"a": (1, 0.8), "b": (1, 0.7)}, 0.9), {})

    def test_below_threshold_empty(self):
        self.assertEqual(apply_char_tag_rules({"a": (2, 0.5), "b": (1, 0.65)}, 0.9), {})

    def test_empty_input(self):
        self.assertEqual(apply_char_tag_rules({}), {})

    def test_boundary_strict(self):
        self.assertEqual(apply_char_tag_rules({"a": (1, 0.65)}, 0.9), {})
        self.assertEqual(apply_char_tag_rules({"a": (1, 0.9)}, 0.9), {})

    def test_custom_thresholds(self):
        stats = {"a": (1, 0.8), "b": (2, 0.55)}
        self.assertEqual(apply_char_tag_rules(stats, 0.7), {"a": 0.8})
        self.assertEqual(apply_char_tag_rules(stats, 0.7, freq_weight=0.2),
                         {"a": 0.8, "b": 0.55})


if __name__ == "__main__":
    unittest.main()
