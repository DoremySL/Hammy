"""apply_char_tag_rules / apply_ip_tag_rules / apply_tag_output_rules：
标签获取后处理规则（v1.0 标签获取的规格）。

char_stats: tag → (初筛内出现次数, 最高置信度)；得分 = 最高置信度 + (次数-1)*0.1，
总分 > 阈值全部保留（返回值为最高置信度，即显示分数），否则为空。
ip_stats: tag → (命中帧数, 最高置信度)；排除 original/real_life 后只输出最高置信度者。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app.pixai_tagger import (apply_char_tag_rules, apply_ip_tag_rules,
                                  apply_tag_output_rules)


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


class TestApplyIpTagRules(unittest.TestCase):
    """排除 original/real_life 后只输出最高置信度者；并列比帧数，再并列全保留。"""

    def test_excludes_non_ip_tags(self):
        stats = {"original": (5, 0.99), "real_life": (3, 0.98), "fate": (2, 0.9)}
        self.assertEqual(apply_ip_tag_rules(stats), {"fate": 0.9})

    def test_only_top_confidence_kept(self):
        stats = {"fate": (4, 0.95), "touhou": (9, 0.8)}
        self.assertEqual(apply_ip_tag_rules(stats), {"fate": 0.95})

    def test_tie_resolved_by_frame_count(self):
        stats = {"fate": (2, 0.9), "touhou": (5, 0.9)}
        self.assertEqual(apply_ip_tag_rules(stats), {"touhou": 0.9})

    def test_full_tie_keeps_all(self):
        stats = {"fate": (3, 0.9), "touhou": (3, 0.9)}
        self.assertEqual(apply_ip_tag_rules(stats), {"fate": 0.9, "touhou": 0.9})

    def test_all_excluded(self):
        self.assertEqual(apply_ip_tag_rules({"original": (2, 0.9)}), {})

    def test_empty_input(self):
        self.assertEqual(apply_ip_tag_rules({}), {})


class TestApplyTagOutputRules(unittest.TestCase):
    """组合规则：角色阈值过滤 + IP 规则 + 角色兜底。"""

    def test_normal_output(self):
        char = {"saber": (3, 0.92)}
        ip = {"fate": (2, 0.9)}
        cs, ips = apply_tag_output_rules(char, ip, char_threshold=0.9)
        self.assertEqual(cs, {"saber": 0.92})
        self.assertEqual(ips, {"fate": 0.9})

    def test_fallback_char_emitted_when_ip_hits_but_char_below(self):
        # IP 命中但角色都低于阈值：补输出置信度最高的角色（低于阈值也输出）
        char = {"saber": (1, 0.5), "rin": (1, 0.7)}
        ip = {"fate": (2, 0.9)}
        cs, ips = apply_tag_output_rules(char, ip, char_threshold=0.9)
        self.assertEqual(cs, {"rin": 0.7})
        self.assertEqual(ips, {"fate": 0.9})

    def test_no_fallback_without_ip_hit(self):
        char = {"saber": (1, 0.5)}
        ip: dict = {}
        cs, ips = apply_tag_output_rules(char, ip, char_threshold=0.9)
        self.assertEqual(cs, {})
        self.assertEqual(ips, {})

    def test_no_fallback_when_char_already_output(self):
        char = {"saber": (3, 0.95)}
        ip = {"fate": (2, 0.9)}
        cs, _ = apply_tag_output_rules(char, ip, char_threshold=0.9)
        self.assertEqual(cs, {"saber": 0.95})

    def test_ip_only_original_excluded_no_fallback(self):
        # 排除后无作品命中：不触发角色兜底
        char = {"saber": (1, 0.5)}
        ip = {"original": (2, 0.99)}
        cs, ips = apply_tag_output_rules(char, ip, char_threshold=0.9)
        self.assertEqual(cs, {})
        self.assertEqual(ips, {})


if __name__ == "__main__":
    unittest.main()
