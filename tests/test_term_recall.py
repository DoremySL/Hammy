"""词条检索测试：模式路由 / 两级词面匹配 / 严格组 CJK 边界 / 排序截断 / 注入段落构建。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_rename.term_recall import (
    TermRecall, _compact, _norm, build_section,
)

ITEMS = [
    {"keyword": "制服", "description": "穿着制服的场景"},
    {"keyword": "office_lady", "description": "职业女性"},
    {"keyword": "1girl", "description": "单人女性"},
    {"keyword": "sunset beach", "description": ""},
    {"keyword": "未命中词", "description": "x"},
]


def _on(items=ITEMS, **kw):
    return TermRecall(items, mode="on", **kw)


class TestNormalize(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(_norm("Office_Lady"), "office lady")
        self.assertEqual(_norm("ＯＬ・装"), "ol・装")  # 全角→半角，NFKC
        self.assertEqual(_norm(None), "")

    def test_compact(self):
        self.assertEqual(_compact("Office-Lady"), "officelady")
        self.assertEqual(_compact("1 girl"), "1girl")


class TestModeRouting(unittest.TestCase):
    """模式显式给定（GUI 三档同名直传），不做条目数自动降级。"""

    def test_explicit_modes(self):
        self.assertEqual(TermRecall(ITEMS, mode="on").mode, "on")
        self.assertEqual(TermRecall(ITEMS, mode="enhanced").mode, "enhanced")
        self.assertEqual(TermRecall(ITEMS, mode="off").mode, "off")
        self.assertEqual(TermRecall(ITEMS).mode, "off")  # 缺省关闭
        self.assertEqual(TermRecall(ITEMS, mode="bogus").mode, "off")  # 非法值按 off

    def test_on_with_single_item(self):
        # 显式 on：1 条也走 on
        self.assertEqual(TermRecall(ITEMS[:1], mode="on").mode, "on")

    def test_empty_items_off(self):
        self.assertEqual(TermRecall([], mode="on").mode, "off")

    def test_has_dense(self):
        self.assertFalse(TermRecall(ITEMS, mode="on").has_dense)          # on 档不用向量
        self.assertFalse(TermRecall(ITEMS, mode="enhanced").has_dense)    # 无后端
        self.assertTrue(TermRecall(ITEMS, mode="enhanced", dense=object()).has_dense)
        self.assertFalse(TermRecall(ITEMS, mode="off", dense=object()).has_dense)

    def test_off_no_injection(self):
        section, kws = TermRecall(ITEMS, mode="off").merged_injection({"filename": "制服"}, [])
        self.assertEqual(section, "")
        self.assertEqual(kws, [])


class TestRecallMatch(unittest.TestCase):
    def test_exact_chinese(self):
        cands = _on().recall_literal({"filename": "【制服】花絮"})
        kws = [c["keyword"] for c in cands]
        self.assertIn("制服", kws)
        self.assertNotIn("未命中词", kws)
        # 词边界语义：中文邻接裸子串不再命中
        self.assertEqual(_on().recall_literal({"filename": "她穿着制服奔跑"}), [])

    def test_compact_match(self):
        cands = _on().recall_literal({"aux": "officelady主题的短剧"})
        self.assertIn("office_lady", [c["keyword"] for c in cands])

    def test_exact_via_separator_unify(self):
        # keyword 分隔符与文本分隔符不同 → 规范化后按 exact 命中
        cands = _on().recall_literal({"aux": "Office Lady 主题的短剧"})
        self.assertIn("office_lady", [c["keyword"] for c in cands])

    def test_token_no_match(self):
        # 分词级已删
        cands = _on().recall_literal({"aux": "the beach at sunset is beautiful"})
        self.assertNotIn("sunset beach", [c["keyword"] for c in cands])

    def test_cjk_boundary(self):
        # 全组严格：邻接汉字/假名拒，数字/标点/拉丁/端点收
        items = [{"keyword": "日常", "group": "Hammy-tagger"},
                 {"keyword": "原神", "group": "Hammy-tagger"},
                 {"keyword": "初音ミク", "group": "Hammy-tagger"}]
        tr = TermRecall(items, mode="on")
        kws = lambda parts: [c["keyword"] for c in tr.recall_literal(parts)]
        # 拒
        self.assertNotIn("日常", kws({"filename": "我的日常vlog记录"}))
        self.assertNotIn("原神", kws({"filename": "原神同人志整理"}))
        self.assertNotIn("初音ミク", kws({"filename": "初音ミク演唱会现场"}))
        # 收
        self.assertIn("日常", kws({"filename": "【日常】制作花絮"}))
        self.assertIn("原神", kws({"filename": "原神2024新春会"}))
        self.assertIn("原神", kws({"filename": "原神"}))
        self.assertIn("初音ミク", kws({"filename": "【初音ミク】演唱会"}))

    def test_cjk_boundary_group_agnostic(self):
        # 词边界与分组无关：无组条目同样拒裸子串
        tr = TermRecall([{"keyword": "日常"}], mode="on")
        self.assertEqual(tr.recall_literal({"filename": "我的日常vlog记录"}), [])

    def test_cjk_boundary_latin_unchanged(self):
        # 拉丁 related 行为不变
        items = [{"keyword": "胶片", "related": "film grain", "group": "Hammy-tagger"}]
        tr = TermRecall(items, mode="on")
        self.assertEqual([c["keyword"] for c in tr.recall_literal(
            {"aux": "很有filmgrain质感"})], ["胶片"])
        self.assertEqual([c["keyword"] for c in tr.recall_literal(
            {"aux": "film grain 质感"})], ["胶片"])

    def test_no_match(self):
        self.assertEqual(_on().recall_literal({"filename": "城市夜景与霓虹灯"}), [])

    def test_kind_score_ordering(self):
        # 同抢最后一个 top_k 名额：精确(1.0) > 紧凑(0.9)
        items = [{"keyword": "office lady", "description": ""},   # 分隔符被打散 → 紧凑命中
                 {"keyword": "制服", "description": ""}]            # 精确命中
        cands = TermRecall(items, top_k=1, mode="on").recall_literal(
            {"filename": "officelady主题【制服】"})
        self.assertEqual([c["keyword"] for c in cands], ["制服"])

    def test_topk_and_stable_order(self):
        items = [{"keyword": "bbb", "description": ""},
                 {"keyword": "aaa", "description": ""},
                 {"keyword": "ccc", "description": ""}]
        cands = TermRecall(items, top_k=2, mode="on").recall_literal({"aux": "aaa bbb ccc"})
        self.assertEqual(len(cands), 2)  # top_k 截断
        # 同为 exact，按词条库原始顺序
        self.assertEqual([c["keyword"] for c in cands], ["bbb", "aaa"])

    def test_multi_part_joined(self):
        # 多 part 值拼接后统一检索（filename 与 aux 各命中一个）
        cands = _on().recall_literal({"filename": "【制服】合集", "aux": "1girl 题材"})
        self.assertEqual([c["keyword"] for c in cands], ["制服", "1girl"])


class TestRelatedRecall(unittest.TestCase):
    """related 单值关联词与关键词同级参与三级词面匹配（不再按逗号拆多值）。"""

    def test_related_exact(self):
        items = [{"keyword": "4K", "description": "", "related": "2160p UHD"}]
        for hay in ("一部2160p UHD的影片", "一部2160p-UHD的影片"):
            cands = TermRecall(items, mode="on").recall_literal({"aux": hay})
            self.assertEqual([c["keyword"] for c in cands], ["4K"], hay)

    def test_related_not_split(self):
        # 单值化：逗号/顿号关联串整体当一个词，逗号后的部分不再独立匹配
        items = [{"keyword": "4K", "description": "", "related": "2160p,超清"}]
        self.assertEqual(TermRecall(items, mode="on").recall_literal({"aux": "这部影片是超清制作"}), [])
        cands = TermRecall(items, mode="on").recall_literal({"aux": "标记为2160p,超清的版本"})
        self.assertEqual([c["keyword"] for c in cands], ["4K"])

    def test_related_compact_match(self):
        items = [{"keyword": "胶片", "description": "", "related": "film grain"},
                 {"keyword": "HDR", "description": "", "related": "dolby vision"}]
        cands = TermRecall(items, mode="on").recall_literal({"aux": "很有filmgrain质感"})
        self.assertIn("胶片", [c["keyword"] for c in cands])
        cands = TermRecall(items, mode="on").recall_literal({"aux": "dolbyvision编码"})
        self.assertIn("HDR", [c["keyword"] for c in cands])

    def test_group_not_matched(self):
        # 分组名仅供显示，不参与词面召回
        items = [{"keyword": "4K", "description": "", "related": "", "group": "画质"}]
        self.assertEqual(TermRecall(items, mode="on").recall_literal({"aux": "画质细腻"}), [])

    def test_section_excludes_related(self):
        items = [{"keyword": "4K", "description": "高分辨率", "related": "2160p"}]
        tr = _on(items)
        section, _ = tr.merged_injection({"aux": "2160p"}, [])
        self.assertIn("- 4K：高分辨率", section)
        self.assertNotIn("2160p", section)
        section2 = tr.build_candidates_section(tr.items)
        self.assertNotIn("2160p", section2)

    def test_related_items_passed_to_dense(self):
        # 向量索引材料应携带关联词（build_index 收到的 items 含 related 字段）
        items = [{"keyword": "4K", "description": "高分辨率", "related": "2160p"}]
        tr = TermRecall(items, mode="enhanced")
        self.assertEqual(tr.items, [{**items[0], "group": ""}])


class TestMergedInjection(unittest.TestCase):
    def test_injection_with_literal_hits(self):
        section, kws = _on().merged_injection({"filename": "制服"}, [])
        self.assertIn("【词条参考】", section)
        self.assertIn("- 制服：穿着制服的场景", section)
        self.assertEqual(kws, ["制服"])

    def test_injection_without_hits(self):
        section, kws = _on().merged_injection({"filename": "无关文件名"}, [])
        self.assertEqual(section, "")
        self.assertEqual(kws, [])

    def test_tagger_hits_priority_and_dedup(self):
        # tagger 条目置顶保序，字面命中排后并去重，共享 top_k
        items = [{"keyword": "制服", "description": "穿着制服"},
                 {"keyword": "海边", "description": ""}]
        hits = [{"keyword": "视觉确认", "description": "A段"},
                {"keyword": "制服", "description": "重复条目"}]
        tr = _on(items=items, top_k=2)
        _, kws = tr.merged_injection({"filename": "【制服】【海边】"}, hits)
        self.assertEqual(kws, ["视觉确认", "制服"])  # 预算被置顶条目吃满，海边挤出
        tr3 = _on(items=items, top_k=3)
        section, kws = tr3.merged_injection({"filename": "【制服】【海边】"}, hits)
        self.assertEqual(kws, ["视觉确认", "制服", "海边"])
        self.assertLess(section.index("- 视觉确认"), section.index("- 海边"))

    def test_tagger_hits_only_when_off(self):
        # 检索关闭时 tagger 条目仍注入（不依赖检索开关）
        tr = TermRecall(ITEMS, mode="off")
        section, kws = tr.merged_injection({"filename": "制服"},
                                           [{"keyword": "视觉确认", "description": ""}])
        self.assertEqual(kws, ["视觉确认"])
        self.assertIn("视觉确认", section)


class TestSections(unittest.TestCase):
    def test_candidates_section(self):
        tr = _on()
        section = tr.build_candidates_section([{"keyword": "制服", "description": "穿着制服"},
                                               {"keyword": "海边", "description": ""}])
        self.assertIn("【词条参考·召回候选】", section)
        self.assertIn("- 制服：穿着制服", section)
        self.assertIn("- 海边", section)

    def test_build_section_header(self):
        s = build_section([{"keyword": "a", "description": ""}], "头")
        self.assertEqual(s, "头\n- a")


if __name__ == "__main__":
    unittest.main()
