"""标签检索测试：模式路由 / 三级词面匹配 / 排序截断 / 首轮注入段落构建。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_rename.tag_recall import (
    TagRecall, _compact, _norm, build_section,
)

ITEMS = [
    {"keyword": "制服", "description": "穿着制服的场景"},
    {"keyword": "office_lady", "description": "职业女性"},
    {"keyword": "1girl", "description": "单人女性"},
    {"keyword": "sunset beach", "description": ""},
    {"keyword": "未命中词", "description": "x"},
]


def _on(items=ITEMS, **kw):
    return TagRecall(items, mode="on", **kw)


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
        self.assertEqual(TagRecall(ITEMS, mode="on").mode, "on")
        self.assertEqual(TagRecall(ITEMS, mode="enhanced").mode, "enhanced")
        self.assertEqual(TagRecall(ITEMS, mode="off").mode, "off")
        self.assertEqual(TagRecall(ITEMS).mode, "off")  # 缺省关闭
        self.assertEqual(TagRecall(ITEMS, mode="bogus").mode, "off")  # 非法值按 off

    def test_on_with_single_item(self):
        # 显式 on：1 条也走 on
        self.assertEqual(TagRecall(ITEMS[:1], mode="on").mode, "on")

    def test_empty_items_off(self):
        self.assertEqual(TagRecall([], mode="on").mode, "off")

    def test_has_dense(self):
        self.assertFalse(TagRecall(ITEMS, mode="on").has_dense)          # on 档不用向量
        self.assertFalse(TagRecall(ITEMS, mode="enhanced").has_dense)    # 无后端
        self.assertTrue(TagRecall(ITEMS, mode="enhanced", dense=object()).has_dense)
        self.assertFalse(TagRecall(ITEMS, mode="off", dense=object()).has_dense)

    def test_off_no_injection(self):
        section, kws = TagRecall(ITEMS, mode="off").literal_injection({"filename": "制服"})
        self.assertEqual(section, "")
        self.assertEqual(kws, [])


class TestRecallMatch(unittest.TestCase):
    def test_exact_chinese(self):
        cands = _on().recall_literal({"filename": "她穿着制服奔跑"})
        kws = [c["keyword"] for c in cands]
        self.assertIn("制服", kws)
        self.assertNotIn("未命中词", kws)

    def test_compact_match(self):
        cands = _on().recall_literal({"aux": "officelady主题的短剧"})
        self.assertIn("office_lady", [c["keyword"] for c in cands])

    def test_exact_via_separator_unify(self):
        # keyword 分隔符与文本分隔符不同 → 规范化后按 exact 命中
        cands = _on().recall_literal({"aux": "Office Lady 主题的短剧"})
        self.assertIn("office_lady", [c["keyword"] for c in cands])

    def test_token_match(self):
        cands = _on().recall_literal({"aux": "the beach at sunset is beautiful"})
        self.assertIn("sunset beach", [c["keyword"] for c in cands])

    def test_no_match(self):
        self.assertEqual(_on().recall_literal({"filename": "城市夜景与霓虹灯"}), [])

    def test_kind_score_ordering(self):
        # 同抢最后一个 top_k 名额：精确(1.0) > 分词(0.7)
        items = [{"keyword": "sunset beach", "description": ""},   # 词序打散 → 分词命中
                 {"keyword": "制服", "description": ""}]            # 精确命中
        cands = TagRecall(items, top_k=1, mode="on").recall_literal(
            {"filename": "制服 at the beach watching sunset"})
        self.assertEqual([c["keyword"] for c in cands], ["制服"])

    def test_topk_and_stable_order(self):
        items = [{"keyword": "bbb", "description": ""},
                 {"keyword": "aaa", "description": ""},
                 {"keyword": "ccc", "description": ""}]
        cands = TagRecall(items, top_k=2, mode="on").recall_literal({"aux": "aaa bbb ccc"})
        self.assertEqual(len(cands), 2)  # top_k 截断
        # 同为 exact，按标签库原始顺序
        self.assertEqual([c["keyword"] for c in cands], ["bbb", "aaa"])

    def test_multi_part_joined(self):
        # 多 part 值拼接后统一检索（filename 与 aux 各命中一个）
        cands = _on().recall_literal({"filename": "制服合集", "aux": "1girl 题材"})
        self.assertEqual([c["keyword"] for c in cands], ["制服", "1girl"])


class TestRelatedRecall(unittest.TestCase):
    """related 单值关联词与关键词同级参与三级词面匹配（不再按逗号拆多值）。"""

    def test_related_exact(self):
        items = [{"keyword": "4K", "description": "", "related": "2160p UHD"}]
        for hay in ("一部2160p UHD的影片", "一部2160p-UHD的影片"):
            cands = TagRecall(items, mode="on").recall_literal({"aux": hay})
            self.assertEqual([c["keyword"] for c in cands], ["4K"], hay)

    def test_related_not_split(self):
        # 单值化：逗号/顿号关联串整体当一个词，逗号后的部分不再独立匹配
        from batch_rename.tag_recall import split_related
        self.assertEqual(split_related("2160p,超清；UHD"), ["2160p,超清；UHD"])
        self.assertEqual(split_related("  "), [])
        items = [{"keyword": "4K", "description": "", "related": "2160p,超清"}]
        self.assertEqual(TagRecall(items, mode="on").recall_literal({"aux": "这部影片是超清制作"}), [])
        cands = TagRecall(items, mode="on").recall_literal({"aux": "标记为2160p,超清的版本"})
        self.assertEqual([c["keyword"] for c in cands], ["4K"])

    def test_related_compact_and_token(self):
        items = [{"keyword": "胶片", "description": "", "related": "film grain"},
                 {"keyword": "HDR", "description": "", "related": "dolby vision"}]
        cands = TagRecall(items, mode="on").recall_literal({"aux": "很有filmgrain质感"})
        self.assertIn("胶片", [c["keyword"] for c in cands])
        cands = TagRecall(items, mode="on").recall_literal({"aux": "dolbyvision编码"})
        self.assertIn("HDR", [c["keyword"] for c in cands])

    def test_group_not_matched(self):
        # 分组名仅供显示，不参与词面召回
        items = [{"keyword": "4K", "description": "", "related": "", "group": "画质"}]
        self.assertEqual(TagRecall(items, mode="on").recall_literal({"aux": "画质细腻"}), [])

    def test_section_excludes_related(self):
        items = [{"keyword": "4K", "description": "高分辨率", "related": "2160p"}]
        tr = _on(items)
        section, _ = tr.literal_injection({"aux": "2160p"})
        self.assertIn("- 4K：高分辨率", section)
        self.assertNotIn("2160p", section)
        section2 = tr.build_candidates_section(tr.items)
        self.assertNotIn("2160p", section2)

    def test_related_items_passed_to_dense(self):
        # 向量索引材料应携带关联词（build_index 收到的 items 含 related 字段）
        items = [{"keyword": "4K", "description": "高分辨率", "related": "2160p"}]
        tr = TagRecall(items, mode="enhanced")
        self.assertEqual(tr.items, items)


class TestLiteralInjection(unittest.TestCase):
    def test_injection_with_hits(self):
        section, kws = _on().literal_injection({"filename": "制服"})
        self.assertIn("【标签检索】", section)
        self.assertIn("- 制服：穿着制服的场景", section)
        self.assertEqual(kws, ["制服"])

    def test_injection_without_hits(self):
        section, kws = _on().literal_injection({"filename": "无关文件名"})
        self.assertEqual(section, "")
        self.assertEqual(kws, [])


class TestSections(unittest.TestCase):
    def test_candidates_section(self):
        tr = _on()
        section = tr.build_candidates_section([{"keyword": "制服", "description": "穿着制服"},
                                               {"keyword": "海边", "description": ""}])
        self.assertIn("【标签检索·召回候选】", section)
        self.assertIn("- 制服：穿着制服", section)
        self.assertIn("- 海边", section)

    def test_build_section_header(self):
        s = build_section([{"keyword": "a", "description": ""}], "头")
        self.assertEqual(s, "头\n- a")


if __name__ == "__main__":
    unittest.main()
