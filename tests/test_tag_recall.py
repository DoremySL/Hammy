"""标签检索召回测试：模式路由 / 三级词面匹配 / 排序截断 / 段落构建。"""
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


def _rag(items=ITEMS, **kw):
    return TagRecall(items, mode="rag", **kw)


class TestNormalize(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(_norm("Office_Lady"), "office lady")
        self.assertEqual(_norm("ＯＬ・装"), "ol・装")  # 全角→半角，NFKC
        self.assertEqual(_norm(None), "")

    def test_compact(self):
        self.assertEqual(_compact("Office-Lady"), "officelady")
        self.assertEqual(_compact("1 girl"), "1girl")


class TestModeRouting(unittest.TestCase):
    """模式显式给定（GUI 三档翻译后传入），不做条目数自动降级。"""

    def test_explicit_modes(self):
        self.assertEqual(TagRecall(ITEMS, mode="rag").mode, "rag")
        self.assertEqual(TagRecall(ITEMS, mode="full").mode, "full")
        self.assertEqual(TagRecall(ITEMS, mode="off").mode, "off")
        self.assertEqual(TagRecall(ITEMS).mode, "off")  # 缺省关闭
        self.assertEqual(TagRecall(ITEMS, mode="bogus").mode, "off")  # 非法值按 off

    def test_rag_with_single_item(self):
        # 显式 rag：1 条也走 rag
        self.assertEqual(TagRecall(ITEMS[:1], mode="rag").mode, "rag")

    def test_empty_items_off(self):
        self.assertEqual(TagRecall([], mode="rag").mode, "off")

    def test_full_section_built(self):
        tr = TagRecall(ITEMS, mode="full")
        self.assertIn("制服", tr.full_section)
        self.assertIn("【标签检索】", tr.full_section)

    def test_rag_no_full_section(self):
        tr = _rag()
        self.assertEqual(tr.mode, "rag")
        self.assertEqual(tr.full_section, "")


class TestRecallMatch(unittest.TestCase):
    def test_exact_chinese(self):
        cands = _rag().recall({"plot": "她穿着制服奔跑"})
        kws = [c["keyword"] for c in cands]
        self.assertIn("制服", kws)
        self.assertNotIn("未命中词", kws)

    def test_compact_match(self):
        cands = _rag().recall({"plot": "officelady主题的短剧"})
        self.assertIn("office_lady", [c["keyword"] for c in cands])

    def test_exact_via_separator_unify(self):
        # keyword 分隔符与文本分隔符不同 → 规范化后按 exact 命中
        cands = _rag().recall({"plot": "Office Lady 主题的短剧"})
        self.assertIn("office_lady", [c["keyword"] for c in cands])

    def test_token_match(self):
        cands = _rag().recall({"plot": "the beach at sunset is beautiful"})
        self.assertIn("sunset beach", [c["keyword"] for c in cands])

    def test_no_match(self):
        self.assertEqual(_rag().recall({"plot": "城市夜景与霓虹灯"}), [])

    def test_kind_score_ordering(self):
        # 同抢最后一个 top_k 名额：精确(1.0) > 分词(0.7)
        items = [{"keyword": "sunset beach", "description": ""},   # 词序打散 → 分词命中
                 {"keyword": "制服", "description": ""}]            # 精确命中
        cands = TagRecall(items, top_k=1, mode="rag").recall(
            {"plot": "制服 at the beach watching sunset"})
        self.assertEqual([c["keyword"] for c in cands], ["制服"])

    def test_topk_and_stable_order(self):
        items = [{"keyword": "bbb", "description": ""},
                 {"keyword": "aaa", "description": ""},
                 {"keyword": "ccc", "description": ""}]
        cands = TagRecall(items, top_k=2, mode="rag").recall({"plot": "aaa bbb ccc"})
        self.assertEqual(len(cands), 2)  # top_k 截断
        # 同为 exact，按标签库原始顺序
        self.assertEqual([c["keyword"] for c in cands], ["bbb", "aaa"])


class TestRelatedRecall(unittest.TestCase):
    """关联词与关键词同级参与三级词面匹配。"""

    def test_related_exact(self):
        items = [{"keyword": "4K", "description": "", "related": "2160p UHD,超清"}]
        for hay in ("一部2160p UHD的影片", "这部影片是超清制作"):
            cands = TagRecall(items, mode="rag").recall({"plot": hay})
            self.assertEqual([c["keyword"] for c in cands], ["4K"], hay)

    def test_related_compact_and_token(self):
        items = [{"keyword": "胶片", "description": "", "related": "film grain"},
                 {"keyword": "HDR", "description": "", "related": "dolby vision"}]
        cands = TagRecall(items, mode="rag").recall({"plot": "很有filmgrain质感"})
        self.assertIn("胶片", [c["keyword"] for c in cands])
        cands = TagRecall(items, mode="rag").recall({"plot": "dolbyvision编码"})
        self.assertIn("HDR", [c["keyword"] for c in cands])

    def test_group_not_matched(self):
        # 分组名仅供显示，不参与词面召回
        items = [{"keyword": "4K", "description": "", "related": "", "group": "画质"}]
        self.assertEqual(TagRecall(items, mode="rag").recall({"plot": "画质细腻"}), [])

    def test_section_excludes_related(self):
        items = [{"keyword": "4K", "description": "高分辨率", "related": "2160p,超清"}]
        tr = TagRecall(items, mode="full")
        self.assertIn("- 4K：高分辨率", tr.full_section)
        self.assertNotIn("2160p", tr.full_section)
        section = tr.build_candidates_section(tr.items)
        self.assertNotIn("2160p", section)

    def test_related_items_passed_to_dense(self):
        # 向量索引材料应携带关联词（build_index 收到的 items 含 related 字段）
        items = [{"keyword": "4K", "description": "高分辨率", "related": "2160p"}]
        tr = TagRecall(items, mode="rag")
        self.assertEqual(tr.items, items)


class TestSections(unittest.TestCase):
    def test_candidates_section(self):
        tr = _rag()
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
