"""向量召回层合并测试：词面+向量去重合并 / 阈值过滤 / top_n 截断 / 失败降级 / 惰性构建。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_rename.tag_recall import TagRecall

ITEMS = [
    {"keyword": "海边", "description": "海边的场景"},
    {"keyword": "制服", "description": "制服"},
    {"keyword": "演播室", "description": "演播室场景"},   # 词面不命中、仅向量命中
    {"keyword": "洛丽塔", "description": "洛丽塔裙装"},   # 同上
    {"keyword": "未命中词", "description": "x"},
]


class FakeDense:
    """预置命中表后端：按 (keyword, score, seg) 返回 idx。"""

    def __init__(self, keyword_scores=None, fail=False):
        self.keyword_scores = keyword_scores or []
        self.fail = fail
        self.build_count = 0
        self.built_keywords = None
        self.last_parts = None

    def build_index(self, items):
        if self.fail:
            raise RuntimeError("boom")
        self.build_count += 1
        self.built_keywords = [it["keyword"] for it in items]
        self._kw2idx = {it["keyword"]: i for i, it in enumerate(items)}

    def search(self, parts, top_k=64):
        if self.fail:
            raise RuntimeError("boom")
        self.last_parts = dict(parts)
        out = []
        for kw, score, seg in self.keyword_scores:
            idx = self._kw2idx.get(kw)
            if idx is not None:
                out.append({"idx": idx, "score": score, "seg": seg})
        return out


def _rag(dense=None, **kw):
    return TagRecall(ITEMS, mode="rag", dense=dense, **kw)


class TestDenseMerge(unittest.TestCase):
    def test_vector_only_candidate_added(self):
        # 「演播室」词面不命中，向量 0.68 命中 → 并入候选
        dense = FakeDense([("演播室", 0.68, "plot")])
        cands = _rag(dense, vec_threshold=0.45).recall({"plot": "女主播在演播室播报"})
        self.assertIn("演播室", [c["keyword"] for c in cands])

    def test_dedupe_with_lexical(self):
        # 「海边」词面已命中，向量重复命中 → 不重复并入
        dense = FakeDense([("海边", 0.71, "plot")])
        cands = _rag(dense).recall({"plot": "她在海边漫步"})
        kws = [c["keyword"] for c in cands]
        self.assertEqual(kws.count("海边"), 1)

    def test_threshold_filters(self):
        dense = FakeDense([("演播室", 0.40, "plot"), ("洛丽塔", 0.62, "plot")])
        cands = _rag(dense, vec_threshold=0.45).recall({"plot": "无关文本"})
        kws = [c["keyword"] for c in cands]
        self.assertNotIn("演播室", kws)
        self.assertIn("洛丽塔", kws)

    def test_top_n_cap(self):
        dense = FakeDense([(kw, 0.9, "plot") for kw in ("海边", "制服", "演播室", "洛丽塔")])
        cands = _rag(dense, vec_threshold=0.45, vec_top_n=1).recall({"plot": "无关文本"})
        # 只有 1 条向量候选并入
        self.assertEqual([c["keyword"] for c in cands], ["海边"])

    def test_dense_failure_degrades_to_lexical(self):
        dense = FakeDense(fail=True)
        tr = _rag(dense)
        cands = tr.recall({"plot": "她在海边漫步"})
        # 后端构建失败 → 禁用向量层，词面照常
        self.assertIn("海边", [c["keyword"] for c in cands])
        self.assertIsNone(tr.dense)

    def test_build_once(self):
        dense = FakeDense([("海边", 0.6, "plot")])
        tr = _rag(dense)
        tr.recall({"plot": "海边"})
        tr.recall({"plot": "海边"})
        self.assertEqual(dense.build_count, 1)

    def test_warmup(self):
        dense = FakeDense([])
        tr = _rag(dense)
        tr.warmup_dense()
        self.assertEqual(dense.build_count, 1)

    def test_filename_segment_included_in_dense(self):
        dense = FakeDense([("演播室", 0.68, "plot")])
        tr = _rag(dense)
        tr.recall({"plot": "女主播在演播室播报", "filename": "测试一下.mp4", "aux": ""})
        self.assertIsNotNone(dense.last_parts)
        self.assertIn("filename", dense.last_parts)
        self.assertEqual(dense.last_parts["filename"], "测试一下.mp4")
        self.assertIn("plot", dense.last_parts)

    def test_full_mode_ignores_dense(self):
        dense = FakeDense([("海边", 0.6, "plot")])
        tr = TagRecall(ITEMS, mode="full", dense=dense)
        self.assertEqual(tr.mode, "full")
        self.assertEqual(dense.build_count, 0)  # full 模式不构建向量索引
        self.assertEqual(tr.recall({"plot": "海边"}), [])  # full 模式不走召回（由 AIWorker 全量注入）


if __name__ == "__main__":
    unittest.main()
