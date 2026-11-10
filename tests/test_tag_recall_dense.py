"""向量召回层测试：纯向量召回 / 阈值过滤 / top_n 截断 / 失败跳过第二轮 / 惰性构建与预热。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_rename.tag_recall import TagRecall

ITEMS = [
    {"keyword": "海边", "description": "海边的场景"},
    {"keyword": "制服", "description": "制服"},
    {"keyword": "演播室", "description": "演播室场景"},
    {"keyword": "洛丽塔", "description": "洛丽塔裙装"},
    {"keyword": "未命中词", "description": "x"},
]


class FakeDense:
    """预置命中表后端：按 (keyword, score) 返回 idx。"""

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
        for kw, score in self.keyword_scores:
            idx = self._kw2idx.get(kw)
            if idx is not None:
                out.append({"idx": idx, "score": score})
        return out


def _enh(dense=None, **kw):
    return TagRecall(ITEMS, mode="enhanced", dense=dense, **kw)


class TestDenseRecall(unittest.TestCase):
    def test_dense_candidate(self):
        dense = FakeDense([("演播室", 0.68)])
        cands = _enh(dense, vec_threshold=0.45).recall_dense({"plot": "女主播在演播室播报"})
        self.assertIn("演播室", [c["keyword"] for c in cands])

    def test_threshold_filters(self):
        dense = FakeDense([("演播室", 0.40), ("洛丽塔", 0.62)])
        cands = _enh(dense, vec_threshold=0.45).recall_dense({"plot": "无关文本"})
        kws = [c["keyword"] for c in cands]
        self.assertNotIn("演播室", kws)
        self.assertIn("洛丽塔", kws)

    def test_top_n_cap(self):
        dense = FakeDense([(kw, 0.9) for kw in ("海边", "制服", "演播室", "洛丽塔")])
        cands = _enh(dense, vec_threshold=0.45, vec_top_n=1).recall_dense({"plot": "无关文本"})
        self.assertEqual([c["keyword"] for c in cands], ["海边"])

    def test_invalid_idx_skipped(self):
        dense = FakeDense()
        tr = _enh(dense)
        out = tr.recall_dense({"plot": "x"})
        # 后端返回空 → 无候选
        self.assertEqual(out, [])

    def test_dense_failure_skips_round2(self):
        # 后端构建失败 → 禁用向量层，后续返回 []（跳过第二轮）
        dense = FakeDense(fail=True)
        tr = _enh(dense)
        self.assertEqual(tr.recall_dense({"plot": "她在海边漫步"}), [])
        self.assertIsNone(tr.dense)
        self.assertFalse(tr.has_dense)

    def test_search_failure_returns_empty(self):
        # search 抛错：本次返回 []，dense 不被禁用
        dense = FakeDense([("海边", 0.6)])
        tr = _enh(dense)
        tr.recall_dense({"plot": "海边"})  # 触发 build
        dense.fail = True
        self.assertEqual(tr.recall_dense({"plot": "海边"}), [])
        self.assertIsNotNone(tr.dense)

    def test_build_once(self):
        dense = FakeDense([("海边", 0.6)])
        tr = _enh(dense)
        tr.recall_dense({"plot": "海边"})
        tr.recall_dense({"plot": "海边"})
        self.assertEqual(dense.build_count, 1)

    def test_warmup(self):
        dense = FakeDense([])
        tr = _enh(dense)
        tr.warmup_dense()
        self.assertEqual(dense.build_count, 1)

    def test_all_parts_passed_to_dense(self):
        dense = FakeDense([("演播室", 0.68)])
        tr = _enh(dense)
        tr.recall_dense({"plot": "女主播在演播室播报", "tags": "演播室",
                         "filename": "测试一下", "aux": "辅助文本"})
        self.assertIsNotNone(dense.last_parts)
        self.assertEqual(set(dense.last_parts), {"plot", "tags", "filename", "aux"})
        self.assertEqual(dense.last_parts["filename"], "测试一下")

    def test_on_mode_ignores_dense(self):
        dense = FakeDense([("海边", 0.6)])
        tr = TagRecall(ITEMS, mode="on", dense=dense)
        self.assertEqual(tr.mode, "on")
        self.assertEqual(dense.build_count, 0)  # on 模式不构建向量索引
        self.assertFalse(tr.has_dense)
        self.assertEqual(tr.recall_dense({"plot": "海边"}), [])  # on 模式不走向量召回

    def test_off_mode_ignores_dense(self):
        dense = FakeDense([("海边", 0.6)])
        tr = TagRecall(ITEMS, mode="off", dense=dense)
        self.assertFalse(tr.has_dense)
        self.assertEqual(tr.recall_dense({"plot": "海边"}), [])

    def test_enhanced_without_dense_degrades(self):
        # enhanced 无后端 → has_dense False，字面召回照常（等同 on）
        tr = _enh()
        self.assertFalse(tr.has_dense)
        section, kws = tr.literal_injection({"filename": "海边漫步"})
        self.assertEqual(kws, ["海边"])


if __name__ == "__main__":
    unittest.main()
