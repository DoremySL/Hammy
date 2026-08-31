"""similarity.py 纯逻辑单测：哈希 / 匹配 / 分箱 / 对齐 / 聚组（合成指纹，不依赖 ffmpeg）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from batch_rename import similarity as sim


def _f(t, h):
    return (float(t), h)


FULL = (1 << 64) - 1


class TestDhash(unittest.TestCase):
    def test_all_decreasing_sets_all_bits(self):
        b = bytes(list(range(8, -1, -1)) * 8)   # 每行 9 像素递减 → 相邻列差全命中
        self.assertEqual(sim.dhash_gray(b), FULL)

    def test_all_increasing_clears_all_bits(self):
        b = bytes(list(range(9)) * 8)
        self.assertEqual(sim.dhash_gray(b), 0)

    def test_is_flat(self):
        self.assertTrue(sim._is_flat(bytes([5] * 72)))
        self.assertTrue(sim._is_flat(bytes([5, 7] * 36)))   # 差 2 ≤ 4
        self.assertFalse(sim._is_flat(bytes([0, 10] * 36)))

    def test_frames_from_raw_skips_flat_and_stamps_time(self):
        f1 = bytes([0, 10] * 36)   # 极差 10 > 4，非平坦
        flat = bytes([3] * 72)
        frames = sim._frames_from_raw(f1 + flat + f1, 10)
        self.assertEqual([t for t, _ in frames], [10.0, 12.0])

    def test_raw_unique_counts_content_changes(self):
        a, b = bytes([1] * 72), bytes([2] * 72)
        self.assertEqual(sim._raw_unique_count(a + a + b + a), 3)


class TestMatchCount(unittest.TestCase):
    @staticmethod
    def _mix(t):
        # 黄金比例散列：不同时刻的哈希汉明距离 ≈32，避免小整数巧合匹配
        return (t * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)

    def test_shift_match(self):
        a = [_f(t, self._mix(t)) for t in range(10)]
        b = [_f(t + 3, self._mix(t)) for t in range(10)]
        self.assertEqual(sim._match_count(a, b, 3.0), 10)
        self.assertEqual(sim._match_count(a, b, 0.0), 0)

    def test_hamming_threshold(self):
        self.assertEqual(sim._match_count([_f(0, 0)], [_f(0, 0b0011)], 0.0), 1)   # 距离 2
        self.assertEqual(sim._match_count([_f(0, 0)], [_f(0, FULL)], 0.0), 0)     # 距离 64

    def test_cap_limits_count(self):
        a = [_f(t, 1) for t in range(10)]
        b = [_f(t, 1) for t in range(10)]
        self.assertEqual(sim._match_count(a, b, 0.0, cap=3), 3)

    def test_empty_bins(self):
        self.assertEqual(sim._match_count([], [_f(0, 1)], 0.0), 0)


class TestBinning(unittest.TestCase):
    def test_bin_by_time_splits_full_span(self):
        frames = [_f(t, t) for t in range(10)]
        bins = sim._bin_by_time(frames, 5)
        self.assertEqual(len(bins), 5)
        self.assertEqual(sum(len(b) for b in bins), 10)

    def test_bin_by_time_zero_span_returns_single(self):
        frames = [_f(0, 0)] * 3
        self.assertEqual(sim._bin_by_time(frames, 5), [frames])


class TestPreDist(unittest.TestCase):
    def test_min_mid_hamming(self):
        fa = [_f(0, 0b0000), _f(1, 0b1111), _f(2, 0b0000)]
        fb = [_f(0, 0b1111), _f(1, 0b0000), _f(2, 0b1111)]
        self.assertEqual(sim.pre_dist(fa, fb), 0)

    def test_short_input(self):
        self.assertEqual(sim.pre_dist([_f(0, 0b0011)], [_f(0, 0b0000)]), 2)


class TestResolution(unittest.TestCase):
    def test_aspect_ratio(self):
        self.assertAlmostEqual(sim.aspect_ratio("1920x1080"), 16 / 9)
        self.assertEqual(sim.aspect_ratio("bogus"), 0.0)
        self.assertEqual(sim.aspect_ratio("100x0"), 0.0)

    def test_res_height(self):
        self.assertEqual(sim._res_height("1920x1080"), 1080)
        self.assertEqual(sim._res_height("bogus"), 0)


class TestScanMode(unittest.TestCase):
    def test_flat_tol_when_not_tiered(self):
        self.assertEqual(sim.NORMAL.dur_tol_for(30.0), 15.0)
        self.assertEqual(sim.EXTREME.dur_tol_for(30.0), 35.0)

    def test_tiered_tol_fast(self):
        self.assertEqual(sim.FAST.dur_tol_for(30.0), 0.0)
        self.assertEqual(sim.FAST.dur_tol_for(120.0), 1.0)
        self.assertEqual(sim.FAST.dur_tol_for(300.0), 2.0)


class TestInferExtras(unittest.TestCase):
    def test_offset_normalized_to_min(self):
        members = ["a", "b"]
        extras = sim.infer_extras(members, [("a", "b", 2.0, 0.0)])
        self.assertAlmostEqual(min(extras.values()), 0.0)
        self.assertAlmostEqual(max(extras.values()) - min(extras.values()), 2.0)

    def test_chain_via_bfs(self):
        extras = sim.infer_extras(["a", "b", "c"],
                                  [("a", "b", 1.0, 0.0), ("b", "c", 3.0, 0.0)])
        self.assertAlmostEqual(max(extras.values()) - min(extras.values()), 4.0)

    def test_isolated_member_zero(self):
        extras = sim.infer_extras(["a", "b"], [])
        self.assertEqual(extras, {"a": 0.0, "b": 0.0})


class TestClusterGroups(unittest.TestCase):
    def _meta(self, res="1920x1080", codec="h264", size=100):
        return {"resolution": res, "codec": codec, "size": size}

    def test_union_of_edges(self):
        metas = {k: self._meta() for k in "abc"}
        r = sim.AlignResult(0, 5, 5, 25, 0, 0)
        groups = sim.cluster_groups(metas, [("a", "b", r), ("b", "c", r)])
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0]["paths"]), ["a", "b", "c"])

    def test_disconnected_stay_separate(self):
        metas = {k: self._meta() for k in "abcd"}
        r = sim.AlignResult(0, 5, 5, 25, 0, 0)
        groups = sim.cluster_groups(metas, [("a", "b", r), ("c", "d", r)])
        self.assertEqual(len(groups), 2)

    def test_keep_prefers_higher_resolution(self):
        metas = {"a": self._meta(res="1280x720"), "b": self._meta(res="1920x1080")}
        r = sim.AlignResult(0, 5, 5, 25, 0, 0)
        groups = sim.cluster_groups(metas, [("a", "b", r)])
        self.assertEqual(groups[0]["keep"], "b")
        self.assertEqual(groups[0]["extras"], {"a": 0.0, "b": 0.0})


class TestAlign(unittest.TestCase):
    def test_quick_align_finds_shift(self):
        fa = [_f(t, 0b11110000) for t in range(20)]
        fb = [_f(t + 3, 0b11110000) for t in range(20)]
        r = sim.align([fa], [fb], 3.0, 20.0, 23.0, quick=True)
        self.assertIsNotNone(r)
        self.assertEqual(r.delta, 3)
        self.assertEqual(r.n_bins, 1)

    def test_quick_align_unmatchable_returns_none(self):
        fa = [_f(t, 0) for t in range(20)]
        fb = [_f(t, FULL) for t in range(20)]
        self.assertIsNone(sim.align([fa], [fb], 0.0, 20.0, 20.0, quick=True))

    def test_full_align_multi_bin_shift(self):
        fa = [_f(t, t) for t in range(200)]
        fb = [_f(t + 5, t) for t in range(200)]
        r = sim.align([fa], [fb], 5.0, 200.0, 205.0)
        self.assertIsNotNone(r)
        self.assertEqual(r.delta, 5)

    def test_empty_segments_return_none(self):
        self.assertIsNone(sim.align([[]], [[_f(0, 1)]], 0.0, 10.0))


if __name__ == "__main__":
    unittest.main()
