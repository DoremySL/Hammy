"""sources.py 相似扫描流水线单测：预筛 / 两阶段 / 终验复用与踢除 / busy / 取消接线。

指纹全部用合成帧（mock fingerprint_windows），不依赖 ffmpeg。
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app.api_mixins.sources import SourcesMixin
from batch_rename import similarity as sim


def _meta(duration, res="1920x1080", codec="h264", size=1000):
    return {"duration": duration, "size": size, "resolution": res, "codec": codec,
            "audio_codec": "aac", "has_audio": True, "video": {"frame_rate": 1.0}}


def _const_frames(h, n=10):
    return [(float(t), h) for t in range(n)]


FULL64 = (1 << 64) - 1


class _SimilarTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        p = mock.patch("gui_app.api_mixins.sources.SIMILAR_CACHE_FILE",
                       self.root / "similar_cache.json")
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)


class TestSimilarPairs(_SimilarTestBase):
    def test_duration_and_aspect_prescreen(self):
        metas = {"a": _meta(100), "b": _meta(105), "c": _meta(200), "d": _meta(100)}
        aspect = {"a": 16 / 9, "b": 16 / 9, "c": 16 / 9, "d": 4 / 3}
        order = ["a", "d", "b", "c"]   # 时长升序
        pairs = SourcesMixin._similar_pairs(order, metas, aspect, sim.NORMAL)
        self.assertEqual(pairs, [("a", "b")])   # d 宽高比不符；c 超容差被 break

    def test_tiered_tol_fast_only_same_duration_below_60s(self):
        metas = {"a": _meta(50), "b": _meta(52), "c": _meta(50)}
        order = ["a", "c", "b"]
        aspect = {p: 16 / 9 for p in order}
        pairs = SourcesMixin._similar_pairs(order, metas, aspect, sim.FAST)
        self.assertEqual(pairs, [("a", "c")])


class TestPhase1(_SimilarTestBase):
    def test_single_segment_pair_accepted_directly(self):
        frames = _const_frames(0b1111, 5)
        get_fp = lambda p, idxs: {i: list(frames) for i in idxs}
        metas = {"a": _meta(60), "b": _meta(62)}   # ≤ FAST.short_sec → single
        fps1, accepted, surv = SourcesMixin._similar_phase1(
            ["a", "b"], [("a", "b")], {"a", "b"}, get_fp, metas, sim.FAST)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(surv, set())

    def test_prefilter_drops_dissimilar(self):
        fps = {"a": {1: _const_frames(0x0000, 5)}, "b": {1: _const_frames(FULL64, 5)}}
        get_fp = lambda p, idxs: fps[p]
        metas = {"a": _meta(60), "b": _meta(62)}
        fps1, accepted, surv = SourcesMixin._similar_phase1(
            ["a", "b"], [("a", "b")], {"a", "b"}, get_fp, metas, sim.FAST)
        self.assertEqual(accepted, [])
        self.assertEqual(surv, set())

    def test_multivindow_pair_becomes_survivor(self):
        frames = _const_frames(0xF0F0, 10)
        get_fp = lambda p, idxs: {i: list(frames) for i in idxs}
        metas = {"a": _meta(200), "b": _meta(200)}   # > short_sec → 非 single
        fps1, accepted, surv = SourcesMixin._similar_phase1(
            ["a", "b"], [("a", "b")], set(), get_fp, metas, sim.FAST)
        self.assertEqual(accepted, [])
        self.assertEqual(surv, {"a", "b"})


class TestPhase2(_SimilarTestBase):
    def test_accepts_and_records_full_edge(self):
        get_fp = lambda p, idxs: {i: _const_frames(0xF0F0, 30) for i in idxs}
        metas = {"a": _meta(200), "b": _meta(200)}
        fpsf, accepted, full_edges = SourcesMixin._similar_phase2(
            {"a", "b"}, [("a", "b")], get_fp, metas, sim.FAST)
        self.assertEqual(len(accepted), 1)
        self.assertIn(frozenset(("a", "b")), full_edges)

    def test_non_survivor_pairs_skipped(self):
        get_fp = lambda p, idxs: {i: _const_frames(0xF0F0, 30) for i in idxs}
        metas = {"a": _meta(200), "b": _meta(200), "c": _meta(200)}
        fpsf, accepted, full_edges = SourcesMixin._similar_phase2(
            {"a", "b"}, [("a", "b"), ("a", "c")], get_fp, metas, sim.FAST)
        self.assertEqual([(a, b) for a, b, _ in accepted], [("a", "b")])
        self.assertEqual(len(accepted), 1)


class TestFinalize(_SimilarTestBase):
    def test_reuses_phase2_edge_without_recompute(self):
        frames = {i: _const_frames(0xF0F0, 30) for i in (1, 2, 3)}
        fpsf = {"a": frames, "b": {i: list(v) for i, v in frames.items()}}
        accepted = [("a", "b", sim.AlignResult(0, 3, 3, 90, 0, 0))]
        full_edges = {frozenset(("a", "b")): accepted[0][2]}
        calls = []
        real_align = sim.align

        def counting(*args, **kwargs):
            calls.append(args)
            return real_align(*args, **kwargs)

        with mock.patch.object(sim, "align", side_effect=counting):
            groups = SourcesMixin._similar_finalize(
                {"a": _meta(200), "b": _meta(200)}, ["a", "b"],
                accepted, {}, fpsf, sim.FAST, full_edges)
        self.assertEqual(calls, [])              # 命中 full_edges，不再重算
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["keep"], "a")

    def test_bridged_member_kicked_by_terminal_verify(self):
        # a≈b、b≈c，但 a vs c 汉明 13 > 阈值 → 桥接链被终验拆开，c 被踢
        hashes = {"a": 0x0000, "b": 0x3FF, "c": 0x1FFF}
        fps1 = {p: {i: _const_frames(h, 10) for i in range(5)} for p, h in hashes.items()}
        metas = {p: _meta(200) for p in "abc"}
        r = sim.AlignResult(0, 3, 3, 30, 0, 0)
        groups = SourcesMixin._similar_finalize(
            metas, ["a", "b", "c"],
            [("a", "b", r), ("b", "c", r)], fps1, {}, sim.NORMAL, {})
        self.assertEqual(len(groups), 1)
        paths = {it["path"] for it in groups[0]["items"]}
        self.assertEqual(paths, {"a", "b"})      # c 被踢
        self.assertNotIn("c", paths)


class TestSimilarScanFlow(_SimilarTestBase):
    def _write_fake_videos(self, names):
        paths = []
        for n in names:
            p = self.root / n
            p.write_bytes(b"x")
            paths.append(str(p))
        return paths

    def _patch_discovery(self, paths, metas):
        import gui_app.discovery as discovery
        for name, side in (("collect_all", lambda: list(paths)),
                           ("probe_video", lambda p: metas[p]),
                           ("prune_probe_cache", lambda ps: None)):
            p = mock.patch.object(discovery, name, side_effect=side)
            p.start()
            self.addCleanup(p.stop)

    def test_busy_when_another_scan_holds_lock(self):
        from gui_app.api_mixins import sources as src
        if not src._similar_lock.acquire(blocking=False):
            self.skipTest("锁被占用")
        self.addCleanup(src._similar_lock.release)
        r = SourcesMixin().find_similar_versions("fast")
        self.assertTrue(r.get("busy"))
        self.assertNotIn("groups", r)

    def test_full_pipeline_groups_identical_pair(self):
        a, b, c = self._write_fake_videos(["a.mp4", "b.mp4", "c.mp4"])
        metas = {a: _meta(200), b: _meta(200), c: _meta(300)}
        self._patch_discovery([a, b, c], metas)
        good = {i: _const_frames(0xF0F0, 10) for i in range(5)}
        bad = {i: _const_frames(0x111 << 20, 10) for i in range(5)}

        def fake_fp(path, duration, is_ts, idxs, stop, mode=None):
            src_frames = good if path in (a, b) else bad
            return {i: list(src_frames[i]) for i in idxs}

        with mock.patch.object(sim, "fingerprint_windows", side_effect=fake_fp):
            r = SourcesMixin().find_similar_versions("fast")
        self.assertEqual(len(r.get("groups", [])), 1)
        paths = {it["path"] for it in r["groups"][0]["items"]}
        self.assertEqual(paths, {a, b})          # c 时长差过大未配对，不进组

    def test_stop_event_wired_and_cleared_after_scan(self):
        from gui_app.api_mixins import sources as src
        a, b = self._write_fake_videos(["a.mp4", "b.mp4"])
        metas = {a: _meta(200), b: _meta(200)}
        self._patch_discovery([a, b], metas)
        started = threading.Event()

        def blocking_fp(path, duration, is_ts, idxs, stop, mode=None):
            started.set()
            stop.wait(5)                          # 挂起直到取消（或 5s 超时兜底）
            return {i: [] for i in idxs}

        result = {}

        def run():
            result["r"] = SourcesMixin().find_similar_versions("fast")

        with mock.patch.object(sim, "fingerprint_windows", side_effect=blocking_fp):
            th = threading.Thread(target=run)
            th.start()
            self.assertTrue(started.wait(5))
            self.assertTrue(SourcesMixin().stop_similar_scan()["stopped"])
            th.join(10)
        self.assertFalse(th.is_alive())
        self.assertEqual(result["r"], {"groups": []})
        self.assertIsNone(src._similar_stop)      # 扫描结束后事件引用被清理
        self.assertFalse(SourcesMixin().stop_similar_scan()["stopped"])


if __name__ == "__main__":
    unittest.main()
