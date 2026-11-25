import json
import sys
import threading
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from batch_rename import video as video_mod
from batch_rename.video import (select_timestamps, select_point_samples,
                                _parse_frame_rate, _parse_format,
                                _anchor_keyframe_indices,
                                _kf_from_csv_line, _point_regions,
                                _probe_keyframes_intervals,
                                _PROBE_KEYFRAMES_TIMEOUT_SEC, probe_video)


class TestParseFormatCreationTime(unittest.TestCase):
    """_parse_format 的 creation_time 过滤：只滤 "0000-" 前缀的占位日期，
    不得误杀带微秒的合法时间戳（ffprobe 常输出 .000000Z）。"""

    def test_microseconds_preserved(self):
        meta = _parse_format({"tags": {"creation_time": "2023-06-01T10:00:00.000000Z"}})
        self.assertEqual(meta["creation_time"], "2023-06-01T10:00:00.000000Z")

    def test_no_microseconds_preserved(self):
        meta = _parse_format({"tags": {"creation_time": "2023-06-01T10:00:00Z"}})
        self.assertEqual(meta["creation_time"], "2023-06-01T10:00:00Z")

    def test_zero_placeholder_filtered(self):
        meta = _parse_format({"tags": {"creation_time": "0000-00-00T00:00:00.000000Z"}})
        self.assertEqual(meta["creation_time"], "")

    def test_empty(self):
        meta = _parse_format({"tags": {}})
        self.assertEqual(meta["creation_time"], "")


class TestParseFrameRate(unittest.TestCase):
    def test_fraction(self):
        self.assertAlmostEqual(_parse_frame_rate("30000/1001"), 29.97, places=2)

    def test_plain(self):
        self.assertEqual(_parse_frame_rate("25"), 25.0)

    def test_empty(self):
        self.assertEqual(_parse_frame_rate(""), 0.0)

    def test_zero_denominator(self):
        self.assertEqual(_parse_frame_rate("0/0"), 0.0)


class TestSelectTimestamps(unittest.TestCase):
    def test_short_video(self):
        self.assertEqual(select_timestamps(5.0, [], 6), [0.0, 2.5, 4.9])

    def test_short_video_with_keyframes(self):
        self.assertEqual(select_timestamps(5.0, [0, 1, 2, 3, 4], 3), [0, 2, 4])

    def test_keyframes_enough(self):
        self.assertEqual(select_timestamps(100.0, [0, 1, 2, 3, 4], 3), [0, 2, 4])

    def test_keyframes_fewer(self):
        self.assertEqual(select_timestamps(100.0, [1.0, 2.0], 5), [1.0, 2.0])

    def test_duration_split(self):
        self.assertEqual(select_timestamps(10.0, [], 3), [2.5, 5.0, 7.5])

    def test_fallback_zero(self):
        self.assertEqual(select_timestamps(0.0, [], 3), [0.0])


class TestAnchors(unittest.TestCase):
    def test_anchor_indices(self):
        self.assertEqual(_anchor_keyframe_indices(100, 5), [0, 25, 50, 74, 99])


class TestKfFromCsvLine(unittest.TestCase):
    def test_keyframe(self):
        self.assertEqual(_kf_from_csv_line(b"1.234567,K_"), 1.234567)

    def test_non_keyframe(self):
        self.assertIsNone(_kf_from_csv_line(b"1.234567,__"))

    def test_invalid_lines(self):
        self.assertIsNone(_kf_from_csv_line(b"N/A,K"))
        self.assertIsNone(_kf_from_csv_line(b"abc,K"))
        self.assertIsNone(_kf_from_csv_line(b"1.0"))
        self.assertIsNone(_kf_from_csv_line(b"-0.5,K"))


class TestProbeKeyframesIntervals(unittest.TestCase):
    def test_duration_nonpositive(self):
        self.assertEqual(
            _probe_keyframes_intervals("Z:\\a.mkv", 0.0, 10, threading.Event()), [])

    def test_parses_and_dedups(self):
        out = b"1.0,K_\n2.0,__\n3.0,K_\n1.0,K_\n"
        with mock.patch.object(video_mod, "run_subprocess_with_cancel",
                               return_value=(out, b"")):
            kf = _probe_keyframes_intervals("Z:\\a.mkv", 100.0, 10, threading.Event())
        self.assertEqual(kf, [1.0, 3.0])

    def test_none_stdout(self):
        with mock.patch.object(video_mod, "run_subprocess_with_cancel",
                               return_value=(None, None)):
            self.assertEqual(
                _probe_keyframes_intervals("Z:\\a.mkv", 100.0, 10,
                                           threading.Event()), [])

    def _read_intervals(self, duration, points, anchors=None):
        specs = []

        def fake_run(cmd, timeout, stop_event):
            spec = cmd[cmd.index("-read_intervals") + 1]
            specs.append(spec)
            if spec.endswith("%+0.001"):
                if anchors is None:
                    return (b"", b"")  # 模拟锚点定位失败
                return ("".join(f"{a:.3f},K__\n" for a in anchors).encode(), b"")
            return (b"", b"")

        with mock.patch.object(video_mod, "run_subprocess_with_cancel",
                               side_effect=fake_run):
            _probe_keyframes_intervals("Z:\\a.mkv", duration, points,
                                       threading.Event())
        return specs

    def test_short_video_windows_merge_to_one(self):
        # 90s、10 点位 → 单区间整段直读（EOF 截断）
        specs = self._read_intervals(90.0, 10)
        self.assertEqual(specs, ["0.000%+90.000"])

    def test_long_video_reads_from_anchors(self):
        # 1000s、5 点位：先锚点定位，再从锚点精确读到分区右界
        specs = self._read_intervals(1000.0, 5,
                                      anchors=[0.0, 205.0, 455.0, 705.0, 955.0])
        self.assertEqual(specs[0],
                         "0.000%+0.001,208.333%+0.001,458.333%+0.001,"
                         "708.333%+0.001,958.333%+0.001")
        self.assertEqual(
            specs[-1], "0.000%+41.717,205.000%+86.717,455.000%+86.717,"
                       "705.000%+86.717,955.000%+45.050")

    def test_anchor_mismatch_falls_back_to_doubled(self):
        # 锚点定位失败 → 时长翻倍兜底
        specs = self._read_intervals(1000.0, 5)
        self.assertEqual(
            specs[-1], "0.000%+83.333,208.333%+166.667,458.333%+166.667,"
                       "708.333%+166.667,958.333%+83.333")

    def test_radius_independent_of_frame_quota(self):
        # 分区半径只由点位间距决定（间距 250s → 半径 = 间距/6），与每点帧数无关
        regions = _point_regions(1000.0, 5)
        self.assertEqual([(round(lo, 3), round(hi, 3)) for _, _, lo, hi in regions],
                         [(0.0, 41.667), (208.333, 291.667), (458.333, 541.667),
                          (708.333, 791.667), (958.333, 1000.0)])


class TestPointRegions(unittest.TestCase):
    def test_short_video_tiles(self):
        # 20s、5 点位：间距 5s，半径 2.5s，分区连续且互斥
        regions = _point_regions(20.0, 5)
        self.assertEqual([(round(lo, 3), round(hi, 3)) for _, _, lo, hi in regions],
                         [(0.0, 2.5), (2.5, 7.5), (7.5, 12.5),
                          (12.5, 17.5), (17.5, 20.0)])

    def test_single_point(self):
        # 单点位取中点；短视频半径收窄到覆盖全片
        regions = _point_regions(6.0, 1)
        self.assertEqual([(round(lo, 3), round(hi, 3)) for _, _, lo, hi in regions],
                         [(0.0, 6.0)])

    def test_radius_grows_with_spacing(self):
        # 30min、9 点位：间距 225s → 半径 = 间距/6 = 37.5s（未顶格）
        regions = _point_regions(1800.0, 9)
        self.assertEqual([(round(lo, 1), round(hi, 1)) for _, _, lo, hi in regions[:2]],
                         [(0.0, 37.5), (187.5, 262.5)])

    def test_radius_capped(self):
        # 2h、9 点位：间距 900s → 半径顶格 60s
        regions = _point_regions(7200.0, 9)
        self.assertEqual([(round(lo, 1), round(hi, 1)) for _, _, lo, hi in regions[:2]],
                         [(0.0, 60.0), (840.0, 960.0)])

    def test_radius_floor_between(self):
        # 间距 10~30s：半径为基础值 5s
        regions = _point_regions(180.0, 9)
        self.assertEqual([(round(lo, 1), round(hi, 1)) for _, _, lo, hi in regions[1:2]],
                         [(17.5, 27.5)])


class TestSelectPointSamples(unittest.TestCase):
    """分区选帧：每个点位在自己的分区内最多取 per_point 个关键帧。"""

    def test_each_position_contributes(self):
        # 点位 0/50/100，各分区内关键帧充足 → 每点 2 帧
        kf = [0.0, 1.5, 49.0, 50.5, 52.0, 99.0, 100.0]
        ts = select_point_samples(100.0, kf, 3, 2)
        self.assertEqual(ts, [(0.0, 0), (1.5, 0), (49.0, 1), (52.0, 1),
                              (99.0, 2), (100.0, 2)])

    def test_cluster_does_not_steal_other_positions(self):
        # 首窗 6 个密集关键帧 + 其余各窗 2 个：5 个点位各自出帧
        kf = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 49.0, 51.0, 99.0, 101.0, 149.0, 151.0, 199.0]
        ts = select_point_samples(200.0, kf, 5, 2)
        near = {p: [t for t, _ in ts if abs(t - p) <= 5] for p in (0, 50, 100, 150, 200)}
        self.assertTrue(all(near[p] for p in near), f"有点位没帧: {near}")
        self.assertEqual(ts, [(0.0, 0), (5.0, 0), (49.0, 1), (51.0, 1), (99.0, 2),
                              (101.0, 2), (149.0, 3), (151.0, 3), (199.0, 4)])

    def test_per_point_one_nearest_to_position(self):
        kf = [0.0, 1.5, 49.0, 50.5, 52.0, 99.0]
        ts = select_point_samples(100.0, kf, 3, 1)
        self.assertEqual(ts, [(0.0, 0), (50.5, 1), (99.0, 2)])

    def test_empty_falls_back_to_duration_split(self):
        ts = select_point_samples(100.0, [], 5, 3)
        self.assertEqual([round(t, 1) for t, _ in ts], [16.7, 33.3, 50.0, 66.7, 83.3])
        self.assertEqual([pi for _, pi in ts], [0, 1, 2, 3, 4])

    def test_short_video_points_do_not_collapse(self):
        # 20s、5 点位：分区互斥，关键帧按最近点位归属
        ts = select_point_samples(20.0, [0.0, 5.0, 10.0, 15.0], 5, 3)
        self.assertEqual(ts, [(0.0, 0), (5.0, 1), (10.0, 2), (15.0, 3)])


class TestProbeVideoSampling(unittest.TestCase):
    def test_sampling_probe_used(self):
        meta_json = json.dumps(
            {"format": {"duration": "90.0", "format_name": "matroska"}}).encode()
        calls = []

        def fake_run(cmd, timeout, stop_event):
            calls.append((cmd, timeout))
            if "-read_intervals" not in cmd:
                return meta_json, b""
            return b"1.0,K_\n5.0,K_\n", b""

        with mock.patch.object(video_mod, "run_subprocess_with_cancel",
                               side_effect=fake_run):
            info, kf = probe_video("Z:\\a.mkv", threading.Event(), 10)
        self.assertEqual(info["duration"], 90.0)
        self.assertEqual(kf, [1.0, 5.0])
        self.assertEqual(len(calls), 2)  # 90s 单区间：元数据 + 整段直读
        self.assertIn("-read_intervals", calls[1][0])
        self.assertEqual(calls[1][1], _PROBE_KEYFRAMES_TIMEOUT_SEC)


if __name__ == "__main__":
    unittest.main()
