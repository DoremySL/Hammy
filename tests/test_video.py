import json
import sys
import threading
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from batch_rename import video as video_mod
from batch_rename.video import (select_timestamps, select_sampled_timestamps,
                                _parse_frame_rate, _parse_format,
                                _sparsify_keyframes, _anchor_keyframe_indices,
                                _kf_from_csv_line,
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


class TestSparsifyAndAnchors(unittest.TestCase):
    def test_sparsify(self):
        self.assertEqual(_sparsify_keyframes([0.0, 0.5, 1.2, 2.0, 2.4, 5.0], 1.0),
                         [0.0, 1.2, 2.4, 5.0])

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
            _probe_keyframes_intervals("Z:\\a.mkv", 0.0, 10, 3, threading.Event()), [])

    def test_parses_and_dedups(self):
        out = b"1.0,K_\n2.0,__\n3.0,K_\n1.0,K_\n"
        with mock.patch.object(video_mod, "run_subprocess_with_cancel",
                               return_value=(out, b"")):
            kf = _probe_keyframes_intervals("Z:\\a.mkv", 100.0, 10, 3, threading.Event())
        self.assertEqual(kf, [1.0, 3.0])

    def test_none_stdout(self):
        with mock.patch.object(video_mod, "run_subprocess_with_cancel",
                               return_value=(None, None)):
            self.assertEqual(
                _probe_keyframes_intervals("Z:\\a.mkv", 100.0, 10, 3, threading.Event()), [])

    def _read_intervals(self, duration, points, per_point):
        captured = {}

        def fake_run(cmd, timeout, stop_event):
            captured["cmd"] = cmd
            return (b"", b"")

        with mock.patch.object(video_mod, "run_subprocess_with_cancel",
                               side_effect=fake_run):
            _probe_keyframes_intervals("Z:\\a.mkv", duration, points, per_point,
                                       threading.Event())
        return captured["cmd"][captured["cmd"].index("-read_intervals") + 1]

    def test_short_video_windows_merge_to_one(self):
        # 90s、10 点位、半径 6s → 窗口两两重叠，合并为覆盖全片的一个区间
        iv = self._read_intervals(90.0, 10, 3)
        self.assertEqual(iv, "0.000%+90.000")

    def test_long_video_keeps_disjoint_windows(self):
        # 1000s、5 点位、半径 6s → 5 个互不重叠的窗口
        iv = self._read_intervals(1000.0, 5, 3)
        self.assertEqual(
            iv, "0.000%+6.000,244.000%+12.000,494.000%+12.000,"
                "744.000%+12.000,994.000%+6.000")

    def test_half_scales_with_per_point(self):
        # 每点帧数 10 → 半径 20s，窗口随之扩大
        iv = self._read_intervals(1000.0, 5, 10)
        self.assertEqual(
            iv, "0.000%+20.000,230.000%+40.000,480.000%+40.000,"
                "730.000%+40.000,980.000%+20.000")


class TestSelectSampledTimestamps(unittest.TestCase):
    """采样模式选帧：每个点位贡献 per_point 个关键帧，密集窗口不挤占其他点位。"""

    def test_each_position_contributes(self):
        # 点位 0/50/100，各窗口内关键帧充足 → 每点 2 帧
        kf = [0.0, 1.5, 49.0, 50.5, 52.0, 99.0, 100.0]
        ts = select_sampled_timestamps(100.0, kf, 3, 2)
        self.assertEqual(ts, [0.0, 1.5, 49.0, 52.0, 99.0])

    def test_cluster_does_not_steal_other_positions(self):
        # 首窗 6 个密集关键帧 + 其余各窗 2 个：5 个点位各自出帧
        kf = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 49.0, 51.0, 99.0, 101.0, 149.0, 151.0, 199.0]
        ts = select_sampled_timestamps(200.0, kf, 5, 2)
        near = {p: [t for t in ts if abs(t - p) <= 5] for p in (0, 50, 100, 150, 200)}
        self.assertTrue(all(near[p] for p in near), f"有点位没帧: {near}")
        self.assertEqual(ts, [0.0, 4.0, 49.0, 51.0, 99.0, 101.0, 149.0, 151.0, 199.0])

    def test_per_point_one_nearest_to_position(self):
        kf = [0.0, 1.5, 49.0, 50.5, 52.0, 99.0]
        ts = select_sampled_timestamps(100.0, kf, 3, 1)
        self.assertEqual(ts, [0.0, 50.5, 99.0])

    def test_empty_falls_back_to_duration_split(self):
        ts = select_sampled_timestamps(100.0, [], 5, 3)
        self.assertEqual([round(t, 1) for t in ts], [16.7, 33.3, 50.0, 66.7, 83.3])

    def test_overlapping_windows_dedup(self):
        ts = select_sampled_timestamps(20.0, [0.0, 5.0, 10.0, 15.0], 5, 3)
        self.assertEqual(len(ts), len(set(ts)))
        self.assertEqual(ts, [0.0, 5.0, 10.0, 15.0])


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
            info, kf = probe_video("Z:\\a.mkv", threading.Event(), 10, 3)
        self.assertEqual(info["duration"], 90.0)
        self.assertEqual(kf, [1.0, 5.0])
        self.assertEqual(len(calls), 2)
        self.assertIn("-read_intervals", calls[1][0])
        self.assertEqual(calls[1][1], _PROBE_KEYFRAMES_TIMEOUT_SEC)


if __name__ == "__main__":
    unittest.main()
