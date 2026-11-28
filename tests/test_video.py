import json
import sys
import threading
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from batch_rename import video as video_mod
from batch_rename.video import (select_point_samples,
                                _parse_frame_rate, _parse_format,
                                _anchor_keyframe_indices,
                                _kf_from_csv_line, _point_regions,
                                _probe_keyframes_intervals,
                                _write_ffconcat_script,
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


class TestSelectPointSamplesNoKeyframes(unittest.TestCase):
    """无关键帧/空分区：同一套点位分区，补普通帧位置。"""

    def test_per_point_one_uses_point_positions(self):
        ts = [t for t, _ in select_point_samples(100.0, [], 5, 1)]
        self.assertEqual([round(t, 3) for t in ts], [0.0, 25.0, 50.0, 75.0, 99.967])

    def test_per_point_multi_even_inside_region(self):
        ts = [t for t, _ in select_point_samples(100.0, [], 5, 3)]
        self.assertEqual([round(t, 2) for t in ts],
                         [1.25, 2.5, 3.75, 22.5, 25.0, 27.5, 47.5, 50.0, 52.5,
                          72.5, 75.0, 77.5, 96.25, 97.5, 98.75])
        self.assertEqual([pi for _, pi in select_point_samples(100.0, [], 5, 3)],
                         [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3)

    def test_mixed_regions_kf_and_fill(self):
        s = select_point_samples(100.0, [1.0], 5, 1)
        self.assertEqual([(round(t, 3), pi) for t, pi in s],
                         [(1.0, 0), (25.0, 1), (50.0, 2), (75.0, 3), (99.967, 4)])

    def test_no_duration_uniform_over_keyframes(self):
        self.assertEqual(select_point_samples(0.0, [2.0, 4.0, 6.0, 8.0], 3, 1),
                         [(2.0, 0), (6.0, 1), (8.0, 2)])
        self.assertEqual(select_point_samples(0.0, [2.0, 4.0], 3, 1),
                         [(2.0, 0), (4.0, 1)])

    def test_no_duration_no_keyframes_first_frame(self):
        self.assertEqual(select_point_samples(0.0, [], 3, 1), [(0.0, 0)])


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
                               return_value=(0, out, b"")):
            kf = _probe_keyframes_intervals("Z:\\a.mkv", 100.0, 10, threading.Event())
        self.assertEqual(kf, [1.0, 3.0])

    def test_none_stdout(self):
        with mock.patch.object(video_mod, "run_subprocess_with_cancel",
                               return_value=(None, None, None)):
            self.assertEqual(
                _probe_keyframes_intervals("Z:\\a.mkv", 100.0, 10,
                                           threading.Event()), [])

    def test_start_time_converted_at_boundary(self):
        # 非零 start_time：位置按 pts 换入，出口换回内容时间
        out = b"101.442,K_\n103.442,K_\n"
        specs = []

        def fake_run(cmd, timeout, stop_event):
            specs.append(cmd[cmd.index("-read_intervals") + 1])
            return 0, out, b""

        with mock.patch.object(video_mod, "run_subprocess_with_cancel",
                               side_effect=fake_run):
            kf = _probe_keyframes_intervals("Z:\\a.ts", 90.0, 10, threading.Event(),
                                            start_time=1.442)
        self.assertEqual(specs, ["1.442%+90.000"])
        self.assertEqual(kf, [100.0, 102.0])

    def _read_intervals(self, duration, points, anchors=None):
        specs = []

        def fake_run(cmd, timeout, stop_event):
            spec = cmd[cmd.index("-read_intervals") + 1]
            specs.append(spec)
            if spec.endswith("%+0.001"):
                if anchors is None:
                    return (0, b"", b"")  # 模拟锚点定位失败
                return (0, "".join(f"{a:.3f},K__\n" for a in anchors).encode(), b"")
            return (0, b"", b"")

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
        # 点位 0/50/100，各分区内关键帧充足 → 每点最多 2 帧；100 与 99 同秒被 1s 间隔过滤
        kf = [0.0, 1.5, 49.0, 50.5, 52.0, 99.0, 100.0]
        ts = select_point_samples(100.0, kf, 3, 2)
        self.assertEqual(ts, [(0.0, 0), (1.5, 0), (49.0, 1), (52.0, 1), (99.0, 2)])

    def test_cluster_does_not_steal_other_positions(self):
        # 首窗密集关键帧经 1s 间隔过滤后隔一取一 + 其余各窗 2 个：5 个点位各自出帧
        kf = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 49.0, 51.0, 99.0, 101.0, 149.0, 151.0, 199.0]
        ts = select_point_samples(200.0, kf, 5, 2)
        near = {p: [t for t, _ in ts if abs(t - p) <= 5] for p in (0, 50, 100, 150, 200)}
        self.assertTrue(all(near[p] for p in near), f"有点位没帧: {near}")
        self.assertEqual(ts, [(0.0, 0), (4.0, 0), (49.0, 1), (51.0, 1), (99.0, 2),
                              (101.0, 2), (149.0, 3), (151.0, 3), (199.0, 4)])

    def test_per_point_one_nearest_to_position(self):
        kf = [0.0, 1.5, 49.0, 50.5, 52.0, 99.0]
        ts = select_point_samples(100.0, kf, 3, 1)
        self.assertEqual(ts, [(0.0, 0), (50.5, 1), (99.0, 2)])

    def test_short_video_points_do_not_collapse(self):
        # 20s、5 点位：分区互斥，关键帧按最近点位归属；末分区无关键帧补普通帧位
        ts = select_point_samples(20.0, [0.0, 5.0, 10.0, 15.0], 5, 3)
        self.assertEqual([(round(t, 3), pi) for t, pi in ts],
                         [(0.0, 0), (5.0, 1), (10.0, 2), (15.0, 3),
                          (18.125, 4), (18.75, 4), (19.375, 4)])

    def test_gap_filter_falls_back_to_fill(self):
        # 20s/3 点位相邻分区共享边界：区域 1 唯一关键帧与 4.9 间隔不足 1s → 被过滤回退普通帧位
        ts = select_point_samples(20.0, [4.9, 5.2], 3, 2)
        self.assertEqual([(round(t, 3), pi) for t, pi in ts],
                         [(4.9, 0), (8.333, 1), (11.667, 1),
                          (16.667, 2), (18.333, 2)])

    def test_no_duration_gap_filter(self):
        # duration 缺失：关键帧先经 1s 间隔过滤再分配点位
        self.assertEqual(select_point_samples(0.0, [1.0, 1.5, 3.0], 3, 1),
                         [(1.0, 0), (3.0, 1)])


class TestWriteFfconcatScript(unittest.TestCase):
    def _read(self, path):
        with open(path, encoding="utf-8") as f:
            content = f.read()
        Path(path).unlink()
        return content

    def test_plain_path(self):
        script = _write_ffconcat_script(r"D:\a\b.mp4", [(10.0, 10.02)])
        self.assertEqual(
            self._read(script),
            "ffconcat version 1.0\n"
            "file 'D:\\a\\b.mp4'\n"
            "inpoint 10.000000\noutpoint 10.020000\n")

    def test_quote_in_path_escaped(self):
        # 路径含 ' 需拼接转义
        script = _write_ffconcat_script(r"D:\x\quote'video.mp4", [(0.0, 0.02)])
        self.assertIn("file 'D:\\x\\quote'\\''video.mp4'",
                      self._read(script))


class TestProbeVideoSampling(unittest.TestCase):
    def test_sampling_probe_used(self):
        meta_json = json.dumps(
            {"format": {"duration": "90.0", "format_name": "matroska"}}).encode()
        calls = []

        def fake_run(cmd, timeout, stop_event):
            calls.append((cmd, timeout))
            if "-read_intervals" not in cmd:
                return 0, meta_json, b""
            return 0, b"1.0,K_\n5.0,K_\n", b""

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
