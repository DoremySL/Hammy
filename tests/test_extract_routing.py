import sys
import threading
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from batch_rename import video as video_mod
from batch_rename.config import Config


def _make_cfg(mode):
    c = Config(sampling_points=5, frames_per_point=3, frame_extract_mode=mode)
    c.validate()
    return c


_META_MP4 = {"duration": 100.0, "start_time": 0.0, "format_name": "matroska,webm",
             "video": {"frame_rate": 24.0}, "audio": []}
_META_TS = {"duration": 100.0, "start_time": 1.442, "format_name": "mpegts",
            "video": {"frame_rate": 24.0}, "audio": []}


class _RouterHarness:
    """mock probe/抽帧底座，记录路由命中。"""

    def __init__(self, meta, key_ts, timestamps, concat_frames=None, per_frames=None):
        self.meta = meta
        self.key_ts = key_ts
        self.timestamp_list = timestamps
        self.concat_calls = []      # (targets, frame_rate)
        self.per_frame_calls = []   # (ts, start_time)
        self.ts_pass_calls = 0
        self.concat_frames = concat_frames
        self.per_frames = per_frames or []

    def _probe(self, path, stop, points, per_point):
        return dict(self.meta), list(self.key_ts)

    def _select(self, duration, key_ts, points, per_point):
        return list(self.timestamp_list)

    def _concat(self, path, targets, config, stop, frame_rate, frame_vf=None):
        self.concat_calls.append((list(targets), frame_rate))
        return list(self.concat_frames) if self.concat_frames is not None else None

    def _per_frame(self, path, ts, config, stop, frame_vf=None, start_time=0.0):
        self.per_frame_calls.append((ts, start_time))
        return self.per_frames.pop(0) if self.per_frames else None

    def _ts_pass(self, path, targets, config, stop, frame_rate, frame_vf=None):
        self.ts_pass_calls += 1
        return [None] * len(targets)

    def run(self, cfg):
        with mock.patch.object(video_mod, "probe_video", self._probe), \
             mock.patch.object(video_mod, "select_sampled_timestamps", self._select), \
             mock.patch.object(video_mod, "_extract_frames_concat", self._concat), \
             mock.patch.object(video_mod, "_extract_frame", self._per_frame), \
             mock.patch.object(video_mod, "_extract_frames_ts", self._ts_pass), \
             mock.patch.object(video_mod.ffmpeg_tools, "ffprobe", "ffprobe-mock"), \
             mock.patch.object(video_mod.ffmpeg_tools, "ffmpeg", "ffmpeg-mock"):
            return video_mod.probe_and_extract_keyframes("x.mp4", cfg, threading.Event())


class TestExtractRouting(unittest.TestCase):
    def test_auto_mp4_keyframe_aligned_uses_concat(self):
        h = _RouterHarness(_META_MP4, [0.0, 50.0], [0.0, 50.0],
                           concat_frames=["F0", "F1"])
        _, frames = h.run(_make_cfg("auto"))
        self.assertEqual(len(h.concat_calls), 1)
        self.assertEqual(h.per_frame_calls, [])
        self.assertEqual(frames, ["F0", "F1"])

    def test_auto_concat_mismatch_falls_back_to_per_frame(self):
        h = _RouterHarness(_META_MP4, [0.0, 50.0], [0.0, 50.0],
                           concat_frames=None)   # 配帧不齐 → None
        _, frames = h.run(_make_cfg("auto"))
        self.assertEqual(len(h.concat_calls), 1)
        self.assertEqual([ts for ts, _ in h.per_frame_calls], [0.0, 50.0])
        self.assertEqual(frames, [])

    def test_auto_non_aligned_targets_go_per_frame(self):
        # key_ts 为空（时长均分兜底点位）不许 concat
        h = _RouterHarness(_META_MP4, [], [25.0, 75.0])
        h.run(_make_cfg("auto"))
        self.assertEqual(h.concat_calls, [])
        self.assertEqual([ts for ts, _ in h.per_frame_calls], [25.0, 75.0])

    def test_auto_partially_aligned_go_per_frame(self):
        h = _RouterHarness(_META_MP4, [0.0, 50.0], [0.0, 61.7])
        h.run(_make_cfg("auto"))
        self.assertEqual(h.concat_calls, [])
        self.assertEqual([ts for ts, _ in h.per_frame_calls], [0.0, 61.7])

    def test_auto_ts_go_per_frame_with_start_time(self):
        h = _RouterHarness(_META_TS, [0.0, 50.0], [0.0, 50.0])
        h.run(_make_cfg("auto"))
        self.assertEqual(h.ts_pass_calls, 0)
        self.assertEqual(h.concat_calls, [])
        self.assertEqual([st for _, st in h.per_frame_calls], [1.442, 1.442])

    def test_legacy_ts_uses_single_pass(self):
        h = _RouterHarness(_META_TS, [0.0, 50.0], [0.0, 50.0])
        h.run(_make_cfg("legacy"))
        self.assertEqual(h.ts_pass_calls, 1)
        self.assertEqual(h.per_frame_calls, [])

    def test_legacy_mp4_no_concat(self):
        h = _RouterHarness(_META_MP4, [0.0, 50.0], [0.0, 50.0])
        h.run(_make_cfg("legacy"))
        self.assertEqual(h.concat_calls, [])
        self.assertEqual(len(h.per_frame_calls), 2)

    def test_per_frame_mode_no_concat(self):
        h = _RouterHarness(_META_MP4, [0.0, 50.0], [0.0, 50.0])
        h.run(_make_cfg("per_frame"))
        self.assertEqual(h.concat_calls, [])
        self.assertEqual(len(h.per_frame_calls), 2)

    def test_invalid_mode_clamped_to_auto(self):
        c = Config(frame_extract_mode="whatever")
        c.validate()
        self.assertEqual(c.frame_extract_mode, "auto")


class TestSingleFrameSubprocess(unittest.TestCase):
    def _run(self, ts, start_time):
        captured = {}

        def fake_run(cmd, timeout, stop):
            captured["cmd"] = cmd
            return b"\xff\xd8\xff\xe0tail", b""

        with mock.patch.object(video_mod, "run_subprocess_with_cancel", fake_run):
            jpeg = video_mod._extract_single_frame("v.mp4", ts, "scale", threading.Event(),
                                                   start_time)
        return jpeg, captured["cmd"]

    def test_start_time_deduction_and_six_decimals(self):
        jpeg, cmd = self._run(5.5, 1.0)
        self.assertIsNotNone(jpeg)
        idx = cmd.index("-ss")
        self.assertEqual(cmd[idx + 1], "4.500000")

    def test_negative_ss_clamped_to_zero(self):
        _, cmd = self._run(0.5, 1.0)
        self.assertEqual(cmd[cmd.index("-ss") + 1], "0.000000")


class TestParseFormatStartTime(unittest.TestCase):
    def test_start_time_parsed(self):
        meta = video_mod._parse_format({"duration": "10", "start_time": "1.442000"})
        self.assertAlmostEqual(meta["start_time"], 1.442)

    def test_start_time_missing_is_zero(self):
        meta = video_mod._parse_format({"duration": "10"})
        self.assertEqual(meta["start_time"], 0.0)

    def test_start_time_negative_clamped(self):
        meta = video_mod._parse_format({"start_time": "-0.5"})
        self.assertEqual(meta["start_time"], 0.0)


if __name__ == "__main__":
    unittest.main()
