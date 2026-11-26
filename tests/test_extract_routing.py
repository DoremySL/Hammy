import sys
import threading
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from batch_rename import video as video_mod
from batch_rename.config import Config
from batch_rename.types import Frame


def _make_cfg(mode):
    c = Config(sampling_points=5, frames_per_point=3, frame_extract_mode=mode)
    c.validate()
    return c


_META_MP4 = {"duration": 100.0, "start_time": 0.0, "format_name": "matroska,webm",
             "video": {"frame_rate": 24.0}, "audio": []}


def _frames(*ts_list):
    """构造测试帧（b64 为标记串，点位索引 = 序号，预处理在 harness 中打桩为透传）。"""
    return [Frame(t, f"F{i}", i) for i, t in enumerate(ts_list)]


class _RouterHarness:
    """mock probe/抽帧底座，记录路由命中。"""

    def __init__(self, meta, key_ts, timestamps, concat_frames=None, per_frames=None):
        self.meta = meta
        self.key_ts = key_ts
        self.timestamp_list = timestamps
        self.concat_calls = []      # (targets, frame_rate, start_time)
        self.per_frame_calls = []   # ts 列表
        self.sample_calls = []      # (duration, key_ts)
        self.concat_frames = concat_frames
        self.per_frames = per_frames or []

    def _probe(self, path, stop, points):
        return dict(self.meta), list(self.key_ts)

    def _samples(self, duration, key_ts, points, per_point, frame_rate=0.0):
        self.sample_calls.append((duration, list(key_ts)))
        return [(t, i) for i, t in enumerate(self.timestamp_list)]

    def _concat(self, path, samples, config, stop, frame_rate, frame_vf=None,
                start_time=0.0):
        self.concat_calls.append(([t for t, _ in samples], frame_rate, start_time))
        return list(self.concat_frames) if self.concat_frames is not None else None

    def _per_frame(self, path, ts, pi, config, stop, frame_vf=None):
        self.per_frame_calls.append(ts)
        return self.per_frames.pop(0) if self.per_frames else None

    def run(self, cfg):
        with mock.patch.object(video_mod, "probe_video", self._probe), \
             mock.patch.object(video_mod, "select_point_samples", self._samples), \
             mock.patch.object(video_mod, "preprocess_frames",
                               lambda frames: list(frames)), \
             mock.patch.object(video_mod, "_extract_frames_concat", self._concat), \
             mock.patch.object(video_mod, "_extract_frame", self._per_frame), \
             mock.patch.object(video_mod.ffmpeg_tools, "ffprobe", "ffprobe-mock"), \
             mock.patch.object(video_mod.ffmpeg_tools, "ffmpeg", "ffmpeg-mock"):
            return video_mod.probe_and_extract_keyframes("x.mp4", cfg, threading.Event())


class TestExtractRouting(unittest.TestCase):
    def test_auto_mp4_keyframe_aligned_uses_concat(self):
        h = _RouterHarness(_META_MP4, [0.0, 50.0], [0.0, 50.0],
                           concat_frames=_frames(0.0, 50.0))
        _, frames = h.run(_make_cfg("auto"))
        self.assertEqual(len(h.concat_calls), 1)
        self.assertEqual(h.per_frame_calls, [])
        self.assertEqual(frames, [Frame(0.0, "F0", 0), Frame(50.0, "F1", 1)])

    def test_auto_concat_mismatch_falls_back_to_per_frame(self):
        h = _RouterHarness(_META_MP4, [0.0, 50.0], [0.0, 50.0],
                           concat_frames=None)   # 配帧不齐 → None
        _, frames = h.run(_make_cfg("auto"))
        self.assertEqual(len(h.concat_calls), 1)
        self.assertEqual(sorted(h.per_frame_calls), [0.0, 50.0])
        self.assertEqual(frames, [])

    def test_auto_concat_receives_start_time(self):
        # start_time 传给 concat 供 inpoint 换算
        meta = dict(_META_MP4, start_time=2.8)
        h = _RouterHarness(meta, [0.0, 50.0], [0.0, 50.0],
                           concat_frames=_frames(0.0, 50.0))
        h.run(_make_cfg("auto"))
        self.assertEqual(h.concat_calls[0][2], 2.8)
        self.assertEqual(h.per_frame_calls, [])

    def test_auto_paired_keyframes_skip_concat(self):
        # 目标后 1 帧处还有关键帧（编码器成对插入）→ concat 段会多吐帧，直接逐帧
        h = _RouterHarness(_META_MP4, [0.0, 50.0, 50.042], [0.0, 50.0])
        h.run(_make_cfg("auto"))
        self.assertEqual(h.concat_calls, [])
        self.assertEqual(sorted(h.per_frame_calls), [0.0, 50.0])

    def test_auto_non_aligned_targets_go_per_frame(self):
        # key_ts 为空（时长均分兜底点位）不许 concat
        h = _RouterHarness(_META_MP4, [], [25.0, 75.0])
        h.run(_make_cfg("auto"))
        self.assertEqual(h.concat_calls, [])
        self.assertEqual(sorted(h.per_frame_calls), [25.0, 75.0])

    def test_auto_partially_aligned_go_per_frame(self):
        h = _RouterHarness(_META_MP4, [0.0, 50.0], [0.0, 61.7])
        h.run(_make_cfg("auto"))
        self.assertEqual(h.concat_calls, [])
        self.assertEqual(sorted(h.per_frame_calls), [0.0, 61.7])

    def test_per_frame_mode_no_concat(self):
        h = _RouterHarness(_META_MP4, [0.0, 50.0], [0.0, 50.0])
        h.run(_make_cfg("per_frame"))
        self.assertEqual(h.concat_calls, [])
        self.assertEqual(len(h.per_frame_calls), 2)

    def test_invalid_mode_clamped_to_auto(self):
        for mode in ("whatever", "legacy"):
            c = Config(frame_extract_mode=mode)
            c.validate()
            self.assertEqual(c.frame_extract_mode, "auto")

    def test_duration_falls_back_to_stream_duration(self):
        # format duration 缺失（0）时用视频流时长选点
        meta = {"duration": 0.0, "start_time": 0.0, "format_name": "matroska,webm",
                "video": {"frame_rate": 24.0, "duration": 123.0}, "audio": []}
        h = _RouterHarness(meta, [1.0], [1.0])
        h.run(_make_cfg("auto"))
        self.assertEqual(h.sample_calls, [(123.0, [1.0])])


class TestSingleFrameSubprocess(unittest.TestCase):
    def test_content_ts_six_decimals(self):
        captured = {}

        def fake_run(cmd, timeout, stop):
            captured["cmd"] = cmd
            return b"\xff\xd8\xff\xe0tail", b""

        with mock.patch.object(video_mod, "run_subprocess_with_cancel", fake_run):
            jpeg = video_mod._extract_single_frame("v.mp4", 5.5, "scale",
                                                  threading.Event())
        self.assertIsNotNone(jpeg)
        idx = captured["cmd"].index("-ss")
        self.assertEqual(captured["cmd"][idx + 1], "5.500000")


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
