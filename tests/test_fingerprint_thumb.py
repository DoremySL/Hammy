import sys
import threading
from pathlib import Path
from subprocess import CompletedProcess
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from batch_rename import similarity as sim
from batch_rename import video as video_mod
from gui_app import discovery


def _gray_raw(n):
    """n 帧非平坦灰度 raw（92 字节占位差异、每帧不同）。"""
    out = b""
    for k in range(n):
        row = bytes([k % 251 + 1] * 9) + bytes([k % 249 + 2] * 8)
        f = row + b"\x10" * (72 - 17)
        out += f
    return out


class TestFingerprintWindows(unittest.TestCase):
    def _capture_run(self):
        calls = []

        def fake_run(cmd, timeout, stop):
            calls.append((list(cmd), timeout))
            return _gray_raw(30), b""

        return calls, fake_run

    def test_short_video_full_decode(self):
        calls, fake_run = self._capture_run()
        with mock.patch.object(sim, "run_subprocess_with_cancel", fake_run):
            frames = sim.fingerprint_windows("v.ts", 100.0, (0, 1, 2, 3, 4),
                                             threading.Event())
        self.assertEqual(len(calls), 1)
        self.assertNotIn("-ss", calls[0][0])
        self.assertEqual(list(frames.keys()), [0])
        self.assertEqual(frames[0][0], (0.0, frames[0][0][1]))

    def test_long_video_windows(self):
        calls, fake_run = self._capture_run()
        with mock.patch.object(sim, "run_subprocess_with_cancel", fake_run):
            frames = sim.fingerprint_windows("v.ts", 2000.0, (0, 1, 2, 3, 4),
                                             threading.Event())
        self.assertEqual(len(calls), 5)
        for (cmd, _), i in zip(calls, range(5)):
            pos = sim.WINDOW_POS[i] * 2000.0
            self.assertEqual(cmd[cmd.index("-ss") + 1], f"{pos:.3f}")
            self.assertEqual(cmd[cmd.index("-t") + 1], "30.000")
            self.assertLess(cmd.index("-ss"), cmd.index("-i"))
            first = frames[i][0]
            self.assertAlmostEqual(first[0], pos)

    def test_ts_long_video_same_windows_as_other_containers(self):
        calls, fake_run = self._capture_run()
        with mock.patch.object(sim, "run_subprocess_with_cancel", fake_run):
            sim.fingerprint_windows("v.ts", 2000.0, (2,), threading.Event())
        cmd = calls[0][0]
        pos = sim.WINDOW_POS[2] * 2000.0
        self.assertEqual(cmd[cmd.index("-ss") + 1], f"{pos:.3f}")


class TestThumbSeekChain(unittest.TestCase):
    def _probe(self, **info):
        base = {"duration": 1200, "start_time": 1.442, "size": 100}
        base.update(info)
        return base

    def test_attempts_long_ts_unified(self):
        with mock.patch.object(discovery, "probe_video",
                               lambda p: self._probe()), \
             mock.patch.object(discovery, "_history_thumb_seconds", lambda vid: None):
            attempts, st = discovery._thumb_seek_attempts("v.ts", "x")
        self.assertEqual(st, 1.442)
        self.assertEqual(attempts, ["120", "1", "0"])

    def test_attempts_thumbs_time_priority(self):
        with mock.patch.object(discovery, "probe_video",
                               lambda p: self._probe()), \
             mock.patch.object(discovery, "_history_thumb_seconds", lambda vid: 300.75):
            attempts, st = discovery._thumb_seek_attempts("v.mkv", "x")
        self.assertEqual(attempts, ["300.750", "120", "1", "0"])

    def test_attempts_tiny_duration(self):
        with mock.patch.object(discovery, "probe_video",
                               lambda p: self._probe(duration=1)), \
             mock.patch.object(discovery, "_history_thumb_seconds", lambda vid: None):
            attempts, st = discovery._thumb_seek_attempts("v.ts", "x")
        self.assertEqual(attempts, ["0"])

    def test_input_seek_deducts_start_time(self):
        captured = {}

        def fake_run(cmd, capture_output, timeout, **kw):
            captured["cmd"] = cmd
            return CompletedProcess(cmd, 0, b"\xff\xd8ok", b"")

        with mock.patch.object(discovery.subprocess, "run", fake_run):
            r = discovery._run_frame_extract("v.ts", "120", 5, start_time=1.442)
        self.assertEqual(r.returncode, 0)
        self.assertEqual(captured["cmd"][captured["cmd"].index("-ss") + 1],
                         f"{120.0 - 1.442:.6f}")
        self.assertLess(captured["cmd"].index("-ss"), captured["cmd"].index("-i"))


if __name__ == "__main__":
    unittest.main()
