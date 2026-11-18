"""transcribe_worker 纯逻辑单测：SRT 时间轴钳制 / VAD 区段判定 / 分窗推理 / 静音闸门。

worker 的业务全局量由 main() 赋值（模块 import 无副作用），测试用 create=True
注入桩值；不触 venv 库（transcribe_cpp/soundfile 仅在 main 内导入）。"""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app import transcribe_worker as w


class _Seg:
    def __init__(self, t0, t1, text):
        self.t0_ms, self.t1_ms, self.text = t0, t1, text


def _vseg(s0, s1):
    return types.SimpleNamespace(start_sample=s0, end_sample=s1)


class _FakeSess:
    """按调用序返回预置结果的推理会话桩。"""

    def __init__(self, results=None, fail=False, segments=None):
        self.results = list(results or [])
        self.fail = fail
        self.segments = segments
        self.calls = []

    def run(self, audio, **kw):
        self.calls.append(len(audio))
        if self.fail:
            raise RuntimeError("boom")
        if self.results:
            return types.SimpleNamespace(
                segments=[types.SimpleNamespace(text=t) for t in self.results.pop(0)],
                text="")
        return types.SimpleNamespace(segments=self.segments, text="")


class TestToSrt(unittest.TestCase):
    def test_clamps_end_and_drops_beyond(self):
        segs = [_Seg(0, 1500, "a"), _Seg(2000, 7000, "b"), _Seg(8000, 9000, "c")]
        srt, idx = w._to_srt(segs, max_ms=6000)
        self.assertEqual(srt.count("-->"), 2)
        self.assertIn("00:00:00,000 --> 00:00:01,500", srt)
        self.assertIn("00:00:02,000 --> 00:00:06,000", srt)
        self.assertNotIn("\nc\n", srt)  # 起始越界段整段丢弃
        self.assertEqual(idx, 3)

    def test_offset_and_idx_continuity(self):
        srt, idx = w._to_srt([_Seg(0, 500, "x")], offset_ms=1000, idx=4)
        self.assertIn("00:00:01,000 --> 00:00:01,500", srt)
        self.assertEqual(idx, 5)

    def test_no_clamp_by_default(self):
        srt, _ = w._to_srt([_Seg(0, 29980, "t")])
        self.assertIn("00:00:29,980", srt)


class TestVadRegions(unittest.TestCase):
    def setUp(self):
        self._old = (w._VAD_SESS, w._VAD_TRIED)

    def tearDown(self):
        w._VAD_SESS, w._VAD_TRIED = self._old

    def _vad(self, sess):
        w._VAD_SESS, w._VAD_TRIED = sess, True

    def test_maps_samples_to_seconds(self):
        self._vad(_FakeSess(segments=[_vseg(100, 3500)]))
        with mock.patch.object(w, "vad_model_path", "m.gguf", create=True):
            self.assertEqual(w._vad_regions([0.0] * 5000, 1000), [(0.1, 3.5)])

    def test_min_speech_threshold(self):
        self._vad(_FakeSess(segments=[_vseg(0, 200)]))
        with mock.patch.object(w, "vad_model_path", "m.gguf", create=True):
            self.assertEqual(w._vad_regions([0.0] * 5000, 1000), [])

    def test_inference_failure_falls_back_to_whole(self):
        self._vad(_FakeSess(fail=True))
        with mock.patch.object(w, "vad_model_path", "m.gguf", create=True):
            self.assertEqual(w._vad_regions([0.0] * 2000, 1000), [(0.0, 2.0)])

    def test_vad_unavailable_returns_whole(self):
        self._vad(None)
        self.assertEqual(w._vad_regions([0.0] * 2000, 1000), [(0.0, 2.0)])

    def test_empty_audio(self):
        self._vad(_FakeSess(segments=[]))
        with mock.patch.object(w, "vad_model_path", "m.gguf", create=True):
            self.assertEqual(w._vad_regions([], 1000), [])


class TestWindowedText(unittest.TestCase):
    def _run(self, sess, audio, regions, win, stop_at=0):
        with mock.patch.object(w, "sess", sess, create=True), \
             mock.patch.object(w, "run_kw", {}, create=True):
            return w._windowed_text(audio, 1, regions, win, stop_at)

    def test_steps_through_region(self):
        sess = _FakeSess(results=[["c0"], ["c1"], ["c2"]])
        out = self._run(sess, [0.0] * 10, [(0.0, 10.0)], 4)
        self.assertEqual(out, "c0 c1 c2")
        self.assertEqual(len(sess.calls), 3)  # 4s/4s/2s 三窗

    def test_stop_at_early_exit(self):
        sess = _FakeSess(results=[["ab"], ["cde"], ["f"]])
        out = self._run(sess, [0.0] * 12, [(0.0, 12.0)], 4, stop_at=5)
        self.assertEqual(out, "ab cde")  # 累计 2+1+3+1=7 ≥5 → 第三窗前停止
        self.assertEqual(len(sess.calls), 2)

    def test_multi_regions(self):
        sess = _FakeSess(results=[["a"], ["b"]])
        out = self._run(sess, [0.0] * 10, [(0.0, 4.0), (6.0, 10.0)], 4)
        self.assertEqual(out, "a b")

    def test_cjk_windows_no_space(self):
        sess = _FakeSess(results=[["你好"], ["，世界"]])
        out = self._run(sess, [0.0] * 10, [(0.0, 10.0)], 4)
        self.assertEqual(out, "你好，世界")

    def test_cjk_within_window_no_space(self):
        sess = _FakeSess(results=[["你好。", "接着说"]])
        out = self._run(sess, [0.0] * 5, [(0.0, 5.0)], 5)
        self.assertEqual(out, "你好。接着说")

    def test_cjk_ascii_mixed(self):
        sess = _FakeSess(results=[["abc"], ["你好"], ["def"]])
        out = self._run(sess, [0.0] * 15, [(0.0, 15.0)], 5)
        self.assertEqual(out, "abc你好def")


class TestTranscribeOneSrt(unittest.TestCase):
    """transcribe_one 的模式分派与静音闸门（抽帧/读盘/会话全部打桩）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._patches = [
            mock.patch.object(w, "_extract_wav", lambda vp, out: None),
            mock.patch.object(w, "params", {"work_dir": self.tmp.name,
                                            "inject_win_sec": 30.0}, create=True),
            mock.patch.object(w, "run_kw", {}, create=True),
            mock.patch.object(w, "max_chars", 300, create=True),
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, mode="srt", has_ts=False, sess=None, vad=None,
             sf_read=None, sr=1000):
        if sf_read is None:
            audio = np.zeros(sr * 3)
            sf_read = lambda p, **kw: (audio, sr)
        if vad is None:
            vad = _FakeSess(segments=[_vseg(0, int(sr * 2))])
        with mock.patch.object(w, "sf", types.SimpleNamespace(read=sf_read), create=True), \
             mock.patch.object(w, "sess", sess, create=True), \
             mock.patch.object(w, "mode", mode, create=True), \
             mock.patch.object(w, "has_ts", has_ts, create=True), \
             mock.patch.object(w, "family", "whisper", create=True), \
             mock.patch.object(w, "language", None, create=True), \
             mock.patch.object(w, "_VAD_SESS", vad), \
             mock.patch.object(w, "_VAD_TRIED", True):
            for p in self._patches:
                p.start()
            try:
                return w.transcribe_one(0, "x.mp4")
            finally:
                for p in reversed(self._patches):
                    p.stop()

    def test_silent_entry_skips_inference(self):
        sess = _FakeSess()
        e = self._run(sess=sess, vad=_FakeSess(segments=[_vseg(0, 100)]))
        self.assertTrue(e["ok"])
        self.assertTrue(e.get("silent"))
        self.assertEqual(e["srt"], "")
        self.assertEqual(sess.calls, [])  # 未送推理

    def test_speech_no_ts_uses_windowed_body(self):
        sess = _FakeSess(results=[["你好"]])
        e = self._run(sess=sess)
        self.assertTrue(e["ok"])
        self.assertIn("你好", e["srt"])
        self.assertEqual(len(sess.calls), 1)
        self.assertNotIn("silent", e)

    def test_speech_with_ts_clamped(self):
        class _R:
            segments = [_Seg(0, 9000, "幻觉"), _Seg(500, 1500, "正常")]
            language = "zh"

        sess = _FakeSess()
        sess.run = lambda audio, **kw: _R()
        e = self._run(sess=sess, has_ts=True)
        self.assertTrue(e["ok"])
        self.assertEqual(e["language"], "zh")
        self.assertIn("00:00:01,500", e["srt"])   # 正常段保留
        self.assertIn("00:00:03,000", e["srt"])   # 越界段结束钳到 3s
        self.assertNotIn("00:00:09,000", e["srt"])

    def test_read_failure_reports_error(self):
        sess = _FakeSess()
        e = self._run(sess=sess, sf_read=lambda p, **kw: (None, 1000))
        # sf.read 返回 None → len(None) 抛异常 → error 条目
        self.assertFalse(e["ok"])
        self.assertTrue(e["error"])


if __name__ == "__main__":
    unittest.main()
