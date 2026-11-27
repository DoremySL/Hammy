"""runner._build_transcribe_meta_map：已有 SRT → 辅助上下文块（缺失视频由 ASR worker 交付）。"""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app.workspace_paths import stable_id
from gui_app.runner import PipelineRunner, _asr_missing_list

EXP = {"enabled": True, "max_chars": 800, "inject_timestamps": False}

SRT_SAMPLE = "1\n00:00:00,000 --> 00:00:05,000\nこんにちは、世界。\n"


class TestBuildTranscribeMetaMap(unittest.TestCase):
    """已有右键转写的 SRT → 辅助上下文块。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        td = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.srt_dir = td / "srt"
        self.tx_file = td / "transcripts.json"
        patchers = [
            mock.patch("gui_app.workspace_paths.TRANSCRIBE_SRT_DIR", self.srt_dir),
            mock.patch("gui_app.workspace_paths.TRANSCRIBE_TRANSCRIPTS_FILE", self.tx_file),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def _make_runner(self):
        return PipelineRunner()

    def _write_srt(self, vp: str, content: str = SRT_SAMPLE):
        self.srt_dir.mkdir(parents=True, exist_ok=True)
        (self.srt_dir / f"{stable_id(vp)}.srt").write_text(content, encoding="utf-8")

    def _write_store(self, vp: str, lang: str = "ja"):
        self.tx_file.parent.mkdir(parents=True, exist_ok=True)
        self.tx_file.write_text(json.dumps(
            {stable_id(vp): {"language": lang}}), encoding="utf-8")

    def test_disabled_returns_empty(self):
        r = self._make_runner()._build_transcribe_meta_map(
            ["C:/a.mp4"], {"enabled": False})
        self.assertEqual(r, {})

    def test_srt_builds_block(self):
        vp = "C:/a.mp4"
        self._write_srt(vp)
        self._write_store(vp, "ja")
        r = self._make_runner()._build_transcribe_meta_map([vp], EXP)
        self.assertIn("こんにちは、世界。", r[vp])
        self.assertIn("【语音辅助参考】", r[vp])
        self.assertIn("- 语言: ja", r[vp])

    def test_truncated_to_max_chars(self):
        vp = "C:/d.mp4"
        self._write_srt(vp, "1\n00:00:00,000 --> 00:00:05,000\n" + "字" * 1200 + "\n")
        r = self._make_runner()._build_transcribe_meta_map([vp], EXP)
        self.assertIn("…（已截断）", r[vp])
        self.assertNotIn("字" * 801, r[vp])

    def test_missing_srt_skipped(self):
        r = self._make_runner()._build_transcribe_meta_map(["C:/e.mp4"], EXP)
        self.assertEqual(r, {})

    def test_empty_srt_skipped(self):
        vp = "C:/c.mp4"
        self._write_srt(vp, "")
        r = self._make_runner()._build_transcribe_meta_map([vp], EXP)
        self.assertEqual(r, {})


class TestAsrMissingList(unittest.TestCase):
    """runner._asr_missing_list：缺 SRT 并行探针 + 无音轨过滤。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.srt_dir = Path(self._tmp.name) / "srt"
        self.srt_dir.mkdir()

    def test_filter_and_srt_skip(self):
        a, b, c, d = "C:/a.mp4", "C:/b.mp4", "C:/c.mp4", "C:/d.mp4"
        (self.srt_dir / f"{stable_id(b)}.srt").write_text("1\nx\n", encoding="utf-8")
        probes = {a: {"has_audio": True}, c: {"has_audio": False}, d: {}}

        def fake_probe(vp):
            return probes[vp]

        with mock.patch("gui_app.discovery.probe_video", side_effect=fake_probe) as pv:
            out = _asr_missing_list([a, b, c, d], self.srt_dir)
        # b 已有 SRT 不探针；c 无音轨过滤；d 探针失败降级保留
        self.assertEqual(out, [a, d])
        self.assertEqual({m.args[0] for m in pv.call_args_list}, {a, c, d})

    def test_stop_returns_empty(self):
        """停止信号置位：提前退出，不收集结果。"""
        stop = threading.Event()
        stop.set()
        with mock.patch("gui_app.discovery.probe_video",
                        return_value={"has_audio": True}):
            out = _asr_missing_list(["C:/a.mp4", "C:/b.mp4"], self.srt_dir, stop)
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
