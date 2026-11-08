"""runner._build_transcribe_meta_map：SRT 优先 / inject_texts 回退 / 截断 / 空文本跳过。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app.workspace_paths import stable_id
from gui_app.runner import PipelineRunner

EXP = {"transcribe_enabled": True, "transcribe_max_chars": 800,
       "transcribe_inject_timestamps": False}

SRT_SAMPLE = "1\n00:00:00,000 --> 00:00:05,000\nこんにちは、世界。\n"


class TestBuildTranscribeMetaMap(unittest.TestCase):
    """注入合并逻辑：有右键转录用 SRT，缺的用 SenseVoice 注入文本。"""

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
            ["C:/a.mp4"], {"transcribe_enabled": False}, inject_texts={"C:/a.mp4": "x"})
        self.assertEqual(r, {})

    def test_srt_preferred_over_inject(self):
        vp = "C:/a.mp4"
        self._write_srt(vp)
        self._write_store(vp, "ja")
        r = self._make_runner()._build_transcribe_meta_map(
            [vp], EXP, inject_texts={vp: " SenseVoice 文本 "})
        self.assertIn("こんにちは、世界。", r[vp])
        self.assertNotIn("SenseVoice 文本", r[vp])
        self.assertIn("- 语言: ja", r[vp])

    def test_inject_fallback_for_missing_srt(self):
        vp = "C:/b.mp4"
        r = self._make_runner()._build_transcribe_meta_map(
            [vp], EXP, inject_texts={vp: " SenseVoice 文本 "})
        self.assertIn("SenseVoice 文本", r[vp])
        self.assertIn("【语音辅助参考】", r[vp])
        self.assertNotIn("- 语言:", r[vp])  # 注入文本无语言信息

    def test_empty_inject_skipped(self):
        vp = "C:/c.mp4"
        r = self._make_runner()._build_transcribe_meta_map(
            [vp], EXP, inject_texts={vp: "   "})
        self.assertEqual(r, {})

    def test_inject_truncated_to_max_chars(self):
        vp = "C:/d.mp4"
        r = self._make_runner()._build_transcribe_meta_map(
            [vp], EXP, inject_texts={vp: "字" * 1200})
        self.assertIn("…（已截断）", r[vp])
        self.assertNotIn("字" * 801, r[vp])

    def test_missing_everything_skipped(self):
        r = self._make_runner()._build_transcribe_meta_map(
            ["C:/e.mp4"], EXP, inject_texts={})
        self.assertEqual(r, {})


if __name__ == "__main__":
    unittest.main()
