import json
import sys
import tempfile
import threading
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from batch_rename.ai import AnalyzeResult, _THUMB_TIME_RE, _call_and_parse
from batch_rename.config import Config
from gui_app import discovery, prompts, runner


def _call_ai(raw: str) -> AnalyzeResult:
    msg = mock.Mock()
    msg.content = raw
    resp = mock.Mock()
    resp.choices = [mock.Mock(message=msg)]
    client = mock.Mock()
    client.chat.completions.create.return_value = resp
    return _call_and_parse(client, "m", [], Config(), threading.Event())


class TestThumbTimeRegex(unittest.TestCase):
    def test_valid(self):
        for s in ("0:00:05", "00:01:23", "123:45:56"):
            self.assertTrue(_THUMB_TIME_RE.fullmatch(s))

    def test_invalid(self):
        for s in ("", "01:23", "1:2:3", "00:01:23.5", "abc", "00:01:0a"):
            self.assertIsNone(_THUMB_TIME_RE.fullmatch(s))


class TestAiExtractThumbTime(unittest.TestCase):
    def test_valid(self):
        r = _call_ai('{"title":"t","plot":"p","tags":["a"],"thumb_time":"00:10:00"}')
        self.assertEqual(r.thumb_time, "00:10:00")

    def test_invalid_format_ignored(self):
        r = _call_ai('{"title":"t","plot":"p","tags":["a"],"thumb_time":"10:00"}')
        self.assertEqual(r.thumb_time, "")

    def test_missing(self):
        r = _call_ai('{"title":"t","plot":"p","tags":["a"]}')
        self.assertEqual(r.thumb_time, "")


class TestHistoryThumbSeconds(unittest.TestCase):
    def _patch(self, entry):
        return mock.patch("gui_app.discovery.get_history_by_id", return_value=entry)

    def test_valid(self):
        with self._patch({"thumb_time": "00:12:34"}):
            self.assertEqual(discovery._history_thumb_seconds("vid"), 754.0)

    def test_no_entry(self):
        with self._patch(None):
            self.assertIsNone(discovery._history_thumb_seconds("vid"))

    def test_invalid_values(self):
        for v in ("01:23", "abc", "", 123, "1:2:3"):
            with self._patch({"thumb_time": v}):
                self.assertIsNone(discovery._history_thumb_seconds("vid"))


class TestInvalidateThumbnail(unittest.TestCase):
    def test_clears_lru_and_disk(self):
        discovery._thumb_lru_put("v1", "data:image/jpeg;base64,AAA")
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "v1.jpg"
            f.write_bytes(b"\xff\xd8fake")
            with mock.patch("gui_app.discovery.thumb_path", return_value=f):
                discovery.invalidate_thumbnail("v1")
                self.assertFalse(f.exists())
        self.assertNotIn("v1", discovery._thumb_lru)


class TestBuildPromptThumbTime(unittest.TestCase):
    PRESET = {"fields": {"plot_guidance": "p", "tags_dim": "", "title_guidance": "t",
                         "plot_example": "pe", "tags_example": '"a"', "title_example": "te",
                         "thumb_time_guidance": "挑选出主体清晰，最能代表该视频适合用作封面的截图时间戳，格式HH:MM:SS",
                         "thumb_time_example": "00:12:34"}}

    def test_disabled_no_thumb(self):
        self.assertNotIn("thumb_time", prompts.build_prompt(self.PRESET))

    def test_enabled_fields(self):
        s = prompts.build_prompt(self.PRESET, with_thumb_time=True)
        self.assertIn('"thumb_time": "挑选出主体清晰，最能代表该视频适合用作封面的截图时间戳，格式HH:MM:SS"', s)
        self.assertIn('"thumb_time": "00:12:34"', s)

    def test_enabled_example_is_valid_json(self):
        s = prompts.build_prompt(self.PRESET, with_thumb_time=True)
        data = json.loads(s.split("示例：")[-1])
        self.assertEqual(data["thumb_time"], "00:12:34")


class TestGetActiveThumbTime(unittest.TestCase):
    def _get(self, with_tt):
        with mock.patch("gui_app.config_store.load_config",
                        return_value={"active_prompt_id": "default"}), \
             mock.patch("gui_app.prompts.list_presets",
                        return_value=prompts._builtin_presets()):
            return prompts.get_active(with_thumb_time=with_tt)

    def test_appends_field_only(self):
        r = self._get(True)
        self.assertIn("挑选出主体清晰", r["prompt"])
        self.assertNotIn("总结最能代表该视频的截图时间戳", r["system_prompt"])

    def test_default_untouched(self):
        r = self._get(False)
        self.assertNotIn("总结最能代表该视频的截图时间戳", r["system_prompt"])
        self.assertNotIn("thumb_time", r["prompt"])


class TestBuildEngineConfigTriState(unittest.TestCase):
    def _build(self, raw):
        c = {"ai": {}, "naming": {}, "experimental": {}, "video": {"frame_time_tags": raw}}
        with mock.patch("gui_app.runner.prompts.get_active",
                        return_value={"preset": {}, "prompt": "p", "system_prompt": "s"}) as ga:
            eng = runner.build_engine_config(c)
        return eng, ga

    def test_mode_0(self):
        eng, ga = self._build(0)
        self.assertEqual(eng.frame_time_tags, 0)
        ga.assert_called_with(with_thumb_time=False)

    def test_mode_1(self):
        eng, ga = self._build(1)
        self.assertEqual(eng.frame_time_tags, 1)
        ga.assert_called_with(with_thumb_time=False)

    def test_mode_2(self):
        eng, ga = self._build(2)
        self.assertEqual(eng.frame_time_tags, 2)
        ga.assert_called_with(with_thumb_time=True)

    def test_string_value(self):
        eng, _ = self._build("2")
        self.assertEqual(eng.frame_time_tags, 2)

    def test_invalid_falls_back(self):
        eng, _ = self._build("x")
        self.assertEqual(eng.frame_time_tags, 0)


if __name__ == "__main__":
    unittest.main()
