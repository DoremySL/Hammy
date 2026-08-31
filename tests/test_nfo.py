import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest
from xml.etree import ElementTree as ET

from batch_rename.nfo import _build_nfo_xml, _safe_xml_text


class TestSafeXmlText(unittest.TestCase):
    def test_strips_control_chars(self):
        self.assertEqual(_safe_xml_text("a\x00b\x08c"), "abc")

    def test_preserves_tab_newline_cr(self):
        self.assertEqual(_safe_xml_text("a\tb\nc\rd"), "a\tb\nc\rd")

    def test_empty(self):
        self.assertEqual(_safe_xml_text(""), "")

    def test_normal_text_unchanged(self):
        self.assertEqual(_safe_xml_text("正常文本 <tag> & more"), "正常文本 <tag> & more")


class TestBuildNfoXml(unittest.TestCase):
    def _parse(self, xml_str: str) -> ET.Element:
        # 去掉 XML 声明行后解析
        return ET.fromstring(xml_str)

    def test_basic_structure(self):
        xml = _build_nfo_xml("标题", "剧情描述", ["tag1", "tag2"], {}, "原文件名")
        root = self._parse(xml)
        self.assertEqual(root.tag, "movie")
        self.assertEqual(root.findtext("title"), "标题")
        self.assertEqual(root.findtext("plot"), "剧情描述")
        self.assertEqual(root.findtext("originaltitle"), "原文件名")
        tags = [e.text for e in root.findall("tag")]
        self.assertEqual(tags, ["tag1", "tag2"])

    def test_runtime_from_duration(self):
        info = {"duration": 150.0}
        xml = _build_nfo_xml("t", "p", [], info, "orig")
        root = self._parse(xml)
        self.assertEqual(root.findtext("runtime"), "2")  # 150/60 = 2

    def test_no_runtime_when_zero_duration(self):
        xml = _build_nfo_xml("t", "p", [], {"duration": 0}, "orig")
        root = self._parse(xml)
        self.assertIsNone(root.find("runtime"))

    def test_creation_time_generates_premiered_and_year(self):
        info = {"creation_time": "2023-05-10T12:30:00Z"}
        xml = _build_nfo_xml("t", "p", [], info, "orig")
        root = self._parse(xml)
        self.assertIsNotNone(root.findtext("premiered"))
        self.assertIsNotNone(root.findtext("year"))
        # year 取决于本地时区转换，但应该是 2023 或 2024（UTC+12 以上）
        self.assertIn(root.findtext("year"), ("2023", "2024"))

    def test_invalid_creation_time_no_premiered(self):
        info = {"creation_time": "1970-01-01T00:00:00Z"}
        xml = _build_nfo_xml("t", "p", [], info, "orig")
        root = self._parse(xml)
        self.assertIsNone(root.find("premiered"))

    def test_video_stream_details(self):
        info = {
            "duration": 120.0,
            "video": {"codec": "h264", "width": 1920, "height": 1080,
                      "frame_rate": 29.97, "field_order": "progressive"},
            "audio": [{"codec": "aac", "channels": 2, "sample_rate": 48000}],
        }
        xml = _build_nfo_xml("t", "p", [], info, "orig")
        root = self._parse(xml)
        sd = root.find("fileinfo/streamdetails")
        self.assertIsNotNone(sd)
        ve = sd.find("video")
        self.assertEqual(ve.findtext("codec"), "h264")
        self.assertEqual(ve.findtext("width"), "1920")
        self.assertEqual(ve.findtext("scantype"), "Progressive")
        ae = sd.find("audio")
        self.assertEqual(ae.findtext("codec"), "aac")
        self.assertEqual(ae.findtext("channels"), "2")

    def test_interlaced_detection(self):
        info = {"video": {"codec": "mpeg2", "field_order": "tt"}}
        xml = _build_nfo_xml("t", "p", [], info, "orig")
        root = self._parse(xml)
        ve = root.find("fileinfo/streamdetails/video")
        self.assertEqual(ve.findtext("scantype"), "Interlaced")

    def test_illegal_chars_in_title_cleaned(self):
        xml = _build_nfo_xml("ti\x00tle", "pl\x08ot", [], {}, "orig")
        root = self._parse(xml)
        self.assertEqual(root.findtext("title"), "title")
        self.assertEqual(root.findtext("plot"), "plot")


if __name__ == "__main__":
    unittest.main()
