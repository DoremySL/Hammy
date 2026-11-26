"""scan_fixable：无后缀补全与流式封装转存合并为单一 items 列表。"""
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from gui_app.api_mixins import sources as sources_mod
from gui_app.api_mixins.sources import SourcesMixin


class TestScanFixable(unittest.TestCase):
    def _scan(self, extensionless, stream):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(sources_mod, "load_manifest",
                                   lambda: {"roots": [root]}), \
                 mock.patch.object(sources_mod.format_fix, "reset_cancel",
                                   lambda: None), \
                 mock.patch.object(sources_mod.format_fix, "scan_extensionless",
                                   lambda roots: extensionless), \
                 mock.patch.object(sources_mod.discovery, "scan_all",
                                   lambda: {"stream": stream}):
                return SourcesMixin().scan_fixable()

    def test_merged_items_with_convert_flag(self):
        extless = [
            {"path": r"D:\x\a", "name": "a", "dir": r"D:\x", "size": 10,
             "ok": True, "ext": ".avi", "codecs": "h264", "reason": ""},
            {"path": r"D:\x\b", "name": "b", "dir": r"D:\x", "size": 20,
             "ok": True, "ext": ".ts", "codecs": "h264", "reason": ""},
            {"path": r"D:\x\c", "name": "c", "dir": r"D:\x", "size": 5,
             "ok": False, "ext": None, "codecs": "", "reason": "无法识别"},
        ]
        stream = [{"path": r"D:\x\d.ts", "name": "d.ts", "dir": r"D:\x", "size": 99}]
        r = self._scan(extless, stream)
        items = {it["name"]: it for it in r["items"]}
        self.assertEqual(len(items), 4)
        self.assertFalse(items["a"]["convert"])          # 普通格式 → 补后缀
        self.assertTrue(items["b"]["convert"])           # 无后缀检出 mpegts → 直接转换
        self.assertFalse(items["c"]["convert"])         # 识别失败 → 不可勾选
        self.assertTrue(items["d.ts"]["convert"] and items["d.ts"]["ok"])
        self.assertTrue(items["a"]["size_str"])
        self.assertEqual(r["roots"], 1)


if __name__ == "__main__":
    unittest.main()
