"""format_fix：_decodable 命令形态与返回值映射。"""
import sys
from pathlib import Path
from subprocess import CompletedProcess
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from batch_rename import format_fix


class TestDecodable(unittest.TestCase):
    def _patch(self, result):
        captured = {}

        def fake_run(cmd, timeout):
            captured.update(cmd=cmd, timeout=timeout)
            return result

        return captured, mock.patch.object(format_fix, "_run", fake_run)

    def test_null_muxer_single_frame(self):
        # null muxer 丢弃输出，不出二进制 stdout
        captured, p = self._patch(CompletedProcess([], 0, "", ""))
        with p:
            self.assertTrue(format_fix._decodable("v.mp4"))
        cmd = captured["cmd"]
        self.assertEqual(cmd[cmd.index("-f") + 1], "null")
        self.assertEqual(cmd[cmd.index("-frames:v") + 1], "1")
        self.assertNotIn("rawvideo", cmd)
        self.assertEqual(captured["timeout"], format_fix._VALIDATE_TIMEOUT_SEC)

    def test_nonzero_or_none_is_undecodable(self):
        for result in (CompletedProcess([], 1, "", ""), None):
            _, p = self._patch(result)
            with p:
                self.assertFalse(format_fix._decodable("v.mp4"))


if __name__ == "__main__":
    unittest.main()
