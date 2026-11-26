"""scan_all 流式封装分桶：.ts/.mts/.m2ts 单列 stream，不进 pending。"""
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from gui_app import discovery


def _mpegts(num_packets=4):
    return b"".join(bytes([0x47]) + bytes(187) for _ in range(num_packets))


class TestScanStreamBucket(unittest.TestCase):
    def _scan(self, root, hist=None):
        with mock.patch.object(discovery, "load_manifest",
                               lambda: {"roots": [root]}), \
             mock.patch.object(discovery, "load_history",
                               lambda: hist if hist is not None else {"entries": []}):
            return discovery.scan_all()

    def test_stream_bucket(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "a.mp4").write_bytes(b"junk")
            Path(root, "b.ts").write_bytes(_mpegts())
            Path(root, "c.mts").write_bytes(_mpegts())
            Path(root, "script.ts").write_text("export const x = 1;")
            r = self._scan(root)
            self.assertEqual(sorted(e["name"] for e in r["stream"]), ["b.ts", "c.mts"])
            self.assertEqual([e["status"] for e in r["stream"]], ["stream"] * 2)
            self.assertEqual([e["name"] for e in r["pending"]], ["a.mp4"])
            self.assertEqual(r["processed"], [])
            self.assertEqual(r["failed"], [])

    def test_processed_ts_stays_processed(self):
        # 已处理的 .ts 不改判：留在 processed，不进 stream
        with tempfile.TemporaryDirectory() as root:
            ts = Path(root, "done.ts")
            ts.write_bytes(_mpegts())
            hist = {"entries": [{"status": "ok", "new_path": str(ts),
                                 "original_path": str(ts)}]}
            r = self._scan(root, hist=hist)
            self.assertEqual(len(r["processed"]), 1)
            self.assertEqual(r["stream"], [])
            self.assertEqual(r["pending"], [])

    def test_is_stream_video(self):
        with tempfile.TemporaryDirectory() as root:
            ts = Path(root, "a.ts")
            ts.write_bytes(_mpegts())
            self.assertTrue(discovery._is_stream_video(str(ts)))
            self.assertFalse(discovery._is_stream_video(str(Path(root, "a.txt"))))


if __name__ == "__main__":
    unittest.main()
