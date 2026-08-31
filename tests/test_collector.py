import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import tempfile
import unittest

from batch_rename.collector import VideoCollector, is_mpeg_ts, is_video_file


class TestCollect(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _touch(self, *names):
        for n in names:
            p = self.root / n
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")

    def _touch_binary(self, name, content: bytes):
        """写入二进制内容（用于构造 MPEG-TS 头等）。"""
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)

    @staticmethod
    def _make_mpeg_ts_header(num_packets=4):
        """构造 MPEG-TS 文件头：num_packets 个 188 字节包，每包首字节 0x47。"""
        packets = []
        for _ in range(num_packets):
            pkt = bytes([0x47]) + bytes(187)
            packets.append(pkt)
        return b"".join(packets)

    @staticmethod
    def _make_typescript_source():
        """构造典型的 TypeScript 源码内容。"""
        return b'import { Component } from "@angular/core";\n\n@Component({\n  selector: "app-root",\n  templateUrl: "./app.component.html"\n})\nexport class AppComponent {\n  title = "my-app";\n}\n'

    def test_collects_video_files(self):
        self._touch("a.mp4", "b.mkv", "c.txt")
        result = VideoCollector.collect([str(self.root)])
        names = sorted(Path(p).name for p in result)
        self.assertEqual(names, ["a.mp4", "b.mkv"])

    def test_skips_failed_dir(self):
        self._touch("ok.mp4", "_failed/bad.mp4")
        result = VideoCollector.collect([str(self.root)])
        names = [Path(p).name for p in result]
        self.assertEqual(names, ["ok.mp4"])

    def test_dedup_overlapping_paths(self):
        self._touch("sub/v.mp4")
        # 传入根目录 + 子目录，同一文件不应重复
        result = VideoCollector.collect([str(self.root), str(self.root / "sub")])
        self.assertEqual(len(result), 1)

    def test_single_file_input(self):
        self._touch("movie.avi")
        result = VideoCollector.collect([str(self.root / "movie.avi")])
        self.assertEqual(len(result), 1)
        self.assertEqual(Path(result[0]).name, "movie.avi")

    def test_non_video_file_ignored(self):
        self._touch("readme.txt", "photo.jpg")
        result = VideoCollector.collect([str(self.root / "readme.txt"),
                                         str(self.root / "photo.jpg")])
        self.assertEqual(result, [])

    # ── .ts 歧义扩展名过滤 ──────────────────────────────
    def test_is_mpeg_ts_detects_sync_byte_alignment(self):
        p = self.root / "real.ts"
        p.write_bytes(self._make_mpeg_ts_header(6))
        self.assertTrue(is_mpeg_ts(str(p)))

    def test_is_mpeg_ts_rejects_typescript_source(self):
        p = self.root / "app.component.ts"
        p.write_bytes(self._make_typescript_source())
        self.assertFalse(is_mpeg_ts(str(p)))

    def test_is_mpeg_ts_rejects_too_small_file(self):
        p = self.root / "tiny.ts"
        p.write_bytes(bytes([0x47]) + bytes(10))
        self.assertFalse(is_mpeg_ts(str(p)))

    def test_is_mpeg_ts_rejects_misaligned_sync(self):
        # 只有首字节是 0x47，后续包首字节不是，说明不是标准 TS
        bad = bytes([0x47]) + bytes(187) + bytes([0x00]) + bytes(187)
        p = self.root / "bad.ts"
        p.write_bytes(bad * 2)
        self.assertFalse(is_mpeg_ts(str(p)))

    def test_is_video_file_certain_ext_no_probe(self):
        # 无歧义扩展名即便内容为空也应通过（不做嗅探）
        self._touch("empty.mp4")
        self.assertTrue(is_video_file(str(self.root / "empty.mp4")))

    def test_collect_includes_real_ts_video(self):
        self._touch_binary("clip.ts", self._make_mpeg_ts_header(8))
        result = VideoCollector.collect([str(self.root)])
        names = [Path(p).name for p in result]
        self.assertEqual(names, ["clip.ts"])

    def test_collect_excludes_typescript_source(self):
        self._touch_binary("app.component.ts", self._make_typescript_source())
        self._touch_binary("service.ts", b"export const x = 1;\n")
        self._touch("ok.mp4")
        result = VideoCollector.collect([str(self.root)])
        names = sorted(Path(p).name for p in result)
        self.assertEqual(names, ["ok.mp4"])

    def test_collect_single_ts_file_input_filtered(self):
        # 直接传入单个 .ts 文件时同样要过嗅探
        self._touch_binary("src/main.ts", self._make_typescript_source())
        self._touch_binary("video.ts", self._make_mpeg_ts_header(5))
        result = VideoCollector.collect([
            str(self.root / "src/main.ts"),
            str(self.root / "video.ts"),
        ])
        names = [Path(p).name for p in result]
        self.assertEqual(names, ["video.ts"])


if __name__ == "__main__":
    unittest.main()
