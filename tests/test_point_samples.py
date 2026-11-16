"""select_point_samples 点位语义 + preprocess 公用预处理测试。"""
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from PIL import Image, ImageDraw

from batch_rename.video import select_point_samples, _anchor_keyframe_indices
from batch_rename.preprocess import preprocess_frame, preprocess_frames


def _jpeg(im, quality=95):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _noise(size):
    return Image.effect_noise(size, 60).convert("RGB")


class TestSelectPointSamples(unittest.TestCase):
    def test_window_sampling_with_point_index(self):
        # duration=100, 5 点位 → positions [0,25,50,75,100], half=5
        s = select_point_samples(100.0, [1.0, 24.9, 51.0, 99.0], 5, 1)
        self.assertEqual(s, [(1.0, 0), (24.9, 1), (51.0, 2), (99.0, 4)])

    def test_empty_window_skipped_no_padding(self):
        # 点位 75 附近无关键帧 → 该点位跳过，不补帧
        s = select_point_samples(100.0, [1.0, 24.9, 51.0, 99.0], 5, 1)
        self.assertFalse(any(pi == 3 for _, pi in s))

    def test_per_point_one_takes_nearest(self):
        # 点位 50 附近两个关键帧 → 取最近者
        s = select_point_samples(100.0, [45.0, 53.0], 3, 1)
        self.assertEqual(s, [(53.0, 1)])

    def test_per_point_multi_uses_anchors(self):
        self.assertEqual(_anchor_keyframe_indices(4, 2), [0, 3])
        s = select_point_samples(100.0, [45.0, 46.5, 53.5, 55.0], 3, 2)
        self.assertEqual(s, [(45.0, 1), (55.0, 1)])

    def test_dedup_belongs_to_first_declaring_point(self):
        # 10.0 同时落在点位 1/2 窗口内 → 归属先声明的点位 1
        s = select_point_samples(10.0, [10.0], 3, 1)
        self.assertEqual(s, [(10.0, 1)])

    def test_fallback_no_keyframes_pseudo_points(self):
        s = select_point_samples(8.0, [], 5, 1)
        self.assertEqual(s, [(0.0, 0), (4.0, 1), (7.9, 2)])
        s = select_point_samples(100.0, [], 5, 1)
        self.assertEqual([pi for _, pi in s], [0, 1, 2, 3, 4])

    def test_fallback_when_all_windows_empty(self):
        # 有关键帧但全部窗口外 → 回退伪点位
        s = select_point_samples(100.0, [50.0], 2, 1)
        self.assertEqual(s, [(50.0, 0)])

    def test_invariant_ascending_and_monotonic(self):
        s = select_point_samples(90.0, [2.0, 30.5, 60.0, 60.9, 89.0], 4, 1)
        ts = [t for t, _ in s]
        pis = [pi for _, pi in s]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(pis, sorted(pis))


class TestPreprocessFrame(unittest.TestCase):
    def test_black_frame_dropped(self):
        self.assertIsNone(preprocess_frame(_jpeg(Image.new("RGB", (672, 378), (0, 0, 0)))))

    def test_letterbox_cropped(self):
        im = _noise((672, 378))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 672, 94], fill=(0, 0, 0))       # 上黑边 25%
        d.rectangle([0, 283, 672, 378], fill=(0, 0, 0))    # 下黑边 25%
        out = preprocess_frame(_jpeg(im))
        self.assertIsNotNone(out)
        w, h = Image.open(io.BytesIO(out)).size
        self.assertEqual(w, 672)
        self.assertTrue(150 < h < 210)   # 上下黑边被裁（JPEG 边界行允许小误差）

    def test_clean_frame_no_reencode(self):
        raw = _jpeg(_noise((672, 378)))
        self.assertEqual(preprocess_frame(raw), raw)

    def test_invalid_bytes_dropped(self):
        self.assertIsNone(preprocess_frame(b"not-a-jpeg"))

    def test_batch_keeps_order(self):
        clean = _jpeg(_noise((320, 180)))
        black = _jpeg(Image.new("RGB", (320, 180), (0, 0, 0)))
        out = preprocess_frames([black, clean, b"bad"])
        self.assertEqual(out, [clean])

    def test_idempotent(self):
        im = _noise((672, 378))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 672, 94], fill=(0, 0, 0))
        d.rectangle([0, 283, 672, 378], fill=(0, 0, 0))
        once = preprocess_frame(_jpeg(im))
        self.assertEqual(preprocess_frame(once), once)


if __name__ == "__main__":
    unittest.main()
