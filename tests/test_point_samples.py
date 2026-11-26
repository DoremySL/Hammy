"""select_point_samples 点位语义 + preprocess 公用预处理测试。"""
import base64
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from PIL import Image, ImageDraw

from batch_rename.video import select_point_samples, _anchor_keyframe_indices
from batch_rename.preprocess import (preprocess_frame_signature, preprocess_frames,
                                     _dedup_point_frames)
from batch_rename.phash import ncc
from batch_rename.types import Frame
from gui_app.tagger import per_point_frames


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
        self.assertEqual(s, [(1.0, 0), (24.9, 1), (51.0, 2), (75.0, 3), (99.0, 4)])

    def test_empty_window_filled_with_frame_position(self):
        # 点位 75 分区无关键帧 → 补点位位置
        s = select_point_samples(100.0, [1.0, 24.9, 51.0, 99.0], 5, 1)
        self.assertIn((75.0, 3), s)

    def test_per_point_one_takes_nearest(self):
        # 点位 50 附近两个关键帧 → 取最近者；空分区补点位位置
        s = select_point_samples(100.0, [45.0, 53.0], 3, 1)
        self.assertEqual([(round(t, 3), pi) for t, pi in s],
                         [(0.0, 0), (53.0, 1), (99.967, 2)])

    def test_per_point_multi_uses_anchors(self):
        self.assertEqual(_anchor_keyframe_indices(4, 2), [0, 3])
        s = select_point_samples(100.0, [45.0, 46.5, 53.5, 55.0], 3, 2)
        self.assertEqual([(round(t, 3), pi) for t, pi in s],
                         [(2.778, 0), (5.556, 0), (45.0, 1), (55.0, 1),
                          (94.444, 2), (97.222, 2)])

    def test_boundary_belongs_to_later_point(self):
        # 10.0 是点位 1/2 分区的边界 → 归后一个点位；空分区补点位位置
        s = select_point_samples(10.0, [10.0], 3, 1)
        self.assertEqual(s, [(0.0, 0), (5.0, 1), (10.0, 2)])

    def test_dense_keyframes_stay_chronological(self):
        # 60s、10 点位、每 2s 一个关键帧：输出升序，且每个点位都有帧
        s = select_point_samples(60.0, [float(t) for t in range(0, 60, 2)], 10, 2)
        ts = [t for t, _ in s]
        points = [pi for _, pi in s]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(points, sorted(points))
        self.assertEqual(len(set(points)), 10)

    def test_no_keyframes_fill_point_positions(self):
        # 8s、5 点位：同套点位分区补普通帧位置，尾点内收防越过末帧
        s = select_point_samples(8.0, [], 5, 1)
        self.assertEqual([(round(t, 3), pi) for t, pi in s],
                         [(0.0, 0), (2.0, 1), (4.0, 2), (6.0, 3), (7.967, 4)])

    def test_keyframes_outside_regions_fill_positions(self):
        # 关键帧全部落在分区间隙 → 各分区补点位位置
        s = select_point_samples(100.0, [50.0], 2, 1)
        self.assertEqual([(round(t, 3), pi) for t, pi in s],
                         [(0.0, 0), (99.967, 1)])

    def test_invariant_ascending_and_monotonic(self):
        s = select_point_samples(90.0, [2.0, 30.5, 60.0, 60.9, 89.0], 4, 1)
        ts = [t for t, _ in s]
        pis = [pi for _, pi in s]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(pis, sorted(pis))


class TestPreprocessFrame(unittest.TestCase):
    def test_black_frame_dropped(self):
        self.assertIsNone(
            preprocess_frame_signature(_jpeg(Image.new("RGB", (672, 378), (0, 0, 0)))))

    def test_solid_color_frame_dropped(self):
        for color in ((255, 255, 255), (255, 0, 0), (128, 128, 128)):
            self.assertIsNone(
                preprocess_frame_signature(_jpeg(Image.new("RGB", (320, 180), color))))

    def test_nearly_solid_frame_kept(self):
        im = _noise((320, 180))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 96, 96], fill=(255, 255, 255))   # 少量白块，不算纯色
        self.assertIsNotNone(preprocess_frame_signature(_jpeg(im)))

    def test_letterbox_cropped(self):
        im = _noise((672, 378))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 672, 94], fill=(0, 0, 0))       # 上黑边 25%
        d.rectangle([0, 283, 672, 378], fill=(0, 0, 0))    # 下黑边 25%
        result = preprocess_frame_signature(_jpeg(im))
        self.assertIsNotNone(result)
        w, h = Image.open(io.BytesIO(result[0])).size
        self.assertEqual(w, 672)
        self.assertTrue(150 < h < 210)   # 上下黑边被裁（JPEG 边界行允许小误差）

    def test_clean_frame_no_reencode(self):
        raw = _jpeg(_noise((672, 378)))
        self.assertEqual(preprocess_frame_signature(raw)[0], raw)

    def test_invalid_bytes_dropped(self):
        self.assertIsNone(preprocess_frame_signature(b"not-a-jpeg"))

    def test_batch_keeps_order(self):
        clean = _jpeg(_noise((320, 180)))
        black = _jpeg(Image.new("RGB", (320, 180), (0, 0, 0)))
        out = [r[0] for r in map(preprocess_frame_signature, [black, clean, b"bad"]) if r]
        self.assertEqual(out, [clean])

    def test_idempotent(self):
        im = _noise((672, 378))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 672, 94], fill=(0, 0, 0))    # 上黑边 25%
        d.rectangle([0, 283, 672, 378], fill=(0, 0, 0)) # 下黑边 25%
        once = preprocess_frame_signature(_jpeg(im))[0]
        self.assertEqual(preprocess_frame_signature(once)[0], once)


class TestSimilaritySignature(unittest.TestCase):
    def test_ncc_identical_and_inverse(self):
        a = bytes([0, 1] * 512)
        self.assertEqual(ncc(a, a), 1.0)
        self.assertEqual(ncc(a, bytes([1, 0] * 512)), -1.0)

    def test_signature_hashes_after_crop(self):
        # 同一画面加黑边后，签名应与无黑边版本几乎一致（先裁边再算哈希）
        content = _noise((320, 120))
        plain = preprocess_frame_signature(_jpeg(content))
        canvas = Image.new("RGB", (320, 240), (0, 0, 0))
        canvas.paste(content, (0, 60))
        boxed = preprocess_frame_signature(_jpeg(canvas))
        self.assertIsNotNone(plain)
        self.assertIsNotNone(boxed)
        self.assertEqual(len(plain[2]), 32 * 32)
        self.assertGreaterEqual(ncc(plain[2], boxed[2]), 0.98)
        self.assertEqual(Image.open(io.BytesIO(boxed[0])).size, (320, 120))


class TestDedupPointFrames(unittest.TestCase):
    def _f(self, ts, pi):
        return Frame(ts, "", pi)

    def test_same_point_similar_dropped(self):
        sig = bytes([10]) * 1024
        items = [(self._f(0.0, 0), 0, sig),
                 (self._f(1.0, 0), 0, sig),
                 (self._f(2.0, 1), 0, sig)]
        out = _dedup_point_frames(items)
        # 同点位第二条被删，跨点位同签名保留
        self.assertEqual([f.ts for f in out], [0.0, 2.0])

    def test_dhash_prefilter_keeps_distinct(self):
        sig = bytes([10]) * 1024
        items = [(self._f(0.0, 0), 0, sig),
                 (self._f(1.0, 0), (1 << 64) - 1, sig)]
        self.assertEqual(len(_dedup_point_frames(items)), 2)


class TestPreprocessDedup(unittest.TestCase):
    def _frame(self, ts, pi, im):
        return Frame(ts, base64.b64encode(_jpeg(im)).decode(), pi)

    def test_identical_same_point_deduped(self):
        a = _noise((320, 180))
        b = _noise((320, 180))
        frames = [self._frame(0.0, 0, a), self._frame(1.0, 0, a),
                  self._frame(2.0, 0, b), self._frame(3.0, 1, a)]
        out = preprocess_frames(frames)
        self.assertEqual([f.ts for f in out], [0.0, 2.0, 3.0])


class TestPerPointFrames(unittest.TestCase):
    def _f(self, pi):
        return Frame(0.0, "b64", pi)

    def test_first_survivor_per_point(self):
        # B1/C1/C2 黑帧剔除后剩 12 帧 → 每点取首个存活帧
        frames = [self._f(0), self._f(0), self._f(0),
                  self._f(1), self._f(1),
                  self._f(2),
                  self._f(3), self._f(3), self._f(3),
                  self._f(4), self._f(4), self._f(4)]
        out = per_point_frames(frames)
        self.assertEqual([f.point_index for f in out], [0, 1, 2, 3, 4])

    def test_point_lost_entirely(self):
        # 点位 2 全灭 → 只剩 4 个点位
        frames = [self._f(0), self._f(0), self._f(0),
                  self._f(1), self._f(1), self._f(1),
                  self._f(3), self._f(3), self._f(3)]
        out = per_point_frames(frames)
        self.assertEqual([f.point_index for f in out], [0, 1, 3])

    def test_order_preserved(self):
        frames = [self._f(1), self._f(0), self._f(1), self._f(0)]
        out = per_point_frames(frames)
        self.assertEqual([f.point_index for f in out], [1, 0])


if __name__ == "__main__":
    unittest.main()
