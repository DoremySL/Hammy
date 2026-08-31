import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest

from batch_rename.config import Config
from batch_rename.utils import safe_int, safe_float


class TestConfigValidate(unittest.TestCase):
    def test_temperature_clamp_high(self):
        c = Config(temperature=5.0); c.validate()
        self.assertEqual(c.temperature, 2.0)

    def test_temperature_clamp_low(self):
        c = Config(temperature=-1.0); c.validate()
        self.assertEqual(c.temperature, 0.0)

    def test_top_p_clamp(self):
        c = Config(top_p=2.0); c.validate()
        self.assertEqual(c.top_p, 1.0)

    def test_max_tokens_min(self):
        c = Config(max_tokens=0); c.validate()
        self.assertEqual(c.max_tokens, 1)

    def test_frame_max_side_min(self):
        c = Config(frame_max_side=10); c.validate()
        self.assertEqual(c.frame_max_side, 64)

    def test_sampling_points_min(self):
        c = Config(sampling_points=0); c.validate()
        self.assertEqual(c.sampling_points, 1)

    def test_frames_per_point_min(self):
        c = Config(frames_per_point=-2); c.validate()
        self.assertEqual(c.frames_per_point, 1)

    def test_ai_workers_min(self):
        c = Config(ai_workers=0); c.validate()
        self.assertEqual(c.ai_workers, 1)

    def test_retry_times_min(self):
        c = Config(retry_times=-5); c.validate()
        self.assertEqual(c.retry_times, 0)


class TestSafeConv(unittest.TestCase):
    def test_safe_int(self):
        self.assertEqual(safe_int("5"), 5)
        self.assertEqual(safe_int(None), 0)
        self.assertEqual(safe_int("x", 7), 7)
        self.assertEqual(safe_int(""), 0)
        self.assertEqual(safe_int(3.9), 3)

    def test_safe_float(self):
        self.assertEqual(safe_float("1.5"), 1.5)
        self.assertEqual(safe_float(None), 0.0)
        self.assertEqual(safe_float("x", 2.5), 2.5)


if __name__ == "__main__":
    unittest.main()
