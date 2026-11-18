import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import threading
import time
import unittest

from batch_rename.stats import PipelineStats


class TestCircuitBreakerAI(unittest.TestCase):
    def test_trips_after_threshold(self):
        stats = PipelineStats(total=10, ai_failure_threshold=3, ai_window_sec=60)
        stop = threading.Event()
        self.assertFalse(stats.record_failure(stop, category="ai"))
        self.assertFalse(stats.record_failure(stop, category="ai"))
        # 第 3 次触发熔断
        self.assertTrue(stats.record_failure(stop, category="ai"))
        self.assertTrue(stop.is_set())
        self.assertTrue(stats.ai_tripped)

    def test_no_trip_below_threshold(self):
        stats = PipelineStats(total=10, ai_failure_threshold=5, ai_window_sec=60)
        stop = threading.Event()
        for _ in range(4):
            self.assertFalse(stats.record_failure(stop, category="ai"))
        self.assertFalse(stop.is_set())

    def test_expired_failures_dont_count(self):
        stats = PipelineStats(total=10, ai_failure_threshold=3, ai_window_sec=0.1)
        stop = threading.Event()
        stats.record_failure(stop, category="ai")
        stats.record_failure(stop, category="ai")
        time.sleep(0.15)  # 等待窗口过期
        # 旧的已过期，这次是窗口内第 1 次
        self.assertFalse(stats.record_failure(stop, category="ai"))
        self.assertFalse(stop.is_set())

    def test_already_tripped_returns_true(self):
        stats = PipelineStats(total=10, ai_failure_threshold=1, ai_window_sec=60)
        stop = threading.Event()
        stats.record_failure(stop, category="ai")
        # 已熔断后再调用仍返回 True
        self.assertTrue(stats.record_failure(stop, category="ai"))


class TestCircuitBreakerFrame(unittest.TestCase):
    def test_frame_category_independent(self):
        stats = PipelineStats(total=10, ai_failure_threshold=2,
                              frame_failure_threshold=2, ai_window_sec=60,
                              frame_window_sec=60)
        stop = threading.Event()
        # AI 失败不影响 frame 窗口
        stats.record_failure(stop, category="ai")
        self.assertFalse(stats.record_failure(stop, category="frame"))
        # frame 第 2 次触发
        self.assertTrue(stats.record_failure(stop, category="frame"))
        self.assertTrue(stats.frame_tripped)


class TestSoftReset(unittest.TestCase):
    def test_consecutive_success_clears_window(self):
        stats = PipelineStats(total=10, ai_failure_threshold=3, ai_window_sec=60)
        stop = threading.Event()
        # 累积 2 次失败（未触发）
        stats.record_failure(stop, category="ai")
        stats.record_failure(stop, category="ai")
        # 3 次连续成功 → 清空窗口
        stats.record_success(category="ai")
        stats.record_success(category="ai")
        stats.record_success(category="ai")
        # 再失败 2 次不应触发（窗口已清空，从 0 重新计数）
        self.assertFalse(stats.record_failure(stop, category="ai"))
        self.assertFalse(stats.record_failure(stop, category="ai"))
        self.assertFalse(stop.is_set())

    def test_failure_resets_success_streak(self):
        stats = PipelineStats(total=10, ai_failure_threshold=4, ai_window_sec=60)
        stop = threading.Event()
        stats.record_failure(stop, category="ai")
        stats.record_success(category="ai")
        stats.record_success(category="ai")
        # 失败打断连续成功计数
        stats.record_failure(stop, category="ai")
        stats.record_success(category="ai")
        stats.record_success(category="ai")
        # 只有 2 次连续成功，未达 3 次，窗口未清空
        # 此时窗口内有 2 次失败
        self.assertFalse(stats.record_failure(stop, category="ai"))
        # 第 4 次失败触发（阈值 4）
        self.assertTrue(stats.record_failure(stop, category="ai"))


class TestIncAndDone(unittest.TestCase):
    def test_inc_counters(self):
        stats = PipelineStats(total=5)
        stats.inc("ok")
        stats.inc("ok")
        stats.inc("error")
        stats.inc("skipped")
        self.assertEqual(stats.ok, 2)
        self.assertEqual(stats.error, 1)
        self.assertEqual(stats.skipped, 1)

    def test_inc_done_returns_progress(self):
        stats = PipelineStats(total=3)
        self.assertEqual(stats.inc_done(), (1, 3))
        self.assertEqual(stats.inc_done(), (2, 3))


if __name__ == "__main__":
    unittest.main()
