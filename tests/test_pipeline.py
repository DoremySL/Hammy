"""管线编排测试：超时估算 / AIWorker 失败分发 / 成功路径 / 发现与熔断阈值。"""
import sys
import threading
import queue
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_rename.ai import AnalyzeResult
from batch_rename.config import Config
from batch_rename.pipeline import (
    _max_ai_seconds_per_video, AIWorker, BatchPipeline,
)
from batch_rename.stats import PipelineStats


class TestMaxAiSecondsPerVideo(unittest.TestCase):
    """单视频最坏耗时估算 = (超时 + 宽限) × (重试 + 1)。"""

    def test_formula(self):
        cfg = Config(ai_timeout=60, retry_times=2)
        self.assertEqual(_max_ai_seconds_per_video(cfg), (60 + 10) * 3)

    def test_zero_retries(self):
        cfg = Config(ai_timeout=30, retry_times=0)
        self.assertEqual(_max_ai_seconds_per_video(cfg), 40)


def _make_worker(config=None, stats=None, stop_event=None):
    cfg = config or Config()
    st = stats or PipelineStats(total=1)
    ev = stop_event or threading.Event()
    return AIWorker(None, cfg, st, queue.Queue(), ev), st, ev


class TestAIWorkerFrameError(unittest.TestCase):
    """抽帧失败分发：计数 + 失败记录 + 回调 status=frame_error。"""

    def test_frame_error_path(self):
        worker, stats, _ = _make_worker()
        calls = []

        def _cb(vp, new_path, status, info, title, plot, tags):
            calls.append((vp, new_path, status))

        worker._on_file_done = _cb
        msg = worker._handle_frame_error("C:/v.mp4")

        self.assertIn("抽帧失败", msg)
        self.assertEqual(stats.error, 1)
        self.assertEqual(stats.error_files, [("C:/v.mp4", "v.mp4")])
        self.assertEqual(calls, [("C:/v.mp4", "C:/v.mp4", "frame_error")])
        self.assertEqual(stats.to_summary(0).processed_count, 0)  # done 未增


class TestAIWorkerFailure(unittest.TestCase):
    """AI 失败分发：计数 + 回调 + 错误提示。"""

    def test_failure_common(self):
        worker, stats, _ = _make_worker()
        calls = []

        def _cb(vp, new_path, status, info, title, plot, tags):
            calls.append(status)

        worker._on_file_done = _cb
        with mock.patch("batch_rename.pipeline.is_context_error", return_value=False):
            msg = worker._handle_failure("C:/v.mp4", 1, "boom", "api")

        self.assertIn("AI失败", msg)
        self.assertEqual(stats.error, 1)
        self.assertEqual(calls, ["error"])

    def test_failure_format_hint(self):
        """格式类错误（format/empty）给用户附加提示。"""
        worker, _, _ = _make_worker()
        with mock.patch("batch_rename.pipeline.is_context_error", return_value=False):
            msg = worker._handle_failure("C:/v.mp4", 0, "JSON 解析失败", "format")
        self.assertIn("JSON 解析失败", msg)

    def test_failure_no_hint_for_api_error(self):
        worker, _, _ = _make_worker()
        with mock.patch("batch_rename.pipeline.is_context_error", return_value=False):
            msg = worker._handle_failure("C:/v.mp4", 0, "502", "server")
        self.assertNotIn("502", msg)

    def test_context_error_marks_stats(self):
        """上下文错误提示标记（show_context_warning 不抛错即可）。"""
        worker, stats, _ = _make_worker()
        with mock.patch("batch_rename.pipeline.is_context_error", return_value=True), \
             mock.patch.object(stats, "show_context_warning") as sw:
            worker._handle_failure("C:/v.mp4", 0, "context too long", "api")
        sw.assert_called_once()

    def test_process_task_cancelled_when_stopped(self):
        """停止后出队的任务计入 cancelled，不触发 AI。"""
        stop = threading.Event()
        stop.set()
        worker, stats, _ = _make_worker(stop_event=stop)
        with mock.patch("batch_rename.pipeline.analyze_frames") as af:
            worker._process_task(("C:/v.mp4", {"duration": 5.0}, ["b64"]))
        af.assert_not_called()
        self.assertEqual(stats.to_summary(0).cancelled, 1)

    def test_process_task_frame_error_dispatch(self):
        """frame_b64s=None → 走抽帧失败分支，不调 AI。"""
        worker, stats, _ = _make_worker()
        with mock.patch("batch_rename.pipeline.analyze_frames") as af:
            worker._process_task(("C:/v.mp4", {}, None))
        af.assert_not_called()
        self.assertEqual(stats.error, 1)


class TestAIWorkerSuccess(unittest.TestCase):
    """成功路径：重命名 + NFO 写入 + 回调 status=ok。"""

    def test_success_rename_ok(self):
        worker, stats, _ = _make_worker()
        calls = []

        def _cb(vp, new_path, status, info, title, plot, tags):
            calls.append((new_path, status, title, tags))

        worker._on_file_done = _cb
        with mock.patch("batch_rename.pipeline.build_new_stem", return_value="新名"), \
             mock.patch("batch_rename.pipeline.rename_video",
                        return_value=("C:/新名.mp4", "ok")):
            msg = worker._handle_success(
                "C:/v.mp4", {"duration": 5.0}, "标题", "剧情", ["tag1"], 2)

        self.assertIn("v.mp4 -> 新名.mp4", msg)
        self.assertEqual(stats.ok, 1)
        self.assertEqual(calls, [("C:/新名.mp4", "ok", "标题", ["tag1"])])

    def test_success_writes_nfo_with_namer(self):
        """NFO 恒生成：缓存模式（nfo_target_dir + nfo_namer）下 nfo_name 透传。"""
        cfg = Config(nfo_target_dir="C:/cache")
        worker, stats, _ = _make_worker(config=cfg)
        worker._nfo_namer = lambda vp: "stable-123"
        with mock.patch("batch_rename.pipeline.build_new_stem", return_value="新名"), \
             mock.patch("batch_rename.pipeline.rename_video",
                        return_value=("C:/新名.mp4", "ok")), \
             mock.patch("batch_rename.pipeline.write_nfo") as wn:
            worker._handle_success("C:/v.mp4", {}, "标题", "剧情", ["t"], 0)
        wn.assert_called_once()
        _, kwargs = wn.call_args
        self.assertEqual(kwargs["nfo_name"], "stable-123")

        # 无 nfo_target_dir 时同样写 NFO（target_dir=None → 视频所在目录）
        worker2, _, _ = _make_worker(config=Config())
        with mock.patch("batch_rename.pipeline.build_new_stem", return_value="新名"), \
             mock.patch("batch_rename.pipeline.rename_video",
                        return_value=("C:/新名.mp4", "ok")), \
             mock.patch("batch_rename.pipeline.write_nfo") as wn2:
            worker2._handle_success("C:/v.mp4", {}, "标题", "剧情", ["t"], 0)
        wn2.assert_called_once()

    def test_success_rename_failed(self):
        """rename 冲突 → status=error 计数 + 回调 error。"""
        worker, stats, _ = _make_worker()
        with mock.patch("batch_rename.pipeline.build_new_stem", return_value="新名"), \
             mock.patch("batch_rename.pipeline.rename_video",
                        return_value=("C:/v.mp4", "error")):
            msg = worker._handle_success("C:/v.mp4", {}, "标题", "剧情", [], 0)
        self.assertIn("重命名失败", msg)
        self.assertEqual(stats.error, 1)


class TestAIWorkerProcessSuccess(unittest.TestCase):
    """_process_task 成功路径：AI 返回标题 → 成功分发。"""

    def test_process_task_success(self):
        worker, stats, _ = _make_worker()
        result = AnalyzeResult("标题", "剧情", ["t"], 1, "", "")
        with mock.patch("batch_rename.pipeline.analyze_frames", return_value=result), \
             mock.patch("batch_rename.pipeline.build_new_stem", return_value="新名"), \
             mock.patch("batch_rename.pipeline.rename_video",
                        return_value=("C:/新名.mp4", "ok")):
            worker._process_task(("C:/v.mp4", {"duration": 5.0}, ["b64"]))
        self.assertEqual(stats.to_summary(0).processed_count, 1)

    def test_process_task_ai_empty_title(self):
        """AI 返回空标题 → 失败分发。"""
        worker, stats, _ = _make_worker()
        result = AnalyzeResult("", "", [], 0, "空内容", "empty")
        with mock.patch("batch_rename.pipeline.analyze_frames", return_value=result), \
             mock.patch("batch_rename.pipeline.is_context_error", return_value=False):
            worker._process_task(("C:/v.mp4", {}, ["b64"]))
        self.assertEqual(stats.error, 1)


class TestBatchPipelineDiscover(unittest.TestCase):
    """发现与过滤：空列表返回 None；有视频则初始化 stats 与熔断阈值。"""

    def test_no_videos_returns_none(self):
        pipe = BatchPipeline([], Config(), None, threading.Event())
        with mock.patch("batch_rename.pipeline.VideoCollector.collect",
                        return_value=[]):
            self.assertIsNone(pipe._discover_and_filter())

    def test_videos_init_stats_and_threshold(self):
        cfg = Config(ai_workers=4)
        pipe = BatchPipeline(["C:/a.mp4", "C:/b.mp4"], cfg, None, threading.Event())
        with mock.patch("batch_rename.pipeline.VideoCollector.collect",
                        return_value=["C:/a.mp4", "C:/b.mp4"]):
            stats = pipe._discover_and_filter()
        self.assertIsNotNone(stats)
        self.assertEqual(stats.total, 2)
        # 阈值 = max(2, min(ai_workers+1, 5)) = 5
        self.assertEqual(stats.ai_failure_threshold, 5)
        # 失败窗口下限 = 熔断器基础窗口
        self.assertGreaterEqual(stats.ai_window_sec, 60.0)


if __name__ == "__main__":
    unittest.main()
