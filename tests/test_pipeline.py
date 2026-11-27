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
    _max_ai_seconds_per_video, _all_queued, AIWorker, BatchPipeline, TaskContext,
    ready_gate,
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


class TestAllQueued(unittest.TestCase):
    """closer 关闭谓词：交付且实际入队才算就绪（防 delivered 先于 put 的竞态）。"""

    def _ctxs(self, n, aux=()):
        return [TaskContext(f"C:/v{i}.mp4", i, aux_stages=set(aux)) for i in range(n)]

    def test_delivered_without_queued_blocks_close(self):
        # 竞态场景：delivered 已置位、尚未入队，closer 不得关闭
        ctxs = self._ctxs(1)
        ctxs[0].complete_stage("frames")
        self.assertFalse(_all_queued(ctxs))

    def test_all_delivered_and_queued_closes(self):
        ctxs = self._ctxs(2)
        for c in ctxs:
            c.complete_stage("frames")
            c.queued = True
        self.assertTrue(_all_queued(ctxs))

    def test_undelivered_blocks_close(self):
        ctxs = self._ctxs(2)
        ctxs[0].complete_stage("frames")
        ctxs[0].queued = True
        self.assertFalse(_all_queued(ctxs))


class TestReadyGate(unittest.TestCase):
    """AI 前置就绪门：并发调用单飞执行 ensure，结果缓存后恒通过。"""

    def test_single_flight_and_cache(self):
        calls = []
        gate = ready_gate(lambda ev: calls.append(1) or True, threading.Event())
        results = []
        threads = [threading.Thread(target=lambda: results.append(gate()))
                   for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(2)
        self.assertEqual(results, [True, True, True])
        self.assertEqual(len(calls), 1)


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

        def _cb(vp, new_path, status, info, title, plot, tags, thumb_lookup=None):
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

        def _cb(vp, new_path, status, info, title, plot, tags, thumb_lookup=None):
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
            ctx = TaskContext("C:/v.mp4", 0)
            ctx.frames = ["b64"]
            worker._process_task(ctx)
        af.assert_not_called()
        self.assertEqual(stats.to_summary(0).cancelled, 1)

    def test_process_task_frame_error_dispatch(self):
        """frame_b64s=None → 走抽帧失败分支，不调 AI。"""
        worker, stats, _ = _make_worker()
        with mock.patch("batch_rename.pipeline.analyze_frames") as af:
            ctx = TaskContext("C:/v.mp4", 0)
            ctx.frames = None
            worker._process_task(ctx)
        af.assert_not_called()
        self.assertEqual(stats.error, 1)


class TestAIWorkerSuccess(unittest.TestCase):
    """成功路径：重命名 + NFO 写入 + 回调 status=ok。"""

    def test_success_rename_ok(self):
        worker, stats, _ = _make_worker()
        calls = []

        def _cb(vp, new_path, status, info, title, plot, tags, thumb_lookup=None):
            calls.append((new_path, status, title, tags))

        worker._on_file_done = _cb
        with mock.patch("batch_rename.pipeline.build_new_stem", return_value="新名"), \
             mock.patch("batch_rename.pipeline.rename_video",
                        return_value=("C:/新名.mp4", "ok")), \
             mock.patch("batch_rename.pipeline.write_nfo"):
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
            ctx = TaskContext("C:/v.mp4", 0)
            ctx.frames = ["b64"]
            worker._process_task(ctx)
        self.assertEqual(stats.to_summary(0).processed_count, 1)

    def test_process_task_ai_empty_title(self):
        """AI 返回空标题 → 失败分发。"""
        worker, stats, _ = _make_worker()
        result = AnalyzeResult("", "", [], 0, "空内容", "empty")
        with mock.patch("batch_rename.pipeline.analyze_frames", return_value=result), \
             mock.patch("batch_rename.pipeline.is_context_error", return_value=False):
            ctx = TaskContext("C:/v.mp4", 0)
            ctx.frames = ["b64"]
            worker._process_task(ctx)
        self.assertEqual(stats.error, 1)


class TestBatchPipelineDiscover(unittest.TestCase):
    """发现与过滤：空列表返回 None；有视频则初始化 stats 与熔断阈值。"""

    def test_no_videos_returns_none(self):
        with mock.patch("batch_rename.pipeline.VideoCollector.collect",
                        return_value=[]):
            pipe = BatchPipeline([], Config(), None, threading.Event())
            self.assertIsNone(pipe._discover_and_filter())

    def test_videos_init_stats_and_threshold(self):
        cfg = Config(ai_workers=4)
        with mock.patch("batch_rename.pipeline.VideoCollector.collect",
                        return_value=["C:/a.mp4", "C:/b.mp4"]):
            pipe = BatchPipeline(["C:/a.mp4", "C:/b.mp4"], cfg, None, threading.Event())
            stats = pipe._discover_and_filter()
        self.assertIsNotNone(stats)
        self.assertEqual(stats.total, 2)
        # 阈值 = max(2, min(ai_workers+1, 5)) = 5
        self.assertEqual(stats.ai_failure_threshold, 5)
        # 失败窗口下限 = 熔断器基础窗口
        self.assertGreaterEqual(stats.ai_window_sec, 60.0)


class TestAIWorkerLlamaGate(unittest.TestCase):
    """AI 前置就绪门：失败停止整批（任务计取消），通过则正常处理。"""

    def _worker_with_tasks(self, gate, n_tasks=2):
        st = PipelineStats(total=n_tasks)
        ev = threading.Event()
        q = queue.Queue()
        for i in range(n_tasks):
            ctx = TaskContext(f"C:/v{i}.mp4", i)
            ctx.frames = ["b64"]
            q.put(ctx)
        q.put(None)
        return AIWorker(None, Config(), st, q, ev, llama_gate=gate), st, ev

    def test_gate_failure_stops_and_cancels(self):
        ensure_calls = []
        gate = ready_gate(lambda ev: ensure_calls.append(1) or False,
                          threading.Event())
        worker, stats, ev = self._worker_with_tasks(gate)
        worker.run()
        self.assertTrue(ev.is_set())
        self.assertEqual(len(ensure_calls), 1)  # 单飞：整批只等待一次
        self.assertEqual(stats.cancelled, 2)    # 首任务 + 排空剩余
        self.assertIsNotNone(stats.fatal_error)  # on_done 据此返回 ok:False

    def test_gate_pass_processes(self):
        gate = ready_gate(lambda ev: True, threading.Event())
        worker, stats, ev = self._worker_with_tasks(gate)
        result = AnalyzeResult("标题", "剧情", ["t"], 0, "", "")
        with mock.patch("batch_rename.pipeline.analyze_frames", return_value=result), \
             mock.patch("batch_rename.pipeline.build_new_stem", return_value="新名"), \
             mock.patch("batch_rename.pipeline.rename_video",
                        return_value=("C:/新名.mp4", "ok")):
            worker.run()
        self.assertEqual(stats.ok, 2)
        self.assertIsNone(stats.fatal_error)


if __name__ == "__main__":
    unittest.main()
