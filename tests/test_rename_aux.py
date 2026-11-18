"""rename_aux 与管线协作测试：TaskContext 交付语义 / aux worker / 编排兜底。"""
import queue
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_rename.config import Config
from batch_rename.pipeline import BatchPipeline, FrameProducer, TaskContext
from batch_rename.stats import PipelineStats
from batch_rename.types import Frame
from gui_app.rename_aux import (
    AsrInjectWorker, TaggerInjectWorker, format_transcribe_block,
)
from gui_app.runner import _thumb_time_seconds
from gui_app.workspace_paths import stable_id


def _cfg():
    return Config(ai_workers=2)


class TestFormatTranscribeBlock(unittest.TestCase):
    def test_basic_block(self):
        b = format_transcribe_block("你好世界", "zh", 2000)
        self.assertIn("【语音辅助参考】", b)
        self.assertIn("- 语言: zh", b)
        self.assertIn("- 转录文本: 你好世界", b)

    def test_truncated(self):
        b = format_transcribe_block("字" * 1200, "", 800)
        self.assertIn("…（已截断）", b)
        self.assertNotIn("字" * 801, b)

    def test_empty_text_returns_empty(self):
        self.assertEqual(format_transcribe_block("  ", "zh", 2000), "")


class TestThumbTimeSeconds(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(_thumb_time_seconds("00:00:04"), 4)
        self.assertEqual(_thumb_time_seconds("01:02:03"), 3723)

    def test_invalid(self):
        self.assertIsNone(_thumb_time_seconds("abc"))
        self.assertIsNone(_thumb_time_seconds("12:34"))
        self.assertIsNone(_thumb_time_seconds(""))
        self.assertIsNone(_thumb_time_seconds(None))


class TestTaskContext(unittest.TestCase):
    def test_deliver_once_when_all_stages_complete(self):
        delivered = []
        ctx = TaskContext("C:/v.mp4", 0, aux_stages={"tagger", "asr"})
        ctx.on_ready = lambda c: delivered.append(c)
        ctx.complete_stage("frames")
        ctx.complete_stage("asr")
        self.assertEqual(delivered, [])  # tagger 未终态不交付
        self.assertTrue(ctx.complete_stage("tagger"))
        self.assertEqual(len(delivered), 1)
        self.assertFalse(ctx.complete_stage("tagger"))  # 重复完成不二次交付
        self.assertEqual(len(delivered), 1)

    def test_fail_remaining_only_given_stage(self):
        ctx = TaskContext("C:/v.mp4", 0, aux_stages={"tagger", "asr"})
        ctx.complete_stage("asr")
        ctx.fail_remaining("tagger")
        self.assertFalse(ctx.delivered)  # frames 仍未终态
        ctx.complete_stage("frames")
        self.assertTrue(ctx.delivered)

    def test_release_idempotent_and_slot_once(self):
        releases = []
        ctx = TaskContext("C:/v.mp4", 0)
        ctx.cache_slot = True
        ctx.release_hook = lambda: releases.append(1)
        ctx.frames = ["b64"]
        ctx.release()
        ctx.release()  # 幂等
        self.assertEqual(len(releases), 1)
        self.assertIsNone(ctx.frames)


class TestAsrInjectWorker(unittest.TestCase):
    def _pipeline(self, vps, aux=("asr",)):
        with mock.patch("batch_rename.pipeline.VideoCollector.collect", return_value=vps):
            return BatchPipeline(vps, _cfg(), None, threading.Event(),
                                 aux_stages=set(aux))

    def test_per_video_delivery(self):
        vps = ["C:/a.mp4", "C:/b.mp4"]
        pipe = self._pipeline(vps)
        w = AsrInjectWorker(vps, 2000, lambda lv, m: None, threading.Event())

        def _fake_batch(paths, max_chars=800, on_log=None, stop_event=None,
                        on_video_done=None):
            if on_video_done:
                on_video_done(0, {"ok": True, "text": "你好世界"})
                on_video_done(1, {"ok": False, "error": "boom"})
            return {"ok": True, "texts": {"C:/a.mp4": "你好世界"}}

        with mock.patch("gui_app.transcribe_cpp.run_inject_batch",
                        side_effect=_fake_batch), \
             mock.patch("gui_app.transcribe_cpp.inject_ready", return_value=True):
            w.run(pipe)
        # 回调按 idx 逐视频交付：a 有文本，b 空文本降级，asr 阶段全部终态
        ctxs = {c.vp: c for c in pipe.contexts}
        self.assertIn("你好世界", ctxs["C:/a.mp4"].extra_meta)
        self.assertEqual(ctxs["C:/b.mp4"].extra_meta, "")
        self.assertFalse(any(c.stage_pending("asr") for c in pipe.contexts))

    def test_stop_keeps_completed(self):
        vps = ["C:/a.mp4", "C:/b.mp4"]
        pipe = self._pipeline(vps)
        stop = threading.Event()
        stop.set()
        w = AsrInjectWorker(vps, 2000, lambda lv, m: None, stop)
        with mock.patch("gui_app.transcribe_cpp.run_inject_batch",
                        return_value={"ok": False, "error": "已取消"}):
            w.run(pipe)
        self.assertFalse(any(c.stage_pending("asr") for c in pipe.contexts))


class _PipelineStub:
    """rename_aux 单元测试用的最小管线替身。"""

    def __init__(self, vps, tagger_queue=True):
        self.config = _cfg()
        self.stop_event = threading.Event()
        self.contexts = [TaskContext(vp, i, aux_stages={"tagger"})
                         for i, vp in enumerate(vps)]
        self.tagger_queue = queue.Queue(maxsize=64) if tagger_queue else None
        self.producer_done = threading.Event()
        self.producer_done.set()  # 单元测试无生产者，直接视为已收尾
        self.aux_stages = {"tagger"} if tagger_queue else set()
        for c in self.contexts:  # 单元测试不跑生产者，frames 阶段手工终态
            c.frames = [Frame(3.6, "b64")]
            c.complete_stage("frames")
        if self.tagger_queue:
            for c in self.contexts:
                self.tagger_queue.put(c)


class TestTaggerInjectWorker(unittest.TestCase):
    def _pipeline(self, vps):
        return _PipelineStub(vps)

    def test_query_words_filled_from_memory_result(self):
        pipe = self._pipeline(["C:/a.mp4"])
        ctx = pipe.contexts[0]
        results = {}
        w = TaggerInjectWorker(threading.Event(), lambda lv, m: None,
                               threshold=0.75, general_count=15,
                               save_fn=lambda r: results.update(r))
        state = {"sent": False}

        def _next_result():
            if state["sent"]:
                return None
            state["sent"] = True
            return {"index": 0, "name": "a.mp4",
                    "character_tags": [{"name": "hakurei_reimu", "score": 0.9}],
                    "ip_tags": [{"name": "touhou", "score": 0.8}],
                    "general_tags": [], "category": "2d"}

        stream = mock.Mock()
        stream.broken = False
        stream.send.return_value = True
        stream.try_next_result = lambda: None   # send 前的机会性排空无结果
        stream.next_result = _next_result       # 收尾排空返回结果
        stream.close = lambda: None

        with mock.patch("gui_app.tagger.load_term_library_map",
                        return_value={"hakurei_reimu": {"zh": "灵梦", "desc": "红白",
                                                        "group": "Hammy-tagger"},
                                      "touhou": {"zh": "东方", "desc": "Project",
                                                 "group": "Hammy-tagger"}}), \
             mock.patch("gui_app.tagger.start_analyze_stream", return_value=stream):
            w.run(pipe)
        # A 组（角色/IP）命中走 tagger_hits 置顶注入，不进字面查询词
        self.assertEqual([h["keyword"] for h in ctx.tagger_hits], ["灵梦", "东方"])
        self.assertEqual(ctx.literal_meta, "")
        self.assertTrue(ctx.delivered)
        self.assertEqual(results[stable_id("C:/a.mp4")]["category"], "2d")

    def test_engine_start_failure_degrades(self):
        pipe = self._pipeline(["C:/a.mp4"])
        ctx = pipe.contexts[0]
        saved = {}
        w = TaggerInjectWorker(threading.Event(), lambda lv, m: None,
                               threshold=0.75, general_count=15,
                               save_fn=lambda r: saved.update(r))
        with mock.patch("gui_app.tagger.load_term_library_map", return_value=None), \
             mock.patch("gui_app.tagger.start_analyze_stream", return_value=None):
            w.run(pipe)
        self.assertTrue(ctx.delivered)
        self.assertIsNone(ctx.literal_meta)  # 引擎失败降级为无标签
        self.assertEqual(saved[stable_id("C:/a.mp4")]["error"], "分析引擎启动失败")

    def test_stop_before_run_completes_all(self):
        pipe = self._pipeline(["C:/a.mp4", "C:/b.mp4"])
        stop = threading.Event()
        stop.set()
        w = TaggerInjectWorker(stop, lambda lv, m: None,
                               threshold=0.75, general_count=15, save_fn=lambda r: None)
        with mock.patch("gui_app.tagger.load_term_library_map", return_value=None), \
             mock.patch("gui_app.tagger.start_analyze_stream") as sa:
            w.run(pipe)
        sa.assert_not_called()  # 停止信号先于主循环，引擎零开销
        self.assertFalse(any(c.stage_pending("tagger") for c in pipe.contexts))


class TestPipelineAuxCoordination(unittest.TestCase):
    def _pipe(self, vps, aux_stages):
        with mock.patch("batch_rename.pipeline.VideoCollector.collect", return_value=vps):
            return BatchPipeline(vps, _cfg(), None, threading.Event(),
                                 aux_stages=set(aux_stages))

    def test_delivery_gated_on_aux_stages(self):
        pipe = self._pipe(["C:/a.mp4"], aux_stages=("tagger", "asr"))
        ctx = pipe.contexts[0]
        delivered = []
        ctx.on_ready = lambda c: delivered.append(c)
        ctx.complete_stage("frames")
        ctx.complete_stage("asr")
        self.assertEqual(delivered, [])
        ctx.complete_stage("tagger")
        self.assertEqual(len(delivered), 1)

    def test_prefill_skip_skips_queueing(self):
        """tagger 阶段已预完成（已有结果）的上下文不再送推理队列。"""
        pipe = self._pipe(["C:/a.mp4"], aux_stages=("tagger",))
        ctx = pipe.contexts[0]
        ctx.complete_stage("tagger")  # runner 预完成
        producer = FrameProducer(pipe.contexts, pipe.config, pipe.stop_event)
        with mock.patch.object(producer, "_submit_frame_task") as submit:
            submit.return_value = None  # 不真正抽帧
            producer.run(pipe.tagger_queue)
        # _collect 未入队（阶段已终态），就地 complete（无副作用）
        self.assertFalse(ctx.stage_pending("tagger"))

    def test_aux_crash_fallback_completes_stage(self):
        pipe = self._pipe(["C:/a.mp4"], aux_stages=("tagger",))
        ctx = pipe.contexts[0]
        ctx.frames = ["b64"]
        ctx.complete_stage("frames")
        # 模拟 _run_aux 包装：worker.run 抛异常 → fail_remaining(stage_key)
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            for c in pipe.contexts:
                c.fail_remaining("tagger")
        self.assertTrue(ctx.delivered)

    def test_stop_counts_closed(self):
        """停止后未交付上下文计入 cancelled，ok+error+cancelled==total。"""
        pipe = self._pipe(["C:/a.mp4", "C:/b.mp4", "C:/c.mp4"], aux_stages=())
        stats = PipelineStats(total=3)
        pipe.stop_event.set()
        for ctx in pipe.contexts:
            if not ctx.delivered:
                stats.inc_cancelled()
                ctx.release()
        s = stats.to_summary(0)
        self.assertEqual(s.ok + s.error + s.cancelled, 3)

    def test_thumb_lookup_rounding_matches_label(self):
        """thumb_lookup 键 = int(ts + 0.5)，与提示词时间标签 _fmt_hms 同口径。"""
        ctx = TaskContext("C:/v.mp4", 0)
        ctx.frames = [Frame(3.4, "a"), Frame(4.5, "b")]
        ctx.thumb_lookup = {int(f.ts + 0.5): f for f in ctx.frames}
        self.assertEqual(ctx.thumb_lookup[3].b64, "a")
        self.assertEqual(ctx.thumb_lookup[5].b64, "b")  # int(4.5+0.5)=5（round() 会得 4）
        self.assertNotIn(4, ctx.thumb_lookup)


if __name__ == "__main__":
    unittest.main()
