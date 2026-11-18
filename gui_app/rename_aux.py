"""重命名流水线辅助阶段 worker：ASR 注入与 Tagger 集成。

辅助 worker 由 BatchPipeline 统一调度（异常即兜底其所辖阶段），
各自负责把阶段推进到终态并回填上下文（extra_meta / literal_meta）。
"""
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from batch_rename.env import logger
from .workspace_paths import stable_id


def format_transcribe_block(text: str, lang: str = "", max_chars: int = 2000) -> str:
    """转录文本 → 提示词辅助上下文块（截断 + 固定声明头，手动/集成共用）。"""
    text = (text or "").strip()
    if not text:
        return ""
    max_chars = max(100, max_chars)
    if len(text) > max_chars:
        text = text[:max_chars] + "…（已截断）"
    parts = ["【语音辅助参考】",
             "以下为自动语音转录结果，仅供参考。请结合视频画面自行判断："]
    if lang:
        parts.append(f"- 语言: {lang}")
    parts.append(f"- 转录文本: {text}")
    parts.append("过长的转录文本可能被截断，注意判断")
    return "\n".join(parts)


class AsrInjectWorker:
    """ASR 注入辅助阶段：单子进程批内逐视频转录，完成即回填上下文。

    仅处理缺失 SRT 的视频（runner 传入 missing 列表）；停止/失败时已完成的
    视频保留转录文本，其余阶段兜底终态（空文本降级）。
    """

    stage_key = "asr"

    def __init__(self, missing: List[str], max_chars: int,
                 on_log: Callable[[str, str], None], stop_event: threading.Event):
        self.missing = missing
        self.max_chars = max(100, int(max_chars))
        self.on_log = on_log
        self.stop_event = stop_event

    def run(self, pipeline) -> None:
        try:
            self._run(pipeline)
        finally:
            for ctx in pipeline.contexts:
                ctx.fail_remaining(self.stage_key)

    def _run(self, pipeline) -> None:
        from . import transcribe_cpp
        ctx_of = {ctx.vp: ctx for ctx in pipeline.contexts}

        def _on_video_done(idx: int, entry: Dict[str, Any]) -> None:
            vp = self.missing[idx] if idx < len(self.missing) else None
            ctx = ctx_of.get(vp)
            if ctx is None:
                return
            text = (entry.get("text") or "").strip() if entry.get("ok") else ""
            ctx.extra_meta = format_transcribe_block(text, "", self.max_chars)
            ctx.complete_stage(self.stage_key)

        r = transcribe_cpp.run_inject_batch(
            self.missing, max_chars=self.max_chars,
            on_log=lambda m: self.on_log("info", str(m)),
            stop_event=self.stop_event,
            on_video_done=_on_video_done)
        if self.stop_event.is_set():
            self.on_log("warn", "[语音辅助] 已停止（已完成视频保留转录文本）")
        elif not r.get("ok"):
            self.on_log("warn", f"[语音辅助] 注入失败（不影响重命名）: {r.get('error')}")
        else:
            logger.debug(f"[语音辅助] 已生成 {len(r.get('texts') or {})} 个视频的参考文本")


class TaggerInjectWorker:
    """Tagger 集成辅助阶段：消费 tagger_queue，常驻引擎流式推理。"""

    stage_key = "tagger"
    _PIPE_DEPTH = 2  # 已 send 未收结果上限（与手动分析一致）

    def __init__(self, stop_event: threading.Event,
                 on_log: Callable[[str, str], None],
                 threshold: float, general_count: int,
                 save_fn: Callable[[Dict[str, Any]], None]):
        self.stop_event = stop_event
        self.on_log = on_log
        self.threshold = threshold
        self.general_count = max(0, int(general_count))
        self.save_fn = save_fn

    def run(self, pipeline) -> None:
        q = pipeline.tagger_queue
        if q is None:
            return
        from . import tagger as tagger_mod
        results: Dict[str, Any] = {}   # vid → tags entry（含 error，仅用于统计）
        stream = None
        sent: Dict[int, Any] = {}
        lib = tagger_mod.load_term_library_map()
        try:
            while not self.stop_event.is_set():
                try:
                    ctx = q.get(timeout=0.5)
                except queue.Empty:
                    if pipeline.producer_done.is_set() and q.empty():
                        break
                    continue
                if ctx is None:
                    break
                if stream is None:
                    stream = self._start_engine(len(pipeline.contexts), tagger_mod)
                    if stream is None:
                        self._mark(ctx, results, None, "分析引擎启动失败")
                        continue
                self._handle_ctx(ctx, stream, sent, results, lib, tagger_mod)
                if stream.broken:
                    self._drain(stream, sent, results, lib, tagger_mod)
                    self.on_log("warn", "[标签辅助] 分析引擎中断，剩余视频降级为无标签")
                    break
            # 收尾只收已 send 的结果：引擎常驻不关 stdin，等 EOF 会挂起
            if stream is not None and not self.stop_event.is_set():
                while sent and not self.stop_event.is_set():
                    vr = stream.next_result()
                    if vr is None:
                        break
                    self._record(vr, sent, results, lib, tagger_mod)
        except Exception as e:
            logger.error(f"Tagger 集成线程异常: {e}", exc_info=True)
        finally:
            if stream is not None:
                stream.close()
            for ctx in pipeline.contexts:
                ctx.fail_remaining(self.stage_key)
            if results:
                self.save_fn(results)
            ok_n = sum(1 for r in results.values() if not r.get("error"))
            if results and not self.stop_event.is_set():
                logger.debug(f"[标签辅助] 集成完成：成功 {ok_n}/{len(results)}")

    def _start_engine(self, total: int, tagger_mod):
        """懒启动引擎：首个待推理任务出现时才加载（全部跳过时零开销）。"""
        self.on_log("info", "[标签辅助] 集成开启，正在启动分析引擎（与抽帧并行）…")
        return tagger_mod.start_analyze_stream(
            total, tag_threshold=self.threshold, ip_threshold=self.threshold,
            stop_event=self.stop_event,
            on_log=lambda m: self.on_log("info", str(m)))

    def _handle_ctx(self, ctx, stream, sent, results, lib, tagger_mod) -> None:
        """单任务：机会性收结果 → 管道背压 → send。"""
        vid = stable_id(ctx.vp)
        name = Path(ctx.vp).name
        self._drain(stream, sent, results, lib, tagger_mod)
        if stream.broken or self.stop_event.is_set():
            return
        if not ctx.frames:
            ctx.complete_stage(self.stage_key)  # 抽帧失败：无辅助标签
            return
        try:
            while len(sent) >= self._PIPE_DEPTH:
                vr = stream.next_result()
                if vr is None:
                    return
                self._record(vr, sent, results, lib, tagger_mod)
            if not stream.send({"index": ctx.index, "name": name,
                                "frames": [f.b64 for f in ctx.frames]}):
                return
            sent[ctx.index] = (ctx, vid, name)
        except Exception as e:
            logger.debug(f"Tagger send 失败 ({name}): {e}")

    def _drain(self, stream, sent, results, lib, tagger_mod) -> None:
        """机会性收取已就绪结果（非阻塞）。"""
        while not self.stop_event.is_set():
            vr = stream.try_next_result()
            if vr is None:
                return
            self._record(vr, sent, results, lib, tagger_mod)

    def _record(self, vr: Dict[str, Any], sent, results, lib, tagger_mod) -> None:
        """收一条推理结果：存 entry + 以内存结果构建召回查询词。"""
        vi = int(vr.get("index", -1))
        item = sent.pop(vi, None)
        if item is None:
            return
        ctx, vid, name = item
        if vr.get("error"):
            self._mark(ctx, results, vid, str(vr["error"]))
            self.on_log("warn", f"[标签辅助] {name} — 识别失败: {str(vr['error'])[:120]}")
            return
        entry = {"character_tags": vr.get("character_tags", []),
                 "ip_tags": vr.get("ip_tags", []),
                 "general_tags": vr.get("general_tags", []),
                 "error": None}
        if vr.get("category"):
            entry["category"] = str(vr["category"])
        results[vid] = entry
        ctx.tagger_hits, ctx.literal_meta = tagger_mod.tagger_injection(
            entry, lib, self.general_count)
        ctx.complete_stage(self.stage_key)

    def _mark(self, ctx, results, vid: Optional[str], error: str) -> None:
        """单任务失败：仅计数（error 条目不落盘），上下文照常交付。"""
        if vid is None:
            vid = stable_id(ctx.vp)
        results[vid] = {"character_tags": [], "ip_tags": [], "general_tags": [],
                        "error": error}
        ctx.complete_stage(self.stage_key)
