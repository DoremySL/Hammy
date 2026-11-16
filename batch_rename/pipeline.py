"""管线编排：抽帧生产者、辅助阶段 worker、AI 消费者、批处理编排器。"""
import time
import queue
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Set
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED, Future

from .config import Config, OpenAIClient
from .env import SEP_INDENT, logger
from .utils import retry_hint, print_summary
from .naming import rename_video, build_new_stem
from .video import probe_and_extract_keyframes
from .nfo import write_nfo
from .ai import analyze_frames, is_context_error
from .stats import PipelineStats, _AI_FAILURE_THRESHOLD, _CIRCUIT_WINDOW_SEC
from .collector import VideoCollector
from .types import Frame
from .failed import move_to_failed

_TIMEOUT_GRACE_SEC = 10  # AI 超时后的额外宽限（网络延迟/进程清理）

# ── 队列与线程池缓冲系数 ──
_FRAME_WORKERS = 4
_FRAME_PENDING_FACTOR = 4
_TASK_QUEUE_FACTOR = 4
_TAGGER_QUEUE_MAX = 64   # 集成 Tagger 待推理队列上限（超出降级为无辅助标签）
# ── join 超时兜底 ──
_MIN_JOIN_TIMEOUT_SEC = 120.0
_JOIN_SLACK_FACTOR = 1.5


def _max_ai_seconds_per_video(config: Config) -> float:
    """单视频 AI 处理的最坏耗时估算。"""
    return (config.ai_timeout + _TIMEOUT_GRACE_SEC) * (config.retry_times + 1)


def _cache_limit(config: Config) -> int:
    """帧缓存上限（任务数）：不小于消费侧在途量，防饿死。"""
    return max(32, config.ai_workers * 4 + 4)


def _ai_seconds_per_video(config: Config, two_round: bool) -> float:
    """管线背压/join 超时用的单视频最坏耗时；增强档两轮按翻倍估算。"""
    secs = _max_ai_seconds_per_video(config)
    return secs * 2 if two_round else secs


def _rag_suffix(recalled: int) -> str:
    """完成日志的召回后缀。"""
    if recalled < 0:
        return ""
    return f"（召回{recalled}个关键词）"


def _worker_slot_ids(n_workers: int, slots: int) -> List[Optional[int]]:
    """worker i → 槽位 i % slots；slots <= 0（远程 API）返回全 None。"""
    if slots <= 0:
        return [None] * n_workers
    return [i % slots for i in range(n_workers)]


# 每文件完成回调签名: (original_path, new_path, status, info, title, plot, tags,
#                      thumb_lookup=None)
# status ∈ {"ok","skipped","error","frame_error"}；new_path 为改名后路径（失败时为原路径）
OnFileDone = Callable[..., None]


# ── 任务上下文 ────────────────────────────────────────
class TaskContext:
    """单视频任务上下文：辅助阶段状态 + LLM 就绪交付。

    阶段集合 = {"frames"} ∪ aux_stages（如 {"tagger", "asr"}）；
    complete_stage() 标记单阶段终态，全部终态时经 on_ready 交付一次且仅一次。
    """

    def __init__(self, vp: str, index: int, aux_stages: Optional[Set[str]] = None,
                 extra_meta: str = "", literal_meta: Optional[str] = None):
        self.vp = vp
        self.index = index
        self.aux_stages = frozenset(aux_stages or ())
        self.info: Dict[str, Any] = {}
        self.frames: Optional[List[Frame]] = None   # None = 抽帧失败
        # LLM 组 prompt 用的辅助上下文（辅助 worker 完成时回填；prefill 直接携带）
        self.extra_meta = extra_meta
        self.literal_meta = literal_meta
        # 缩略图缓存：int(ts + 0.5) 秒 → Frame（与提示词时间标签同一取整口径）
        self.thumb_lookup: Optional[Dict[int, Frame]] = None
        # 帧缓存额度（FrameProducer 信号量）：提交抽帧前置位，release 时归位
        self.cache_slot = False
        self.release_hook: Optional[Callable[[Any], None]] = None
        self.on_ready: Optional[Callable[["TaskContext"], None]] = None
        self._stages_left = {"frames", *self.aux_stages}
        self._lock = threading.Lock()
        self._released = False
        self._delivered = False

    @property
    def delivered(self) -> bool:
        with self._lock:
            return self._delivered

    def stage_pending(self, key: str) -> bool:
        """阶段是否尚未终态。"""
        with self._lock:
            return key in self._stages_left

    def complete_stage(self, key: str) -> bool:
        """标记阶段终态；返回 True 表示本次调用触发了交付。"""
        deliver = False
        with self._lock:
            self._stages_left.discard(key)
            if not self._stages_left and not self._delivered:
                self._delivered = True
                deliver = True
        if deliver and self.on_ready is not None:
            self.on_ready(self)
        return deliver

    def fail_remaining(self, stage_key: Optional[str] = None) -> None:
        """把未终态的阶段标记终态，避免「永久等待」。

        stage_key 指定时只兜底该辅助阶段（worker 退出用，不越权其他阶段）；
        缺省时兜底除 frames 外的全部阶段（管线 join 后用，frames 已由生产者
        线程保证终态）。
        """
        with self._lock:
            if stage_key is not None:
                left = {stage_key} if stage_key in self._stages_left else set()
            else:
                left = set(self._stages_left)
                left.discard("frames")
        for k in sorted(left):
            self.complete_stage(k)

    def release(self) -> None:
        """LLM 消费完毕/取消后释放帧缓存（幂等，线程安全）。"""
        with self._lock:
            if self._released:
                return
            self._released = True
            owned = self.cache_slot
            self.cache_slot = False
        self.frames = None
        self.thumb_lookup = None
        if owned and self.release_hook is not None:
            try:
                self.release_hook()
            except Exception:
                pass


# ── 抽帧生产者 ────────────────────────────────────────
class FrameProducer:
    """抽帧生产者：ThreadPoolExecutor 并发抽帧，结果写入任务上下文。

    帧缓存信号量限制「已抽帧但未交付」的任务数（等待辅助阶段期间的内存上限）；
    额度在任务提交前获取、在 LLM 消费完毕或取消清理时经 ctx.release() 归位。
    """

    def __init__(self, contexts: List[TaskContext], config: Config,
                 stop_event: threading.Event, ai_secs: Optional[float] = None):
        self.contexts = contexts
        self.config = config
        self.stop_event = stop_event
        self._ai_secs = ai_secs if ai_secs is not None else _max_ai_seconds_per_video(config)
        self._cache_sem = threading.Semaphore(_cache_limit(config))

    def release_slot_for(self) -> None:
        """缓存额度归位（TaskContext.release 触发，无参钩子）。"""
        self._cache_sem.release()

    def _submit_frame_task(self, exec_: ThreadPoolExecutor,
                           ctx: TaskContext) -> Optional[Future]:
        """获取缓存额度后提交抽帧任务；停止或额度等待超限返回 None。

        额度获取成功才置位 ctx.cache_slot，release 才归位（未提交不占额度）。"""
        deadline = time.time() + self._ai_secs
        while True:
            if self.stop_event.is_set():
                return None
            if self._cache_sem.acquire(blocking=False):
                ctx.cache_slot = True
                return exec_.submit(self._extract_ctx, ctx)
            if time.time() > deadline:
                logger.error(f"帧缓存额度等待超过 {self._ai_secs:.0f} 秒，强制停止管线。")
                self.stop_event.set()
                return None
            self.stop_event.wait(0.1)

    def _extract_ctx(self, ctx: TaskContext) -> None:
        """抽帧 worker 主体：抽帧 → 写入上下文 → frames 阶段终态。"""
        try:
            info, frames = probe_and_extract_keyframes(ctx.vp, self.config, self.stop_event)
            ctx.info = info or {}
            if frames and not self.stop_event.is_set():
                ctx.frames = list(frames)
                if self.config.frame_time_tags == 2:
                    ctx.thumb_lookup = {int(f.ts + 0.5): f for f in frames}
        except Exception as e:
            logger.error(f"抽帧线程异常: {e}")
        finally:
            # 抽帧失败也是 frames 阶段终态（交付后由 LLM 侧 frame_error 处理）
            ctx.complete_stage("frames")

    def _collect(self, future: Future, ctx: TaskContext, tagger_queue) -> None:
        """单个抽帧 future 完成：需要标签推理的送 tagger 队列，其余就地终态。"""
        if self.stop_event.is_set():
            return
        if tagger_queue is not None and ctx.frames and ctx.stage_pending("tagger"):
            try:
                tagger_queue.put_nowait(ctx)
                return
            except queue.Full:
                # Tagger 积压达上限：本次集成未覆盖该视频（降级为无辅助标签）
                logger.debug(f"Tagger 队列已满，跳过 {Path(ctx.vp).name} 的标签推理")
        ctx.complete_stage("tagger")

    def run(self, tagger_queue=None):
        """并发抽帧全部上下文；tagger_queue 非空时成功任务按完成顺序送入队列。"""
        pending = [c for c in self.contexts if not self.stop_event.is_set()]
        try:
            frame_workers = min(_FRAME_WORKERS, len(pending))
            if frame_workers <= 0:
                return
            with ThreadPoolExecutor(max_workers=frame_workers,
                                    thread_name_prefix="frame") as frame_exec:
                futures: Dict[Future, TaskContext] = {}
                for ctx in pending:
                    if self.stop_event.is_set():
                        break
                    fut = self._submit_frame_task(frame_exec, ctx)
                    if fut is None:
                        break
                    futures[fut] = ctx
                    if len(futures) >= frame_workers * _FRAME_PENDING_FACTOR:
                        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                        for f in done:
                            self._collect(f, futures.pop(f), tagger_queue)
                if futures and not self.stop_event.is_set():
                    for future in as_completed(list(futures.keys())):
                        self._collect(future, futures.pop(future), tagger_queue)
                elif futures:
                    # f.cancel() 只取消未开始任务，运行中的抽帧靠内部轮询 stop 退出
                    for f in futures:
                        f.cancel()
        except Exception as e:
            logger.error(f"生产者线程发生致命异常: {e}", exc_info=True)
        finally:
            # 停止/异常路径兜底：frames 阶段未终态的上下文标记终态
            for ctx in self.contexts:
                ctx.complete_stage("frames")


# ── AI 任务处理器 ─────────────────────────────────────
class AIWorker:
    """AI 任务消费者：出队 → 调 AI → 重命名 → 更新统计。"""

    def __init__(self, client: OpenAIClient, config: Config, stats: PipelineStats,
                 task_queue: "queue.Queue[Optional[TaskContext]]",
                 stop_event: threading.Event,
                 on_file_done: Optional[OnFileDone] = None,
                 nfo_namer: Optional[Callable[[str], str]] = None,
                 slot_id: Optional[int] = None,
                 tag_recall=None):
        self.client = client
        self.config = config
        self.stats = stats
        self.task_queue = task_queue
        self.stop_event = stop_event
        self._on_file_done = on_file_done
        # 可选 NFO 目标命名函数（缓存模式防同 stem 冲突）
        self._nfo_namer = nfo_namer
        self.slot_id = slot_id   # llama 集成时 worker i → slot i，两轮同槽命中缓存
        # on/enhanced 都下发给 analyze_frames；两轮耗时估算只看 enhanced+向量可用
        self.tag_recall = (tag_recall
                           if tag_recall is not None
                           and getattr(tag_recall, "mode", "") in ("on", "enhanced")
                           else None)
        self.two_round = bool(self.tag_recall is not None
                              and getattr(self.tag_recall, "has_dense", False))

    def _move_failed_hint(self) -> str:
        return "，稍后将移至 _failed"

    def _emit_file_done(self, vp: str, new_path, status: str, info: dict,
                        title: str = "", plot: str = "", tags: list = None,
                        thumb_lookup=None) -> None:
        """安全触发 on_file_done 回调，异常不影响管线。"""
        if self._on_file_done is None:
            return
        try:
            self._on_file_done(vp, new_path, status, info, title, plot, tags or [],
                               thumb_lookup=thumb_lookup)
        except Exception as e:
            logger.debug(f"on_file_done 回调异常: {e}")

    def _handle_success(self, vp: str, info: Dict[str, Any], title: str,
                        plot: str, tags: List[str], retries: int,
                        recalled: int = -1, recalled_keywords=(),
                        thumb_lookup=None) -> str:
        name = Path(vp).name
        final_stem = build_new_stem(vp, info, title, self.config)
        new_path, status = rename_video(vp, final_stem, self.config)

        if status in ("ok", "skipped"):
            # 缓存模式：唯一命名防同 stem 覆盖
            nfo_name = self._nfo_namer(vp) if self._nfo_namer else None
            write_nfo(video_path=new_path, title=title, plot=plot,
                       tags=tags, info=info, original_name=name,
                       target_dir=self.config.nfo_target_dir,
                       nfo_name=nfo_name, status=status)

        self.stats.inc(status)
        if status == "error":
            self.stats.add_error_file(vp)
        self.stats.add_result(status, name, Path(new_path).name)
        self.stats.record_success(category="ai")

        # 每文件完成回调（GUI 用于写 history / 精确归类 / 缩略图缓存刷新）
        self._emit_file_done(vp, new_path, status, info, title, plot, tags,
                             thumb_lookup=thumb_lookup)

        if status == "ok":
            # 增强模式（recalled ≥ 0）追加召回数：词面+向量去重后的候选关键词数
            rag_suffix = _rag_suffix(recalled)
            if recalled_keywords:
                logger.debug(f"{name} 召回关键词: {'、'.join(recalled_keywords)}")
            return f"{name} -> {Path(new_path).name}{rag_suffix}{retry_hint(retries, success=True)}"
        elif status == "skipped":
            return f"{name}（文件名未变化）"
        else:
            return f"{name} — 重命名失败{self._move_failed_hint()}"

    def _handle_failure(self, vp: str, retries: int, err_msg: str, error_kind: str) -> str:
        name = Path(vp).name
        ctx_error = is_context_error(err_msg)
        if ctx_error:
            self.stats.show_context_warning(self.config)
        self.stats.inc("error")
        self.stats.add_error_file(vp)
        self.stats.add_result("error", name, name, f"AI 分析失败: {err_msg[:80]}")
        tripped = self.stats.record_failure(self.stop_event, "处理失败", category="ai")
        if tripped:
            if ctx_error:
                logger.critical("原因: 模型上下文不足或配置错误。")
            else:
                logger.critical("原因: AI 服务可能已崩溃或无响应。")
        # 失败回调：status="error"，new_path=原路径，info/title/plot/tags 为空
        self._emit_file_done(vp, vp, "error", {})
        msg = f"{name} — AI失败{retry_hint(retries)}{self._move_failed_hint()}"
        # 格式类错误（JSON 解析失败 / 空内容）给用户提示
        if error_kind in ("format", "empty"):
            msg += f"（提示：{err_msg}）"
        return msg

    def _handle_frame_error(self, vp: str) -> str:
        """抽帧失败处理：更新统计 + 触发回调，返回展示消息。"""
        name = Path(vp).name
        self.stats.inc("error")
        self.stats.add_error_file(vp)
        self.stats.add_result("error", name, name, "抽帧失败")
        self.stats.record_failure(self.stop_event, "抽帧失败", category="frame")
        # 抽帧失败回调：status="frame_error"
        self._emit_file_done(vp, vp, "frame_error", {})
        return f"{name} — 抽帧失败{self._move_failed_hint()}"

    def _handle_unexpected_error(self, vp: str, e: Exception) -> str:
        """意外异常处理：更新统计 + 触发回调，返回展示消息。"""
        name = Path(vp).name
        self.stats.inc("error")
        self.stats.add_error_file(vp)
        self.stats.add_result("error", name, name, str(e))
        self.stats.record_failure(self.stop_event, "处理异常", category="ai")
        # 处理异常回调
        self._emit_file_done(vp, vp, "error", {})
        return f"{name} — {e}{self._move_failed_hint()}"

    def _process_task(self, ctx: TaskContext) -> None:
        # 停止时已出队的任务不处理，计入 cancelled（保证计数可核对）
        if self.stop_event.is_set():
            self.stats.inc_cancelled()
            ctx.release()
            return

        display_msg = ""
        try:
            if ctx.frames is None:
                display_msg = self._handle_frame_error(ctx.vp)
            else:
                self.stats.record_success(category="frame")
                # AI 调用 → 成功/失败分发
                try:
                    result = analyze_frames(
                        self.client, self.config.model, ctx.frames, self.config,
                        self.stop_event, ctx.vp, (ctx.info or {}).get("duration", 0.0),
                        container_title=(ctx.info or {}).get("container_title", ""),
                        extra_meta=ctx.extra_meta,
                        literal_meta=ctx.literal_meta,
                        slot_id=self.slot_id,
                        tag_recall=self.tag_recall,
                    )
                    if result.title:
                        if result.thumb_time:
                            ctx.info["thumb_time"] = result.thumb_time
                        display_msg = self._handle_success(
                            ctx.vp, ctx.info, result.title, result.plot,
                            result.tags, result.retries,
                            result.recalled, result.recalled_keywords,
                            thumb_lookup=ctx.thumb_lookup)
                    else:
                        display_msg = self._handle_failure(ctx.vp, result.retries,
                                                           result.err_msg, result.error_kind)
                except Exception as e:
                    display_msg = self._handle_unexpected_error(ctx.vp, e)
        finally:
            ctx.release()
        current, total = self.stats.inc_done()
        if display_msg:
            logger.info(f"{SEP_INDENT}[{current}/{total}] {display_msg}")

    def run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    task = self.task_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if task is None:
                    break
                self._process_task(task)
        except Exception as e:
            logger.exception(f"消费者线程异常退出: {e}")
            self.stop_event.set()
        finally:
            # 提前停止后排空队列剩余任务，计入 cancelled 并释放缓存；
            # None 哨兵放回并停止，保住其它 worker 的退出信号
            while True:
                try:
                    task = self.task_queue.get_nowait()
                except queue.Empty:
                    break
                if task is None:
                    self.task_queue.put_nowait(None)
                    break
                self.stats.inc_cancelled()
                task.release()


# ── 管线编排 ──────────────────────────────────────────
class BatchPipeline:
    """批量重命名管线编排器：组装各组件并执行。

    aux_stages/aux_workers：辅助阶段（tagger/asr）与对应 worker（rename_aux）；
    prefill：视频路径 → {"extra_meta"/"literal_meta"}（复用已有结果的上下文）。
    """

    def __init__(self, paths: List[str], config: Config, client: OpenAIClient,
                 stop_event: threading.Event,
                 on_file_done: Optional[OnFileDone] = None,
                 nfo_namer: Optional[Callable[[str], str]] = None,
                 tag_recall=None,
                 aux_stages: Optional[Set[str]] = None,
                 aux_workers: Optional[List[Any]] = None,
                 prefill: Optional[Dict[str, Dict[str, Any]]] = None):
        self.paths = paths
        self.config = config
        self.client = client
        self.stop_event = stop_event
        self.on_file_done = on_file_done
        self.nfo_namer = nfo_namer
        self.tag_recall = tag_recall
        self.aux_stages = set(aux_stages or ())
        self.aux_workers = aux_workers or []
        self._prefill = prefill or {}
        self.tagger_queue: Optional[queue.Queue] = None
        self.producer: Optional[FrameProducer] = None
        self.producer_done = threading.Event()  # 抽帧生产者收尾信号
        # 两轮耗时估算：enhanced 且向量后端可用（不可用时增强档降级为单轮）
        self._two_round = tag_recall is not None and getattr(tag_recall, "mode", "") == "enhanced" \
            and getattr(tag_recall, "has_dense", False)
        # 暴露 stats 给外部（GUI 完成后读取真实计数）
        self.stats: Optional[PipelineStats] = None
        self._videos: List[str] = []
        self.contexts: List[TaskContext] = []
        self._start_time: float = 0.0
        # 收集视频并在构造期建立任务上下文（调用方可在 run() 前对
        # 「已有结果跳过推理」的上下文预完成对应辅助阶段）
        self._videos = VideoCollector.collect(paths)
        self.contexts = self._build_contexts(self._videos)

    def _build_contexts(self, videos: List[str]) -> List[TaskContext]:
        return [TaskContext(vp=vp, index=i, aux_stages=self.aux_stages,
                            extra_meta=(self._prefill.get(vp) or {}).get("extra_meta", ""),
                            literal_meta=(self._prefill.get(vp) or {}).get("literal_meta"))
                for i, vp in enumerate(videos)]

    def run(self):
        # 入口防御：ai_workers=0 等非法配置会假死
        self.config.validate()
        stats = self._discover_and_filter()
        self.stats = stats  # 暴露给外部读取
        if stats is None:
            return
        try:
            self._run_workers(stats)
        finally:
            # 无论是否正常结束都必须收尾（移动失败文件 + 输出报告）
            self._cleanup(stats)

    def _discover_and_filter(self):
        """返回 PipelineStats（视频已在构造期收集）；无可处理视频时返回 None。"""
        videos = self._videos
        if not videos:
            logger.info(f"\n{SEP_INDENT}未找到待重命名的视频文件")
            return None

        n = len(videos)
        stats = PipelineStats(total=n)

        # 熔断阈值：下限防偶发失败误熔断，上限防大并发迟迟不熔断
        stats.ai_failure_threshold = max(2, min(self.config.ai_workers + 1, _AI_FAILURE_THRESHOLD))

        # 失败窗口 = 连续失败所需轮数 × 每轮耗时，下限取熔断器基础窗口
        cycles_needed = (stats.ai_failure_threshold + self.config.ai_workers - 1) // self.config.ai_workers
        per_failure = _max_ai_seconds_per_video(self.config)
        stats.ai_window_sec = max(float(_CIRCUIT_WINDOW_SEC), cycles_needed * per_failure)

        logger.info(f"{SEP_INDENT}正在自动重命名：{n}个视频")
        logger.info(f"{SEP_INDENT}正在提取关键帧发送AI分析，请耐心等待\n")

        return stats

    def _run_workers(self, stats):
        """启动辅助 worker + 抽帧生产者 + 就绪队列收尾 + AI 消费者。"""
        videos = self._videos
        ai_max_per_video = _ai_seconds_per_video(self.config, self._two_round)
        ready_queue: queue.Queue = queue.Queue(maxsize=self.config.ai_workers * _TASK_QUEUE_FACTOR)
        self.tagger_queue = (queue.Queue(maxsize=_TAGGER_QUEUE_MAX)
                             if "tagger" in self.aux_stages else None)
        producer = FrameProducer(self.contexts, self.config, self.stop_event,
                                 ai_secs=ai_max_per_video)
        self.producer = producer

        def _put_ready(ctx: TaskContext) -> None:
            """阶段全终态 → 进入 LLM 就绪队列；停止/超限直接计取消。"""
            deadline = time.time() + _max_ai_seconds_per_video(self.config)
            while not self.stop_event.is_set():
                try:
                    ready_queue.put(ctx, timeout=0.5)
                    return
                except queue.Full:
                    if time.time() > deadline:
                        logger.error("LLM 就绪队列阻塞超时，强制停止管线。")
                        self.stop_event.set()
                        break
            if self.stop_event.is_set():
                stats.inc_cancelled()
                ctx.release()

        for ctx in self.contexts:
            ctx.on_ready = _put_ready
            ctx.release_hook = producer.release_slot_for

        def _queue_closer() -> None:
            """全部上下文交付（或停止）后投递哨兵，让 AI worker 退出主循环。"""
            while True:
                if self.stop_event.is_set():
                    break
                if all(ctx.delivered for ctx in self.contexts):
                    break
                time.sleep(0.2)
            for _ in range(self.config.ai_workers):
                while not self.stop_event.is_set():
                    try:
                        ready_queue.put(None, timeout=1.0)
                        break
                    except queue.Full:
                        continue

        def _run_aux(worker) -> None:
            try:
                worker.run(self)
            except Exception as e:
                # worker 崩溃立即兜底其阶段，避免管线挂到 join 超时
                logger.error(f"辅助阶段 {getattr(worker, 'stage_key', 'aux')} 异常退出: {e}",
                             exc_info=True)
                for ctx in self.contexts:
                    ctx.fail_remaining(getattr(worker, "stage_key", None))

        def _run_producer() -> None:
            try:
                producer.run(self.tagger_queue)
            finally:
                self.producer_done.set()  # aux worker 据此判断抽帧收尾

        aux_threads = [threading.Thread(target=_run_aux, args=(w,),
                                        name=f"aux-{getattr(w, 'stage_key', 'aux')}",
                                        daemon=True)
                       for w in self.aux_workers]
        producer_thread = threading.Thread(target=_run_producer, name="producer", daemon=False)
        closer_thread = threading.Thread(target=_queue_closer, name="queue-closer", daemon=True)

        slots = max(0, int(getattr(self.config, "llama_slots", 0) or 0))
        slot_ids = _worker_slot_ids(self.config.ai_workers, slots)
        if slots > 0 and slots < self.config.ai_workers:
            logger.warning(f"[RAG] 槽位数({slots})少于 AI 并发({self.config.ai_workers})，"
                           f"同槽视频将交替覆盖缓存，两轮同槽收益下降")
        workers = [AIWorker(self.client, self.config, stats, ready_queue, self.stop_event,
                            on_file_done=self.on_file_done, nfo_namer=self.nfo_namer,
                            slot_id=slot_ids[i], tag_recall=self.tag_recall)
                   for i in range(self.config.ai_workers)]
        # AI worker 设为 daemon：网络调用失效时可防进程退出挂死
        worker_threads = [threading.Thread(target=w.run, name=f"ai-{i}", daemon=True)
                          for i, w in enumerate(workers)]

        for t in aux_threads:
            t.start()
        producer_thread.start()
        closer_thread.start()
        for t in worker_threads:
            t.start()

        start_time = time.time()
        per_thread_timeout = max(_MIN_JOIN_TIMEOUT_SEC,
            (len(videos) / max(1, self.config.ai_workers)) * ai_max_per_video * _JOIN_SLACK_FACTOR)
        # 所有线程共享一个总 deadline（closer 随交付/停止退出，不参与 hung 判定）
        deadline = time.time() + per_thread_timeout
        join_threads = aux_threads + [producer_thread] + worker_threads
        hung = False
        for t in join_threads:
            remaining = max(0, deadline - time.time())
            t.join(timeout=remaining)
            if t.is_alive():
                hung = True
        if hung:
            logger.error(f"管线线程在 {per_thread_timeout:.0f} 秒内未全部退出，强制停止。")
            self.stop_event.set()
            for t in join_threads:
                if t.is_alive():
                    t.join(timeout=5.0)
        closer_thread.join(timeout=2.0)

        # 辅助阶段兜底：join 后仍未终态的阶段标记终态（防永久等待）
        for ctx in self.contexts:
            ctx.fail_remaining()
        # 取消核对：从未交付的上下文统一计入 cancelled（含辅助阶段卡死兜底）
        for ctx in self.contexts:
            if not ctx.delivered:
                stats.inc_cancelled()
                ctx.release()

        self._start_time = start_time

    def _cleanup(self, stats):
        """移动失败文件 + 输出报告。"""
        move_to_failed(stats.error_files)

        elapsed = time.time() - self._start_time
        print_summary(stats.to_summary(elapsed))
