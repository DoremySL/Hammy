"""管线编排：抽帧生产者、AI 消费者、批处理编排器。"""
import time
import queue
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
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
from .failed import move_to_failed

_TIMEOUT_GRACE_SEC = 10  # AI 超时后的额外宽限（网络延迟/进程清理）

# ── 队列与线程池缓冲系数 ──
_FRAME_WORKERS = 4
_FRAME_PENDING_FACTOR = 4
_TASK_QUEUE_FACTOR = 4
# ── join 超时兜底 ──
_MIN_JOIN_TIMEOUT_SEC = 120.0
_JOIN_SLACK_FACTOR = 1.5


def _max_ai_seconds_per_video(config: Config) -> float:
    """单视频 AI 处理的最坏耗时估算。"""
    return (config.ai_timeout + _TIMEOUT_GRACE_SEC) * (config.retry_times + 1)


# 每文件完成回调签名: (original_path, new_path, status, info, title, plot, tags)
# status ∈ {"ok","skipped","error","frame_error"}；new_path 为改名后路径（失败时为原路径）
OnFileDone = Callable[[str, Optional[str], str, Dict[str, Any], str, str, List[str]], None]


# ── 抽帧生产者 ────────────────────────────────────────
class FrameProducer:
    """抽帧生产者：ThreadPoolExecutor 并发抽帧，结果入任务队列。"""

    def __init__(self, videos: List[str], task_queue: queue.Queue,
                 config: Config, stop_event: threading.Event):
        self.videos = videos
        self.task_queue = task_queue
        self.config = config
        self.stop_event = stop_event

    def _safe_put(self, item, max_wait: Optional[float] = None) -> bool:
        if max_wait is None:
            max_wait = _max_ai_seconds_per_video(self.config)
        deadline = time.time() + max_wait
        while not self.stop_event.is_set():
            try:
                self.task_queue.put(item, timeout=1.0)
                return True
            except queue.Full:
                if time.time() > deadline:
                    logger.error(f"任务队列阻塞超过 {max_wait:.0f} 秒，强制停止管线。")
                    self.stop_event.set()
                    return False
                self.stop_event.wait(0.1)
        return False

    def _handle_frame_result(self, future: Future, vp: Optional[str]):
        if vp is None or self.stop_event.is_set():
            return
        try:
            info, frames = future.result()
            if not frames:
                self._safe_put((vp, None, None))
            else:
                self._safe_put((vp, info, frames))
        except Exception as e:
            logger.error(f"抽帧线程异常: {e}")
            self._safe_put((vp, None, None))

    def run(self):
        try:
            frame_workers = min(_FRAME_WORKERS, len(self.videos))
            max_pending = frame_workers * _FRAME_PENDING_FACTOR
            with ThreadPoolExecutor(max_workers=frame_workers, thread_name_prefix="frame") as frame_exec:
                futures: Dict[Future, str] = {}
                for vp in self.videos:
                    if self.stop_event.is_set():
                        break
                    future = frame_exec.submit(probe_and_extract_keyframes, vp, self.config, self.stop_event)
                    futures[future] = vp
                    if len(futures) >= max_pending:
                        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                        done_list = list(done)
                        for f in done_list:
                            self._handle_frame_result(f, futures.pop(f, None))
                if futures:
                    if not self.stop_event.is_set():
                        for future in as_completed(list(futures.keys())):
                            if self.stop_event.is_set():
                                for f in list(futures):
                                    f.cancel()
                                break
                            self._handle_frame_result(future, futures.pop(future, None))
                        futures.clear()
                    else:
                        # f.cancel() 只取消未开始任务，运行中的抽帧靠内部
                        # 轮询 stop_event 自行退出，join 才不会挂死
                        for f in futures:
                            f.cancel()
        except Exception as e:
            logger.error(f"生产者线程发生致命异常: {e}", exc_info=True)
        finally:
            for _ in range(self.config.ai_workers):
                if not self._safe_put(None):
                    self.stop_event.set()
                    break


# ── AI 任务处理器 ─────────────────────────────────────
class AIWorker:
    """AI 任务消费者：出队 → 调 AI → 重命名 → 更新统计。"""

    def __init__(self, client: OpenAIClient, config: Config, stats: PipelineStats,
                 task_queue: queue.Queue, stop_event: threading.Event,
                 on_file_done: Optional[OnFileDone] = None,
                 nfo_namer: Optional[Callable[[str], str]] = None,
                 extra_meta_map: Optional[Dict[str, str]] = None):
        self.client = client
        self.config = config
        self.stats = stats
        self.task_queue = task_queue
        self.stop_event = stop_event
        self._on_file_done = on_file_done
        # 可选 NFO 目标命名函数（缓存模式防同 stem 冲突）
        self._nfo_namer = nfo_namer
        # 每个视频的额外上下文（扩展功能注入，key=视频路径）
        self.extra_meta_map = extra_meta_map or {}

    def _move_failed_hint(self) -> str:
        return "，稍后将移至 _failed"

    def _emit_file_done(self, vp: str, new_path, status: str, info: dict,
                        title: str = "", plot: str = "", tags: list = None) -> None:
        """安全触发 on_file_done 回调，异常不影响管线。"""
        if self._on_file_done is None:
            return
        try:
            self._on_file_done(vp, new_path, status, info, title, plot, tags or [])
        except Exception as e:
            logger.debug(f"on_file_done 回调异常: {e}")

    def _handle_success(self, vp: str, info: Dict[str, Any], title: str,
                        plot: str, tags: List[str], retries: int) -> str:
        name = Path(vp).name
        final_stem = build_new_stem(vp, info, title, self.config)
        new_path, status = rename_video(vp, final_stem, self.config)

        if status in ("ok", "skipped"):
            # 缓存模式：唯一命名防同 stem 覆盖
            nfo_name = self._nfo_namer(vp) if self._nfo_namer else None
            write_nfo(video_path=new_path, title=title, plot=plot,
                       tags=tags, info=info, original_name=name,
                       target_dir=self.config.nfo_target_dir,
                       nfo_name=nfo_name)

        self.stats.inc(status)
        if status == "error":
            self.stats.add_error_file(vp)
        self.stats.add_result(status, name, Path(new_path).name)
        self.stats.record_success(category="ai")

        # 每文件完成回调（GUI 用于写 history / 精确归类）
        self._emit_file_done(vp, new_path, status, info, title, plot, tags)

        if status == "ok":
            return f"{name} -> {Path(new_path).name}{retry_hint(retries, success=True)}"
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

    def _process_task(self, task):
        vp, info, frames = task

        # 停止时已出队的任务不处理，计入 cancelled（保证计数可核对）
        if self.stop_event.is_set():
            self.stats.inc_cancelled()
            return

        # 抽帧失败分发
        if frames is None:
            display_msg = ""
            if vp:
                display_msg = self._handle_frame_error(vp)
            current, total = self.stats.inc_done()
            if display_msg:
                logger.info(f"{SEP_INDENT}[{current}/{total}] {display_msg}")
            return

        self.stats.record_success(category="frame")

        # AI 调用 → 成功/失败分发
        display_msg = ""
        try:
            result = analyze_frames(
                self.client, self.config.model, frames, self.config,
                self.stop_event, vp, info.get("duration", 0.0),
                container_title=info.get("container_title", ""),
                extra_meta=self.extra_meta_map.get(vp, ""),
            )
            if result.title:
                display_msg = self._handle_success(vp, info, result.title, result.plot,
                                                  result.tags, result.retries)
            else:
                display_msg = self._handle_failure(vp, result.retries,
                                                   result.err_msg, result.error_kind)
        except Exception as e:
            display_msg = self._handle_unexpected_error(vp, e)
        finally:
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
            # 提前停止后排空队列剩余任务，计入 cancelled；
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


# ── 管线编排 ──────────────────────────────────────────
class BatchPipeline:
    """批量重命名管线编排器：组装各组件并执行。"""

    def __init__(self, paths: List[str], config: Config, client: OpenAIClient,
                 stop_event: threading.Event,
                 on_file_done: Optional[OnFileDone] = None,
                 nfo_namer: Optional[Callable[[str], str]] = None,
                 extra_meta_map: Optional[Dict[str, str]] = None):
        self.paths = paths
        self.config = config
        self.client = client
        self.stop_event = stop_event
        self.on_file_done = on_file_done
        self.nfo_namer = nfo_namer
        # 每个视频的额外上下文（扩展功能注入，key=视频路径）
        self.extra_meta_map = extra_meta_map or {}
        # 暴露 stats 给外部（GUI 完成后读取真实计数）
        self.stats: Optional[PipelineStats] = None
        self._videos: List[str] = []
        self._start_time: float = 0.0

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
        """发现视频，返回 PipelineStats；无可处理视频时返回 None。"""
        videos = VideoCollector.collect(self.paths)
        if not videos:
            logger.info(f"\n{SEP_INDENT}未找到可处理的视频文件")
            return None

        n = len(videos)
        stats = PipelineStats(total=n)

        # 熔断阈值：下限防偶发失败误熔断，上限防大并发迟迟不熔断
        stats.ai_failure_threshold = max(2, min(self.config.ai_workers + 1, _AI_FAILURE_THRESHOLD))

        # 失败窗口 = 连续失败所需轮数 × 每轮耗时，下限取熔断器基础窗口
        cycles_needed = (stats.ai_failure_threshold + self.config.ai_workers - 1) // self.config.ai_workers
        per_failure = _max_ai_seconds_per_video(self.config)
        stats.ai_window_sec = max(float(_CIRCUIT_WINDOW_SEC), cycles_needed * per_failure)

        logger.info(f"{SEP_INDENT}正在处理：{n}个视频")
        logger.info(f"{SEP_INDENT}正在提取关键帧发送AI分析，请耐心等待\n")

        self._videos = videos
        return stats

    def _run_workers(self, stats):
        """启动生产者 + 消费者线程，等待完成或超时。"""
        videos = self._videos
        task_queue: queue.Queue = queue.Queue(maxsize=self.config.ai_workers * _TASK_QUEUE_FACTOR)
        producer = FrameProducer(videos, task_queue, self.config, self.stop_event)
        producer_thread = threading.Thread(target=producer.run, name="producer", daemon=False)
        producer_thread.start()

        workers = [AIWorker(self.client, self.config, stats, task_queue, self.stop_event,
                            on_file_done=self.on_file_done, nfo_namer=self.nfo_namer,
                            extra_meta_map=self.extra_meta_map)
                   for _ in range(self.config.ai_workers)]
        # AI worker 设为 daemon：网络调用失效时可防进程退出挂死
        worker_threads = [threading.Thread(target=w.run, name=f"ai-{i}", daemon=True)
                          for i, w in enumerate(workers)]
        all_threads = [producer_thread] + worker_threads
        for t in worker_threads:
            t.start()

        start_time = time.time()
        ai_max_per_video = _max_ai_seconds_per_video(self.config)
        per_thread_timeout = max(_MIN_JOIN_TIMEOUT_SEC,
            (len(videos) / max(1, self.config.ai_workers)) * ai_max_per_video * _JOIN_SLACK_FACTOR)
        # 所有线程共享一个总 deadline
        deadline = time.time() + per_thread_timeout
        hung = False
        for t in all_threads:
            remaining = max(0, deadline - time.time())
            t.join(timeout=remaining)
            if t.is_alive():
                hung = True
        if hung:
            logger.error(f"管线线程在 {per_thread_timeout:.0f} 秒内未全部退出，强制停止。")
            self.stop_event.set()
            for t in all_threads:
                if t.is_alive():
                    t.join(timeout=5.0)

        self._start_time = start_time

    def _cleanup(self, stats):
        """移动失败文件 + 输出报告。"""
        move_to_failed(stats.error_files)

        elapsed = time.time() - self._start_time
        print_summary(stats.to_summary(elapsed))
