"""管线共享状态：统计计数与熔断器。"""
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

from .types import RenameRecord, SummaryStats
from .config import Config
from .env import SEP_CHAR, SEP_WIDTH, logger


# ── 熔断器常量 ──
_CIRCUIT_WINDOW_SEC = 120        # 失败时间窗（秒）
_AI_FAILURE_THRESHOLD = 5        # AI 失败熔断阈值
_FRAME_FAILURE_THRESHOLD = 10    # 抽帧失败熔断阈值
_SUCCESS_RESET_STREAK = 3        # 连续成功达到此数则清空对应失败窗口（软复位）

# inc() 允许的统计键
_VALID_STAT_KEYS = frozenset({"ok", "error", "skipped"})


@dataclass
class PipelineStats:
    """生产者-消费者管线共享状态，线程安全。"""
    total: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    ok: int = 0
    error: int = 0
    skipped: int = 0
    cancelled: int = 0  # 停止时已入队但未处理的任务数（用于计数核对）
    results: List[RenameRecord] = field(default_factory=list)
    error_files: List[Tuple[str, str]] = field(default_factory=list)
    done: int = 0

    ai_window: deque = field(default_factory=deque, init=False, repr=False)      # 失败时间戳
    frame_window: deque = field(default_factory=deque, init=False, repr=False)  # 失败时间戳
    ai_tripped: bool = False
    frame_tripped: bool = False
    context_warning_shown: bool = False

    # 熔断窗口大小（秒）和阈值：均根据 config 动态调整
    ai_window_sec: float = _CIRCUIT_WINDOW_SEC
    ai_failure_threshold: int = _AI_FAILURE_THRESHOLD
    frame_window_sec: float = _CIRCUIT_WINDOW_SEC
    frame_failure_threshold: int = _FRAME_FAILURE_THRESHOLD

    # 连续成功计数（用于复位熔断窗口）
    _ai_success_streak: int = field(default=0, init=False, repr=False)
    _frame_success_streak: int = field(default=0, init=False, repr=False)

    def inc(self, key: str):
        with self._lock:
            if key == "ok":
                self.ok += 1
            elif key == "error":
                self.error += 1
            elif key == "skipped":
                self.skipped += 1
            else:
                logger.warning(f"无效的统计键: {key}（有效值: {', '.join(sorted(_VALID_STAT_KEYS))}）")

    def add_result(self, status: str, old_name: str, new_name: str, message: str = ""):
        with self._lock:
            self.results.append(RenameRecord(status, old_name, new_name, message))

    def add_error_file(self, vp: str):
        with self._lock:
            self.error_files.append((vp, Path(vp).name))

    def inc_cancelled(self) -> None:
        """停止时清点出队/队列中未处理的任务。"""
        with self._lock:
            self.cancelled += 1

    def inc_done(self) -> Tuple[int, int]:
        with self._lock:
            self.done += 1
            return self.done, self.total

    def record_failure(self, stop_event: threading.Event, scenario: str = "处理失败",
                       category: str = "ai") -> bool:
        """记录一次失败。基于时间窗口判断最近一段时间内的失败率是否超过阈值。"""
        now = time.time()
        tripped_msg: Optional[str] = None
        with self._lock:
            if category == "frame":
                self._frame_success_streak = 0
                if self.frame_tripped:
                    return True
                window = self.frame_window_sec
                threshold = self.frame_failure_threshold
                while self.frame_window and now - self.frame_window[0] > window:
                    self.frame_window.popleft()
                self.frame_window.append(now)
                failures = len(self.frame_window)
                if failures >= threshold:
                    stop_event.set()
                    self.frame_tripped = True
                    tripped_msg = f"最近 {window:.0f} 秒内抽帧失败 {failures} 次（{scenario}）！已停止处理。"
            else:
                self._ai_success_streak = 0
                if self.ai_tripped:
                    return True
                window = self.ai_window_sec
                threshold = self.ai_failure_threshold
                while self.ai_window and now - self.ai_window[0] > window:
                    self.ai_window.popleft()
                self.ai_window.append(now)
                failures = len(self.ai_window)
                if failures >= threshold:
                    stop_event.set()
                    self.ai_tripped = True
                    tripped_msg = f"最近 {window:.0f} 秒内 AI 任务失败 {failures} 次（{scenario}）！已停止处理。"
        # 日志在锁外输出
        if tripped_msg:
            logger.critical(tripped_msg)
            return True
        return False

    def record_success(self, category: str = "ai"):
        """记录一次成功；连续成功达阈值则软复位对应熔断窗口。"""
        with self._lock:
            if category == "frame":
                self._frame_success_streak += 1
                if self._frame_success_streak >= _SUCCESS_RESET_STREAK:
                    self.frame_window.clear()
                    self.frame_tripped = False
                    self._frame_success_streak = 0
            else:
                self._ai_success_streak += 1
                if self._ai_success_streak >= _SUCCESS_RESET_STREAK:
                    self.ai_window.clear()
                    self.ai_tripped = False
                    self._ai_success_streak = 0

    def show_context_warning(self, config: Config):
        with self._lock:
            if self.context_warning_shown:
                return
            self.context_warning_shown = True
        bar = SEP_CHAR * SEP_WIDTH
        logger.warning(f"\n{bar}")
        logger.warning("检测到上下文窗口溢出错误 (Context Length Exceeded)")
        logger.warning("建议方案：")
        logger.warning(f" 1. 减少采样点位 (当前: {config.sampling_points})")
        logger.warning(f" 2. 减少每点帧数 (当前: {config.frames_per_point})")
        logger.warning(f" 3. 降低抽帧分辨率 (当前长边: {config.frame_max_side})")
        logger.warning(" 4. 检查本地模型配置是否限制了 Context Window")
        logger.warning(f"{bar}\n")

    def to_summary(self, elapsed: float) -> SummaryStats:
        with self._lock:
            return SummaryStats(
                results=list(self.results),
                ok=self.ok, skipped=self.skipped, error=self.error,
                cancelled=self.cancelled,
                elapsed=elapsed,
                # 实际处理数用 done（停止时未处理的任务不计入，避免虚报）
                processed_count=self.done,
            )
