"""js_push.py — JS 推送队列。"""
from __future__ import annotations

import json
import sys
import threading
import time
from typing import List

from .constants import PUSH_QUEUE_MAX

# flush 连续失败达到该阈值后暂停推送一段时间（WebView 疑似断开，避免无效重试刷屏）
_FAIL_PAUSE_THRESHOLD = 50
_FAIL_PAUSE_SEC = 5.0


class JSPushQueue:
    """线程安全的 JS 推送队列。"""

    def __init__(self):
        self._queue: List[str] = []
        self._lock = threading.Lock()
        # 兜底串行化 evaluate_js：即使未来出现第二个 flush 调用方也不会并发
        self._eval_lock = threading.Lock()
        self._window = None
        self._consecutive_failures = 0
        self._paused_until = 0.0

    def set_window(self, window) -> None:
        self._window = window

    def push(self, method: str, *args) -> None:
        """构造 window.__ui.method(arg1, arg2, ...) 推入队列。"""
        try:
            js_args = ", ".join(json.dumps(a, ensure_ascii=False) for a in args)
            js = f"window.__ui.{method}({js_args});"
        except Exception as e:
            sys.stderr.write(f"[js_push] push 序列化失败 ({method}): {e}\n")
            return
        with self._lock:
            self._queue.append(js)
            # 防止积压过多
            if len(self._queue) > PUSH_QUEUE_MAX:
                self._queue = self._queue[-PUSH_QUEUE_MAX:]

    def flush(self) -> None:
        """把队列中的 JS 全部 evaluate。应由单一 flush 线程周期调用。"""
        if self._window is None:
            return
        if time.monotonic() < self._paused_until:
            # 暂停期内丢弃推送，避免向疑似断开的 WebView 无效重试
            with self._lock:
                self._queue.clear()
            return
        with self._lock:
            queue = self._queue[:]
            self._queue.clear()
        if not queue:
            return
        failed = 0
        first_err = None
        with self._eval_lock:
            for js in queue:
                try:
                    self._window.evaluate_js(js)
                except Exception as e:
                    failed += 1
                    if first_err is None:
                        first_err = e
        if not failed:
            self._consecutive_failures = 0
            return
        self._consecutive_failures += failed
        sys.stderr.write(
            f"[js_push] {failed}/{len(queue)} 条 JS 推送失败: {first_err}\n")
        if self._consecutive_failures >= _FAIL_PAUSE_THRESHOLD:
            sys.stderr.write(
                f"[js_push] 连续失败 {self._consecutive_failures} 条，"
                f"WebView 可能已断开，暂停推送 {_FAIL_PAUSE_SEC:.0f}s\n")
            self._consecutive_failures = 0
            self._paused_until = time.monotonic() + _FAIL_PAUSE_SEC


js_pusher = JSPushQueue()
