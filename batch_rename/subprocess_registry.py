"""子进程注册表：用于关窗/退出时统一终止 ffmpeg 等子进程。"""
import sys
import time
import threading
import subprocess
from typing import Set

from .job_object import init_job_object, assign_process

_active_subprocesses: Set[subprocess.Popen] = set()
_subprocess_lock = threading.Lock()


def register_subprocess(p: subprocess.Popen) -> None:
    with _subprocess_lock:
        _active_subprocesses.add(p)
    # 挂入作业对象：程序以任意方式退出时由内核自动终止该进程；失败静默降级
    try:
        if init_job_object():
            assign_process(p.pid)
    except Exception:
        pass


def unregister_subprocess(p: subprocess.Popen) -> None:
    with _subprocess_lock:
        _active_subprocesses.discard(p)


def _safe_call(func, *args) -> None:
    try:
        func(*args)
    except Exception as e:
        sys.stderr.write(f"[subprocess_registry] {func.__name__} 失败: {e}\n")


def terminate_all_subprocesses(timeout: float = 2.0) -> None:
    """终止所有活跃子进程。只清除已终止的条目，避免抹掉并发新注册的进程。"""
    with _subprocess_lock:
        procs = list(_active_subprocesses)

    for p in procs:
        if p.poll() is None:
            _safe_call(p.terminate)

    deadline = time.time() + timeout
    for p in procs:
        if p.poll() is None:
            try:
                p.wait(timeout=max(0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                _safe_call(p.kill)
            except Exception:
                pass

    # 只移除已处理的进程，不清空整个集合
    with _subprocess_lock:
        for p in procs:
            _active_subprocesses.discard(p)
