"""installer.py — 扩展模块共享安装基础设施（UV + 镜像 + GPU 检测 + 子进程工具）。

"""
from __future__ import annotations

import collections
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .env import APP_ROOT, PYTHON_EXE, make_logger
from batch_rename.env import SUBPROCESS_KWARGS

from batch_rename.subprocess_registry import register_subprocess, unregister_subprocess

# ── UV 包管理工具路径 ──
UV_RUNTIME_DIR = APP_ROOT / "UV-Tool"      # UV 二进制 + 包缓存
UV_CACHE_DIR = UV_RUNTIME_DIR / "cache"    # UV 下载缓存（UV_CACHE_DIR 指向这里）

# ── 通用 PyPI 镜像（timm/Pillow/faster-whisper 等）──
PYPI_MIRRORS = {
    "nju": {"name": "南京大学", "url": "https://mirrors.nju.edu.cn/pypi/web/simple"},
    "tsinghua": {"name": "清华大学", "url": "https://pypi.tuna.tsinghua.edu.cn/simple"},
    "aliyun": {"name": "阿里云", "url": "https://mirrors.aliyun.com/pypi/simple"},
}
DEFAULT_PYPI_MIRROR = "nju"

# ── PyTorch CUDA wheels 专用镜像（仅 PixAI 需要）──
PYTORCH_MIRRORS = {
    "nju-cu128": {"name": "南京大学 (CUDA 12.8)", "url": "https://mirrors.nju.edu.cn/pytorch/whl/cu128"},
    "nju-cu121": {"name": "南京大学 (CUDA 12.1)", "url": "https://mirrors.nju.edu.cn/pytorch/whl/cu121"},
    "nju-cu118": {"name": "南京大学 (CUDA 11.8)", "url": "https://mirrors.nju.edu.cn/pytorch/whl/cu118"},
    "nju-cpu": {"name": "南京大学 (CPU)", "url": "https://mirrors.nju.edu.cn/pytorch/whl/cpu"},
    "official-cu128": {"name": "PyTorch 官方 (CUDA 12.8)", "url": "https://download.pytorch.org/whl/cu128"},
}
DEFAULT_PYTORCH_MIRROR = "nju-cu128"

# ── 超时 ──
UV_TIMEOUT_SEC = 120.0

# run_venv_script 返回值约定
RC_OK = 0
RC_TIMEOUT = -1
RC_CANCEL = -2


# 输出日志到回调（前端推送）或 stderr
log = make_logger("installer")


# ── 镜像 / GPU 检测 ──

def detect_gpu() -> Dict[str, Any]:
    """检测 NVIDIA GPU，返回型号/驱动版本/驱动支持的最高 CUDA 版本/推荐的 torch 镜像 ID。"""
    result = {
        "has_nvidia": False,
        "gpu_name": "",
        "driver_version": "",
        "cuda_max": "",        # 驱动支持的最高 CUDA 版本（如 "12.8"），来自 nvidia-smi 表头
        "recommended": "cpu",
    }
    try:
        p = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            **SUBPROCESS_KWARGS,
        )
        out, _ = p.communicate(timeout=5)
        if p.returncode == 0 and out.strip():
            line = out.strip().splitlines()[0]
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                result["has_nvidia"] = True
                result["gpu_name"] = parts[0]
                result["driver_version"] = parts[1]
                result["cuda_max"] = _query_cuda_version()
                result["recommended"] = _recommend_cuda(result["cuda_max"], result["driver_version"])
    except Exception:
        pass
    return result


def _query_cuda_version() -> str:
    """从 nvidia-smi 表头解析驱动支持的最高 CUDA 版本（如 "12.8"）。失败返回空串。"""
    try:
        p = subprocess.Popen(
            ["nvidia-smi"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            **SUBPROCESS_KWARGS,
        )
        out, _ = p.communicate(timeout=5)
        if p.returncode == 0:
            # 新驱动(R580+)表头为 CUDA UMD Version: 13.3；旧版英文 CUDA Version: 13.3；中文 Windows CUDA 版本: 13.3
            patterns = [
                r"CUDA\s+UMD\s+Version\s*:?\s*([\d.]+)",
                r"CUDA\s+Version\s*:?\s*([\d.]+)",
                r"CUDA\s*版本\s*:?\s*([\d.]+)",
            ]
            for _pat in patterns:
                m = re.search(_pat, out)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return ""


def _recommend_cuda(cuda_max: str, driver_version: str = "") -> str:
    """根据驱动支持的最高 CUDA 版本推荐 PyTorch CUDA 镜像 ID。"""
    def _by_ceiling(ver: str) -> str:
        try:
            parts = ver.split(".")
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            return ""
        if (major, minor) >= (12, 8):
            return "nju-cu128"
        if (major, minor) >= (12, 1):
            return "nju-cu121"
        if (major, minor) >= (11, 8):
            return "nju-cu118"
        return "cpu"

    def _by_driver(ver: str) -> str:
        try:
            major = int(ver.split(".")[0])
        except (ValueError, IndexError):
            return "cpu"
        if major >= 570:
            return "nju-cu128"
        if major >= 525:
            return "nju-cu121"
        if major >= 452:
            return "nju-cu118"
        return "cpu"

    if cuda_max:
        r = _by_ceiling(cuda_max)
        if r:
            return r
    if driver_version:
        return _by_driver(driver_version)
    return "cpu"


def get_mirror_groups(groups: List[str]) -> Dict[str, Any]:
    """返回指定镜像分组 + GPU 检测（供前端动态渲染选择弹窗）。
        Args:
        groups: 需要的镜像组，如 ["pytorch", "pypi"]（PixAI）或 ["pypi"]（Whisper）
    """
    result: Dict[str, Any] = {}
    if "pypi" in groups:
        result["pypi"] = [{"id": k, "name": v["name"], "url": v["url"]} for k, v in PYPI_MIRRORS.items()]
        result["default_pypi"] = DEFAULT_PYPI_MIRROR
    if "pytorch" in groups:
        result["pytorch"] = [{"id": k, "name": v["name"], "url": v["url"]} for k, v in PYTORCH_MIRRORS.items()]
        result["default_pytorch"] = DEFAULT_PYTORCH_MIRROR
        result["gpu"] = detect_gpu()  # 仅 PyTorch 安装需要 GPU 检测
    return result


# ── 子进程工具 ──

def _subprocess_env() -> Dict[str, str]:
    """返回注入了 UV_CACHE_DIR 的环境变量（让 UV 缓存落到 UV-Tool 目录）。"""
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(UV_CACHE_DIR)
    return env


def _terminate_proc(p: subprocess.Popen, wait_sec: float = 3.0) -> None:
    """terminate → 短等 → kill 的统一收尾。"""
    try:
        p.terminate()
        p.wait(timeout=wait_sec)
    except subprocess.TimeoutExpired:
        try:
            p.kill()
        except Exception:
            pass
    except Exception:
        pass


def start_cancel_watcher(cancel_event: threading.Event,
                         closer: Callable[[], None]) -> Callable[[], None]:
    """启动取消监视线程并返回停止回调（finally 中调用）。"""
    quit_ev = threading.Event()

    def _watch() -> None:
        while not quit_ev.wait(0.1):
            if cancel_event.is_set():
                try:
                    closer()
                except Exception:
                    pass
                return

    threading.Thread(target=_watch, name="dl-cancel-watcher", daemon=True).start()
    return quit_ev.set


def run_subprocess(cmd: list, timeout: float) -> Tuple[int, str, str]:
    """运行子进程并返回 (returncode, stdout, stderr)。超时/异常返回 -1。"""
    p = None
    try:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            env=_subprocess_env(), **SUBPROCESS_KWARGS,
        )
        register_subprocess(p)
        stdout, stderr = p.communicate(timeout=timeout)
        return p.returncode, stdout or "", stderr or ""
    except subprocess.TimeoutExpired:
        if p is not None:
            _terminate_proc(p)
        return -1, "", "超时"
    except Exception as e:
        return -1, "", str(e)
    finally:
        if p is not None:
            unregister_subprocess(p)


def run_subprocess_streaming(
    cmd: list,
    timeout: float,
    log_fn: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Tuple[int, str]:
    """运行子进程，实时逐行推送输出（stderr 并入）到 log_fn。返回 (returncode, 全部stdout)。"""
    try:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=_subprocess_env(), **SUBPROCESS_KWARGS,
        )
    except Exception as e:
        log(f"启动失败: {e}", log_fn)
        return -1, ""

    register_subprocess(p)
    q: "queue.Queue[Optional[str]]" = queue.Queue()

    def _reader():
        assert p.stdout is not None
        try:
            for line in p.stdout:
                q.put(line)
        except Exception:
            pass
        q.put(None)

    threading.Thread(target=_reader, name="proc-stream-reader", daemon=True).start()

    lines: List[str] = []
    deadline = time.time() + timeout
    rc: Optional[int] = None
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                _terminate_proc(p, wait_sec=1.0)
                log("已取消", log_fn)
                rc = RC_CANCEL
                break
            if time.time() > deadline:
                _terminate_proc(p)
                log("操作超时", log_fn)
                rc = RC_TIMEOUT
                break
            try:
                line = q.get(timeout=0.25)
            except queue.Empty:
                continue
            if line is None:
                break
            lines.append(line)
            if log_fn and line.strip():
                log_fn(line.rstrip())
        if rc is None:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _terminate_proc(p)
            rc = p.returncode if p.returncode is not None else -1
        return rc, "".join(lines)
    finally:
        unregister_subprocess(p)


def venv_python_path(venv_dir: Path) -> Path:
    """返回指定虚拟环境中的 python 可执行文件路径。"""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run_venv_script(
    venv_dir: Path,
    script: str,
    params: Dict[str, Any],
    *,
    timeout: float,
    stop_event: Optional[threading.Event] = None,
    on_line: Optional[Callable[[str], None]] = None,
) -> Tuple[int, str, str]:
    """在模块 venv 内执行内联脚本，返回 (returncode, stdout最后一行, stderr尾部)。"""
    py = venv_python_path(venv_dir)
    if not py.is_file():
        return -1, "", f"模块环境不存在: {py}"

    fd, params_path = tempfile.mkstemp(suffix=".json", prefix="modjob_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False)
    except Exception as e:
        try:
            os.unlink(params_path)
        except OSError:
            pass
        return -1, "", f"参数文件写入失败: {e}"

    try:
        p = subprocess.Popen(
            [str(py), "-c", script, params_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            **SUBPROCESS_KWARGS,
        )
    except Exception as e:
        _unlink_quiet(params_path)
        return -1, "", f"启动失败: {e}"

    register_subprocess(p)
    out_q: "queue.Queue[Optional[str]]" = queue.Queue()
    err_buf: "collections.deque[str]" = collections.deque(maxlen=200)

    def _drain_stdout():
        assert p.stdout is not None
        try:
            for line in p.stdout:
                out_q.put(line)
        except Exception:
            pass
        out_q.put(None)

    def _drain_stderr():
        assert p.stderr is not None
        try:
            for line in p.stderr:
                err_buf.append(line.rstrip())
        except Exception:
            pass

    threading.Thread(target=_drain_stdout, name="venv-out-reader", daemon=True).start()
    threading.Thread(target=_drain_stderr, name="venv-err-reader", daemon=True).start()

    last_line = ""
    deadline = time.time() + timeout
    rc: Optional[int] = None
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                _terminate_proc(p, wait_sec=1.0)
                rc = RC_CANCEL
                break
            if time.time() > deadline:
                _terminate_proc(p)
                rc = RC_TIMEOUT
                break
            try:
                line = out_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if line is None:
                break
            last_line = line.rstrip("\n")
            if on_line is not None:
                try:
                    on_line(last_line)
                except Exception:
                    pass  # 回调异常不影响子进程收尾
        if rc is None:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _terminate_proc(p)
            rc = p.returncode if p.returncode is not None else -1
        return rc, last_line, "\n".join(list(err_buf)[-20:])
    finally:
        unregister_subprocess(p)
        _unlink_quiet(params_path)


def _unlink_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# ── UV 管理 ──

def find_uv() -> Optional[str]:
    """查找 UV 可执行文件：UV-Tool/ → 系统 PATH。"""
    ext = ".exe" if sys.platform == "win32" else ""
    uv_shared = UV_RUNTIME_DIR / f"uv{ext}"
    if uv_shared.exists():
        return str(uv_shared)
    return shutil.which("uv")


def ensure_uv(pypi_url: str, log_fn: Optional[Callable[[str], None]] = None,
              stop_event: Optional[threading.Event] = None) -> Optional[str]:
    """确保 UV 可用，不存在则安装到 UV-Tool 目录。"""
    uv = find_uv()
    if uv:
        return uv
    log("UV 未找到，正在下载…", log_fn)
    ext = ".exe" if sys.platform == "win32" else ""
    UV_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp_target = UV_RUNTIME_DIR / "_uv_tmp"
    rc, out = run_subprocess_streaming(
        [PYTHON_EXE, "-m", "pip", "install", "uv", "-i", pypi_url,
         "--target", str(tmp_target), "--quiet"],
        UV_TIMEOUT_SEC, log_fn, stop_event,
    )
    if stop_event is not None and stop_event.is_set():
        shutil.rmtree(tmp_target, ignore_errors=True)
        return None
    if rc != 0:
        log(f"pip 安装 UV 失败: {out[:200]}", log_fn)
        shutil.rmtree(tmp_target, ignore_errors=True)
        return None
    src = tmp_target / "Scripts" / f"uv{ext}"
    if not src.exists():
        src = tmp_target / "bin" / f"uv{ext}"
    if not src.exists():
        src = tmp_target / f"uv{ext}"
    if src.exists():
        dest = UV_RUNTIME_DIR / f"uv{ext}"
        shutil.copy2(src, dest)
        shutil.rmtree(tmp_target, ignore_errors=True)
        return str(dest)
    shutil.rmtree(tmp_target, ignore_errors=True)
    log("UV 安装后未找到可执行文件", log_fn)
    return None


# ── 缓存管理 ──

def clean_uv_cache(log_fn: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """清理 UV 包缓存，释放磁盘空间。"""
    if not UV_CACHE_DIR.is_dir():
        return {"ok": True, "message": "缓存目录不存在，无需清理"}
    uv = find_uv()
    cleaned = False
    if uv:
        rc, _, _ = run_subprocess([uv, "cache", "clean"], UV_TIMEOUT_SEC)
        cleaned = (rc == 0)
    if not cleaned:
        # UV 不可用或命令失败：直接删除缓存目录
        shutil.rmtree(UV_CACHE_DIR, ignore_errors=True)
        UV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log("已清理 UV 包缓存", log_fn)
    return {"ok": True, "message": "已清理 UV 包缓存"}


def uninstall_uv(log_fn: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """卸载 UV：删除整个 UV-Tool/（含缓存）。"""
    if not UV_RUNTIME_DIR.exists():
        return {"ok": True, "message": "UV-Tool 不存在，无需卸载"}
    try:
        shutil.rmtree(UV_RUNTIME_DIR)
        log("已卸载 UV（UV-Tool 已删除）", log_fn)
        return {"ok": True, "message": "已卸载 UV 并清理缓存"}
    except Exception as e:
        return {"ok": False, "error": f"卸载失败: {e}"}


def get_uv_status() -> Dict[str, Any]:
    """获取 UV 包管理工具状态（是否安装、是否位于 UV-Tool 目录）。"""
    uv = find_uv()
    in_runtime = False
    if uv:
        try:
            in_runtime = Path(uv).resolve().is_relative_to(UV_RUNTIME_DIR.resolve())
        except ValueError:
            in_runtime = False
    return {
        "installed": uv is not None,
        "in_runtime": in_runtime,
        "uv_path": uv or "",
        "runtime_dir": str(UV_RUNTIME_DIR),
    }
