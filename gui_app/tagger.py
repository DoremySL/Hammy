"""Hammy Tagger：安装、模型管理、抽帧复用与推理子进程管理。

推理/引擎构建逻辑在模型仓库（REPO_ID）内：hammy_server.py 流式分析服务、
build_trt.py 引擎构建（构建后自动删除 ONNX 节省磁盘）。
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from batch_rename.subprocess_registry import register_subprocess, unregister_subprocess
from batch_rename.env import SUBPROCESS_KWARGS
from batch_rename.config import Config
from batch_rename.utils import fmt_size
from batch_rename.video import probe_and_extract_keyframes
from .env import APP_ROOT, PYTHON_EXE, make_logger
from . import installer
from .installer import (
    DEFAULT_TRT_SITE,
    ensure_uv as _ensure_uv,
    run_subprocess_streaming as _run_subprocess_streaming,
    InstallSteps,
    UV_TIMEOUT_SEC,
    venv_python_path,
    _terminate_proc,
)

# ── 路径常量 ──
TAGGER_DIR = APP_ROOT / "Tagger"
VENV_DIR = TAGGER_DIR / "venv"
MODEL_DIR = TAGGER_DIR / "models"  # 模型根目录（魔搭 作者/仓库 布局）

REPO_ID = "DoremySL/Hammy-tagger-dual"  # 双模型仓库：H 504×336 横 / V 336×504 竖

TAG_THRESHOLD = 0.75
TAG_ZH_FILE = "tag_zh.json"  # 标签中文翻译表

# 仓库文件：ONNX 仅安装期需要（引擎构建后自动删除），共享文件长期保留
_ONNX_FILES = {
    "fp16": ("fp16-onnx/tagger-H.onnx", "fp16-onnx/tagger-V.onnx"),
    "fp32": ("fp32-onnx/tagger-H.onnx", "fp32-onnx/tagger-H.onnx.data",
             "fp32-onnx/tagger-V.onnx", "fp32-onnx/tagger-V.onnx.data"),
}
SHARED_FILES = ("config.json", "char_ip_map.json", "tag_zh.json", "manifest.json",
                "hammy_tagger.py", "hammy_server.py", "build_trt.py")
_CRITICAL_SHARED = ("config.json", "char_ip_map.json", "hammy_tagger.py", "hammy_server.py")
_ENGINES = ("tagger-H.trt", "tagger-V.trt")

# 子进程超时
_INSTALL_TIMEOUT_SEC = 2400.0
_BUILD_TIMEOUT_SEC = 1800.0
_INFERENCE_BASE_SEC = 240.0
_INFERENCE_PER_VIDEO_SEC = 120.0

# 抽帧滤镜：短边压到 400（只缩不放）
FRAME_VF = ("scale=trunc(iw*min(1\\,400/min(iw\\,ih))/2)*2"
            ":trunc(ih*min(1\\,400/min(iw\\,ih))/2)*2")


_log = make_logger("tagger")


def _repo_dir() -> Path:
    """模型仓库本地目录。"""
    return MODEL_DIR / REPO_ID


def _engines_dir() -> Path:
    return _repo_dir() / "engines"


def _engine_paths() -> Tuple[Path, ...]:
    return tuple(_engines_dir() / n for n in _ENGINES)


def _server_script() -> Path:
    return _repo_dir() / "hammy_server.py"


def get_mirrors_info() -> Dict[str, Any]:
    """返回 tagger 安装弹窗数据（站点/GPU 检测）。"""
    gpu = installer.detect_gpu()
    return {
        "sites": [{"id": k, "name": v["name"], "desc": installer.mirror_host(v["pypi"])}
                  for k, v in installer.TRT_SITES.items()],
        "default_site": installer.DEFAULT_TRT_SITE,
        "gpu": gpu,
        "gpu_issue": installer.trt_gpu_issue(gpu),
    }


def resolve_precision(setting: str = "auto", gpu: Optional[Dict[str, Any]] = None) -> str:
    """把精度设置解析为实际引擎精度（无 Tensor Core 的卡用 fp32）。"""
    setting = (setting or "auto").lower()
    if setting in ("fp16", "fp32"):
        return setting
    if gpu is None:
        gpu = installer.detect_gpu_cached()
    if installer.gpu_has_tensor_core(gpu.get("gpu_name") or ""):
        return "fp16"
    return "fp32"


_zh_cache: Dict[str, Any] = {"mtime": None, "map": {}}


def load_tag_zh() -> Dict[str, str]:
    """标签中文翻译表（danbooru 标签 → 中文名）。"""
    global _zh_cache
    path = _repo_dir() / TAG_ZH_FILE
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _zh_cache = {"mtime": None, "map": {}}
        return {}
    if _zh_cache["mtime"] != mtime:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            m = data.get("map") if isinstance(data, dict) else None
            zh = {str(k): str(v) for k, v in m.items() if v} if isinstance(m, dict) else {}
        except Exception:
            zh = {}
        _zh_cache = {"mtime": mtime, "map": zh}
    return _zh_cache["map"]


def get_status() -> Dict[str, Any]:
    """获取 Hammy-tagger 安装状态（精度读安装时写入的配置值）。"""
    from .config_store import load_tagger_config
    venv_python = venv_python_path(VENV_DIR)
    venv_ok = VENV_DIR.is_dir() and venv_python.exists()
    precision = load_tagger_config().get("precision") or ""
    ready = (venv_ok
             and all((_repo_dir() / n).is_file() for n in _CRITICAL_SHARED)
             and all(p.is_file() for p in _engine_paths()))
    return {
        "dir_exists": TAGGER_DIR.is_dir(),
        "precision": precision,
        "ready": ready,
    }


def install_dependencies(
    site: str = DEFAULT_TRT_SITE,
    log_fn: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """安装全部依赖（uv + venv + 推理依赖 + 模型下载 + TRT 引擎构建），返回 {"ok", "error"}。"""
    plan = installer.resolve_trt_site(site)
    steps = InstallSteps(5)

    TAGGER_DIR.mkdir(parents=True, exist_ok=True)

    def _cancelled() -> Optional[Dict[str, Any]]:
        if stop_event is not None and stop_event.is_set():
            _log("安装已取消", log_fn)
            return {"ok": False, "cancelled": True, "error": "安装已取消"}
        return None

    # ── 步骤 1：显卡预检 ──
    steps.push(1)
    _log("━━ 步骤 1/5：检测显卡与驱动 ━━", log_fn)
    gpu = installer.detect_gpu()
    issue = installer.trt_gpu_issue(gpu)
    if issue:
        return {"ok": False, "error": issue}
    precision = resolve_precision("auto", gpu)
    _log(f"GPU: {gpu.get('gpu_name')}（驱动 {gpu.get('driver_version')}）"
         f" → 引擎精度 {precision.upper()}", log_fn)

    # ── 步骤 2 ──
    steps.push(2)
    _log("━━ 步骤 2/5：创建虚拟环境 ━━", log_fn)
    uv = _ensure_uv(plan["pypi_url"], log_fn, stop_event)
    r = _cancelled()
    if r:
        return r
    if not uv:
        return {"ok": False, "error": "UV 安装失败"}
    if not venv_python_path(VENV_DIR).exists():
        rc, out = _run_subprocess_streaming(
            [uv, "venv", str(VENV_DIR), "--python", PYTHON_EXE],
            UV_TIMEOUT_SEC, log_fn, stop_event,
        )
        r = _cancelled()
        if r:
            return r
        if rc != 0:
            return {"ok": False, "error": f"创建虚拟环境失败: {out[-300:]}"}
    _log("虚拟环境就绪", log_fn)
    venv_py = str(venv_python_path(VENV_DIR))

    # ── 步骤 3 ──
    steps.push(3)
    _log(f"━━ 步骤 3/5：安装推理依赖（{plan['site_name']}）━━", log_fn)
    rc, out = _run_subprocess_streaming(
        [uv, "pip", "install", "--python", venv_py,
         "numpy", "pillow", "cuda-bindings",
         "nvidia-cuda-runtime", "nvidia-cuda-nvrtc",
         "--index-url", plan["pypi_url"]],
        _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
    )
    r = _cancelled()
    if r:
        return r
    if rc != 0:
        return {"ok": False, "error": f"安装推理依赖失败: {out[-500:]}"}
    _log("→ tensorrt（NVIDIA 索引 + PyPI 主索引，约 2GB）…", log_fn)
    rc, out = _run_subprocess_streaming(
        [uv, "pip", "install", "--python", venv_py,
         "tensorrt>=11,<12", "--extra-index-url", plan["nvidia_url"]],
        _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
    )
    r = _cancelled()
    if r:
        return r
    if rc != 0:
        return {"ok": False, "error": f"安装 tensorrt 失败: {out[-500:]}"}

    # ── 步骤 4：模型下载 ──
    from . import models_downloader
    if not models_downloader.begin_download():
        return {"ok": False, "busy": True, "error": "已有下载任务进行中，请稍后再试"}
    try:
        last_pct = {"v": -1}

        def _model_progress(ev: Dict[str, Any]):
            if ev.get("type") == "progress" and ev.get("total") and ev.get("pct") is not None:
                pct5 = int(ev["pct"] * 100 // 5)
                if pct5 > last_pct["v"]:
                    last_pct["v"] = pct5
                    _log(f"  {Path(ev['file']).name} {int(ev['pct'] * 100)}%"
                         f"（{fmt_size(ev['done'])}/{fmt_size(ev['total'])}）", log_fn)
                steps.push(4, ((ev.get("idx", 1) - 1) + ev["pct"]) / max(1, ev.get("count", 1)))
            elif ev.get("type") in ("file_done", "file_skip"):
                steps.push(4, ev.get("idx", 0) / max(1, ev.get("count", 1)))

        steps.push(4)
        files = [*_ONNX_FILES[precision], *SHARED_FILES]
        _log(f"━━ 步骤 4/5：从魔搭下载 Hammy tagger 双模型与脚本"
             f"（{precision.upper()}，共 {len(files)} 个文件）━━", log_fn)
        dl = models_downloader.download_files(
            "ms", REPO_ID, files, str(MODEL_DIR),
            log_fn=lambda m: _log(m, log_fn), progress_cb=_model_progress,
            cancel_event=stop_event, cleanup_on_cancel=True)
        if dl.get("cancelled"):
            return {"ok": False, "cancelled": True, "error": "模型下载已取消"}
        if not dl.get("ok"):
            core_failed = [f for f in dl.get("failed") or [] if f.get("file") != TAG_ZH_FILE]
            if dl.get("error") or core_failed:
                err = dl.get("error") or str(core_failed[0].get("error", ""))[:200]
                return {"ok": False, "error": f"模型下载失败: {err}"}
            _log("标签翻译表下载失败（不影响标签生成，中文显示将回退英文）", log_fn)
    finally:
        models_downloader.end_download()

    # ── 步骤 5：构建 TRT 引擎（build_trt.py 构建成功后自动删除 ONNX）──
    steps.push(5)
    _log("━━ 步骤 5/5：构建 TensorRT 引擎（H/V 双模型）━━", log_fn)
    from .llama_cpp import pause_for_task
    pause_for_task(log_fn=log_fn)  # 让出显存
    rc, out = _run_subprocess_streaming(
        [venv_py, str(_repo_dir() / "build_trt.py"), "--precision", precision],
        _BUILD_TIMEOUT_SEC, log_fn, stop_event,
    )
    if stop_event is not None and stop_event.is_set():
        return {"ok": False, "cancelled": True, "error": "安装已取消"}
    if rc != 0:
        return {"ok": False, "error": f"引擎构建失败: {out[-300:]}"}
    missing = [p.name for p in _engine_paths() if not p.is_file()]
    if missing:
        return {"ok": False, "error": f"引擎构建失败: {'、'.join(missing)} 未生成"}

    from .config_store import update_tagger_config
    update_tagger_config(lambda c: c.update(precision=precision))
    _log("━━ 全部完成 ━━", log_fn)
    return {"ok": True, "error": None}


def ensure_model_files(log_fn: Optional[Callable[[str], None]] = None,
                       stop_event: Optional[threading.Event] = None) -> Optional[str]:
    """分析前预检：缺失的共享文件从魔搭补下载；引擎缺失直接报重装。返回错误信息或 None。"""
    from . import models_downloader
    if any(not p.is_file() for p in _engine_paths()):
        return "TRT 引擎缺失，请卸载后重新安装 Hammy-tagger"
    missing = [n for n in SHARED_FILES if not (_repo_dir() / n).is_file()]
    if not missing:
        return None
    if not models_downloader.begin_download():
        return "已有下载任务进行中，请稍后再试"
    try:
        _log(f"补充下载仓库文件: {', '.join(missing)}", log_fn)
        dl = models_downloader.download_files(
            "ms", REPO_ID, missing, str(MODEL_DIR),
            log_fn=lambda m: _log(m, log_fn), cancel_event=stop_event,
            cleanup_on_cancel=True)
        if dl.get("cancelled"):
            return "模型下载已取消"
        failed = [f.get("file") for f in dl.get("failed") or []]
        if not dl.get("ok") and (dl.get("error") or any(f in _CRITICAL_SHARED for f in failed)):
            err = dl.get("error") or str((dl.get("failed") or [{}])[0].get("error", ""))[:200]
            return f"仓库文件补下载失败: {err or '未知错误'}"
        if any(f in _CRITICAL_SHARED for f in failed):
            return "仓库文件补下载失败，请卸载后重新安装 Hammy-tagger"
    finally:
        models_downloader.end_download()
    return None


def remove_tagger() -> Dict[str, Any]:
    """删除 Tagger 文件夹与模块数据文件夹（_workspace/tagger/）。"""
    existed = TAGGER_DIR.exists()
    if existed:
        try:
            shutil.rmtree(TAGGER_DIR)
        except Exception as e:
            return {"ok": False, "error": f"删除失败: {e}"}
    # 模块配置/开关/标签结果一并删除
    try:
        from .workspace_paths import TAGGER_CONFIG_FILE
        shutil.rmtree(TAGGER_CONFIG_FILE.parent, ignore_errors=True)
    except Exception:
        pass
    return {"ok": True,
            "message": "已删除 Tagger 文件夹" if existed else "目录不存在，无需删除"}


# ── 抽帧与推理 ──

_NO_STOP = threading.Event()


def extract_frames_for_tagger(video_path: str,
                              stop_event: Optional[threading.Event] = None,
                              frame_count: int = 15) -> List[str]:
    """复用主流程抽帧（FRAME_VF：短边压到 448），返回 JPEG base64 帧列表。"""
    if stop_event is None:
        stop_event = _NO_STOP
    if stop_event.is_set():
        return []
    cfg = Config(sampling_points=max(1, frame_count), frames_per_point=1)
    cfg.validate()
    _, frames = probe_and_extract_keyframes(video_path, cfg, stop_event, frame_vf=FRAME_VF)
    return [f.b64 for f in frames]


def _inference_timeout(video_count: int) -> float:
    return _INFERENCE_BASE_SEC + _INFERENCE_PER_VIDEO_SEC * max(1, video_count)


class AnalyzeStream:
    """交互式分析子进程（模型仓库 hammy_server.py）：stdin 逐视频送入，stdout 流式回传。"""

    def __init__(self, params: Dict[str, Any], video_count: int,
                 stop_event=None,
                 on_log: Optional[Callable[[str], None]] = None):
        self.stop_event = stop_event
        self.on_log = on_log
        self.broken = False  # 子进程异常/停止/超时后置位
        self._deadline = time.time() + _inference_timeout(video_count)
        self._result_q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._err_tail: "deque[str]" = deque(maxlen=200)
        self._error = ""

        py = venv_python_path(VENV_DIR)
        if not py.is_file():
            raise RuntimeError(f"模块环境不存在: {py}")
        script = _server_script()
        if not script.is_file():
            raise RuntimeError(f"分析服务脚本不存在: {script}")

        fd, self._params_path = tempfile.mkstemp(suffix=".json", prefix="modjob_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False)
        except Exception as e:
            os.unlink(self._params_path)
            raise RuntimeError(f"参数文件写入失败: {e}")

        try:
            self.p = subprocess.Popen(
                [str(py), str(script), self._params_path],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                **SUBPROCESS_KWARGS,
            )
        except Exception as e:
            try:
                os.unlink(self._params_path)
            except OSError:
                pass
            raise RuntimeError(f"启动失败: {e}")
        register_subprocess(self.p)

        def _read_stdout():
            assert self.p.stdout is not None
            try:
                for line in self.p.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    if obj.get("log"):
                        if self.on_log is not None:
                            try:
                                self.on_log(str(obj["log"]))
                            except Exception:
                                pass
                    elif "video_result" in obj and isinstance(obj["video_result"], dict):
                        self._result_q.put(obj["video_result"])
                    elif obj.get("error"):
                        self._error = str(obj["error"])
            except Exception:
                pass
            self._result_q.put(None)  # stdout EOF 哨兵

        def _read_stderr():
            assert self.p.stderr is not None
            try:
                for line in self.p.stderr:
                    self._err_tail.append(line.rstrip())
            except Exception:
                pass

        self._out_thread = threading.Thread(target=_read_stdout, name="tagger-out-reader", daemon=True)
        self._err_thread = threading.Thread(target=_read_stderr, name="tagger-err-reader", daemon=True)
        self._out_thread.start()
        self._err_thread.start()

    def send(self, item: Dict[str, Any]) -> bool:
        """发送一个视频请求，子进程已退出或推理超时返回 False。"""
        if self.broken or self.p.poll() is not None:
            return False
        if time.time() > self._deadline:  # 超时后不再发送，防管道阻塞无限等
            self.broken = True
            self._error = "推理超时"
            return False
        try:
            assert self.p.stdin is not None
            self.p.stdin.write(json.dumps(item, ensure_ascii=False) + "\n")
            self.p.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            self.broken = True
            return False

    def next_result(self) -> Optional[Dict[str, Any]]:
        """阻塞等待下一个 video_result，停止/超时/子进程退出时返回 None。"""
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                self.broken = True
                return None
            if time.time() > self._deadline:
                self.broken = True
                self._error = "推理超时"
                _terminate_proc(self.p)
                return None
            try:
                vr = self._result_q.get(timeout=0.25)
            except queue.Empty:
                continue  # 只信 EOF 哨兵，避免 poll 竞态丢尾部结果
            if vr is None:  # stdout EOF
                self.broken = True
                return None
            return vr

    def try_next_result(self) -> Optional[Dict[str, Any]]:
        """非阻塞版 next_result：无就绪结果时返回 None（不置位 broken）。"""
        if self.broken:
            return None
        try:
            vr = self._result_q.get(timeout=0.02)
        except queue.Empty:
            return None
        if vr is None:  # stdout EOF
            self.broken = True
            return None
        return vr

    def close(self) -> Dict[str, Any]:
        """关闭子进程并回收。Returns: {"ok": bool, "error": str|None}。"""
        stopped = self.stop_event is not None and self.stop_event.is_set()
        if stopped or self.broken:
            try:
                _terminate_proc(self.p, wait_sec=1.0)
            except Exception:
                pass
        else:
            try:
                if self.p.stdin is not None:
                    self.p.stdin.close()
            except (OSError, ValueError):
                pass
        if stopped:
            wait_sec = 2.0
            join_sec = 2.0
        else:
            wait_sec = max(10.0, self._deadline - time.time())
            join_sec = 10
        try:
            self.p.wait(timeout=wait_sec)
        except subprocess.TimeoutExpired:
            _terminate_proc(self.p)
        try:
            self._out_thread.join(timeout=join_sec)
            self._err_thread.join(timeout=join_sec)
        except Exception:
            pass
        unregister_subprocess(self.p)
        try:
            os.unlink(self._params_path)
        except OSError:
            pass

        rc = self.p.returncode if self.p.returncode is not None else -1
        if self.stop_event is not None and self.stop_event.is_set():
            return {"ok": False, "error": "已停止"}
        if rc != 0:
            if self._error:
                return {"ok": False, "error": self._error}
            tail = "\n".join(list(self._err_tail)[-20:])
            return {"ok": False,
                    "error": f"推理失败: {tail[-500:] if tail else '子进程异常退出'}"}
        return {"ok": True, "error": None}


def start_analyze_stream(
    video_count: int,
    tag_threshold: float = TAG_THRESHOLD,
    ip_threshold: float = TAG_THRESHOLD,
    stop_event=None,
    on_log: Optional[Callable[[str], None]] = None,
) -> Optional[AnalyzeStream]:
    """启动交互式分析子进程（模型仓库 hammy_server.py）。"""
    status = get_status()
    if not status["ready"]:
        return None
    params = {
        "backend": "trt",
        "precision": status["precision"] or "fp16",
        "char_threshold": tag_threshold,
        "ip_threshold": ip_threshold,
    }
    try:
        return AnalyzeStream(params, video_count,
                             stop_event=stop_event, on_log=on_log)
    except Exception:
        return None
