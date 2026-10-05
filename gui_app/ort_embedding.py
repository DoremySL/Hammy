"""ort_embedding.py — 标签向量检索（ONNX Runtime 后端）：安装与嵌入 worker 客户端。"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from batch_rename.env import SUBPROCESS_KWARGS
from batch_rename.subprocess_registry import register_subprocess
from batch_rename.utils import fmt_size
from .env import APP_ROOT, PYTHON_EXE, make_logger
from . import installer
from .installer import (
    PYPI_MIRRORS, DEFAULT_PYPI_MIRROR,
    ensure_uv as _ensure_uv,
    run_subprocess_streaming as _run_subprocess_streaming,
    _terminate_proc,
    venv_python_path,
    InstallSteps,
)

ORT_DIR = APP_ROOT / "ort-embedding"
VENV_DIR = ORT_DIR / "venv"
MODEL_DIR = ORT_DIR / "models"
INDEX_PATH = ORT_DIR / "index_cache.npz"
WORKER_SCRIPT = Path(__file__).resolve().parent / "ort_embed_worker.py"

EMBED_REPO = "DoremySL/embeddinggemma-300m-ort"
EMBED_TITLE = "EmbeddingGemma-300m"

# 各变体的下载清单
_VARIANT_FILES = {
    "gpu": ["model.onnx", "model.onnx_data", "tokenizer.json"],
    "cpu": ["model_int8_qat.onnx", "tokenizer.json"],
}
_VARIANT_MODEL = {"gpu": "model.onnx", "cpu": "model_int8_qat.onnx"}
_VARIANT_LABEL = {"gpu": "GPU · fp32", "cpu": "CPU · int8"}

_INSTALL_TIMEOUT_SEC = 1800.0      # 依赖安装超时
_WORKER_READY_TIMEOUT_SEC = 300.0  # worker 冷启动超时
_BUILD_TIMEOUT_SEC = 14400.0       # 索引构建超时
_QUERY_TIMEOUT_SEC = 180.0

_log = make_logger("ort_embedding")


def _norm_device(value: Any) -> str:
    return "cpu" if str(value or "").strip().lower() == "cpu" else "gpu"


def _repo_dir() -> Path:
    """模型目录（作者/仓库 布局）。"""
    return MODEL_DIR / "DoremySL" / EMBED_REPO.split("/", 1)[1]


def _variant_files(device: str) -> List[str]:
    return list(_VARIANT_FILES[_norm_device(device)])


def variant_complete(device: str) -> bool:
    d = _repo_dir()
    return all((d / f).is_file() for f in _variant_files(device))


def installed_variant() -> Optional[str]:
    for v in ("gpu", "cpu"):
        if variant_complete(v):
            return v
    return None


def _ort_flavor() -> str:
    """venv 内实际安装的 onnxruntime 包（gpu = onnxruntime-gpu / cpu = onnxruntime）。"""
    sites = [VENV_DIR / "Lib" / "site-packages"]
    sites += sorted(VENV_DIR.glob("lib/python*/site-packages"))
    for sp in sites:
        if not sp.is_dir():
            continue
        if any(sp.glob("onnxruntime_gpu-*.dist-info")):
            return "gpu"
        if any(sp.glob("onnxruntime-*.dist-info")):
            return "cpu"
    return ""


def get_status() -> Dict[str, Any]:
    """向量检索模块安装状态。"""
    venv_python = venv_python_path(VENV_DIR)
    venv_ok = VENV_DIR.is_dir() and venv_python.exists()
    variant = installed_variant()
    flavor = _ort_flavor()
    return {
        "dir_exists": ORT_DIR.is_dir(),
        "venv_exists": venv_ok,
        "model_exists": variant is not None,
        "ready": venv_ok and variant is not None,
        "dir_path": str(ORT_DIR),
        "model_dir_path": str(MODEL_DIR),
        "worker_running": _is_worker_alive(),
        "variant": variant or "",
        "ort_flavor": flavor,
        "device_match": bool(variant and flavor) and variant == flavor,
        "model_title": EMBED_TITLE,
        "repo": EMBED_REPO,
    }


def _download_model(device: str,
                    log_fn: Optional[Callable[[str], None]] = None,
                    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
                    stop_event: Optional[threading.Event] = None) -> Dict[str, Any]:
    """按设备下载模型变体文件（魔搭；取消时清理未完成文件）。"""
    from . import models_downloader
    device = _norm_device(device)
    stop_event = stop_event or models_downloader.get_cancel_event()
    if variant_complete(device):
        return {"ok": True, "downloaded": 0, "message": "模型已安装"}

    if not models_downloader.begin_download():
        return {"ok": False, "busy": True, "error": "已有下载任务进行中，请稍后再试"}
    try:
        dl = models_downloader.download_files(
            "ms", EMBED_REPO, _variant_files(device), str(MODEL_DIR),
            log_fn=log_fn, progress_cb=progress_cb,
            cancel_event=stop_event, cleanup_on_cancel=True)
        if dl.get("cancelled"):
            return {"ok": False, "cancelled": True, "error": "模型下载已取消"}
        if not dl.get("ok"):
            err = dl.get("error")
            if not err and dl.get("failed"):
                err = str(dl["failed"][0].get("error", ""))[:200]
            return {"ok": False, "error": f"模型下载失败: {err or '未知错误'}"}
    finally:
        models_downloader.end_download()
    if not variant_complete(device):
        return {"ok": False, "error": "下载完成但模型文件不完整"}
    return {"ok": True, "downloaded": len(_VARIANT_FILES[device]), "device": device}


def install_dependencies(
    device: str = "gpu",
    site: str = DEFAULT_PYPI_MIRROR,
    log_fn: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """安装依赖并下载所选设备的模型文件（device: gpu/cpu，site: PyPI 站点 ID）。"""
    device = _norm_device(device)
    mir = PYPI_MIRRORS.get(site) or PYPI_MIRRORS[DEFAULT_PYPI_MIRROR]
    pypi_url = mir["url"]
    ort_pkg = "onnxruntime-gpu" if device == "gpu" else "onnxruntime"
    steps = InstallSteps(4)

    ORT_DIR.mkdir(parents=True, exist_ok=True)

    def _cancelled() -> Optional[Dict[str, Any]]:
        if stop_event is not None and stop_event.is_set():
            _log("安装已取消", log_fn)
            return {"ok": False, "cancelled": True, "error": "安装已取消"}
        return None

    steps.push(1)
    _log("━━ 步骤 1/4：检测 UV ━━", log_fn)
    uv = _ensure_uv(pypi_url, log_fn, stop_event)
    if r := _cancelled():
        return r
    if not uv:
        return {"ok": False, "error": "UV 安装失败"}

    steps.push(2)
    _log("━━ 步骤 2/4：创建虚拟环境 ━━", log_fn)
    if not VENV_DIR.is_dir():
        rc, out = _run_subprocess_streaming(
            [uv, "venv", str(VENV_DIR), "--python", PYTHON_EXE],
            installer.UV_TIMEOUT_SEC, log_fn, stop_event,
        )
        if r := _cancelled():
            return r
        if rc != 0:
            return {"ok": False, "error": f"创建虚拟环境失败: {out[-300:]}"}
    venv_py = str(venv_python_path(VENV_DIR))
    _log("虚拟环境就绪", log_fn)

    steps.push(3)
    _log(f"━━ 步骤 3/4：安装 {ort_pkg} + numpy/tokenizers（{mir['name']}）━━", log_fn)
    # 换设备重装前先卸旧包（含 nvidia CUDA 库，未装时 uv 跳过并告警）
    rc, out = _run_subprocess_streaming(
        [uv, "pip", "uninstall", "--python", venv_py,
         "onnxruntime", "onnxruntime-gpu",
         "nvidia-cuda-runtime", "nvidia-cuda-nvrtc", "nvidia-cufft",
         "nvidia-curand", "nvidia-cudnn-cu13"],
        _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
    )
    if r := _cancelled():
        return r
    rc, out = _run_subprocess_streaming(
        [uv, "pip", "install", "--python", venv_py,
         ort_pkg, "numpy", "tokenizers", "--index-url", pypi_url],
        _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
    )
    if r := _cancelled():
        return r
    if rc != 0:
        return {"ok": False, "error": f"安装依赖失败: {out[-500:]}"}
    steps.push(3, 0.5)

    if device == "gpu":
        # CUDA 运行库固定走 NVIDIA 源（通用镜像不装这些）
        _log("→ 安装 CUDA 运行库（NVIDIA 中国源）…", log_fn)
        rc, out = _run_subprocess_streaming(
            [uv, "pip", "install", "--python", venv_py,
             "nvidia-cuda-runtime", "nvidia-cuda-nvrtc", "nvidia-cufft",
             "nvidia-curand", "nvidia-cudnn-cu13",
             "--index-url", installer.NVIDIA_PYPI_URL],
            _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
        )
        if r := _cancelled():
            return r
        if rc != 0:
            return {"ok": False, "error": f"安装 CUDA 运行库失败: {out[-500:]}"}

    stop_worker()  # 依赖变更后丢弃旧 worker

    # ── 步骤 4：下载模型 ──
    label = "fp32（GPU 用）" if device == "gpu" else "int8（CPU 用）"
    _log(f"━━ 步骤 4/4：下载模型（{EMBED_TITLE} · {label}，魔搭）━━", log_fn)
    last_mark = {"v": -1}

    def _install_progress(ev: Dict[str, Any]):
        if ev.get("type") == "progress" and ev.get("total") and ev.get("pct") is not None:
            mark = int(ev["pct"] * 100 // 5)
            if mark > last_mark["v"]:
                last_mark["v"] = mark
                _log(f"  {Path(ev['file']).name} {int(ev['pct'] * 100)}%"
                     f"（{fmt_size(ev['done'])}/{fmt_size(ev['total'])}）", log_fn)
            steps.push(4, ((ev.get("idx", 1) - 1) + ev["pct"]) / max(1, ev.get("count", 1)))
        elif ev.get("type") in ("file_done", "file_skip"):
            steps.push(4, ev.get("idx", 0) / max(1, ev.get("count", 1)))

    steps.push(4)
    dl = _download_model(device, log_fn=log_fn, progress_cb=_install_progress,
                         stop_event=stop_event)
    if r := _cancelled():
        return r
    if dl.get("cancelled"):
        return {"ok": False, "cancelled": True, "error": "模型下载已取消"}
    if not dl.get("ok"):
        return {"ok": False, "error": f"模型下载失败: {dl.get('error') or '未知错误'}"}
    if dl.get("downloaded") == 0:
        _log("模型已安装，跳过下载", log_fn)
    try:
        from .config_store import update_config

        def _set_device(c):
            c.setdefault("experimental", {}).update(rag_vec_device=device)
            return c

        update_config(_set_device)
    except Exception as e:
        _log(f"设置运行设备失败（忽略）: {e}", log_fn)
    _log("━━ 全部完成 ━━", log_fn)
    return {"ok": True, "error": None}


def remove_ort_embedding() -> Dict[str, Any]:
    """删除 ort-embedding 文件夹（venv + 模型 + 索引缓存）并复位模块配置。"""
    stop_worker()
    existed = ORT_DIR.exists()
    if existed:
        try:
            import shutil
            shutil.rmtree(ORT_DIR)
        except Exception as e:
            return {"ok": False, "error": f"删除失败: {e}"}
    # 启用开关与设备选择一并复位
    try:
        from .config_store import update_config

        def _reset(c):
            c.setdefault("experimental", {}).update(rag_vec_enabled=False,
                                                    rag_vec_device="")
            return c

        update_config(_reset)
    except Exception:
        pass
    return {"ok": True,
            "message": "已删除 ort-embedding 文件夹" if existed else "目录不存在，无需删除"}


_proc_lock = threading.RLock()         # 可重入锁
_proc: Optional[subprocess.Popen] = None
_proc_key: Optional[tuple] = None

def _is_worker_alive() -> bool:
    return _proc is not None and _proc.poll() is None

def stop_worker() -> None:
    """停止 worker 子进程。"""
    global _proc, _proc_key
    with _proc_lock:
        proc = _proc
        _proc, _proc_key = None, None
        if proc is not None:
            params_path = getattr(proc, "_ort_params_path", None)
            if params_path:
                _unlink_quiet(params_path)
            if proc.poll() is None:
                _terminate_proc(proc, wait_sec=2.0)

def _items_hash(items: List[Dict[str, str]]) -> str:
    """条目指纹：JSON 的 md5。"""
    payload = json.dumps(items, ensure_ascii=False, sort_keys=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def _embed_text(it: Dict[str, str]) -> str:
    """条目 → 向量编码文本。"""
    base = f"{it.get('keyword', '')}：{it.get('description', '')}".strip("：")
    related = str(it.get("related", "") or "").strip()
    return f"{base}。关联词：{related}" if base and related else (base or related)

class OrtEmbedBackend:
    """TagRecall.dense 协议实现：build_index(items) + search(parts, top_k)。"""

    def __init__(self, device: str = "",
                 log_fn: Optional[Callable[[str], None]] = None):
        if str(device or "").strip():
            self.device = _norm_device(device)
        else:
            self.device = installed_variant() or "cpu"
        self.model_title = f"{EMBED_TITLE}（{_VARIANT_LABEL[self.device]}）"
        self._model_path = _repo_dir()
        self._log_fn = log_fn

    @property
    def name(self) -> str:
        return f"ort:{self.model_title}:{self.device}"

    def _log(self, msg: str) -> None:
        if self._log_fn:
            try:
                self._log_fn(msg)
            except Exception:
                pass

    def _spawn(self) -> None:
        global _proc, _proc_key
        model_path = self._model_path
        if not ((model_path / _VARIANT_MODEL["gpu"]).is_file()
                or (model_path / _VARIANT_MODEL["cpu"]).is_file()):
            raise RuntimeError("嵌入模型未下载，请重装依赖（安装时同时下载模型）")
        py = venv_python_path(VENV_DIR)
        if not py.is_file():
            raise RuntimeError("ort-embedding venv 不存在，请先安装")
        params = {
            "model_path": str(model_path),
            "device": self.device,
            "index_path": str(INDEX_PATH),
        }
        params_path = _tempfile_params(params)
        try:
            proc = subprocess.Popen(
                [str(py), str(WORKER_SCRIPT), params_path],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                **SUBPROCESS_KWARGS,
            )
        except Exception as e:
            _unlink_quiet(params_path)
            raise RuntimeError(f"worker 启动失败: {e}")
        register_subprocess(proc)
        _proc, _proc_key = proc, (self.device,)
        proc._ort_params_path = params_path  # 进程退出后统一清理

        ev_q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        err_tail: List[str] = []

        def _reader():
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev_q.put(json.loads(line))
                    except Exception:
                        ev_q.put({"event": "log", "message": line})
            except Exception:
                pass
            ev_q.put(None)

        def _err_reader():
            assert proc.stderr is not None
            try:
                for line in proc.stderr:
                    line = line.rstrip()
                    if line:
                        err_tail.append(line)
                        del err_tail[:-50]
            except Exception:
                pass

        threading.Thread(target=_reader, name="ort-embed-reader", daemon=True).start()
        threading.Thread(target=_err_reader, name="ort-embed-err", daemon=True).start()
        proc._ort_ev_q = ev_q
        proc._ort_err_tail = err_tail

        deadline = time.time() + _WORKER_READY_TIMEOUT_SEC
        while time.time() < deadline:
            if proc.poll() is not None:
                tail = "；".join(err_tail[-3:])
                raise RuntimeError(f"worker 启动即退出: {tail or '无输出'}")
            try:
                ev = ev_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if ev is None:
                raise RuntimeError("worker 输出流中断")
            if ev.get("event") == "ready":
                actual = str(ev.get("device", ""))
                note = ("；CUDA 不可用，已回退 CPU（fp32 模型仍可运行，速度较慢）"
                        if self.device == "gpu" and "cpu" in actual.lower() else "")
                self._log(f"嵌入 worker 就绪（device={actual}，模型 {ev.get('file')}，"
                          f"dim={ev.get('dim')}，加载 {ev.get('load_s')}s{note}）")
                return
            if ev.get("event") == "error":
                raise RuntimeError(str(ev.get("message", "worker 初始化失败")))
        raise RuntimeError("worker 在 300 秒内未就绪")

    def _ensure(self) -> subprocess.Popen:
        global _proc
        proc = _proc
        key = (self.device,)
        if proc is None or proc.poll() is not None or _proc_key != key:
            stop_worker()
            self._spawn()
            proc = _proc
        return proc

    def _request(self, req: Dict[str, Any], want_event: str, timeout: float,
                 on_event: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        with _proc_lock:
            proc = self._ensure()
            try:
                proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
                proc.stdin.flush()
            except Exception as e:
                stop_worker()
                raise RuntimeError(f"worker 请求写入失败: {e}")
            deadline = time.time() + timeout
            while time.time() < deadline:
                if proc.poll() is not None:
                    stop_worker()
                    tail = "；".join(list(getattr(proc, "_ort_err_tail", []))[-3:])
                    raise RuntimeError(f"worker 已退出: {tail or '无输出'}")
                try:
                    ev = proc._ort_ev_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if ev is None:
                    stop_worker()
                    raise RuntimeError("worker 输出流中断")
                if ev.get("event") == "error":
                    raise RuntimeError(str(ev.get("message", "worker 错误")))
                if ev.get("event") == want_event:
                    return ev
                if on_event is not None:
                    try:
                        on_event(ev)
                    except Exception:
                        pass
            stop_worker()
            raise RuntimeError(f"worker 响应超时（{int(timeout)}s，等待 {want_event}）")

    def build_index(self, items: List[Dict[str, str]]) -> None:
        """编码全部条目并缓存（hash 一致时直接载入 npz）。"""
        last = {"t": 0.0}

        def _on_event(ev: Dict[str, Any]) -> None:
            if ev.get("event") == "progress":
                now = time.time()
                if now - last["t"] >= 5.0 or ev.get("done") == ev.get("total"):
                    last["t"] = now
                    self._log(f"向量索引编码中… {ev.get('done')}/{ev.get('total')}")

        ev = self._request({"op": "build", "hash": _items_hash(items),
                            "texts": [_embed_text(it) for it in items]},
                           "built", _BUILD_TIMEOUT_SEC, _on_event)
        extra = ""
        if not ev.get("cached") and ev.get("encode_s") is not None:
            extra = f"，编码 {ev['encode_s']}s"
        self._log(f"向量索引就绪：{ev.get('count')} 条（dim={ev.get('dim')}，"
                  f"{'缓存命中' if ev.get('cached') else '全新编码'}{extra}）")

    def search(self, parts: Dict[str, str], top_k: int) -> List[Dict[str, Any]]:
        """分段编码查询，返回 top-K 命中 [{"idx","score","seg"}]。"""
        ev = self._request({"op": "query", "parts": parts, "top_k": int(top_k)},
                           "matches", _QUERY_TIMEOUT_SEC)
        return list(ev.get("matches", []) or [])

def _tempfile_params(params: Dict[str, Any]) -> str:
    fd, path = tempfile.mkstemp(suffix=".json", prefix="ortembed_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False)
    except Exception:
        _unlink_quiet(path)
        raise
    return path

def _unlink_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
