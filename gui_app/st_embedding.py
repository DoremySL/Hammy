"""st_embedding.py — 标签向量检索（Sentence-Transformers 后端）：依赖/模型安装与嵌入 worker 客户端。"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from batch_rename.env import SUBPROCESS_KWARGS
from batch_rename.subprocess_registry import register_subprocess, unregister_subprocess
from .env import APP_ROOT, PYTHON_EXE, make_logger
from . import installer
from .installer import (
    PYPI_MIRRORS, PYTORCH_MIRRORS, DEFAULT_PYPI_MIRROR, DEFAULT_PYTORCH_MIRROR,
    ensure_uv as _ensure_uv,
    run_subprocess_streaming as _run_subprocess_streaming,
    _terminate_proc,
    venv_python_path,
)

ST_DIR = APP_ROOT / "st-embedding"
VENV_DIR = ST_DIR / "venv"
MODEL_DIR = ST_DIR / "models"
INDEX_PATH = ST_DIR / "index_cache.npz"
WORKER_SCRIPT = Path(__file__).resolve().parent / "st_embed_worker.py"

ST_EMBED_MODELS = [
    {"key": "qwen3-06b", "title": "Qwen3-Embedding-0.6B",
     "desc": "多语言，建议GPU运行",
     "repo": "Qwen/Qwen3-Embedding-0.6B", "backend": "torch",
     "size_label": "约 1.2GB", "recommended": True,
     "files": ["1_Pooling/config.json", "config.json", "config_sentence_transformers.json",
               "configuration.json", "generation_config.json", "merges.txt",
               "model.safetensors", "modules.json", "tokenizer.json",
               "tokenizer_config.json", "vocab.json"]},
    {"key": "e5-small", "title": "multilingual-e5-small",
     "desc": "多语言，CPU运行可用",
     "repo": "intfloat/multilingual-e5-small", "backend": "torch",
     "size_label": "约 450MB", "recommended": False,
     "files": ["1_Pooling/config.json", "config.json", "configuration.json",
               "model.safetensors", "modules.json", "sentence_bert_config.json",
               "sentencepiece.bpe.model", "special_tokens_map.json",
               "tokenizer.json", "tokenizer_config.json"]},
    {"key": "bge-small-zh", "title": "bge-small-zh-v1.5",
     "desc": "中文向，CPU运行可用",
     "repo": "BAAI/bge-small-zh-v1.5", "backend": "torch",
     "size_label": "约 92MB", "recommended": False,
     "files": ["1_Pooling/config.json", "config.json", "config_sentence_transformers.json",
               "configuration.json", "model.safetensors", "modules.json",
               "sentence_bert_config.json", "special_tokens_map.json",
               "tokenizer.json", "tokenizer_config.json", "vocab.txt"]},
]

_INSTALL_TIMEOUT_SEC = 1800.0      # torch + sentence-transformers 安装
_DOWNLOAD_TIMEOUT_SEC = 7200.0     # 模型下载
_WORKER_READY_TIMEOUT_SEC = 600.0  # worker 冷启动（含 torch import + 模型加载）
_BUILD_TIMEOUT_SEC = 14400.0       # 索引构建（大库全新编码可能很久，设宽上限）
_QUERY_TIMEOUT_SEC = 180.0

_log = make_logger("st_embedding")

def _model_def(value: str = "") -> Dict[str, Any]:
    """按 key 解析模型定义；空/未知回退默认。"""
    v = str(value or "").strip()
    if v:
        for m in ST_EMBED_MODELS:
            if m["key"] == v:
                return m
    return ST_EMBED_MODELS[0]

def _model_dir(model: Dict[str, Any]) -> Optional[Path]:
    """模型目录：download_files 的 作者/仓库 子目录布局。"""
    author, repo_name = model["repo"].split("/", 1)
    path = MODEL_DIR / author / repo_name
    return path if path.is_dir() and any(path.glob("*.safetensors")) else None

def model_installed(model: Dict[str, Any]) -> bool:
    return _model_dir(model) is not None

def _download_model(model_key: str,
                    log_fn: Optional[Callable[[str], None]] = None,
                    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
                    stop_event: Optional[threading.Event] = None) -> Dict[str, Any]:
    """按清单下载嵌入模型（魔搭）。"""
    from . import models_downloader
    model = _model_def(model_key)
    stop_event = stop_event or models_downloader.get_cancel_event()
    if not venv_python_path(VENV_DIR).is_file():
        return {"ok": False, "error": "请先安装依赖（安装按钮）"}
    if model_installed(model):
        return {"ok": True, "downloaded": 0, "message": "模型已安装"}

    if not models_downloader.begin_download():
        return {"ok": False, "busy": True, "error": "已有下载任务进行中，请稍后再试"}
    try:
        dl = models_downloader.download_files(
            "ms", model["repo"], model["files"], str(MODEL_DIR),
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
    if not model_installed(model):
        return {"ok": False, "error": "下载完成但未找到模型权重文件"}
    return {"ok": True, "downloaded": len(model["files"]), "model": model["key"]}

def get_status() -> Dict[str, Any]:
    """st-embedding 模块安装状态（含模型清单与安装位图）。"""
    venv_python = venv_python_path(VENV_DIR)
    venv_ok = VENV_DIR.is_dir() and venv_python.exists()
    models: List[Dict[str, Any]] = []
    installed: Dict[str, bool] = {}
    any_model = False
    for m in ST_EMBED_MODELS:
        inst = model_installed(m)
        installed[m["key"]] = inst
        any_model = any_model or inst
        models.append({"key": m["key"], "title": m["title"], "desc": m["desc"],
                       "repo": m["repo"], "size_label": m["size_label"],
                       "recommended": bool(m.get("recommended", False)),
                       "installed": inst})
    return {
        "dir_exists": ST_DIR.is_dir(),
        "venv_exists": venv_ok,
        "model_exists": any_model,
        "ready": venv_ok and any_model,
        "dir_path": str(ST_DIR),
        "model_dir_path": str(MODEL_DIR),
        "worker_running": _is_worker_alive(),
        "models": models,
        "installed": installed,
    }

def _fmt_size(num: float) -> str:
    """字节数 -> （B/KB/MB/GB）。"""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}TB"

def install_dependencies(
    pytorch_mirror: str = DEFAULT_PYTORCH_MIRROR,
    pypi_mirror: str = DEFAULT_PYPI_MIRROR,
    model: str = "",
    log_fn: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """安装依赖（uv + venv + torch + sentence-transformers/modelscope）+ 下载所选模型。"""
    torch_url = PYTORCH_MIRRORS.get(pytorch_mirror, PYTORCH_MIRRORS[DEFAULT_PYTORCH_MIRROR])["url"]
    pypi_url = PYPI_MIRRORS.get(pypi_mirror, PYPI_MIRRORS[DEFAULT_PYPI_MIRROR])["url"]
    meta = _model_def(model)
    model_key = meta["key"]

    ST_DIR.mkdir(parents=True, exist_ok=True)

    def _cancelled() -> Optional[Dict[str, Any]]:
        if stop_event is not None and stop_event.is_set():
            _log("安装已取消", log_fn)
            return {"ok": False, "cancelled": True, "error": "安装已取消"}
        return None

    _log("━━ 步骤 1/5：检测 UV ━━", log_fn)
    uv = _ensure_uv(pypi_url, log_fn, stop_event)
    if r := _cancelled():
        return r
    if not uv:
        return {"ok": False, "error": "UV 安装失败"}
    _log(f"UV 就绪: {uv}", log_fn)

    _log("━━ 步骤 2/5：创建虚拟环境 ━━", log_fn)
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

    _log(f"━━ 步骤 3/5：安装 torch（{'CPU' if 'cpu' in torch_url else 'CUDA'} 版）━━", log_fn)
    rc, out = _run_subprocess_streaming(
        [uv, "pip", "install", "--python", venv_py,
         "torch", "--index-url", torch_url],
        _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
    )
    if r := _cancelled():
        return r
    if rc != 0:
        return {"ok": False, "error": f"安装 torch 失败: {out[-500:]}"}

    _log("━━ 步骤 4/5：安装 sentence-transformers + modelscope ━━", log_fn)
    rc, out = _run_subprocess_streaming(
        [uv, "pip", "install", "--python", venv_py,
         "sentence-transformers", "modelscope",
         "--index-url", pypi_url],
        _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
    )
    if r := _cancelled():
        return r
    if rc != 0:
        return {"ok": False, "error": f"安装依赖失败: {out[-500:]}"}

    stop_worker()  # 依赖变更（如 transformers 版本）后丢弃旧 worker

    # ── 步骤 5：下载所选嵌入模型（复用 _download_model，取消时清理未完成文件）──
    # 安装时把块级进度折算成日志（每 5% 一行）：模型下载期间日志面板不能全程静默，
    # 否则用户误以为卡住
    _log(f"━━ 步骤 5/5：下载模型（{meta['title']}，魔搭）━━", log_fn)
    last_mark = {"v": -1}

    def _install_progress(ev: Dict[str, Any]):
        if ev.get("type") == "progress" and ev.get("total") and ev.get("pct") is not None:
            mark = int(ev["pct"] * 100 // 5)
            if mark > last_mark["v"]:
                last_mark["v"] = mark
                _log(f"  {Path(ev['file']).name} {int(ev['pct'] * 100)}%"
                     f"（{_fmt_size(ev['done'])}/{_fmt_size(ev['total'])}）", log_fn)

    dl = _download_model(model_key, log_fn=log_fn, progress_cb=_install_progress,
                         stop_event=stop_event)
    if r := _cancelled():
        return r
    if dl.get("cancelled"):
        _log("模型下载已取消，未完成的文件已清理", log_fn)
        return {"ok": False, "cancelled": True, "error": "模型下载已取消"}
    if not dl.get("ok"):
        return {"ok": False, "error": f"模型下载失败: {dl.get('error') or '未知错误'}"}
    _log("模型下载完成", log_fn)
    # 安装时选定的模型设为当前模型：若用户选了非默认模型而不同步，「嵌入模型」
    # 下拉与向量检索会指向未下载的默认模型，ready 判定与界面状态随之错位
    try:
        from .config_store import update_config

        def _set_model(c):
            c.setdefault("experimental", {}).update(rag_vec_model=model_key)
            return c

        update_config(_set_model)
        _log(f"当前模型已设为: {meta['title']}", log_fn)
    except Exception as e:
        _log(f"设置当前模型失败（忽略）: {e}", log_fn)
    _log("━━ 全部完成 ━━", log_fn)
    return {"ok": True, "error": None}

def remove_st_embedding() -> Dict[str, Any]:
    """删除 st-embedding 文件夹（venv + 模型 + 索引缓存）并复位模块配置。"""
    stop_worker()
    existed = ST_DIR.exists()
    if existed:
        try:
            import shutil
            shutil.rmtree(ST_DIR)
        except Exception as e:
            return {"ok": False, "error": f"删除失败: {e}"}
    # 配置不留痕迹：启用开关与模型选择一并复位（与 faster-whisper 删除模块
    # 配置目录等价），避免重装依赖后开关自动回到卸载前的开启状态
    try:
        from .config_store import update_config

        def _reset(c):
            c.setdefault("experimental", {}).update(rag_vec_enabled=False,
                                                    rag_vec_model="")
            return c

        update_config(_reset)
    except Exception:
        pass
    return {"ok": True,
            "message": "已删除 st-embedding 文件夹" if existed else "目录不存在，无需删除"}

_proc_lock = threading.RLock()         # 可重入：请求路径内会嵌套调用 stop_worker/_ensure
_proc: Optional[subprocess.Popen] = None
_proc_key: Optional[tuple] = None

def _is_worker_alive() -> bool:
    return _proc is not None and _proc.poll() is None

def stop_worker() -> None:
    """停止 worker 子进程（安装/卸载/参数变化/异常时调用，保证状态与队列重同步）。"""
    global _proc, _proc_key
    with _proc_lock:
        proc = _proc
        _proc, _proc_key = None, None
        if proc is not None:
            params_path = getattr(proc, "_st_params_path", None)
            if params_path:
                _unlink_quiet(params_path)
            if proc.poll() is None:
                _terminate_proc(proc, wait_sec=2.0)

def _items_hash(items: List[Dict[str, str]]) -> str:
    """条目指纹：规范化 JSON 的 md5（顺序敏感——idx 对齐依赖稳定顺序）。"""
    payload = json.dumps(items, ensure_ascii=False, sort_keys=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def _embed_text(it: Dict[str, str]) -> str:
    """条目 → 向量编码文本：关键词：描述；有关联词时追加（分组名不参与）。"""
    base = f"{it.get('keyword', '')}：{it.get('description', '')}".strip("：")
    related = str(it.get("related", "") or "").strip()
    return f"{base}。关联词：{related}" if base and related else (base or related)

class StEmbedBackend:
    """TagRecall.dense 协议实现：build_index(items) + search(parts, top_k)。

    worker 子进程懒启动；设备/模型参数变化时自动重启；任何失败抛异常，
    由 TagRecall 捕获并降级为纯词面召回。
    """

    def __init__(self, device: str = "auto", model_key: str = "",
                 log_fn: Optional[Callable[[str], None]] = None):
        model = _model_def(model_key)
        self.device = str(device or "auto")
        self.model_key = model["key"]
        self.model_title = model["title"]
        self._model_path = _model_dir(model)  # 未下载时为 None，_spawn 时报错
        self._log_fn = log_fn

    @property
    def name(self) -> str:
        return f"st:{self.model_title}:{self.device}"

    def _log(self, msg: str) -> None:
        if self._log_fn:
            try:
                self._log_fn(msg)
            except Exception:
                pass

    def _spawn(self) -> None:
        global _proc, _proc_key
        model_path = self._model_path
        if model_path is None:
            raise RuntimeError("嵌入模型未下载（卡片下拉中选择模型下载）")
        py = venv_python_path(VENV_DIR)
        if not py.is_file():
            raise RuntimeError("st-embedding venv 不存在，请先安装")
        params = {
            "model_path": str(model_path),
            "device": self.device,
            "index_path": str(INDEX_PATH),
        }
        params_path = tempfile_params(params)
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
        _proc, _proc_key = proc, (self.device, self.model_key)
        proc._st_params_path = params_path  # 进程退出后统一清理

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

        threading.Thread(target=_reader, name="st-embed-reader", daemon=True).start()
        threading.Thread(target=_err_reader, name="st-embed-err", daemon=True).start()
        proc._st_ev_q = ev_q
        proc._st_err_tail = err_tail

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
                self._log(f"嵌入 worker 就绪（device={ev.get('device')}，dim={ev.get('dim')}，"
                          f"加载 {ev.get('load_s')}s）")
                return
            if ev.get("event") == "error":
                raise RuntimeError(str(ev.get("message", "worker 初始化失败")))
        raise RuntimeError("worker 在 600 秒内未就绪")

    def _ensure(self) -> subprocess.Popen:
        global _proc
        proc = _proc
        key = (self.device, self.model_key)
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
                    tail = "；".join(list(getattr(proc, "_st_err_tail", []))[-3:])
                    raise RuntimeError(f"worker 已退出: {tail or '无输出'}")
                try:
                    ev = proc._st_ev_q.get(timeout=0.5)
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
            # 超时：事件队列可能与请求错位，丢弃 worker 强制重建
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

def tempfile_params(params: Dict[str, Any]) -> str:
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".json", prefix="stembed_")
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
