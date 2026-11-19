"""transcribe.cpp 语音转录：安装、GGUF 模型管理，对视频转录输出 SRT 字幕。"""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from batch_rename.utils import fmt_size
from .env import APP_ROOT, PYTHON_EXE, FFMPEG_EXE, make_logger
from .installer import (
    PYPI_MIRRORS, DEFAULT_PYPI_MIRROR, UV_TIMEOUT_SEC,
    RC_CANCEL, RC_TIMEOUT,
    ensure_uv as _ensure_uv,
    venv_python_path,
    run_subprocess_streaming as _run_subprocess_streaming,
    run_venv_script as _run_venv_script,
    InstallSteps,
)

# ── 路径常量 ──
TRANSCRIBE_DIR = APP_ROOT / "transcribe"
VENV_DIR = TRANSCRIBE_DIR / "venv"
MODEL_DIR = TRANSCRIBE_DIR / "models"  # GGUF 模型根目录

# ── 内置模型元数据 ──
TRANSCRIBE_MODELS = {
    "whisper-turbo": {
        "title": "whisper-turbo",
        "family": "whisper",
        "gguf": "whisper-large-v3-turbo-Q8_0.gguf",
        "hf_repo": "handy-computer/whisper-large-v3-turbo-gguf",
        "desc": "多语言快速转录",
        "size_label": "~850MB",
        "recommended": True,
        "timestamps": True,   # 是否有逐句时间戳（SRT 导出/脚本分支的唯一真源）
    },
    "moss": {
        "title": "MOSS-Diarize",
        "family": "moss",
        "gguf": "MOSS-Transcribe-Diarize-Q8_0.gguf",
        "hf_repo": "handy-computer/moss-transcribe-diarize-gguf",
        "desc": "多语言，配置要求略高，适合生成字幕",
        "size_label": "~1GB",
        "recommended": False,
        "timestamps": True,
    },
    "sensevoice": {
        "title": "SenseVoice-Small",
        "family": "sensevoice",
        "gguf": "SenseVoiceSmall-Q8_0.gguf",
        "hf_repo": "handy-computer/SenseVoiceSmall-gguf",
        "desc": "中英日韩粤，无逐句时间戳，导出字幕不可用",
        "size_label": "~240MB",
        "recommended": False,
        "timestamps": False,
    },
}
DEFAULT_MODEL = "whisper-turbo"
DEFAULT_INJECT_MODEL = "sensevoice"

# 注入模式分窗时长（秒）
INJECT_WIN_SEC = {"sensevoice": 30.0, "whisper": 60.0, "moss": 300.0}

# ── 官方 Silero VAD（transcribe.cpp 内置，与 ASR 同引擎零新依赖）──
# 区段级推理跳过静音/底噪/纯音乐（防幻觉）；模型缺失时整段推理
VAD_MODEL_META = {
    "gguf": "silero-vad-v6.2-F32.gguf",
    "hf_repo": "handy-computer/silero-vad-v6.2-gguf",
    "size_label": "~1.2MB",
}


def vad_model_path() -> Optional[Path]:
    """官方 VAD 模型 GGUF 路径；未下载返回 None。"""
    author, _, name = VAD_MODEL_META["hf_repo"].partition("/")
    p = MODEL_DIR / author / name / VAD_MODEL_META["gguf"]
    return p if p.is_file() else None


def vad_ready() -> bool:
    return vad_model_path() is not None


def _download_vad(log_fn=None, stop_event=None) -> bool:
    """下载官方 Silero VAD 模型；成功返回 True。"""
    from . import models_downloader as _md
    dl = _md.download_files(
        "hf", VAD_MODEL_META["hf_repo"], [VAD_MODEL_META["gguf"]], str(MODEL_DIR),
        log_fn=lambda m: _log(m, log_fn), cancel_event=stop_event,
        cleanup_on_cancel=True)
    return bool(dl.get("ok"))

# ── 引擎安装 ──
_INSTALL_TIMEOUT_SEC = 600.0
# 整批挂起兜底预算：Σ(视频时长 × 0.25s) + 120s；时长未知的视频按 600s
_TIMEOUT_PER_VIDEO_SEC = 0.25
_TIMEOUT_FLAT_SEC = 120.0
_TIMEOUT_NO_DURATION_SEC = 600.0

_log = make_logger("transcribe_cpp")


def _model_path(model_key: str) -> Optional[Path]:
    """已安装模型 GGUF 的实际路径。"""
    meta = TRANSCRIBE_MODELS.get(model_key)
    if not meta:
        return None
    author, _, name = meta["hf_repo"].partition("/")
    p = MODEL_DIR / author / name / meta["gguf"]
    return p if p.is_file() else None


def is_model_installed(model_key: str) -> bool:
    """模型完整性检测。"""
    return _model_path(model_key) is not None


def get_current_model() -> str:
    """右键转录模型 key。"""
    from .config_store import load_transcribe_config
    key = load_transcribe_config()["model"]
    return key if key in TRANSCRIBE_MODELS else DEFAULT_MODEL


def get_inject_model() -> str:
    """辅助重命名模型 key。"""
    from .config_store import load_transcribe_config
    key = load_transcribe_config()["inject_model"]
    return key if key in TRANSCRIBE_MODELS else DEFAULT_INJECT_MODEL


def get_installed_models() -> Dict[str, bool]:
    """扫描全部内置模型的安装状态 {key: bool}。"""
    return {key: is_model_installed(key) for key in TRANSCRIBE_MODELS}


def get_status() -> Dict[str, Any]:
    """获取安装状态。models 为全部候选模型（转录/注入共用一份候选）。"""
    venv_python = venv_python_path(VENV_DIR)
    venv_ok = VENV_DIR.is_dir() and venv_python.exists()
    current = get_current_model()
    installed = get_installed_models()
    all_models = [{**meta, "key": key} for key, meta in TRANSCRIBE_MODELS.items()]
    return {
        "dir_exists": TRANSCRIBE_DIR.is_dir(),
        "venv_exists": venv_ok,
        "model_exists": installed.get(current, False),
        "ready": venv_ok and installed.get(current, False),
        "dir_path": str(TRANSCRIBE_DIR),
        "models": all_models,
        "installed": installed,
        "current_model": current,
        "inject_model": get_inject_model(),
    }


def inject_ready() -> bool:
    """辅助重命名注入是否就绪。"""
    venv_python = venv_python_path(VENV_DIR)
    return VENV_DIR.is_dir() and venv_python.exists() and is_model_installed(get_inject_model())


def _download_model(model_key: str,
                    log_fn: Optional[Callable[[str], None]] = None,
                    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
                    cancel_event: Optional[threading.Event] = None,
                    cleanup_on_cancel: bool = False) -> Dict[str, Any]:
    """下载指定模型 GGUF。"""
    meta = TRANSCRIBE_MODELS.get(model_key)
    if not meta:
        return {"ok": False, "error": f"未知模型: {model_key}"}

    repo = meta["hf_repo"]
    from . import models_downloader
    if not models_downloader.begin_download():
        return {"ok": False, "busy": True, "error": "已有下载任务进行中，请稍后再试"}
    log_fn = log_fn or (lambda s: None)
    cancel = cancel_event or models_downloader.get_cancel_event()
    try:
        log_fn(f"模型 {meta['title']}：{meta['gguf']}（{meta['size_label']}）")
        return models_downloader.download_files(
            "hf", repo, [meta["gguf"]], str(MODEL_DIR),
            log_fn=log_fn, progress_cb=progress_cb,
            cancel_event=cancel,
            cleanup_on_cancel=cleanup_on_cancel,
        )
    except models_downloader.DownloadCancelled:
        return {"ok": False, "cancelled": True, "error": "已取消"}
    except Exception as e:
        return {"ok": False, "error": f"下载模型失败: {e}"}
    finally:
        models_downloader.end_download()


def install_dependencies(pypi_mirror: str = DEFAULT_PYPI_MIRROR,
                         model: str = DEFAULT_MODEL,
                         inject_model: str = DEFAULT_INJECT_MODEL,
                         log_fn: Optional[Callable[[str], None]] = None,
                         stop_event: Optional[threading.Event] = None) -> Dict[str, Any]:
    """安装 transcribe.cpp 全部依赖（model=右键转录模型，inject_model=辅助重命名模型）。"""
    pypi_url = PYPI_MIRRORS.get(pypi_mirror, PYPI_MIRRORS[DEFAULT_PYPI_MIRROR])["url"]
    TRANSCRIBE_DIR.mkdir(parents=True, exist_ok=True)
    unknown = [k for k in (model, inject_model) if k not in TRANSCRIBE_MODELS]
    if unknown:
        return {"ok": False, "error": f"未知模型: {', '.join(unknown)}"}
    # 去重：两处选同一模型只下载一次
    models_to_install = list(dict.fromkeys([model, inject_model]))
    steps = InstallSteps(4)

    def _cancelled() -> Optional[Dict[str, Any]]:
        if stop_event is not None and stop_event.is_set():
            _log("安装已取消", log_fn)
            return {"ok": False, "cancelled": True, "error": "安装已取消"}
        return None

    # ── 步骤 1 ──
    steps.push(1)
    _log("━━ 步骤 1/4：检测 UV ━━", log_fn)
    uv = _ensure_uv(pypi_url, log_fn, stop_event)
    r = _cancelled()
    if r:
        return r
    if not uv:
        return {"ok": False, "error": "UV 安装失败"}

    # ── 步骤 2 ──
    steps.push(2)
    _log("━━ 步骤 2/4：创建虚拟环境 ━━", log_fn)
    if not VENV_DIR.is_dir():
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

    # ── 步骤 3：安装引擎 ──
    steps.push(3)
    _log("━━ 步骤 3/4：安装引擎 ━━", log_fn)
    rc, out = _run_subprocess_streaming(
        [uv, "pip", "install", "--python", venv_py,
         "transcribe_cpp", "numpy", "soundfile",
         "--index-url", pypi_url],
        _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
    )
    r = _cancelled()
    if r:
        return r
    if rc != 0:
        return {"ok": False, "error": f"安装引擎失败: {out[-500:]}"}

    # ── 步骤 4：下载模型 ──
    steps.push(4)
    if not vad_ready():
        _log("  ── Silero VAD ──", log_fn)
        if not _download_vad(log_fn=log_fn, stop_event=stop_event):
            r = _cancelled()
            if r:
                return r
            return {"ok": False, "error": "Silero VAD 模型下载失败"}
    todo = [k for k in models_to_install if not is_model_installed(k)]
    if not todo:
        _log("━━ 步骤 4/4：模型已存在，跳过下载 ━━", log_fn)
    else:
        names = " + ".join(TRANSCRIBE_MODELS[k]["title"] for k in todo)
        _log(f"━━ 步骤 4/4：下载模型（{names}）━━", log_fn)
        last_pct10 = {"v": -1}
        n_total = len(todo)

        for n, mk in enumerate(todo):
            meta = TRANSCRIBE_MODELS[mk]
            _log(f"  ── {meta['title']}（{meta['size_label']}）──", log_fn)

            def _install_progress(ev: Dict[str, Any]):
                if ev.get("type") == "progress" and ev.get("total") and ev.get("pct") is not None:
                    pct10 = int(ev["pct"] * 100 // 5)
                    if pct10 > last_pct10["v"]:
                        last_pct10["v"] = pct10
                        _log(f"  {Path(ev['file']).name} {int(ev['pct'] * 100)}%"
                             f"（{fmt_size(ev['done'])}/{fmt_size(ev['total'])}）", log_fn)
                    frac = ((ev.get("idx", 1) - 1) + ev["pct"]) / max(1, ev.get("count", 1))
                    steps.push(4, (n + frac) / n_total)
                elif ev.get("type") in ("file_done", "file_skip"):
                    frac = ev.get("idx", 0) / max(1, ev.get("count", 1))
                    steps.push(4, (n + frac) / n_total)

            dl = _download_model(mk, log_fn=log_fn, progress_cb=_install_progress,
                                 cancel_event=stop_event, cleanup_on_cancel=True)
            r = _cancelled()
            if r:
                return r
            if dl.get("cancelled"):
                return {"ok": False, "cancelled": True, "error": "模型下载已取消"}
            if not dl.get("ok"):
                return {"ok": False, "error": f"模型下载失败: {dl.get('error') or '未知错误'}"}
    try:
        from .config_store import update_transcribe_config
        update_transcribe_config(lambda c: c.update(model=model, inject_model=inject_model))
        _log(f"当前模型已设为: {TRANSCRIBE_MODELS[model]['title']}"
             f"（辅助重命名: {TRANSCRIBE_MODELS[inject_model]['title']}）", log_fn)
    except Exception as e:
        _log(f"设置当前模型失败（忽略）: {e}", log_fn)
    _log("━━ 全部完成 ━━", log_fn)
    return {"ok": True, "error": None}


def remove_transcribe() -> Dict[str, Any]:
    """删除 transcribe 文件夹与模块数据。"""
    existed = TRANSCRIBE_DIR.exists()
    if existed:
        try:
            shutil.rmtree(TRANSCRIBE_DIR)
        except Exception as e:
            return {"ok": False, "error": f"删除失败: {e}"}
    # 模块配置与转录结果一并删除
    from .workspace_paths import TRANSCRIBE_CONFIG_FILE
    shutil.rmtree(TRANSCRIBE_CONFIG_FILE.parent, ignore_errors=True)
    return {"ok": True,
            "message": "已删除 transcribe 文件夹" if existed else "目录不存在，无需删除"}


def _batch_timeout(video_paths: List[str]) -> float:
    """按 ffprobe 时长累加的整批超时预算。"""
    from .discovery import probe_video
    total = _TIMEOUT_FLAT_SEC
    for vp in video_paths:
        dur = probe_video(vp).get("duration") or 0
        total += dur * _TIMEOUT_PER_VIDEO_SEC if dur else _TIMEOUT_NO_DURATION_SEC
    return total


def _run_batch_script(params: Dict[str, Any], per_video: List[Dict[str, Any]],
                      on_video_done: Optional[Callable[[int, Dict[str, Any]], None]],
                      on_log: Optional[Callable[[str], None]],
                      stop_event, err_label: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """运行转录子进程并解析逐行 JSON 输出。返回 (error, items)，error=None 表示成功。

    临时目录由父进程持有（work_dir 传入脚本）：子进程被杀后的残留由 finally 统一清理。"""

    def _on_line(line: str):
        line = line.strip()
        if not line:
            return
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return
        if "idx" in obj:
            idx = obj["idx"]
            entry = {k: v for k, v in obj.items() if k != "idx"}
            while len(per_video) <= idx:
                per_video.append({"ok": False, "error": "缺失"})
            per_video[idx] = entry
            if on_video_done:
                on_video_done(idx, entry)
        elif obj.get("log") and on_log:
            on_log(obj["log"])

    work_dir = tempfile.mkdtemp(prefix="transcribe_job_")
    params["work_dir"] = work_dir
    try:
        timeout = _batch_timeout(params["video_paths"])
        rc, _last, err_tail = _run_venv_script(
            VENV_DIR, _WORKER_SCRIPT, params,
            timeout=timeout, stop_event=stop_event, on_line=_on_line,
        )

        if rc == RC_CANCEL:
            return "已取消", per_video
        if rc == RC_TIMEOUT:
            return f"{err_label}超时", per_video
        if rc != 0:
            return f"{err_label}失败: {err_tail[-500:] if err_tail else '子进程异常退出'}", per_video
        if per_video:
            return None, per_video
        return f"{err_label}无输出", []
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def run_transcription_batch(video_paths: List[str],
                            language: str = "",
                            on_video_done: Optional[Callable[[int, Dict[str, Any]], None]] = None,
                            on_log: Optional[Callable[[str], None]] = None,
                            stop_event=None) -> Dict[str, Any]:
    """对多个视频一次性转录。

    Returns: {"ok": bool, "per_video": [...], "error": str|None}
    """
    current = get_current_model()
    venv_python = venv_python_path(VENV_DIR)
    if not (VENV_DIR.is_dir() and venv_python.exists()):
        return {"ok": False, "per_video": [], "error": "语音转录未安装或未就绪"}
    model_path = _model_path(current)
    if not model_path:
        return {"ok": False, "per_video": [],
                "error": f"当前模型（{TRANSCRIBE_MODELS[current]['title']}）未下载，请先在扩展功能页下载"}
    if not FFMPEG_EXE:
        return {"ok": False, "per_video": [], "error": "未找到 ffmpeg，无法提取音频"}

    vad = vad_model_path()
    params = {
        "video_paths": video_paths,
        "model_path": str(model_path),
        "family": TRANSCRIBE_MODELS[current]["family"],
        "timestamps": bool(TRANSCRIBE_MODELS[current].get("timestamps", True)),
        "inject_win_sec": INJECT_WIN_SEC[TRANSCRIBE_MODELS[current]["family"]],
        "language": language,
        "ffmpeg": FFMPEG_EXE,
        "venv_site_packages": str(VENV_DIR / "Lib" / "site-packages"),
        # 官方 Silero VAD（缺失则整段推理/不做静音判断）
        "vad_model": str(vad) if vad else "",
        "mode": "srt",
    }

    per_video: List[Dict[str, Any]] = []
    error, items = _run_batch_script(params, per_video, on_video_done, on_log,
                                     stop_event, "转录")
    if error:
        return {"ok": False, "per_video": per_video, "error": error}
    return {"ok": True, "per_video": items, "error": None}


def run_inject_batch(video_paths: List[str],
                     max_chars: int = 800,
                     on_log: Optional[Callable[[str], None]] = None,
                     stop_event=None,
                     on_video_done: Optional[Callable[[int, Dict[str, Any]], None]] = None
                     ) -> Dict[str, Any]:
    """注入文本批量推理，达到 max_chars 提前停止，不落盘。

    on_video_done(idx, entry)：单视频完成即回调（同一子进程批内逐视频交付，
    供重命名流水线在整批结束前拿到已完成视频的注入文本）。
    Returns: {"ok": bool, "texts": {video_path: text}, "error": str|None}
    """
    if not inject_ready():
        return {"ok": False, "texts": {}, "error": "注入模型未安装或未就绪"}
    if not FFMPEG_EXE:
        return {"ok": False, "texts": {}, "error": "未找到 ffmpeg，无法提取音频"}

    inject_key = get_inject_model()
    vad_path = vad_model_path()
    params = {
        "video_paths": video_paths,
        "model_path": str(_model_path(inject_key)),
        "family": TRANSCRIBE_MODELS[inject_key]["family"],
        "language": "",
        "ffmpeg": FFMPEG_EXE,
        "venv_site_packages": str(VENV_DIR / "Lib" / "site-packages"),
        "max_chars": int(max_chars),
        "inject_win_sec": INJECT_WIN_SEC[TRANSCRIBE_MODELS[inject_key]["family"]],
        # 官方 Silero VAD（缺失则脚本内整段推理，静音不跳过）
        "vad_model": str(vad_path) if vad_path else "",
        "mode": "text",
    }

    per_video: List[Dict[str, Any]] = []
    error, items = _run_batch_script(params, per_video, on_video_done, on_log,
                                     stop_event, "推理")
    if error:
        return {"ok": False, "texts": {}, "error": error}

    texts: Dict[str, str] = {}
    for idx, vp in enumerate(video_paths):
        tr = items[idx] if idx < len(items) else {"ok": False}
        if tr.get("ok") and tr.get("text"):
            texts[vp] = tr["text"]
    return {"ok": True, "texts": texts, "error": None}


# ── 转录 worker（真实文件：可 lint/可单测，venv 内按路径执行）──
_WORKER_SCRIPT = Path(__file__).resolve().parent / "transcribe_worker.py"
