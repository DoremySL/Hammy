"""transcribe.cpp 语音转录：安装、GGUF 模型管理，对视频转录输出 SRT 字幕。"""
from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from batch_rename.utils import fmt_size
from .env import APP_ROOT, PYTHON_EXE, FFMPEG_EXE, make_logger
from .installer import (
    PYPI_MIRRORS, DEFAULT_PYPI_MIRROR, UV_TIMEOUT_SEC,
    RC_CANCEL, RC_TIMEOUT,
    detect_gpu,
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
WHEELS_DIR = TRANSCRIBE_DIR / "wheels"  # 手动投放的引擎 wheel

# ── 内置模型元数据 ──
TRANSCRIBE_MODELS = {
    "whisper-turbo": {
        "title": "whisper-turbo",
        "family": "whisper",
        "gguf": "whisper-large-v3-turbo-Q8_0.gguf",
        "hf_repo": "handy-computer/whisper-large-v3-turbo-gguf",
        "ms_repo": "",  # 待填入魔搭仓库
        "desc": "多语言快速转录（99 语种），速度最快",
        "size_label": "~850MB",
        "recommended": True,
    },
    "moss": {
        "title": "MOSS-Diarize",
        "family": "moss",
        "gguf": "MOSS-Transcribe-Diarize-Q8_0.gguf",
        "hf_repo": "handy-computer/moss-transcribe-diarize-gguf",
        "ms_repo": "",  # 待填入魔搭仓库
        "desc": "中日英转录 + 说话人分离，适合影视字幕",
        "size_label": "~1GB",
        "recommended": False,
    },
    "sensevoice": {
        "title": "SenseVoice-Small",
        "family": "sensevoice",
        "gguf": "SenseVoiceSmall-Q8_0.gguf",
        "hf_repo": "handy-computer/SenseVoiceSmall-gguf",
        "ms_repo": "",  # 待填入魔搭仓库
        "desc": "最快档：中英日韩粤，自带标点（无逐句时间戳，不作字幕模型）",
        "size_label": "~240MB",
        "recommended": False,
    },
}
DEFAULT_MODEL = "whisper-turbo"
DEFAULT_INJECT_MODEL = "sensevoice"

# 安装组合：字幕主模型 + 注入模型
INSTALL_BUNDLES = {
    "whisper+sv": ["whisper-turbo", "sensevoice"],
    "moss+sv": ["moss", "sensevoice"],
}
DEFAULT_BUNDLE = "whisper+sv"

# 注入模式分窗时长（秒）
INJECT_WIN_SEC = {"sensevoice": 30.0, "whisper": 60.0, "moss": 300.0}

# ── 引擎 wheel ──
GITHUB_RELEASE_API = "https://api.github.com/repos/handy-computer/transcribe.cpp/releases?per_page=5"
_BINDING_WHEEL_RE = re.compile(r"^transcribe_cpp-(\d+\.\d+\.\d+)-py3-none-any\.whl$")
_NATIVE_WHEEL_RE = re.compile(r"^transcribe_cpp_native_cu12-(\d+\.\d+\.\d+)-py3-none-win_amd64\.whl$")

_INSTALL_TIMEOUT_SEC = 600.0
_TRANSCRIBE_TIMEOUT_PER_VIDEO_SEC = 600.0

_log = make_logger("transcribe_cpp")


def _model_candidates(model_key: str) -> List[Path]:
    """模型 GGUF 的候选磁盘路径。"""
    meta = TRANSCRIBE_MODELS.get(model_key)
    if not meta:
        return []
    out = []
    for repo in (meta.get("ms_repo"), meta.get("hf_repo")):
        if repo:
            author, _, name = repo.partition("/")
            out.append(MODEL_DIR / author / name / meta["gguf"])
    return out


def _model_path(model_key: str) -> Optional[Path]:
    """已安装模型 GGUF 的实际路径。"""
    for p in _model_candidates(model_key):
        if p.is_file():
            return p
    return None


def is_model_installed(model_key: str) -> bool:
    """模型完整性检测。"""
    return _model_path(model_key) is not None


def get_current_model() -> str:
    """字幕模型 key。"""
    try:
        from .config_store import load_transcribe_config
        key = load_transcribe_config().get("model") or DEFAULT_MODEL
    except Exception:
        key = DEFAULT_MODEL
    return key if key in TRANSCRIBE_MODELS else DEFAULT_MODEL


def get_inject_model() -> str:
    """辅助重命名模型 key。"""
    try:
        from .config_store import load_transcribe_config
        key = load_transcribe_config().get("inject_model") or DEFAULT_INJECT_MODEL
    except Exception:
        key = DEFAULT_INJECT_MODEL
    return key if key in TRANSCRIBE_MODELS else DEFAULT_INJECT_MODEL


def get_installed_models() -> Dict[str, bool]:
    """扫描全部内置模型的安装状态 {key: bool}。"""
    return {key: is_model_installed(key) for key in TRANSCRIBE_MODELS}


def get_status() -> Dict[str, Any]:
    """获取安装状态。models=字幕模型，inject_models=注入模型候选。"""
    venv_python = venv_python_path(VENV_DIR)
    venv_ok = VENV_DIR.is_dir() and venv_python.exists()
    current = get_current_model()
    subtitle = {k: m for k, m in TRANSCRIBE_MODELS.items()
                if m.get("family") in ("whisper", "moss")}
    installed = get_installed_models()
    return {
        "dir_exists": TRANSCRIBE_DIR.is_dir(),
        "venv_exists": venv_ok,
        "model_exists": installed.get(current, False),
        "ready": venv_ok and installed.get(current, False),
        "dir_path": str(TRANSCRIBE_DIR),
        # 模型元数据
        "models": [{**meta, "key": key} for key, meta in subtitle.items()],
        "inject_models": [{**meta, "key": key} for key, meta in TRANSCRIBE_MODELS.items()],
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

    source = "ms" if meta.get("ms_repo") else "hf"
    repo = meta.get("ms_repo") or meta.get("hf_repo")
    from . import models_downloader
    if not models_downloader.begin_download():
        return {"ok": False, "busy": True, "error": "已有下载任务进行中，请稍后再试"}
    log_fn = log_fn or (lambda s: None)
    cancel = cancel_event or models_downloader.get_cancel_event()
    try:
        log_fn(f"模型 {meta['title']}：{meta['gguf']}（{meta['size_label']}，"
               f"{'魔搭' if source == 'ms' else 'hf-mirror'}）")
        return models_downloader.download_files(
            source, repo, [meta["gguf"]], str(MODEL_DIR),
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


def _fetch_release_wheels(log_fn: Callable[[str], None],
                          stop_event: Optional[threading.Event]) -> Dict[str, Dict[str, Any]]:
    """从最新 release 解析引擎 wheel。"""
    from .llama_cpp import _fetch_json
    releases = _fetch_json(GITHUB_RELEASE_API, cancel_event=stop_event)
    if not isinstance(releases, list):
        raise RuntimeError("GitHub release 响应异常")
    for rel in releases:
        if rel.get("draft"):
            continue
        binding = native = None
        for a in rel.get("assets", []):
            name = a.get("name", "")
            if not binding and _BINDING_WHEEL_RE.match(name):
                binding = a
            elif not native and _NATIVE_WHEEL_RE.match(name):
                native = a
        if binding and native:
            log_fn(f"使用 release {rel.get('tag_name')} 的引擎 wheel")
            return {"binding": binding, "native": native}
    raise RuntimeError("release 中未找到引擎 wheel（transcribe_cpp + transcribe_cpp_native_cu12）")


def _local_wheels() -> Optional[List[Path]]:
    """本地手动投放的引擎 wheel。"""
    if not WHEELS_DIR.is_dir():
        return None
    binding = sorted(WHEELS_DIR.glob("transcribe_cpp-*-py3-none-any.whl"))
    native = sorted(WHEELS_DIR.glob("transcribe_cpp_native_cu12-*-py3-none-win_amd64.whl"))
    if binding and native:
        return [binding[-1], native[-1]]
    return None


def _download_engine_wheels(proxy: str, log_fn: Callable[[str], None],
                            stop_event: Optional[threading.Event],
                            progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[Path]:
    """下载引擎 wheel 到 wheels/，已有本地 wheel 直接使用。"""
    local = _local_wheels()
    if local:
        log_fn("使用本地引擎 wheel：" + "、".join(p.name for p in local))
        return local

    from .llama_cpp import _proxy_url
    from . import models_downloader
    assets = _fetch_release_wheels(log_fn, stop_event)
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for key in ("binding", "native"):
        a = assets[key]
        items.append({"url": _proxy_url(proxy, a["browser_download_url"]),
                      "filename": a["name"], "size": a.get("size", 0),
                      "label": a["name"]})
    r = models_downloader.download_urls_parallel(items, str(WHEELS_DIR),
                                                 log_fn=log_fn,
                                                 progress_cb=progress_cb,
                                                 cancel_event=stop_event)
    if not r.get("ok"):
        failed = "; ".join(f"{f.get('file')}: {f.get('error')}" for f in r.get("failed", []))
        raise RuntimeError("引擎 wheel 下载失败" + (f": {failed}" if failed else ""))
    return [Path(r["dir"]) / it["filename"] for it in items]


def install_dependencies(pypi_mirror: str = DEFAULT_PYPI_MIRROR,
                         bundle: str = DEFAULT_BUNDLE,
                         github_proxy: str = "",
                         log_fn: Optional[Callable[[str], None]] = None,
                         stop_event: Optional[threading.Event] = None) -> Dict[str, Any]:
    """安装 transcribe.cpp 全部依赖。"""
    pypi_url = PYPI_MIRRORS.get(pypi_mirror, PYPI_MIRRORS[DEFAULT_PYPI_MIRROR])["url"]
    TRANSCRIBE_DIR.mkdir(parents=True, exist_ok=True)
    models_to_install = INSTALL_BUNDLES.get(bundle, INSTALL_BUNDLES[DEFAULT_BUNDLE])
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
    has_nvidia = bool(detect_gpu().get("has_nvidia"))
    _lf = (lambda m: _log(m, log_fn)) if log_fn else _log
    if has_nvidia:
        _log("━━ 步骤 3/4：安装引擎（CUDA，GitHub release）━━", log_fn)

        def _engine_progress(ev: Dict[str, Any]):
            if ev.get("type") == "progress" and ev.get("total") and ev.get("pct") is not None:
                steps.push(3, ((ev.get("idx", 1) - 1) + ev["pct"]) / max(1, ev.get("count", 1)))
            elif ev.get("type") in ("file_done", "file_skip"):
                steps.push(3, ev.get("idx", 0) / max(1, ev.get("count", 1)))

        wheels = _download_engine_wheels(github_proxy, _lf, stop_event,
                                         progress_cb=_engine_progress)
        r = _cancelled()
        if r:
            return r
        rc, out = _run_subprocess_streaming(
            [uv, "pip", "install", "--python", venv_py,
             *[str(p) for p in wheels], "numpy", "soundfile",
             "--index-url", pypi_url],
            _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
        )
    else:
        _log("━━ 步骤 3/4：安装引擎（CPU/Vulkan，PyPI）━━", log_fn)
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
    # 安装成功后清理临时 wheel 目录
    shutil.rmtree(WHEELS_DIR, ignore_errors=True)

    # ── 步骤 4：下载模型 ──
    steps.push(4)
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
            src_label = "魔搭" if meta.get("ms_repo") else "hf-mirror"
            _log(f"  ── {meta['title']}（{src_label}，{meta['size_label']}）──", log_fn)

            def _install_progress(ev: Dict[str, Any]):
                # 下载同步执行，闭包读到的是当前循环的 n
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
        update_transcribe_config(lambda c: c.update(model=models_to_install[0]))
        _log(f"当前模型已设为: {TRANSCRIBE_MODELS[models_to_install[0]]['title']}", log_fn)
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
    # 模块配置/开关/转录结果一并删除
    try:
        from .workspace_paths import TRANSCRIBE_CONFIG_FILE
        shutil.rmtree(TRANSCRIBE_CONFIG_FILE.parent, ignore_errors=True)
    except Exception:
        pass
    return {"ok": True,
            "message": "已删除 transcribe 文件夹" if existed else "目录不存在，无需删除"}


def run_transcription_batch(video_paths: List[str],
                            language: str = "",
                            on_video_done: Optional[Callable[[int, Dict[str, Any]], None]] = None,
                            on_log: Optional[Callable[[str], None]] = None,
                            stop_event=None) -> Dict[str, Any]:
    """对多个视频一次性转录。

    Returns: {"ok": bool, "per_video": [...], "error": str|None}
    """
    current = get_current_model()
    if TRANSCRIBE_MODELS[current].get("family") not in ("whisper", "moss"):
        return {"ok": False, "per_video": [],
                "error": "当前模型不支持字幕转录（仅 whisper-turbo / MOSS 可用）"}
    status = get_status()
    if not status["ready"]:
        return {"ok": False, "per_video": [], "error": "语音转录未安装或未就绪"}

    model_path = _model_path(current)
    if not model_path:
        return {"ok": False, "per_video": [],
                "error": f"当前模型（{TRANSCRIBE_MODELS[current]['title']}）未下载，请先在扩展功能页下载"}
    if not FFMPEG_EXE:
        return {"ok": False, "per_video": [], "error": "未找到 ffmpeg，无法提取音频"}

    backend = "cuda" if detect_gpu().get("has_nvidia") else "cpu"
    params = {
        "video_paths": video_paths,
        "model_path": str(model_path),
        "family": TRANSCRIBE_MODELS[current]["family"],
        "language": language,
        "backend": backend,
        "ffmpeg": FFMPEG_EXE,
        "venv_site_packages": str(VENV_DIR / "Lib" / "site-packages"),
        "mode": "srt",
    }

    per_video: List[Dict[str, Any]] = []
    final_result: Optional[Dict[str, Any]] = None

    def _on_line(line: str):
        nonlocal final_result
        line = line.strip()
        if not line:
            return
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return
        if obj.get("done"):
            final_result = obj
        elif "idx" in obj:
            idx = obj["idx"]
            entry = {k: v for k, v in obj.items() if k != "idx"}
            while len(per_video) <= idx:
                per_video.append({"ok": False, "srt": "", "error": "缺失"})
            per_video[idx] = entry
            if on_video_done:
                on_video_done(idx, entry)
        elif obj.get("log") and on_log:
            on_log(obj["log"])

    timeout = _TRANSCRIBE_TIMEOUT_PER_VIDEO_SEC * max(1, len(video_paths))
    rc, _last, err_tail = _run_venv_script(
        VENV_DIR, _TRANSCRIBE_SCRIPT, params,
        timeout=timeout, stop_event=stop_event, on_line=_on_line,
    )

    if rc == RC_CANCEL:
        return {"ok": False, "per_video": per_video, "error": "已取消"}
    if rc == RC_TIMEOUT:
        return {"ok": False, "per_video": per_video, "error": "转录超时"}
    if rc != 0:
        return {"ok": False, "per_video": per_video,
                "error": f"转录失败: {err_tail[-500:] if err_tail else '子进程异常退出'}"}
    if final_result:
        return {"ok": True, "per_video": final_result.get("per_video", per_video), "error": None}
    if per_video:
        return {"ok": True, "per_video": per_video, "error": None}
    return {"ok": False, "per_video": [], "error": "转录无输出"}


def run_inject_batch(video_paths: List[str],
                     max_chars: int = 800,
                     on_log: Optional[Callable[[str], None]] = None,
                     stop_event=None) -> Dict[str, Any]:
    """注入文本批量推理，达到 max_chars 提前停止，不落盘。

    Returns: {"ok": bool, "texts": {video_path: text}, "error": str|None}
    """
    if not video_paths:
        return {"ok": True, "texts": {}, "error": None}
    if not inject_ready():
        return {"ok": False, "texts": {}, "error": "注入模型未安装或未就绪"}
    if not FFMPEG_EXE:
        return {"ok": False, "texts": {}, "error": "未找到 ffmpeg，无法提取音频"}

    inject_key = get_inject_model()
    backend = "cuda" if detect_gpu().get("has_nvidia") else "cpu"
    params = {
        "video_paths": video_paths,
        "model_path": str(_model_path(inject_key)),
        "family": TRANSCRIBE_MODELS[inject_key]["family"],
        "language": "",
        "backend": backend,
        "ffmpeg": FFMPEG_EXE,
        "venv_site_packages": str(VENV_DIR / "Lib" / "site-packages"),
        "max_chars": max(0, int(max_chars or 0)),
        "mode": "text",
    }

    texts: Dict[str, str] = {}
    per_video: List[Dict[str, Any]] = []
    final_result: Optional[Dict[str, Any]] = None

    def _on_line(line: str):
        nonlocal final_result
        line = line.strip()
        if not line:
            return
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return
        if obj.get("done"):
            final_result = obj
        elif "idx" in obj:
            idx = obj["idx"]
            entry = {k: v for k, v in obj.items() if k != "idx"}
            while len(per_video) <= idx:
                per_video.append({"ok": False, "text": "", "error": "缺失"})
            per_video[idx] = entry
        elif obj.get("log") and on_log:
            on_log(obj["log"])

    timeout = _TRANSCRIBE_TIMEOUT_PER_VIDEO_SEC * max(1, len(video_paths))
    rc, _last, err_tail = _run_venv_script(
        VENV_DIR, _TRANSCRIBE_SCRIPT, params,
        timeout=timeout, stop_event=stop_event, on_line=_on_line,
    )

    if rc == RC_CANCEL:
        return {"ok": False, "texts": {}, "error": "已取消"}
    if rc == RC_TIMEOUT:
        return {"ok": False, "texts": {}, "error": "推理超时"}
    if rc != 0:
        return {"ok": False, "texts": {},
                "error": f"推理失败: {err_tail[-500:] if err_tail else '子进程异常退出'}"}

    items = (final_result or {}).get("per_video", per_video)
    for idx, vp in enumerate(video_paths):
        tr = items[idx] if idx < len(items) else {"ok": False, "text": "", "error": "无结果"}
        if tr.get("ok") and tr.get("text"):
            texts[vp] = tr["text"]
    return {"ok": True, "texts": texts, "error": None}


# ── 转录脚本 ──
_TRANSCRIBE_SCRIPT = r"""
import sys, json, os, subprocess, tempfile

params_file = sys.argv[1]
with open(params_file, 'r', encoding='utf-8') as f:
    params = json.load(f)

video_paths = params['video_paths']
model_path = params['model_path']
family = params.get('family', 'whisper')
language = params.get('language', '') or None
backend = params.get('backend', 'cpu')
ffmpeg = params.get('ffmpeg', '')
site_packages = params.get('venv_site_packages', '')
max_chars = int(params.get('max_chars', 0) or 0)  # 注入模式的提前停止阈值
mode = params.get('mode', 'srt')  # srt=字幕转录 / text=注入文本

FAMILY_CHUNK_SEC = {'moss': 1800}
INJECT_WIN_SEC = {'sensevoice': 30.0, 'whisper': 60.0, 'moss': 300.0}

def plog(msg):
    print(json.dumps({'log': msg}, ensure_ascii=False), flush=True)

# nvidia pip 包的 CUDA DLL 注入 PATH
if site_packages:
    for sub in ('cublas', 'cuda_runtime', 'cuda_nvrtc', 'nvjitlink'):
        dll_dir = os.path.join(site_packages, 'nvidia', sub, 'bin')
        if os.path.isdir(dll_dir):
            os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')

import transcribe_cpp as tc
import soundfile as sf

# 按后端优先级初始化，失败逐级回退
backends = ['cuda', 'cpu'] if backend == 'cuda' else ['vulkan', 'cpu']
model = None
for be in backends:
    try:
        model = tc.Model(model_path, backend=be)
        plog(f'引擎后端: {be}')
        break
    except Exception as e:
        plog(f'{be} 初始化失败: {str(e).splitlines()[0]}')
if model is None:
    sys.exit(1)
sess = model.session()

# whisper/moss 出 SRT；sensevoice 出纯文本
run_kw = {}
if family in ('whisper', 'moss'):
    run_kw['timestamps'] = 'segment'
if family == 'moss':
    run_kw['diarize'] = 'on'
if language:
    run_kw['language'] = language


def _srt_ts(ms):
    h, rem = divmod(int(ms) // 1000, 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d},{int(ms) % 1000:03d}'


def _to_srt(segments, offset_ms=0):
    lines = []
    idx = 1
    for seg in segments:
        t = (getattr(seg, 'text', '') or '').strip()
        if not t:
            continue
        lines.append(str(idx))
        lines.append(f'{_srt_ts(seg.t0_ms + offset_ms)} --> {_srt_ts(seg.t1_ms + offset_ms)}')
        lines.append(t)
        lines.append('')
        idx += 1
    return '\n'.join(lines)


def _extract_wav(vp, out_wav):
    cmd = [ffmpeg, '-y', '-v', 'error', '-i', vp,
           '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', out_wav]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'ffmpeg 提取音频失败: {(r.stderr or "")[-300:]}')


def transcribe_one(i, vp):
    try:
        with tempfile.TemporaryDirectory(prefix='transcribe_') as td:
            full = os.path.join(td, 'a.wav')
            _extract_wav(vp, full)
            audio, sr = sf.read(full, dtype='float32', always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            dur = len(audio) / sr
            if mode == 'text':
                # 注入模式：分窗推理，达标即停
                win = INJECT_WIN_SEC.get(family, 30.0)
                texts = []
                acc = 0
                pos = 0.0
                while pos < dur - 0.1:
                    seg_audio = audio[int(pos * sr): int((pos + win) * sr)]
                    res = sess.run(seg_audio, **run_kw)
                    t = ' '.join(s.text.strip() for s in (res.segments or [])
                                 if s.text.strip()) or (res.text or '').strip()
                    if t:
                        texts.append(t)
                        acc += len(t) + 1
                    if max_chars and acc >= max_chars:
                        break
                    pos += win
                entry = {'ok': True, 'text': ' '.join(texts), 'srt': '',
                         'language': '', 'duration': round(dur, 1), 'error': None}
            else:
                chunk = FAMILY_CHUNK_SEC.get(family, 0)
                if chunk and dur <= chunk + 0.5:
                    chunk = 0
                if not chunk:
                    res = sess.run(audio, **run_kw)
                    srt = _to_srt(res.segments)
                    lang = getattr(res, 'language', '') or ''
                else:
                    parts = []
                    lang = ''
                    pos = 0.0
                    while pos < dur - 0.1:
                        seg_audio = audio[int(pos * sr): int((pos + chunk) * sr)]
                        res = sess.run(seg_audio, **run_kw)
                        parts.append(_to_srt(res.segments, offset_ms=int(pos * 1000)))
                        lang = lang or (getattr(res, 'language', '') or '')
                        pos += chunk
                    srt = '\n'.join(parts)
                entry = {'ok': True, 'srt': srt, 'language': lang, 'text': '',
                         'duration': round(dur, 1), 'error': None}
    except Exception as e:
        entry = {'ok': False, 'srt': '', 'text': '', 'language': '', 'duration': 0, 'error': str(e)}
    print(json.dumps({'idx': i, **entry}, ensure_ascii=False), flush=True)
    return entry

per_video = [transcribe_one(i, vp) for i, vp in enumerate(video_paths)]

# 最终汇总行
print(json.dumps({'done': True, 'per_video': per_video}, ensure_ascii=False), flush=True)
"""
