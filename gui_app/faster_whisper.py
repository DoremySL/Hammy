"""Faster-Whisper 语音转录：安装、CTranslate2 模型管理，对视频转录输出 SRT 字幕。"""
from __future__ import annotations

import json
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .env import APP_ROOT, PYTHON_EXE, make_logger
from .installer import (
    PYPI_MIRRORS, DEFAULT_PYPI_MIRROR, UV_TIMEOUT_SEC,
    RC_CANCEL, RC_TIMEOUT,
    detect_gpu,
    ensure_uv as _ensure_uv,
    venv_python_path,
    run_subprocess_streaming as _run_subprocess_streaming,
    run_venv_script as _run_venv_script,
)

# ── 路径常量 ──
WHISPER_DIR = APP_ROOT / "faster-whisper"
VENV_DIR = WHISPER_DIR / "venv"
# hf 下载根目录，布局 models/<author>/<repo>/（与 llama 模型下载组织一致）
MODEL_DIR = WHISPER_DIR / "models"

# ── 内置模型元数据（全部 CTranslate2 格式，faster-whisper 直接加载目录）──
# key 为模型唯一标识，持久化到 whisper/config.json 的 "model" 键
WHISPER_MODELS = {
    "v3-turbo": {
        "title": "v3-turbo",
        "repo": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "desc": "平衡速度与精度，支持多种语言",
        "size_label": "~1.6GB",
        "recommended": True,
    },
    "large-v3": {
        "title": "large-v3",
        "repo": "Systran/faster-whisper-large-v3",
        "desc": "牺牲速度追求精度，支持多种语言",
        "size_label": "~3.0GB",
        "recommended": False,
    },
    "ja-1.5B": {
        "title": "ja-1.5B",
        "repo": "TransWithAI/whisper-ja-1.5B-ct2",
        "desc": "仅支持日语，用于转录高质量日文字幕",
        "size_label": "~3.0GB",
        "recommended": False,
    },
}
DEFAULT_MODEL = "v3-turbo"
# 仅支持日语、转录时强制语言的模型
JA_ONLY_MODELS = frozenset({"ja-1.5B"})

_INSTALL_TIMEOUT_SEC = 600.0
_TRANSCRIBE_TIMEOUT_PER_VIDEO_SEC = 600.0  # 单视频上限，按数量线性放大


_log = make_logger("faster_whisper")


def _fmt_size(num: float) -> str:
    """字节数 -> 人类可读（B/KB/MB/GB）。"""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}TB"


def _model_subdir(model_key: str) -> Optional[Path]:
    """模型在磁盘上的目录（hf 布局：models/<author>/<repo>/）。"""
    meta = WHISPER_MODELS.get(model_key)
    if not meta:
        return None
    author, _, name = meta["repo"].partition("/")
    return MODEL_DIR / author / name


def is_model_installed(model_key: str) -> bool:
    """模型完整性检测：目录存在且含 model.bin（CTranslate2 模型文件）。"""
    sub = _model_subdir(model_key)
    return bool(sub and sub.is_dir() and (sub / "model.bin").is_file())


def get_current_model() -> str:
    """当前使用模型 key（配置持久化；异常值/缺失回退默认）。"""
    try:
        from .config_store import load_whisper_config
        key = load_whisper_config().get("model") or DEFAULT_MODEL
    except Exception:
        key = DEFAULT_MODEL
    return key if key in WHISPER_MODELS else DEFAULT_MODEL


def get_installed_models() -> Dict[str, bool]:
    """扫描全部内置模型的安装状态 {key: bool}。"""
    return {key: is_model_installed(key) for key in WHISPER_MODELS}


def get_status() -> Dict[str, Any]:
    """获取 faster-whisper 安装状态（含模型元数据、安装状态与当前模型）。"""
    venv_python = venv_python_path(VENV_DIR)
    venv_ok = VENV_DIR.is_dir() and venv_python.exists()
    current = get_current_model()
    installed = get_installed_models()
    return {
        "dir_exists": WHISPER_DIR.is_dir(),
        "venv_exists": venv_ok,
        "model_exists": installed.get(current, False),
        "ready": venv_ok and installed.get(current, False),
        "dir_path": str(WHISPER_DIR),
        # 模型元数据（前端安装弹窗 / 面板下拉 / 管理弹窗共用）
        "models": [{**meta, "key": key} for key, meta in WHISPER_MODELS.items()],
        "installed": installed,
        "current_model": current,
    }


def _download_model(model_key: str,
                    log_fn: Optional[Callable[[str], None]] = None,
                    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
                    cancel_event: Optional[threading.Event] = None,
                    cleanup_on_cancel: bool = False) -> Dict[str, Any]:
    """下载指定模型（hf-mirror.com，全仓库文件）。"""
    meta = WHISPER_MODELS.get(model_key)
    if not meta:
        return {"ok": False, "error": f"未知模型: {model_key}"}

    from . import models_downloader
    if not models_downloader.begin_download():
        return {"ok": False, "busy": True, "error": "已有下载任务进行中，请稍后再试"}
    log_fn = log_fn or (lambda s: None)
    cancel = cancel_event or models_downloader.get_cancel_event()
    try:
        # 全仓库文件列表：CTranslate2 格式需要 model.bin / config.json /
        # vocabulary / tokenizer 等全部文件，缺一不可加载
        files = models_downloader.api_list_files("hf", meta["repo"], cancel_event=cancel)
        names = [f["path"] for f in files
                 if f.get("type") != "directory" and f.get("path")]
        if not names:
            return {"ok": False, "error": f"仓库 {meta['repo']} 无可用文件"}
        log_fn(f"模型 {meta['title']}：共 {len(names)} 个文件，开始下载…")
        return models_downloader.download_files(
            "hf", meta["repo"], names, str(MODEL_DIR),
            log_fn=log_fn, progress_cb=progress_cb,
            cancel_event=cancel,
            cleanup_on_cancel=cleanup_on_cancel,
        )
    except models_downloader.DownloadCancelled:
        # 文件列表拉取阶段取消（下载阶段的取消由 download_files 内部处理）
        return {"ok": False, "cancelled": True, "error": "已取消"}
    except Exception as e:
        return {"ok": False, "error": f"下载模型失败: {e}"}
    finally:
        models_downloader.end_download()


def install_dependencies(pypi_mirror: str = DEFAULT_PYPI_MIRROR,
                         model: str = DEFAULT_MODEL,
                         log_fn: Optional[Callable[[str], None]] = None,
                         stop_event: Optional[threading.Event] = None) -> Dict[str, Any]:
    """安装 faster-whisper 全部依赖。"""
    pypi_url = PYPI_MIRRORS.get(pypi_mirror, PYPI_MIRRORS[DEFAULT_PYPI_MIRROR])["url"]
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    model_key = model if model in WHISPER_MODELS else DEFAULT_MODEL

    def _cancelled() -> Optional[Dict[str, Any]]:
        """用户取消：返回取消结果；未取消返回 None。"""
        if stop_event is not None and stop_event.is_set():
            _log("安装已取消", log_fn)
            return {"ok": False, "cancelled": True, "error": "安装已取消"}
        return None

    # ── 步骤 1 ──
    _log("━━ 步骤 1/4：检测 UV ━━", log_fn)
    uv = _ensure_uv(pypi_url, log_fn, stop_event)
    r = _cancelled()
    if r:
        return r
    if not uv:
        return {"ok": False, "error": "UV 安装失败"}
    _log(f"UV 就绪: {uv}", log_fn)

    # ── 步骤 2 ──
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

    # ── 步骤 3 ──
    _log(f"━━ 步骤 3/4：安装依赖（镜像: {pypi_url}）━━", log_fn)
    _log("→ faster-whisper + ctranslate2…", log_fn)
    rc, out = _run_subprocess_streaming(
        [uv, "pip", "install", "--python", venv_py,
         "faster-whisper", "--index-url", pypi_url],
        _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
    )
    r = _cancelled()
    if r:
        return r
    if rc != 0:
        return {"ok": False, "error": f"安装 faster-whisper 失败: {out[-500:]}"}
    # ctranslate2 需要额外 CUDA 运行时（自带 cuDNN 但不含 cuBLAS）；
    # 无 NVIDIA GPU 的机器跳过（~500MB），转录脚本自动走 CPU (int8) 模式
    if detect_gpu().get("has_nvidia"):
        _log("→ nvidia-cublas/cufft/curand（CUDA 运行时）…", log_fn)
        rc, out = _run_subprocess_streaming(
            [uv, "pip", "install", "--python", venv_py,
             "nvidia-cublas-cu12", "nvidia-cufft-cu12", "nvidia-curand-cu12",
             "--index-url", pypi_url],
            _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
        )
        r = _cancelled()
        if r:
            return r
        if rc != 0:
            return {"ok": False, "error": f"安装 CUDA 运行时失败: {out[-500:]}"}
    else:
        _log("→ 未检测到 NVIDIA GPU，跳过 CUDA 运行时（转录将使用 CPU 模式）", log_fn)
    _log("依赖安装完成", log_fn)

    # ── 步骤 4：从 hf-mirror 下载所选模型（复用 models_downloader）──
    _log(f"━━ 步骤 4/4：下载模型（{WHISPER_MODELS[model_key]['title']}，hf-mirror）━━", log_fn)
    # 安装时把块级进度折算成日志（每 5% 一行）：1.6GB+ 的 model.bin 下载期间
    # 日志面板不能全程静默，否则用户误以为卡住
    last_pct10 = {"v": -1}

    def _install_progress(ev: Dict[str, Any]):
        if ev.get("type") == "progress" and ev.get("total") and ev.get("pct") is not None:
            pct10 = int(ev["pct"] * 100 // 5)
            if pct10 > last_pct10["v"]:
                last_pct10["v"] = pct10
                _log(f"  {Path(ev['file']).name} {int(ev['pct'] * 100)}%"
                     f"（{_fmt_size(ev['done'])}/{_fmt_size(ev['total'])}）", log_fn)

    dl = _download_model(model_key, log_fn=log_fn, progress_cb=_install_progress,
                         cancel_event=stop_event, cleanup_on_cancel=True)
    r = _cancelled()
    if r:
        return r
    if dl.get("cancelled"):
        _log("模型下载已取消，未完成的文件已清理", log_fn)
        return {"ok": False, "cancelled": True, "error": "模型下载已取消"}
    if not dl.get("ok"):
        return {"ok": False, "error": f"模型下载失败: {dl.get('error') or '未知错误'}"}
    _log("模型下载完成", log_fn)
    # 安装时选定的模型设为当前模型：配置默认值是 v3-turbo，若用户选了其他模型
    # 而不同步，ready 判定会指向未下载的 v3-turbo，界面误报「找不到模型」
    try:
        from .config_store import update_whisper_config
        update_whisper_config(lambda c: c.update(model=model_key) or c)
        _log(f"当前模型已设为: {WHISPER_MODELS[model_key]['title']}", log_fn)
    except Exception as e:
        _log(f"设置当前模型失败（忽略）: {e}", log_fn)
    _log("━━ 全部完成 ━━", log_fn)
    return {"ok": True, "error": None}


def remove_faster_whisper() -> Dict[str, Any]:
    """删除 faster-whisper 文件夹与模块数据文件夹（_workspace/whisper/）。"""
    existed = WHISPER_DIR.exists()
    if existed:
        try:
            shutil.rmtree(WHISPER_DIR)
        except Exception as e:
            return {"ok": False, "error": f"删除失败: {e}"}
    # 模块配置/开关/转录结果一并删除（可插拔：删除模块后不留任何模块痕迹）
    try:
        from .workspace_paths import WHISPER_CONFIG_FILE
        shutil.rmtree(WHISPER_CONFIG_FILE.parent, ignore_errors=True)
    except Exception:
        pass
    return {"ok": True,
            "message": "已删除 faster-whisper 文件夹" if existed else "目录不存在，无需删除"}


def run_transcription_batch(video_paths: List[str],
                            vad: bool = True, language: str = "",
                            batch: bool = False, workers: int = 0,
                            on_video_done: Optional[Callable[[int, Dict[str, Any]], None]] = None,
                            on_log: Optional[Callable[[str], None]] = None,
                            stop_event=None) -> Dict[str, Any]:
    """对多个视频一次性转录（模型只加载一次，NDJSON 流式输出每个结果）。
        Args:
        workers: 视频间并发数（0=自动：GPU 4 路 / CPU 串行），同一模型实例
        多线程并发解码，结果按 idx 索引不依赖完成顺序
        on_video_done: 回调 (idx, result_dict)，每完成一个视频触发
        on_log: 子进程阶段日志回调（加载模型 / 开始转录等）
        Returns:
        {"ok": bool, "per_video": [{"ok", "srt", "language", "duration", "error"}], "error": str|None}
    """
    status = get_status()
    if not status["ready"]:
        return {"ok": False, "per_video": [], "error": "faster-whisper 未安装或未就绪"}

    current = get_current_model()
    model_dir = _model_subdir(current)
    if not model_dir or not model_dir.is_dir():
        return {"ok": False, "per_video": [],
                "error": f"当前模型（{WHISPER_MODELS[current]['title']}）未下载，请先在扩展功能页下载"}
    # ja-1.5B 仅支持日语：强制 language=ja，忽略面板语言设置，避免识别错乱
    if current in JA_ONLY_MODELS:
        language = "ja"

    params = {
        "video_paths": video_paths,
        "model_dir": str(model_dir),
        "vad": vad,
        "language": language,
        "batch": batch,
        "workers": workers,
        "venv_site_packages": str(VENV_DIR / "Lib" / "site-packages"),
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


# ── 转录脚本（在 venv python 中执行，支持多视频批量）──
# 进度信息经 stdout 的 {"log": ...} JSON 行回传（stderr 仅用于异常诊断），
# 每完成一个视频输出一行 {"idx": i, ...}，最后输出 {"done": true, ...} 汇总。
_TRANSCRIBE_SCRIPT = r"""
import sys, json, os
from pathlib import Path

params_file = sys.argv[1]
with open(params_file, 'r', encoding='utf-8') as f:
    params = json.load(f)

video_paths = params['video_paths']
model_dir = params['model_dir']
use_vad = params.get('vad', True)
language = params.get('language', '') or None
use_batch = params.get('batch', False)
site_packages = params.get('venv_site_packages', '')

def plog(msg):
    print(json.dumps({'log': msg}, ensure_ascii=False), flush=True)

# Windows: ctranslate2 需要 nvidia DLL 在 PATH 中（os.add_dll_directory 无效）
if site_packages:
    for sub in ['cublas', 'cufft', 'curand', 'cuda_nvrtc', 'nvjitlink']:
        dll_dir = os.path.join(site_packages, 'nvidia', sub, 'bin')
        if os.path.isdir(dll_dir):
            os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')

from faster_whisper import WhisperModel
try:
    from faster_whisper import BatchedInferencePipeline
except ImportError:
    BatchedInferencePipeline = None

# 日志极简：不推加载/开始/完成类日志，逐视频结果经 {"idx": i, ...} 行回传，
# 由父进程按完成数统一打日志 + 推进度条；仅保留降级/异常类提示。

# 显式探测 NVIDIA GPU：device='auto' + int8_float16 在无 GPU 机器上会因
# 计算类型不兼容而抛错（错误信息不固定，关键词匹配不可靠），
# 这里直接按探测结果选择设备与计算类型，任何 CUDA 失败都兜底 CPU。
try:
    import ctranslate2
    gpu_count = ctranslate2.get_cuda_device_count()
except Exception:
    gpu_count = 0
if gpu_count > 0:
    try:
        model = WhisperModel(model_dir, device='cuda', compute_type='int8_float16')
    except Exception as e:
        plog(f'CUDA 初始化失败，回退 CPU: {str(e).splitlines()[0]}')
        model = WhisperModel(model_dir, device='cpu', compute_type='int8')
        gpu_count = 0
else:
    model = WhisperModel(model_dir, device='cpu', compute_type='int8')

workers = int(params.get('workers', 0) or 0) or (4 if gpu_count > 0 else 1)
workers = max(1, min(workers, len(video_paths)))

if use_batch:
    if BatchedInferencePipeline is None:
        plog('当前 faster-whisper 版本不支持批处理，回退普通模式')
        use_batch = False
    else:
        model = BatchedInferencePipeline(model=model)

vad_params = dict(
    threshold=0.35,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=200,
) if use_vad else None

def transcribe_one(i, vp):
    # 转录单个视频并即时输出 {"idx": i, ...} 行（线程安全，供主进程流式解析）
    try:
        if use_batch:
            segments, info = model.transcribe(
                vp, beam_size=5, language=language,
                batch_size=16,
                vad_filter=use_vad, vad_parameters=vad_params,
            )
        else:
            segments, info = model.transcribe(
                vp, beam_size=5, language=language,
                vad_filter=use_vad, vad_parameters=vad_params,
            )
        srt_lines = []
        idx = 1
        for seg in segments:
            t = seg.text.strip()
            if not t:
                continue
            start_ms = int(seg.start * 1000)
            end_ms = int(seg.end * 1000)
            sh, sm = divmod(start_ms // 1000, 3600)
            sm, ss = divmod(sm, 60)
            sms = start_ms % 1000
            eh, em = divmod(end_ms // 1000, 3600)
            em, es = divmod(em, 60)
            ems = end_ms % 1000
            srt_lines.append(f'{idx}')
            srt_lines.append(f'{sh:02d}:{sm:02d}:{ss:02d},{sms:03d} --> {eh:02d}:{em:02d}:{es:02d},{ems:03d}')
            srt_lines.append(t)
            srt_lines.append('')
            idx += 1
        srt_content = '\n'.join(srt_lines)
        entry = {'ok': True, 'srt': srt_content, 'language': info.language,
                 'duration': round(info.duration, 1), 'error': None}
    except Exception as e:
        entry = {'ok': False, 'srt': '', 'language': '', 'duration': 0, 'error': str(e)}
    print(json.dumps({'idx': i, **entry}, ensure_ascii=False), flush=True)
    return entry

if workers > 1:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix='whisper') as ex:
        per_video = list(ex.map(lambda iv: transcribe_one(*iv),
                                enumerate(video_paths)))
else:
    per_video = [transcribe_one(i, vp) for i, vp in enumerate(video_paths)]

# 最终汇总行
print(json.dumps({'done': True, 'per_video': per_video}, ensure_ascii=False), flush=True)
"""
