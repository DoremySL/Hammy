"""PixAI Tagger：安装、模型管理，以及对抽帧做二次元预筛 + 角色/IP 标签获取。"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from batch_rename.subprocess_registry import register_subprocess, unregister_subprocess
from batch_rename.env import SUBPROCESS_KWARGS
from batch_rename.config import Config
from batch_rename.utils import fmt_size, safe_int
from batch_rename.video import probe_and_extract_keyframes
from .env import APP_ROOT, PYTHON_EXE, make_logger
from . import installer
from .installer import (
    DEFAULT_TRT_SITE,
    ensure_uv as _ensure_uv,
    run_subprocess_streaming as _run_subprocess_streaming,
    run_venv_script as _run_venv_script,
    InstallSteps,
    UV_TIMEOUT_SEC,
    venv_python_path,
    _terminate_proc,
)

# ── 路径常量 ──
PIXAI_TAGGER_DIR = APP_ROOT / "pixai-tagger"
VENV_DIR = PIXAI_TAGGER_DIR / "venv"
MODEL_DIR = PIXAI_TAGGER_DIR / "models"  # 模型根目录（hf 布局）

# tagger：ONNX 导出仓库，TRT 引擎现场构建
MAIN_REPO = "DoremySL/pixai-tagger-v1.0-onnx-1008-672"
CLS_REPO = "deepghs/anime_real_cls"  # 二次元/真实二分类预筛模型

TAG_THRESHOLD = 0.66
CHAR_FREQ_WEIGHT = 0.1  # 角色得分频次权重
ANIME_CLS_THRESHOLD = 0.72  # 预筛二次元阈值（按帧分数中位数）
REAL_CLS_THRESHOLD = 0.50  # 预筛真实阈值

# 支持的模型输入尺寸
SUPPORTED_SIZES = (1008, 672)
DEFAULT_INPUT_SIZE = 1008

# 仓库文件名规则：tagger_<size>_<fp16|fp32>.onnx
_ENGINE_META_FILE = "engine_meta.json"  # 引擎指纹文件
_CLS_MODEL_SUBDIR = "mobilenetv3_v1.4_dist"
_CLS_ONNX_FILE = "model.onnx"      # 预筛模型，索引 0 = anime
_CLS_ENGINE_NAME = "cls_384.trt"   # cls 引擎文件名
_CLS_SHAPE_RANGE = (1, 2, 2)       # cls 动态 batch 范围（opt=2 为实测吞吐/显存平衡点）
_TRT_WORKSPACE_GB = 4              # TRT 构建临时显存预算
TAG_ZH_FILE = "tag_zh.json"  # 标签中文翻译表

# 子进程超时
_INSTALL_TIMEOUT_SEC = 2400.0
_BUILD_TIMEOUT_SEC = 1200.0
_INFERENCE_BASE_SEC = 240.0
_INFERENCE_PER_VIDEO_SEC = 120.0


_log = make_logger("pixai_tagger")


def normalize_size(value: Any) -> int:
    size = safe_int(value, DEFAULT_INPUT_SIZE)
    return size if size in SUPPORTED_SIZES else DEFAULT_INPUT_SIZE


def get_mirrors_info() -> Dict[str, Any]:
    """返回 pixai 安装弹窗数据（尺寸/站点/GPU 检测）。"""
    gpu = installer.detect_gpu()
    return {
        "sizes": [
            {"id": 1008, "name": "1008 × 1008（精准）", "desc": "模型默认分辨率", "size_label": "约 1 GB"},
            {"id": 672, "name": "672 × 672（快速）", "desc": "约 2 倍速，误判率略有提升", "size_label": "约 1 GB"},
        ],
        "sites": [{"id": k, "name": v["name"], "desc": v["desc"]}
                  for k, v in installer.TRT_SITES.items()],
        "default_site": installer.DEFAULT_TRT_SITE,
        "gpu": gpu,
        "gpu_issue": installer.trt_gpu_issue(gpu),
        "pypi": [{"id": k, "name": v["name"], "url": v["url"]} for k, v in installer.PYPI_MIRRORS.items()],
        "default_pypi": installer.DEFAULT_PYPI_MIRROR,
    }


def tagger_model_files(size: int, precision: str) -> Tuple[str, ...]:
    """tagger 仓库文件清单。"""
    files = ["config.json", f"tagger_{int(size)}_{precision}.onnx"]
    if precision == "fp32":
        files.append(f"tagger_{int(size)}_{precision}.onnx.data")
    return tuple(files)


def _model_dir(size: int, precision: str) -> Optional[Path]:
    d = MODEL_DIR / MAIN_REPO
    return d if all((d / f).is_file() for f in tagger_model_files(size, precision)) else None


def _cls_model_dir() -> Optional[Path]:
    p = MODEL_DIR / CLS_REPO / _CLS_MODEL_SUBDIR / _CLS_ONNX_FILE
    return p if p.is_file() else None


def _tagger_engine_path(model_dir: Path, size: int, precision: str) -> Path:
    return model_dir / f"tagger_{int(size)}_{precision}.trt"


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


def _configured_size_precision() -> Tuple[int, str]:
    """从 pixai 配置读取（输入尺寸, 实际引擎精度）。"""
    from .config_store import load_pixai_config
    return (normalize_size(load_pixai_config().get("input_size")),
            resolve_precision("auto"))


_zh_cache: Dict[str, Any] = {}


def load_tag_zh() -> Dict[str, str]:
    """标签中文翻译表（danbooru 标签 → 中文名）。"""
    path = MODEL_DIR / MAIN_REPO / TAG_ZH_FILE
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _zh_cache.clear()
        return {}
    if _zh_cache.get("mtime") != mtime:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            m = data.get("map") if isinstance(data, dict) else None
            zh = {str(k): str(v) for k, v in m.items() if v} if isinstance(m, dict) else {}
        except Exception:
            zh = {}
        _zh_cache.clear()
        _zh_cache["mtime"] = mtime
        _zh_cache["map"] = zh
    return _zh_cache["map"]


def get_status() -> Dict[str, Any]:
    """获取 pixai-tagger 安装状态。"""
    venv_python = venv_python_path(VENV_DIR)
    size, precision = _configured_size_precision()
    model_dir = _model_dir(size, precision)
    engine_ok = model_dir is not None and _tagger_engine_path(model_dir, size, precision).is_file()
    return {
        "dir_exists": PIXAI_TAGGER_DIR.is_dir(),
        "venv_exists": VENV_DIR.is_dir() and venv_python.exists(),
        "model_exists": model_dir is not None,
        "engine_exists": engine_ok,
        "input_size": size,
        "precision": precision,
        "ready": VENV_DIR.is_dir() and venv_python.exists() and model_dir is not None,
        "dir_path": str(PIXAI_TAGGER_DIR),
    }


def install_dependencies(
    input_size: int = DEFAULT_INPUT_SIZE,
    site: str = DEFAULT_TRT_SITE,
    log_fn: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """安装全部依赖（uv + venv + tensorrt + 模型下载 + TRT 引擎构建），返回 {"ok", "error"}。"""
    plan = installer.resolve_trt_site(site)
    size = normalize_size(input_size)
    steps = InstallSteps(6)

    PIXAI_TAGGER_DIR.mkdir(parents=True, exist_ok=True)

    def _cancelled() -> Optional[Dict[str, Any]]:
        if stop_event is not None and stop_event.is_set():
            _log("安装已取消", log_fn)
            return {"ok": False, "cancelled": True, "error": "安装已取消"}
        return None

    # ── 步骤 1：显卡预检 ──
    steps.push(1)
    _log("━━ 步骤 1/6：检测显卡与驱动 ━━", log_fn)
    gpu = installer.detect_gpu()
    issue = installer.trt_gpu_issue(gpu)
    if issue:
        return {"ok": False, "error": issue}
    precision = resolve_precision("auto", gpu)
    _log(f"GPU: {gpu.get('gpu_name')}（驱动 {gpu.get('driver_version')}）"
         f" → 引擎精度 {precision.upper()}"
         + ("（无 Tensor Core，FP32 引擎更快）" if precision == "fp32" else ""), log_fn)

    # ── 步骤 2 ──
    steps.push(2)
    _log("━━ 步骤 2/6：创建虚拟环境 ━━", log_fn)
    uv = _ensure_uv(plan["pypi_url"], log_fn, stop_event)
    r = _cancelled()
    if r:
        return r
    if not uv:
        return {"ok": False, "error": "UV 安装失败"}
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
    steps.push(3)
    _log(f"━━ 步骤 3/6：安装推理依赖（{plan['site_name']}）━━", log_fn)
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

    # ── 步骤 4 + 5：模型下载 ──
    from . import models_downloader
    if not models_downloader.begin_download():
        return {"ok": False, "busy": True, "error": "已有下载任务进行中，请稍后再试"}
    try:
        last_pct = {"v": -1}

        def _dl_log(msg: str):
            _log(msg, log_fn)

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

        def _cls_progress(ev: Dict[str, Any]):
            if ev.get("type") == "progress" and ev.get("pct") is not None:
                steps.push(5, ev["pct"])
            elif ev.get("type") in ("file_done", "file_skip"):
                steps.push(5, 1.0)

        steps.push(4)
        _log(f"━━ 步骤 4/6：从魔搭下载 PixAI Tagger ONNX 模型（{size}px · {precision.upper()}）━━", log_fn)
        dl = models_downloader.download_files(
            "ms", MAIN_REPO, [*tagger_model_files(size, precision), TAG_ZH_FILE],
            str(MODEL_DIR), log_fn=_dl_log, progress_cb=_model_progress,
            cancel_event=stop_event, cleanup_on_cancel=True)
        if dl.get("cancelled"):
            return {"ok": False, "cancelled": True, "error": "模型下载已取消"}
        if not dl.get("ok"):
            core_failed = [f for f in dl.get("failed") or [] if f.get("file") != TAG_ZH_FILE]
            if dl.get("error") or core_failed:
                err = dl.get("error") or str(core_failed[0].get("error", ""))[:200]
                return {"ok": False, "error": f"模型下载失败: {err}"}
            _log("标签翻译表下载失败（不影响标签获取，中文显示将回退英文）", log_fn)

        steps.push(5)
        _log("━━ 步骤 5/6：下载二次元预筛模型（anime_real_cls，约 17MB）━━", log_fn)
        cls_dl = models_downloader.download_files(
            "ms", CLS_REPO, [f"{_CLS_MODEL_SUBDIR}/{_CLS_ONNX_FILE}"],
            str(MODEL_DIR), log_fn=_dl_log, progress_cb=_cls_progress,
            cancel_event=stop_event, cleanup_on_cancel=True)
        if cls_dl.get("cancelled"):
            return {"ok": False, "cancelled": True, "error": "安装已取消"}
        if not cls_dl.get("ok"):
            err = cls_dl.get("error") or str((cls_dl.get("failed") or [{}])[0].get("error", ""))[:200]
            return {"ok": False, "error": f"预筛模型下载失败: {err}"}
    finally:
        models_downloader.end_download()

    # ── 步骤 6：构建 TRT 引擎 ──
    steps.push(6)
    _log("━━ 步骤 6/6：构建 TensorRT 引擎（约 1-3 分钟）━━", log_fn)
    err = _build_engines(size, precision, gpu_name=gpu.get("gpu_name") or "",
                         log_fn=log_fn, stop_event=stop_event)
    if stop_event is not None and stop_event.is_set():
        return {"ok": False, "cancelled": True, "error": "安装已取消"}
    if err:
        return {"ok": False, "error": err}

    from .config_store import update_pixai_config
    update_pixai_config(lambda c: c.update(input_size=int(size)) or c)
    _log("━━ 全部完成 ━━", log_fn)
    return {"ok": True, "error": None}


def ensure_model_files(size: int, precision: str,
                       log_fn: Optional[Callable[[str], None]] = None,
                       stop_event: Optional[threading.Event] = None) -> Optional[str]:
    """分析前预检：缺失的模型/翻译表从魔搭补下载，返回错误信息或 None。"""
    from . import models_downloader
    main_dir = MODEL_DIR / MAIN_REPO
    missing_tagger = [f for f in tagger_model_files(size, precision)
                      if not (main_dir / f).is_file()]
    missing_zh = [] if (main_dir / TAG_ZH_FILE).is_file() else [TAG_ZH_FILE]
    missing_cls = [] if _cls_model_dir() is not None else [f"{_CLS_MODEL_SUBDIR}/{_CLS_ONNX_FILE}"]
    if not missing_tagger and not missing_zh and not missing_cls:
        return None
    if not models_downloader.begin_download():
        return "已有下载任务进行中，请稍后再试"
    try:
        if missing_tagger or missing_zh:
            _log(f"补充下载模型文件: {', '.join([*missing_tagger, *missing_zh])}", log_fn)
            dl = models_downloader.download_files(
                "ms", MAIN_REPO, [*missing_tagger, *missing_zh], str(MODEL_DIR),
                log_fn=lambda m: _log(m, log_fn), cancel_event=stop_event,
                cleanup_on_cancel=True)
            if dl.get("cancelled"):
                return "模型下载已取消"
            core_failed = [f for f in dl.get("failed") or [] if f.get("file") != TAG_ZH_FILE]
            if not dl.get("ok") and (dl.get("error") or (core_failed and missing_tagger)):
                err = dl.get("error") or str((core_failed or [{}])[0].get("error", ""))[:200]
                return f"模型文件补下载失败: {err or '未知错误'}"
            if not missing_tagger and not core_failed and not missing_cls:
                return None  # 只补了翻译表
        if missing_cls:
            _log("补充下载预筛模型…", log_fn)
            dl = models_downloader.download_files(
                "ms", CLS_REPO, missing_cls, str(MODEL_DIR),
                log_fn=lambda m: _log(m, log_fn), cancel_event=stop_event,
                cleanup_on_cancel=True)
            if dl.get("cancelled"):
                return "模型下载已取消"
            if not dl.get("ok"):
                err = dl.get("error") or str((dl.get("failed") or [{}])[0].get("error", ""))[:200]
                return f"预筛模型补下载失败: {err or '未知错误'}"
    finally:
        models_downloader.end_download()
    return None


def _build_engines(size: int, precision: str, *, gpu_name: str = "",
                   log_fn: Optional[Callable[[str], None]] = None,
                   stop_event: Optional[threading.Event] = None) -> Optional[str]:
    """在 venv 内现场构建 tagger 与预筛 cls TRT 引擎并写指纹文件。"""
    model_dir = MODEL_DIR / MAIN_REPO
    cls_onnx = _cls_model_dir()
    if cls_onnx is None:
        return "预筛模型缺失，无法构建引擎"
    jobs = [{"onnx": str(model_dir / f"tagger_{size}_{precision}.onnx"),
             "out": str(_tagger_engine_path(model_dir, size, precision)),
             "workspace_gb": _TRT_WORKSPACE_GB},
            {"onnx": str(cls_onnx), "out": str(model_dir / _CLS_ENGINE_NAME),
             "workspace_gb": _TRT_WORKSPACE_GB, "shape_range": list(_CLS_SHAPE_RANGE)}]

    def _on_line(line: str):
        line = line.strip()
        if not line:
            return
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return
        if isinstance(obj, dict) and obj.get("log"):
            _log(str(obj["log"]), log_fn)

    from .llama_cpp import pause_for_task, resume_after_task
    pause_for_task(log_fn=log_fn)  # 让出显存
    try:
        rc, _last, err_tail = _run_venv_script(
            VENV_DIR, _TRT_BUILD_SCRIPT,
            {"jobs": jobs, "gpu_name": gpu_name, "size": int(size),
             "precision": precision, "cls_shape": list(_CLS_SHAPE_RANGE),
             "meta_file": str(model_dir / _ENGINE_META_FILE)},
            timeout=_BUILD_TIMEOUT_SEC, stop_event=stop_event, on_line=_on_line)
    finally:
        resume_after_task(log_fn=log_fn)
    if stop_event is not None and stop_event.is_set():
        return "安装已取消"
    if rc != 0:
        return f"引擎构建失败: {(err_tail or '子进程异常退出')[-300:]}"
    if not _tagger_engine_path(model_dir, size, precision).is_file():
        return "引擎构建失败: 引擎文件未生成"
    return None


# ── venv 内联脚本公共段：日志 + TRT 引擎构建 ──

_TRT_CORE = r'''
import gc, glob, json, os, shutil, sys, tempfile, time
from pathlib import Path

def _add_nvidia_dll_dirs():
    """把 pip 安装的 nvidia-* wheel 的 DLL 目录加入搜索路径。"""
    for base in {p for p in sys.path if p and Path(p).name.lower() == 'site-packages'}:
        for d in glob.glob(str(Path(base) / 'nvidia' / '*' / 'bin')):
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass
            os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')

_add_nvidia_dll_dirs()

def plog(msg):
    print(json.dumps({'log': str(msg)}, ensure_ascii=False), flush=True)

def _stage_ascii(onnx_path):
    """把 ONNX 与外部权重拷到 ASCII 临时目录。"""
    for stale in Path(tempfile.gettempdir()).glob('trt_stage_*'):
        shutil.rmtree(stale, ignore_errors=True)
    tmpdir = Path(tempfile.mkdtemp(prefix='trt_stage_'))
    if not str(tmpdir).isascii():
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError('系统 TEMP 目录不是 ASCII 路径，无法构建引擎')
    for side in onnx_path.parent.glob(onnx_path.name + '*'):
        shutil.copy2(side, tmpdir / side.name)
    return tmpdir / onnx_path.name, tmpdir

def trt_build(onnx_path, engine_path, workspace_gb=4, shape_range=None):
    """从 ONNX 构建本机 TRT 引擎（动态维度钉在 1；shape_range=(min,opt,max) 按范围构建）。"""
    import tensorrt as trt
    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    data_file = Path(str(onnx_path) + '.data')
    tmpdir, src = None, onnx_path
    try:
        if data_file.exists():
            if not str(onnx_path).isascii():
                src, tmpdir = _stage_ascii(onnx_path)
            ok = parser.parse_from_file(str(src))
        else:
            ok = parser.parse(onnx_path.read_bytes())
        if not ok:
            errs = '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f'ONNX 解析失败: {errs[:300]}')
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb) << 30)
        profile = None
        for i in range(network.num_inputs):
            name = network.get_input(i).name
            shape = [int(d) for d in network.get_input(i).shape]
            if any(d < 0 for d in shape):
                if profile is None:
                    profile = builder.create_optimization_profile()
                if shape_range is None:
                    fixed = tuple(1 if d < 0 else d for d in shape)
                    profile.set_shape(name, fixed, fixed, fixed)
                else:
                    lo, opt, hi = (int(v) for v in shape_range)
                    profile.set_shape(name,
                                      tuple(lo if d < 0 else d for d in shape),
                                      tuple(opt if d < 0 else d for d in shape),
                                      tuple(hi if d < 0 else d for d in shape))
        if profile is not None:
            config.add_optimization_profile(profile)
        t0 = time.perf_counter()
        ser = builder.build_serialized_network(network, config)
        if ser is None:
            raise RuntimeError('引擎构建失败')
        engine_path.write_bytes(ser)
        plog(f'{engine_path.name} 构建完成（{time.perf_counter() - t0:.0f}s，'
             f'{engine_path.stat().st_size / 2 ** 20:.0f} MB）')
    finally:
        del network, parser, builder  # 先释放对象再删临时目录（Windows 文件锁）
        gc.collect()
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
'''

_TRT_BUILD_SCRIPT = _TRT_CORE + r'''
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    params = json.load(f)

try:
    import tensorrt as trt
    _gpu_name = params.get('gpu_name') or ''  # 指纹 GPU 名与分析脚本同源（CUDA 设备 0）
    try:
        from cuda.bindings import runtime as _cudart
        _e, _p = _cudart.cudaGetDeviceProperties(0)
        if _e == 0 and _p is not None:
            _gpu_name = _p.name.decode(errors='replace').strip()
    except Exception:
        pass
    for job in params['jobs']:
        trt_build(job['onnx'], job['out'], job.get('workspace_gb', 4),
                  job.get('shape_range'))
    meta = {
        'gpu_name': _gpu_name,
        'trt_version': trt.__version__,
        'tagger': {'size': params.get('size'), 'precision': params.get('precision')},
        'cls': {'shape': params.get('cls_shape') or []},
    }
    Path(params['meta_file']).write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
except Exception as e:
    print(json.dumps({'error': str(e)}, ensure_ascii=False), flush=True)
    sys.exit(1)
'''


def remove_pixai_tagger() -> Dict[str, Any]:
    """删除 pixai-tagger 文件夹与模块数据文件夹（_workspace/pixai/）。"""
    existed = PIXAI_TAGGER_DIR.exists()
    if existed:
        try:
            shutil.rmtree(PIXAI_TAGGER_DIR)
        except Exception as e:
            return {"ok": False, "error": f"删除失败: {e}"}
    # 模块配置/开关/标签结果一并删除
    try:
        from .workspace_paths import PIXAI_CONFIG_FILE
        shutil.rmtree(PIXAI_CONFIG_FILE.parent, ignore_errors=True)
    except Exception:
        pass
    return {"ok": True,
            "message": "已删除 pixai-tagger 文件夹" if existed else "目录不存在，无需删除"}


# ── 抽帧与推理 ──

_NO_STOP = threading.Event()


def extract_frames_for_tagger(video_path: str,
                              stop_event: Optional[threading.Event] = None,
                              frame_count: int = 15,
                              max_side: int = DEFAULT_INPUT_SIZE) -> List[str]:
    """复用主流程抽帧，返回 JPEG base64 帧列表。"""
    if stop_event is None:
        stop_event = _NO_STOP
    if stop_event.is_set():
        return []
    cfg = Config(sampling_points=max(1, frame_count), frames_per_point=1,
                 frame_max_side=max_side)
    cfg.validate()
    _, frames = probe_and_extract_keyframes(video_path, cfg, stop_event)
    return [f.b64 for f in frames]


def _inference_timeout(video_count: int) -> float:
    return _INFERENCE_BASE_SEC + _INFERENCE_PER_VIDEO_SEC * max(1, video_count)


class AnalyzeStream:
    """交互式分析子进程：stdin 逐视频送入，stdout 流式回传 {"video_result": {...}}。"""

    def __init__(self, params: Dict[str, Any], video_count: int,
                 stop_event=None,
                 on_log: Optional[Callable[[str], None]] = None,
                 on_video_result: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.stop_event = stop_event
        self.on_log = on_log
        self.on_video_result = on_video_result
        self.per_video: List[Dict[str, Any]] = []
        self.broken = False  # 子进程异常/停止/超时后置位
        self._deadline = time.time() + _inference_timeout(video_count)
        self._result_q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._err_tail: "deque[str]" = deque(maxlen=200)
        self._error = ""

        py = venv_python_path(VENV_DIR)
        if not py.is_file():
            raise RuntimeError(f"模块环境不存在: {py}")

        fd, self._params_path = tempfile.mkstemp(suffix=".json", prefix="modjob_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False)
        except Exception as e:
            os.unlink(self._params_path)
            raise RuntimeError(f"参数文件写入失败: {e}")

        try:
            self.p = subprocess.Popen(
                [str(py), "-c", _ANALYZE_SCRIPT, self._params_path],
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
                        vr = obj["video_result"]
                        self.per_video.append(vr)
                        if self.on_video_result is not None:
                            try:
                                self.on_video_result(vr)
                            except Exception:
                                pass
                        self._result_q.put(vr)
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

        self._out_thread = threading.Thread(target=_read_stdout, name="pixai-out-reader", daemon=True)
        self._err_thread = threading.Thread(target=_read_stderr, name="pixai-err-reader", daemon=True)
        self._out_thread.start()
        self._err_thread.start()

    def send(self, item: Dict[str, Any]) -> bool:
        """发送一个视频请求，子进程已退出时返回 False。"""
        if self.broken or self.p.poll() is not None:
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
                if self.p.poll() is not None:
                    self.broken = True
                    return None
                continue
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
        """关闭子进程并回收。

        Returns: {"ok": bool, "per_video": [...], "error": str|None}
        """
        stopped = self.stop_event is not None and self.stop_event.is_set()
        if stopped:
            try:
                _terminate_proc(self.p, wait_sec=1.0)
            except Exception:
                pass
        elif not self.broken:
            try:
                if self.p.stdin is not None:
                    self.p.stdin.close()
            except (OSError, ValueError):
                pass
        else:
            try:
                _terminate_proc(self.p)
            except Exception:
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
        except Exception:
            pass
        unregister_subprocess(self.p)
        try:
            os.unlink(self._params_path)
        except OSError:
            pass

        rc = self.p.returncode if self.p.returncode is not None else -1
        if self.stop_event is not None and self.stop_event.is_set():
            return {"ok": False, "per_video": self.per_video, "error": "已停止"}
        if time.time() > self._deadline:
            return {"ok": False, "per_video": self.per_video, "error": "推理超时"}
        if rc != 0:
            tail = self._error or "\n".join(list(self._err_tail)[-20:])
            return {"ok": False, "per_video": self.per_video,
                    "error": f"推理失败: {tail[-500:] if tail else '子进程异常退出'}"}
        return {"ok": True, "per_video": self.per_video, "error": None}


def start_analyze_stream(
    video_count: int,
    skip_real: bool = False,
    anime_threshold: float = ANIME_CLS_THRESHOLD,
    real_threshold: float = REAL_CLS_THRESHOLD,
    tag_threshold: float = TAG_THRESHOLD,
    ip_threshold: float = TAG_THRESHOLD,
    char_freq_weight: float = CHAR_FREQ_WEIGHT,
    input_size: int = DEFAULT_INPUT_SIZE,
    stop_event=None,
    on_log: Optional[Callable[[str], None]] = None,
    on_video_result: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Optional[AnalyzeStream]:
    """启动交互式分析子进程。"""
    status = get_status()
    if not status["ready"]:
        return None
    size = normalize_size(input_size)
    prec = resolve_precision("auto")
    model_dir = _model_dir(size, prec)
    if not model_dir:
        return None
    cls_onnx = _cls_model_dir()
    if cls_onnx is None:
        return None  # 预筛模型缺失
    params = {
        "model_dir": str(model_dir),
        "cls_model_path": str(cls_onnx),
        "cls_engine": _CLS_ENGINE_NAME,
        "cls_shape": list(_CLS_SHAPE_RANGE),
        "meta_file": str(model_dir / _ENGINE_META_FILE),
        "size": size,
        "precision": prec,
        "workspace_gb": _TRT_WORKSPACE_GB,
        "skip_real": skip_real,
        "anime_threshold": anime_threshold,
        "real_threshold": real_threshold,
        "tag_threshold": tag_threshold,
        "ip_threshold": ip_threshold,
        "char_freq_weight": char_freq_weight,
    }
    try:
        return AnalyzeStream(params, video_count,
                             stop_event=stop_event, on_log=on_log,
                             on_video_result=on_video_result)
    except Exception:
        return None


# ── 分析脚本（venv 中执行）──
_ANALYZE_SCRIPT = _TRT_CORE + r"""
import base64, io, queue, threading
import numpy as np
from PIL import Image, ImageStat
import tensorrt as trt
from cuda.bindings import runtime as cudart

def report(video_result):
    print(json.dumps({'video_result': video_result}, ensure_ascii=False), flush=True)

def _fatal(msg):
    print(json.dumps({'error': str(msg)}, ensure_ascii=False), flush=True)
    sys.exit(1)

def _cerr(res):
    '''取 cuda-python 返回的错误码。'''
    return res[0] if isinstance(res, tuple) else res

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    params = json.load(f)

model_dir = Path(params['model_dir'])
cls_onnx = Path(params.get('cls_model_path') or '')
cls_engine_path = model_dir / (params.get('cls_engine') or 'cls_384.trt')
meta_file = Path(params.get('meta_file') or (model_dir / 'engine_meta.json'))
skip_real = bool(params.get('skip_real', False))
cls_threshold = float(params.get('anime_threshold', 0.72))
real_threshold = float(params.get('real_threshold', 0.50))
char_threshold = float(params.get('tag_threshold', 0.66))
char_freq_w = float(params.get('char_freq_weight', 0.1))
char_prefilter = char_threshold * 0.3
ip_threshold = float(params.get('ip_threshold', 0.66))
size = int(params.get('size') or 1008)
precision = str(params.get('precision') or 'fp16')
workspace_gb = float(params.get('workspace_gb') or 4)
cls_shape = tuple(int(v) for v in params.get('cls_shape') or (1, 2, 2))

# ── 标签表与类别切分 ──
try:
    cfg = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    TAGS, SPLIT = cfg['tags'], cfg['tags_split']
except Exception as e:
    _fatal(f'标签表读取失败: {e}')
_seg_off, _off = {}, 0
for _cat, _cnt in SPLIT:
    _seg_off[_cat] = (_off, int(_cnt))
    _off += int(_cnt)
if 'character' not in _seg_off or 'copyright' not in _seg_off:
    _fatal('config.json 缺少 character/copyright 类别切分')

# ── CUDA 初始化 ──
try:
    _cerr(cudart.cudaFree(0))  # 初始化上下文
    _err, _props = cudart.cudaGetDeviceProperties(0)
    if _err != 0 or _props is None:
        raise RuntimeError(f'cudaGetDeviceProperties 错误 {_err}')
    GPU_NAME = _props.name.decode(errors='replace').strip()
except Exception as e:
    _fatal(f'未检测到可用的 NVIDIA GPU: {e}')

tagger_engine = model_dir / f'tagger_{size}_{precision}.trt'

def _read_meta():
    try:
        return json.loads(meta_file.read_text(encoding='utf-8'))
    except Exception:
        return {}

def _meta_base_fresh(meta):
    '''引擎指纹是否匹配当前 GPU/TRT 版本。'''
    return (str(meta.get('gpu_name') or '').strip().lower() == GPU_NAME.lower()
            and str(meta.get('trt_version') or '') == trt.__version__)

def _write_engine_meta():
    meta = {'gpu_name': GPU_NAME, 'trt_version': trt.__version__,
            'tagger': {'size': size, 'precision': precision},
            'cls': {'shape': list(cls_shape)}}
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')

_meta = _read_meta()
_base_fresh = _meta_base_fresh(_meta)
_tag = _meta.get('tagger') or {}

# tagger 引擎：缺失或指纹过期则重建
if not (_base_fresh and _tag.get('size') == size and _tag.get('precision') == precision
        and tagger_engine.is_file()):
    onnx_file = model_dir / f'tagger_{size}_{precision}.onnx'
    if not onnx_file.is_file():
        _fatal(f'缺少 {onnx_file.name}，请重新安装 pixai-tagger')
    tagger_engine.unlink(missing_ok=True)  # 删除过期引擎
    plog('TRT 引擎缺失或与当前 GPU/TRT 版本不符，正在构建（约 1-3 分钟）…')
    try:
        trt_build(onnx_file, tagger_engine, workspace_gb)
        _write_engine_meta()
    except Exception as e:
        _fatal(f'引擎构建失败: {e}')

class TrtRunner:
    '''单引擎封装：设备缓冲 + 异步前向（静态与 range 动态 batch 引擎通用）。'''

    def __init__(self, engine_path):
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
            Path(engine_path).read_bytes())
        if engine is None:
            raise RuntimeError(f'引擎反序列化失败: {Path(engine_path).name}')
        self.ctx = engine.create_execution_context()
        self.in_name = self.out_name = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT and self.in_name is None:
                self.in_name = name
            elif mode == trt.TensorIOMode.OUTPUT and self.out_name is None:
                self.out_name = name
        if not self.in_name or not self.out_name:
            raise RuntimeError(f'引擎 IO 不完整: {Path(engine_path).name}')
        self.in_dtype = trt.nptype(engine.get_tensor_dtype(self.in_name))
        self._out_dtype = trt.nptype(engine.get_tensor_dtype(self.out_name))
        self._out_itemsize = np.dtype(self._out_dtype).itemsize
        self._dyn = any(d < 0 for d in self.ctx.get_tensor_shape(self.in_name))
        self.d_in = self.d_out = self.stream = None
        self.host_out = None
        self._in_bytes = self._out_bytes = 0
        if not self._dyn:
            in_shape = tuple(int(d) for d in self.ctx.get_tensor_shape(self.in_name))
            out_shape = tuple(int(d) for d in self.ctx.get_tensor_shape(self.out_name))
            self.host_out = np.empty(out_shape, dtype=self._out_dtype)
            self._realloc(in_shape, out_shape)

    def _realloc(self, in_shape, out_shape):
        '''按需分配设备缓冲（只增不减），绑定张量地址。'''
        if self.stream is None:
            _err, self.stream = cudart.cudaStreamCreate()
            if _err != 0:
                raise RuntimeError('CUDA 流创建失败')
        in_bytes = int(np.prod(in_shape)) * np.dtype(self.in_dtype).itemsize
        out_bytes = int(np.prod(out_shape)) * self._out_itemsize
        if in_bytes > self._in_bytes:
            if self.d_in is not None:
                cudart.cudaFree(self.d_in)
            _err, self.d_in = cudart.cudaMalloc(in_bytes)
            if _err != 0:
                raise RuntimeError('显存分配失败')
            self._in_bytes = in_bytes
            self.ctx.set_tensor_address(self.in_name, self.d_in)
        if out_bytes > self._out_bytes:
            if self.d_out is not None:
                cudart.cudaFree(self.d_out)
            _err, self.d_out = cudart.cudaMalloc(out_bytes)
            if _err != 0:
                raise RuntimeError('显存分配失败')
            self._out_bytes = out_bytes
            self.ctx.set_tensor_address(self.out_name, self.d_out)

    def run(self, host_in):
        x = np.ascontiguousarray(host_in, dtype=self.in_dtype)
        if self._dyn:
            if not self.ctx.set_input_shape(self.in_name, x.shape):
                raise RuntimeError('set_input_shape 失败')
            out_shape = tuple(int(d) for d in self.ctx.get_tensor_shape(self.out_name))
            self._realloc(x.shape, out_shape)
            host_out = np.empty(out_shape, dtype=self._out_dtype)
        else:
            if x.nbytes != self._in_bytes:
                raise RuntimeError(f'输入字节数不符: {x.nbytes} != {self._in_bytes}')
            host_out = self.host_out
        if _cerr(cudart.cudaMemcpyAsync(self.d_in, x.ctypes.data, x.nbytes,
                                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                                        self.stream)) != 0:
            raise RuntimeError('主机→显存拷贝失败')
        if not self.ctx.execute_async_v3(self.stream):
            raise RuntimeError('推理执行失败')
        if _cerr(cudart.cudaMemcpyAsync(host_out.ctypes.data, self.d_out, host_out.nbytes,
                                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                                        self.stream)) != 0:
            raise RuntimeError('显存→主机拷贝失败')
        if _cerr(cudart.cudaStreamSynchronize(self.stream)) != 0:
            raise RuntimeError('CUDA 流同步失败')
        return host_out

try:
    tagger_runner = TrtRunner(tagger_engine)
except Exception as e:
    tagger_engine.unlink(missing_ok=True)  # 坏引擎自愈：删除后下次重建
    _fatal(f'tagger 引擎加载失败: {e}')

# cls 引擎（必装）：缺失、指纹过期或 shape 范围不符则重建
if not (_base_fresh and list(((_meta.get('cls') or {}).get('shape') or [])) == list(cls_shape)
        and cls_engine_path.is_file()):
    cls_engine_path.unlink(missing_ok=True)  # 删除过期引擎
    if not cls_onnx.is_file():
        _fatal('预筛模型缺失，请重新安装 pixai-tagger')
    try:
        trt_build(cls_onnx, cls_engine_path, workspace_gb, shape_range=cls_shape)
        _write_engine_meta()
    except Exception as e:
        _fatal(f'预筛引擎构建失败: {e}')
try:
    cls_runner = TrtRunner(cls_engine_path)
except Exception as e:
    cls_engine_path.unlink(missing_ok=True)  # 坏引擎自愈：删除后下次重建
    _fatal(f'预筛引擎加载失败: {e}')

# ── 坏帧过滤 ──
_DARK_BRIGHTNESS = 15.0  # 亮度低于此值视为坏帧

def _mean_brightness(image):
    '''粗略亮度（0-255）。'''
    return ImageStat.Stat(image.convert('L').resize((64, 64))).mean[0]

def _is_flat_frame(image):
    '''纯色/低信息量帧判定。'''
    g = image.convert('L').resize((64, 64))
    stddev = ImageStat.Stat(g).stddev[0]
    if stddev >= 60:
        return False
    if stddev < 8:
        return True
    rgb = image.convert('RGB').resize((32, 32))
    r, gr, b = ImageStat.Stat(rgb).mean
    return (max(r, gr, b) - min(r, gr, b)) / 255.0 < 0.05

def tag_preprocess(img, size, dtype):
    '''等比缩放 + 居中黑边填充到 size×size，归一化 [-1,1]。'''
    w, h = img.size
    if (h, w) != (size, size):
        r = min(size / h, size / w)
        img = img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (size, size), (0, 0, 0))
        canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
        img = canvas
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - 0.5) / 0.5
    return np.ascontiguousarray(np.transpose(a, (2, 0, 1))[None].astype(dtype))

def cls_preprocess(img):
    '''Resize 384×384 + 归一化 [-1,1] + CHW。'''
    im = img.resize((384, 384), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - 0.5) / 0.5
    return np.ascontiguousarray(np.transpose(a, (2, 0, 1))[None])

# ── warmup ──
try:
    cls_runner.run(cls_preprocess(Image.new('RGB', (384, 384))))
except Exception as e:
    _fatal(f'预筛引擎 warmup 失败: {e}')
try:
    tagger_runner.run(tag_preprocess(Image.new('RGB', (384, 216)), size, tagger_runner.in_dtype))
except Exception as e:
    plog(f'tagger warmup 失败（忽略）: {e}')

def pil_to_rgb(image):
    if image.mode in ('RGBA', 'P'):
        if image.mode == 'P':
            image = image.convert('RGBA')
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    return image.convert('RGB')

def decode_frames(b64_list):
    '''base64 → PIL Image，失败返回 None。'''
    frames = []
    for b64 in b64_list:
        try:
            frames.append(pil_to_rgb(Image.open(io.BytesIO(base64.b64decode(b64)))))
        except Exception:
            frames.append(None)
    return frames

def _run_cls_frames(images):
    '''预筛：按 opt batch 分块前向，返回逐帧 anime 分数（ONNX 输出已是 softmax 概率，索引 0 = anime）。'''
    xs = [cls_preprocess(im) for im in images]
    scores = []
    for i in range(0, len(xs), cls_shape[1]):
        probs = cls_runner.run(np.concatenate(xs[i:i + cls_shape[1]]))
        scores.extend(float(v) for v in probs[:, 0])
    return scores

# ── 标签输出规则 ──

def _accum_stats(stats, probs, off, cnt, threshold):
    '''单帧单类别的 (出现次数, 最高分) 统计。'''
    seg = probs[off:off + cnt]
    for j in np.nonzero(seg > threshold)[0]:
        t, p = TAGS[off + int(j)], float(seg[j])
        c, mx = stats.get(t, (0, 0.0))
        stats[t] = (c + 1, max(mx, p))

def _apply_char_rules(char_stats):
    '''角色标签：频次加权得分过滤。'''
    return {t: mx for t, (cnt, mx) in char_stats.items()
            if mx + (cnt - 1) * char_freq_w > char_threshold}

_EXCLUDED_IP = frozenset(('original', 'real_life'))

def _apply_ip_rules(ip_stats):
    '''IP 标签：只留最高置信度作品。'''
    cand = {t: s for t, s in ip_stats.items() if t not in _EXCLUDED_IP}
    if not cand:
        return {}
    key = max((mx, cnt) for cnt, mx in cand.values())
    return {t: mx for t, (cnt, mx) in cand.items() if (mx, cnt) == key}

def _char_fallback(char_scores, ip_scores, char_stats):
    '''IP 命中且角色全空时，补输出最高置信度角色。'''
    if ip_scores and not char_scores and char_stats:
        best = min(char_stats.items(), key=lambda kv: (-kv[1][1], kv[0]))
        return {best[0]: best[1][1]}
    return char_scores

def _run_tag_frames(imgs):
    '''标签获取：逐帧前向，character/IP 跨帧统计。'''
    char_stats, ip_stats = {}, {}
    for im in imgs:
        logits = tagger_runner.run(tag_preprocess(im, size, tagger_runner.in_dtype))[0]
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
        _accum_stats(char_stats, probs, *_seg_off['character'], char_prefilter)
        _accum_stats(ip_stats, probs, *_seg_off['copyright'], ip_threshold)
    return char_stats, ip_stats

def _tag_list(scores):
    '''{标签: 分数} → 按分数降序的输出列表。'''
    return [{'name': t, 'score': round(s, 4)}
            for t, s in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]

# ── 预取线程：stdin → 解码/坏帧过滤入有界队列 ──
_prefetch_q = queue.Queue(maxsize=2)

def _prefetch_reader():
    for _line in sys.stdin:
        _line = _line.strip()
        if not _line:
            continue
        try:
            req = json.loads(_line)
        except json.JSONDecodeError:
            plog(f'忽略无效请求行: {_line[:80]}')
            continue
        if not isinstance(req, dict):
            continue
        vi = int(req.get('index', 0))
        name = req.get('name') or f'视频{vi + 1}'
        b64s = req.get('frames') or []
        if not b64s:
            _prefetch_q.put({'report': {'index': vi, 'name': name, 'error': '无帧数据'}})
            continue
        try:
            imgs = [f for f in decode_frames(b64s) if f is not None]
        finally:
            del b64s
        if not imgs:
            _prefetch_q.put({'report': {'index': vi, 'name': name, 'error': '帧解码失败'}})
            continue
        # 坏帧过滤：保留不足 2 帧时回退全量
        bright = [im for im in imgs if _mean_brightness(im) >= _DARK_BRIGHTNESS
                  and not _is_flat_frame(im)]
        if len(bright) >= 2:
            imgs = bright
        _prefetch_q.put({'vi': vi, 'name': name, 'imgs': imgs})
    _prefetch_q.put(None)  # stdin EOF

threading.Thread(target=_prefetch_reader, name='prefetch', daemon=True).start()

while True:
    item = _prefetch_q.get()
    if item is None:
        break  # stdin EOF
    if 'report' in item:
        report(item['report'])
        continue
    vi, name = item['vi'], item['name']
    imgs = item['imgs']

    is_anime = None  # None = 不确定
    anime_score = 0.0
    try:
        scores = _run_cls_frames(imgs)
    except Exception as e:
        _fatal(f'预筛推理失败 {name}: {e}')
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    med = (sorted_scores[n // 2] + sorted_scores[(n - 1) // 2]) / 2
    if med >= cls_threshold:
        is_anime = True
        verdict = '二次元'
    elif med <= real_threshold:
        is_anime = False
        verdict = '非二次元'
    else:
        verdict = '不确定'
    anime_score = round(med, 4)
    if is_anime is not True:
        plog(f'预筛 {name}: 中位数={anime_score} → {verdict}')

    if skip_real and is_anime is False:
        report({'index': vi, 'name': name, 'anime_score': anime_score,
                'is_anime': False, 'character_tags': [], 'ip_tags': []})
        del imgs
        continue

    # ── 标签获取 ──
    try:
        char_stats, ip_stats = _run_tag_frames(imgs)
    except Exception as e:
        report({'index': vi, 'name': name, 'error': f'标签获取异常: {e}'})
        del imgs
        continue
    del imgs
    ip_scores = _apply_ip_rules(ip_stats)
    char_scores = _char_fallback(_apply_char_rules(char_stats), ip_scores, char_stats)
    result = {
        'index': vi, 'name': name,
        'character_tags': _tag_list(char_scores),
        'ip_tags': _tag_list(ip_scores),
        'anime_score': anime_score,  # 分类信息（前端角标用）
        'is_anime': is_anime,
    }
    report(result)
"""
