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
from typing import Any, Callable, Dict, List, Optional

from batch_rename.subprocess_registry import register_subprocess, unregister_subprocess
from batch_rename.env import SUBPROCESS_KWARGS
from .env import APP_ROOT, PYTHON_EXE, make_logger
from . import installer
from .installer import (
    PYPI_MIRRORS, PYTORCH_MIRRORS, DEFAULT_PYPI_MIRROR, DEFAULT_PYTORCH_MIRROR,
    ensure_uv as _ensure_uv,
    run_subprocess_streaming as _run_subprocess_streaming,
    UV_TIMEOUT_SEC,
    venv_python_path,
    _terminate_proc,
)

# ── 路径常量 ──
PIXAI_TAGGER_DIR = APP_ROOT / "pixai-tagger"
VENV_DIR = PIXAI_TAGGER_DIR / "venv"
MODEL_DIR = PIXAI_TAGGER_DIR / "model"

MODEL_ID = "pixai-labs/pixai-tagger-v0.9"
CLS_MODEL_ID = "deepghs/anime_real_cls"  # 二次元/真实二分类预筛模型

CHARACTER_THRESHOLD = 0.9
# 预筛三分层阈值：中位数 >= 高阈值 → anime；<= 低阈值 → real；
# 中间为不确定（照常打标，附 UNC 角标）。
# 高阈值留足余量以容忍异常帧（纯色转场/动漫元素贴片等）被误打高分；
# 低阈值略降以扩大「不确定」窗口，拿不准时走照常打标而非错误跳过。
ANIME_CLS_THRESHOLD = 0.72
REAL_CLS_THRESHOLD = 0.50

_WEIGHTS_FILE = "model_v0.9.pth"
_TAGS_FILE = "tags_v0.9_13k.json"
_MAPPING_FILE = "char_ip_map.json"
_CLS_CKPT_FILE = "model.ckpt"
_CLS_MODEL_SUBDIR = "mobilenetv3_v1.4_dist"  # 模型子目录名（轻量版，约 32MB）

# 子进程超时（推理含双模型加载，按视频数量线性放大）
_INSTALL_TIMEOUT_SEC = 900.0    # torch 体积大，安装可能很慢
_MODEL_TIMEOUT_SEC = 900.0
_INFERENCE_BASE_SEC = 240.0     # 推理基础超时（双模型加载 + 少量帧）
_INFERENCE_PER_VIDEO_SEC = 60.0  # 每增加一个视频放宽的余量（CPU 推理 ~1.4s/帧）


_log = make_logger("pixai_tagger")


def get_mirrors_info() -> Dict[str, Any]:
    """返回可选镜像站列表 + GPU 检测结果（供前端渲染选择弹窗）。"""
    return installer.get_mirror_groups(["pytorch", "pypi"])


def _find_model_snapshot_dir() -> Optional[Path]:
    """查找主模型快照目录（ModelScope 布局：models/<org>/snapshots/master）。"""
    snapshots_base = MODEL_DIR / "models"
    if snapshots_base.is_dir():
        for org_dir in snapshots_base.iterdir():
            snap_master = org_dir / "snapshots" / "master"
            if snap_master.is_dir() and (snap_master / _WEIGHTS_FILE).exists():
                return snap_master
    if (MODEL_DIR / _WEIGHTS_FILE).exists():
        return MODEL_DIR
    return None


def _find_cls_model_path() -> Optional[Path]:
    """查找 anime_real_cls 模型检查点路径。"""
    # 优先：安装时统一放置的 model/anime_real_cls/mobilenetv3_v1.4_dist/model.ckpt
    direct = MODEL_DIR / "anime_real_cls" / _CLS_MODEL_SUBDIR / _CLS_CKPT_FILE
    if direct.exists():
        return direct
    # 兜底：ModelScope snapshot 结构
    snapshots_base = MODEL_DIR / "models"
    if snapshots_base.is_dir():
        for org_dir in snapshots_base.iterdir():
            snap_master = org_dir / "snapshots" / "master"
            for candidate in (snap_master / _CLS_MODEL_SUBDIR / _CLS_CKPT_FILE,
                              snap_master / _CLS_CKPT_FILE):
                if candidate.exists():
                    return candidate
    return None


def cls_model_available() -> bool:
    """二次元预筛模型是否可用。"""
    return _find_cls_model_path() is not None


def get_status() -> Dict[str, Any]:
    """获取 pixai-tagger 安装状态。"""
    venv_python = venv_python_path(VENV_DIR)
    snap_dir = _find_model_snapshot_dir()
    model_ready = (
        snap_dir is not None
        and (snap_dir / _WEIGHTS_FILE).exists()
        and (snap_dir / _TAGS_FILE).exists()
        and (snap_dir / _MAPPING_FILE).exists()
    )
    return {
        "dir_exists": PIXAI_TAGGER_DIR.is_dir(),
        "venv_exists": VENV_DIR.is_dir() and venv_python.exists(),
        "model_exists": model_ready,
        "cls_model_exists": cls_model_available(),
        "ready": VENV_DIR.is_dir() and venv_python.exists() and model_ready,
        "dir_path": str(PIXAI_TAGGER_DIR),
    }


def install_dependencies(
    pytorch_mirror: str = DEFAULT_PYTORCH_MIRROR,
    pypi_mirror: str = DEFAULT_PYPI_MIRROR,
    log_fn: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """安装 pixai-tagger 全部依赖（uv + venv + torch/timm + 两个模型）。
        Args:
        pytorch_mirror: PyTorch CUDA 镜像 ID（nju-cu128/nju-cpu/…）
        pypi_mirror: 通用 PyPI 镜像 ID（nju/tsinghua/aliyun）
        log_fn: 实时日志回调
        stop_event: 安装取消事件，取消返回 {"cancelled": True}
        Returns:
        {"ok": bool, "error": str|None}
    """
    torch_url = PYTORCH_MIRRORS.get(pytorch_mirror, PYTORCH_MIRRORS[DEFAULT_PYTORCH_MIRROR])["url"]
    pypi_url = PYPI_MIRRORS.get(pypi_mirror, PYPI_MIRRORS[DEFAULT_PYPI_MIRROR])["url"]

    PIXAI_TAGGER_DIR.mkdir(parents=True, exist_ok=True)

    def _cancelled() -> Optional[Dict[str, Any]]:
        """用户取消：返回取消结果；未取消返回 None。"""
        if stop_event is not None and stop_event.is_set():
            _log("安装已取消", log_fn)
            return {"ok": False, "cancelled": True, "error": "安装已取消"}
        return None

    def _cleanup_model_download(*paths: Path) -> None:
        """取消时清理未完成的模型下载目录，下次安装从头开始。"""
        for p in paths:
            shutil.rmtree(str(p), ignore_errors=True)

    # ── 步骤 1 ──
    _log("━━ 步骤 1/6：检测 UV ━━", log_fn)
    uv = _ensure_uv(pypi_url, log_fn, stop_event)
    r = _cancelled()
    if r:
        return r
    if not uv:
        return {"ok": False, "error": "UV 安装失败"}
    _log(f"UV 就绪: {uv}", log_fn)

    # ── 步骤 2 ──
    _log("━━ 步骤 2/6：创建虚拟环境 ━━", log_fn)
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
    is_cpu = "cpu" in torch_url
    _log(f"━━ 步骤 3/6：安装 torch（{'CPU' if is_cpu else 'CUDA'} 版）━━", log_fn)
    rc, out = _run_subprocess_streaming(
        [uv, "pip", "install", "--python", venv_py,
         "torch", "--index-url", torch_url],
        _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
    )
    r = _cancelled()
    if r:
        return r
    if rc != 0:
        return {"ok": False, "error": f"安装 torch 失败: {out[-500:]}"}

    # ── 步骤 4 ──
    _log("━━ 步骤 4/6：安装 torchvision + timm/Pillow/requests/modelscope ━━", log_fn)
    if is_cpu:
        rc, out = _run_subprocess_streaming(
            [uv, "pip", "install", "--python", venv_py,
             "torchvision", "--index-url", torch_url],
            _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
        )
    else:
        # CUDA 版 torchvision 必须单独 force-reinstall，否则装到 CPU 版报 nms 错误
        _log("→ torchvision (force-reinstall, CUDA 版)…", log_fn)
        rc, out = _run_subprocess_streaming(
            [uv, "pip", "install", "--python", venv_py,
             "torchvision", "--force-reinstall", "--no-deps", "--upgrade",
             "--index-url", torch_url],
            _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
        )
    r = _cancelled()
    if r:
        return r
    if rc != 0:
        return {"ok": False, "error": f"安装 torchvision 失败: {out[-500:]}"}
    _log("→ timm, Pillow, requests, modelscope…", log_fn)
    rc, out = _run_subprocess_streaming(
        [uv, "pip", "install", "--python", venv_py,
         "timm", "Pillow", "requests", "modelscope",
         "--index-url", pypi_url],
        _INSTALL_TIMEOUT_SEC, log_fn, stop_event,
    )
    r = _cancelled()
    if r:
        return r
    if rc != 0:
        return {"ok": False, "error": f"安装依赖失败: {out[-500:]}"}
    _log("全部依赖安装完成", log_fn)

    # ── 步骤 5 ──
    _log("━━ 步骤 5/6：从魔搭下载 PixAI Tagger 模型 ━━", log_fn)
    rc, out = _run_subprocess_streaming(
        [venv_py, "-c", _build_model_download_script()],
        _MODEL_TIMEOUT_SEC, log_fn, stop_event,
    )
    r = _cancelled()
    if r:
        _cleanup_model_download(PIXAI_TAGGER_DIR / "ms_cache", MODEL_DIR / "models")
        return r
    if rc != 0:
        return {"ok": False, "error": f"模型下载失败: {out[-500:]}"}
    _log("PixAI Tagger 模型下载完成", log_fn)

    # ── 步骤 6 ──
    _log("━━ 步骤 6/6：下载预筛模型（anime_real_cls，约 32MB）━━", log_fn)
    rc, out = _run_subprocess_streaming(
        [venv_py, "-c", _build_cls_model_download_script()],
        _MODEL_TIMEOUT_SEC, log_fn, stop_event,
    )
    r = _cancelled()
    if r:
        # 预筛模型下载缓存独立于主模型，取消只清 cls_cache，
        # 不动已装好的主模型（models/）与共享 blob 缓存（残留 blob 下次可续传）
        _cleanup_model_download(MODEL_DIR / "cls_cache")
        return r
    if rc != 0:
        # 预筛模型下载失败不阻断安装，只是标签获取时无法跳过非二次元视频
        _log(f"预筛模型下载失败（不影响标签获取主功能）: {out[-200:]}", log_fn)
    else:
        _log("二次元预筛模型下载完成", log_fn)

    _log("━━ 全部完成 ━━", log_fn)
    return {"ok": True, "error": None}


def _build_model_download_script() -> str:
    """构建主模型下载脚本（在 venv python 中执行）。"""
    return f"""
import os, sys
os.environ['MODELSCOPE_CACHE'] = r'{str(PIXAI_TAGGER_DIR / "ms_cache")}'
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('{MODEL_ID}', cache_dir=r'{str(MODEL_DIR)}')
print(f'模型已下载到: {{model_dir}}')
"""


def _build_cls_model_download_script() -> str:
    """构建 anime_real_cls 模型下载脚本。"""
    target_dir = MODEL_DIR / "anime_real_cls" / _CLS_MODEL_SUBDIR
    allow_file = f"{_CLS_MODEL_SUBDIR}/{_CLS_CKPT_FILE}"
    return f"""
import os, sys, shutil
from pathlib import Path
os.environ['MODELSCOPE_CACHE'] = r'{str(PIXAI_TAGGER_DIR / "ms_cache")}'
from modelscope.hub.snapshot_download import snapshot_download
print('正在下载预筛模型（仅 {allow_file}）…')
model_dir = snapshot_download(
    '{CLS_MODEL_ID}',
    cache_dir=r'{str(MODEL_DIR / "cls_cache")}',
    allow_patterns=['{allow_file}'],
)
print(f'下载完成: {{model_dir}}')
# 将 ckpt 复制到统一目录（方便查找，不依赖 snapshot 布局）
src = Path(model_dir) / '{_CLS_MODEL_SUBDIR}' / '{_CLS_CKPT_FILE}'
if not src.exists():
    src = Path(model_dir) / '{_CLS_CKPT_FILE}'
if src.exists():
    dest_dir = Path(r'{str(target_dir)}')
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / '{_CLS_CKPT_FILE}'
    if not dest.exists():
        shutil.copy2(str(src), str(dest))
    print(f'模型已放置到: {{dest}}')
else:
    print(f'未找到 {_CLS_CKPT_FILE}，列出已下载文件:')
    for p in Path(model_dir).rglob('*'):
        if p.is_file():
            print(f'  {{p}}')
    sys.exit(1)
# 复制完成即删下载缓存：ckpt 已放统一目录（_find_cls_model_path 只查那里），
# cls_cache 留着只会让模型双份占盘
try:
    shutil.rmtree(r'{str(MODEL_DIR / "cls_cache")}', ignore_errors=True)
    print('已清理下载缓存 cls_cache')
except Exception as e:
    print(f'清理 cls_cache 失败（可手动删除）: {{e}}')
"""


def remove_pixai_tagger() -> Dict[str, Any]:
    """删除 pixai-tagger 文件夹与模块数据文件夹（_workspace/pixai/）。"""
    existed = PIXAI_TAGGER_DIR.exists()
    if existed:
        try:
            shutil.rmtree(PIXAI_TAGGER_DIR)
        except Exception as e:
            return {"ok": False, "error": f"删除失败: {e}"}
    # 模块配置/开关/标签结果一并删除（可插拔：删除模块后不留任何模块痕迹）
    try:
        from .workspace_paths import PIXAI_CONFIG_FILE
        shutil.rmtree(PIXAI_CONFIG_FILE.parent, ignore_errors=True)
    except Exception:
        pass
    return {"ok": True,
            "message": "已删除 pixai-tagger 文件夹" if existed else "目录不存在，无需删除"}


# ── 推理（标签获取 / 二次元预筛） ──

def _inference_timeout(video_count: int) -> float:
    return _INFERENCE_BASE_SEC + _INFERENCE_PER_VIDEO_SEC * max(1, video_count)


class AnalyzeStream:
    """交互式分析子进程：启动即加载双模型（与抽帧并行），stdin 逐视频送入、stdout 流式回传。
        {"video_result": {...}}（单子进程串行，响应顺序 = 请求顺序）
    """

    def __init__(self, params: Dict[str, Any], video_count: int,
                 stop_event=None,
                 on_log: Optional[Callable[[str], None]] = None,
                 on_video_result: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.stop_event = stop_event
        self.on_log = on_log
        self.on_video_result = on_video_result
        self.per_video: List[Dict[str, Any]] = []
        self.broken = False  # 子进程异常/停止/超时后置位，调用方应停止发送
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
                                pass  # 回调异常不影响子进程收尾
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
        """发送一个视频请求（帧 base64，一行 JSON）。子进程已退出时返回 False。"""
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
        """阻塞等待下一个 video_result（响应顺序 = 请求顺序）。

        停止/超时/子进程提前退出时返回 None 并置位 broken（close 负责 kill 收尾）。
        """
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
                # 队列空才检查子进程是否已退出，避免丢弃已入队结果
                if self.p.poll() is not None:
                    self.broken = True
                    return None
                continue
            if vr is None:  # stdout EOF
                self.broken = True
                return None
            return vr

    def try_next_result(self) -> Optional[Dict[str, Any]]:
        """非阻塞版 next_result：无就绪结果时返回 None 且不置位 broken
        （供 send-ahead 管道机会性排空）；仅在引擎终止（EOF/提前退出）
        时置位 broken——超时/停止仍由阻塞版 next_result / close 判定。"""
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
        """关闭 stdin（EOF 触发子进程处理完已发送请求后退出）并回收。
            Returns:
            {"ok": bool, "per_video": [...], "error": str|None}
        """
        stopped = self.stop_event is not None and self.stop_event.is_set()
        if stopped:
            try:
                _terminate_proc(self.p, wait_sec=1.0)
            except Exception:
                pass
        elif not self.broken:
            # 正常收尾：EOF → 子进程读完 stdin 缓冲的请求后退出
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
    tag_threshold: float = CHARACTER_THRESHOLD,
    stop_event=None,
    on_log: Optional[Callable[[str], None]] = None,
    on_video_result: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Optional[AnalyzeStream]:
    """启动交互式分析子进程（点击后立即加载双模型，与抽帧并行）。"""
    status = get_status()
    if not status["ready"]:
        return None
    snap_dir = _find_model_snapshot_dir()
    if not snap_dir:
        return None
    cls_path = _find_cls_model_path()
    if not cls_path:
        # 预筛模型缺失（安装第 6 步失败被容忍）：脚本内降级为不预筛直接标签获取
        cls_path = ""
    params = {
        "model_dir": str(snap_dir),
        "cls_model_path": str(cls_path),
        "skip_real": skip_real,
        "anime_threshold": anime_threshold,
        "real_threshold": real_threshold,
        "tag_threshold": tag_threshold,
    }
    try:
        return AnalyzeStream(params, video_count,
                             stop_event=stop_event, on_log=on_log,
                             on_video_result=on_video_result)
    except Exception:
        return None  # 启动失败：调用方按「分析引擎启动失败」整体上报


# ── 分析脚本（venv 中执行）：双模型各加载一次，逐视频流式回传 {"video_result": {...}}；失败输出 {"error": ...} 非零退出 ──
_ANALYZE_SCRIPT = r"""
import sys, json, os, io, base64, queue, time
from pathlib import Path

# 强制离线模式，禁止任何网络请求
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageStat
import timm

def plog(msg):
    print(json.dumps({'log': msg}, ensure_ascii=False), flush=True)

def report(video_result):
    print(json.dumps({'video_result': video_result}, ensure_ascii=False), flush=True)

params_file = sys.argv[1]
with open(params_file, 'r', encoding='utf-8') as f:
    params = json.load(f)

model_dir = Path(params['model_dir'])
cls_model_path = params.get('cls_model_path') or ''
skip_real = bool(params.get('skip_real', False))
cls_threshold = float(params.get('anime_threshold', 0.72))
real_threshold = float(params.get('real_threshold', 0.50))
tag_threshold = float(params.get('tag_threshold', 0.9))

weights_file = model_dir / 'model_v0.9.pth'
tags_file = model_dir / 'tags_v0.9_13k.json'
mapping_file = model_dir / 'char_ip_map.json'

for f, label in ((weights_file, '模型权重'), (tags_file, '标签文件'), (mapping_file, 'IP映射文件')):
    if not f.exists():
        print(json.dumps({'error': f'{label}不存在: {f}'}))
        sys.exit(1)

# ── 加载标签与 IP 映射 ──
with tags_file.open('r', encoding='utf-8') as f:
    tag_info = json.load(f)
tag_map = tag_info['tag_map']
gen_tag_count = tag_info['tag_split']['gen_tag_count']
index_to_tag = {v: k for k, v in tag_map.items()}
with mapping_file.open('r', encoding='utf-8') as f:
    char_ip_mapping = json.load(f)

# ── 构建标签获取模型 ──
class TaggingHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(input_dim, num_classes))
    def forward(self, x):
        return torch.nn.functional.sigmoid(self.head(x))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# use_fp16 在 warmup 精度自测后确定（GTX 16 系等无 Tensor Core 的卡 fp16 反而慢）

# ── 预筛模型（与主模型总是一起安装；缺失时降级为不预筛直接标签获取）──
has_cls = bool(cls_model_path) and Path(cls_model_path).exists()
cls_model = None
if has_cls:
    cls_model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=2)
    # ckpt 为 PyTorch Lightning 格式且含 numpy 标量，可信来源，weights_only=False
    ckpt = torch.load(str(cls_model_path), map_location=device, weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    # 过滤非权重项（total_ops/total_params 等）+ 逐层去除可能的前缀
    filtered = {}
    for k, v in state_dict.items():
        if 'total_ops' in k or 'total_params' in k or not isinstance(v, torch.Tensor):
            continue
        new_key = k
        for prefix in ('model.', 'backbone.', 'module.'):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        filtered[new_key] = v
    cls_model.load_state_dict(filtered, strict=False)
    cls_model.to(device)
    cls_model.eval()
else:
    plog('未找到预筛模型，跳过预筛直接标签获取')

# ── 标签获取模型 ──
encoder = timm.create_model('eva02_large_patch14_448', pretrained=False)
encoder.reset_classifier(0)
tag_model = nn.Sequential(encoder, TaggingHead(1024, 13461))
states_dict = torch.load(str(weights_file), map_location=device, weights_only=True)
tag_model.load_state_dict(states_dict)
tag_model.to(device)
tag_model.eval()

# ── 预处理 ──
tag_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
cls_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ── 坏帧过滤：过暗帧（片头尾黑场/转场/字幕底）与纯色帧（纯色转场/虚焦/灰场）
# 会让预筛误判；纯色帧对二分类模型是分布外输入，打分不可信
_DARK_BRIGHTNESS = 15.0  # 亮度低于此值的帧视为坏帧

def _mean_brightness(image):
    '''粗略亮度（0-255）：缩到 64px 灰度取均值，成本极低。'''
    return ImageStat.Stat(image.convert('L').resize((64, 64))).mean[0]

def _is_flat_frame(image):
    '''纯色/低信息量帧：方差<8（纯色/虚焦）或 方差<60 且饱和度<0.05（灰阶转场）。
    正常画面帧方差通常远大于此，误伤风险极低。'''
    g = image.convert('L').resize((64, 64))
    stddev = ImageStat.Stat(g).stddev[0]
    if stddev >= 60:
        return False
    if stddev < 8:
        return True
    rgb = image.convert('RGB').resize((32, 32))
    r, gr, b = ImageStat.Stat(rgb).mean
    return (max(r, gr, b) - min(r, gr, b)) / 255.0 < 0.05

# ── warmup：双模型各跑一次 dummy 前向，预热 CUDA kernel/cuDNN 与显存分配器，
# 避免首个视频推理撞上初始化开销造成首次毛刺。
# CUDA 下同时按前向耗时在 fp32/fp16 间自适应选优（无 Tensor Core 的卡 fp16 反而更慢，
# 不能盲目启用）。──
use_fp16 = False
try:
    _dummy = torch.zeros(1, 3, 384, 384).to(device)
    with torch.inference_mode():
        if cls_model is not None:
            cls_model(_dummy)
    _dummy = torch.zeros(1, 3, 448, 448).to(device)
    with torch.inference_mode():
        tag_model(_dummy)
    if device == 'cuda':
        def _time_fwd(model, x, dtype, iters=2):
            model.to(dtype)
            x = x.to(dtype)
            with torch.inference_mode():
                model(x)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    model(x)
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters
        t_fp32 = _time_fwd(tag_model, _dummy, torch.float32)
        t_fp16 = _time_fwd(tag_model, _dummy, torch.float16)
        use_fp16 = t_fp16 < t_fp32
        plog(f'精度自测: fp32 {t_fp32*1000:.0f}ms/次, fp16 {t_fp16*1000:.0f}ms/次'
             f' → 使用 {"fp16" if use_fp16 else "fp32"}')
    del _dummy
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
except Exception as _e:
    plog(f'warmup 失败（忽略）: {_e}')
    use_fp16 = False
# 自测结论统一应用：双模型保持同一 dtype
if use_fp16:
    if cls_model is not None:
        cls_model.half()
    tag_model.half()
else:
    if cls_model is not None:
        cls_model.float()
    tag_model.float()

def pil_to_rgb(image):
    if image.mode in ('RGBA', 'P'):
        if image.mode == 'P':
            image = image.convert('RGBA')
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    return image.convert('RGB')

def decode_frames(b64_list):
    '''base64 → PIL Image（内存解码，不落盘）；坏帧返回 None 占位。
    单帧失败不记日志：该视频最终会以「分析失败: 帧解码失败」在父进程呈现。'''
    frames = []
    for b64 in b64_list:
        try:
            frames.append(pil_to_rgb(Image.open(io.BytesIO(base64.b64decode(b64)))))
        except Exception:
            frames.append(None)
    return frames

def _run_cls_frames(tensors):
    '''预筛：全部帧一次 batch 前向，返回逐帧 anime 分数（预处理已在预取线程完成）；
    CUDA OOM 时对半拆分重试。'''
    try:
        batch = torch.stack(tensors)
        if use_fp16:
            batch = batch.half()
        if device == 'cuda':
            batch = batch.pin_memory().to(device, non_blocking=True)
        else:
            batch = batch.to(device)
        with torch.inference_mode():
            probs = torch.softmax(cls_model(batch), dim=1)
        return [float(p[0]) for p in probs]  # 索引 0 = anime
    except torch.cuda.OutOfMemoryError:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        mid = len(tensors) // 2
        if mid == 0:
            raise
        return _run_cls_frames(tensors[:mid]) + _run_cls_frames(tensors[mid:])

def _run_tag_frames(tensors):
    '''标签获取：全部帧一次 batch 前向，逐帧 sigmoid + 逐标签跨帧 max 合并
    （预处理已在预取线程完成）；CUDA OOM 时对半拆分重试。'''
    merged = {}
    try:
        batch = torch.stack(tensors)
        if use_fp16:
            batch = batch.half()
        if device == 'cuda':
            batch = batch.pin_memory().to(device, non_blocking=True)
        else:
            batch = batch.to(device)
        with torch.inference_mode():
            probs = tag_model(batch)
        for prob in probs:
            char_probs = prob[gen_tag_count:]
            indices = (char_probs > tag_threshold).nonzero(as_tuple=True)[0]
            for idx in indices:
                global_idx = idx.item() + gen_tag_count
                tag_name = index_to_tag.get(global_idx)
                score = float(char_probs[idx])
                if tag_name and (tag_name not in merged or score > merged[tag_name]):
                    merged[tag_name] = score
        return merged
    except torch.cuda.OutOfMemoryError:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        mid = len(tensors) // 2
        if mid == 0:
            raise
        left = _run_tag_frames(tensors[:mid])
        right = _run_tag_frames(tensors[mid:])
        for k, v in right.items():
            if k not in left or v > left[k]:
                left[k] = v
        return left

# ── 预取：后台线程读 stdin → JSON 解析 → 帧解码 → 双模型 transform，产出就绪
# tensor 入有界队列；主线程只做 GPU 推理，视频间不留 CPU 间隙。
# 父进程逐行发送 {"index","name","frames"}；stdin EOF（close）即正常结束。
# 队列项：就绪请求 {'vi','name','cls_t','tag_t'} / 待回传错误 {'report'} / EOF 哨兵 None。
import threading
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
            del b64s  # 无论成败先释放 base64
        if not imgs:
            _prefetch_q.put({'report': {'index': vi, 'name': name, 'error': '帧解码失败'}})
            continue
        # 坏帧过滤：剔除过暗帧与纯色帧；剩余不足 2 帧时回退全量（极端暗视频仍有帧可用）
        bright = [im for im in imgs if _mean_brightness(im) >= _DARK_BRIGHTNESS
                  and not _is_flat_frame(im)]
        if len(bright) >= 2:
            imgs = bright
        try:
            cls_t = ([cls_transform(im) for im in imgs]
                     if cls_model is not None else [])
            tag_t = [tag_transform(im) for im in imgs]
        except Exception as e:
            _prefetch_q.put({'report': {'index': vi, 'name': name,
                                        'error': f'帧预处理异常: {e}'}})
            continue
        finally:
            del imgs  # transform 完成即释放 PIL 帧
        _prefetch_q.put({'vi': vi, 'name': name,
                         'cls_t': cls_t, 'tag_t': tag_t})
    _prefetch_q.put(None)  # stdin EOF → 正常结束

threading.Thread(target=_prefetch_reader, name='prefetch', daemon=True).start()

while True:
    item = _prefetch_q.get()
    if item is None:
        break  # stdin EOF：已预取的视频均已处理完
    if 'report' in item:
        report(item['report'])
        continue
    vi, name = item['vi'], item['name']
    cls_t, tag_t = item['cls_t'], item['tag_t']

    is_anime = None  # None = 不确定
    anime_score = 0.0
    cls_ok = False
    if cls_model is not None:
        try:
            scores = _run_cls_frames(cls_t)
        except Exception as e:
            plog(f'预筛推理异常 {name}: {e}（继续标签获取，不标记分类）')
            scores = []
        finally:
            del cls_t  # 预筛完成即释放该套 tensor
        if scores:
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            med = (sorted_scores[n // 2] + sorted_scores[(n - 1) // 2]) / 2  # 中位数
            if med >= cls_threshold:
                is_anime = True
                verdict = '二次元'
            elif med <= real_threshold:
                is_anime = False
                verdict = '非二次元'
            else:
                verdict = '不确定'
            cls_ok = True
            anime_score = round(med, 4)
            plog(f'预筛 {name}: 中位数={anime_score} → {verdict}')

    if skip_real and cls_ok and is_anime is False:
        report({'index': vi, 'name': name, 'anime_score': anime_score,
                'is_anime': False, 'character_tags': [], 'ip_tags': []})
        del tag_t
        continue

    # ── 标签获取（逐标签跨帧 max 合并 + char_ip_map IP 并集）──
    try:
        char_scores = _run_tag_frames(tag_t)
    except Exception as e:
        report({'index': vi, 'name': name, 'error': f'标签获取异常: {e}'})
        del tag_t
        continue
    del tag_t  # 处理完成即释放该视频 tensor
    ip_set = set()
    for char_tag in char_scores:
        if char_tag in char_ip_mapping:
            ip_set.update(char_ip_mapping[char_tag])
    result = {
        'index': vi, 'name': name,
        'character_tags': [{'name': k, 'score': round(v, 4)}
                           for k, v in sorted(char_scores.items(), key=lambda x: -x[1])],
        'ip_tags': [{'name': ip} for ip in sorted(ip_set)],
    }
    # 预筛成功时才附带分类信息（供前端 ANIME/REAL/UNC 角标）
    if cls_ok:
        result['anime_score'] = anime_score
        result['is_anime'] = is_anime
    report(result)
"""
