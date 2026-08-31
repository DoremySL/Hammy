"""llama_cpp.py — 简易 llama.cpp 安装/启动器：拉起 llama-server 并注册到子进程表，主程序退出自动终止。"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .env import APP_ROOT
from .installer import detect_gpu, start_cancel_watcher
from .js_push import js_pusher
from batch_rename.env import SUBPROCESS_KWARGS
from batch_rename.subprocess_registry import register_subprocess, unregister_subprocess

# ── 路径 ──
LLAMA_DIR = APP_ROOT / "llama.cpp"
# 模型文件夹与 llama.cpp 同级（卸载 llama.cpp 时不删除，可复用）
MODELS_DIR = LLAMA_DIR.parent / "models"
EXE_NAME = "llama-server.exe"


def get_models_dir() -> Path:
    """返回模型文件夹路径。"""
    try:
        from .config_store import load_llama_config
        custom = load_llama_config().get("models_dir", "")
        if custom and str(custom).strip():
            p = Path(str(custom).strip())
            if not p.is_absolute():
                p = APP_ROOT / p
            return p
    except Exception:
        pass
    return MODELS_DIR


# 只需第一个含 win 构建的 release，per_page 过大会多传数倍 JSON
GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5"

# ── 默认启动参数 ──
_CPU = 6
DEFAULTS: Dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8080,
    "threads": _CPU,
    "threads_batch": _CPU,
    "ngl": 999,
    "ctx": 10240,
    "batch": 4096,
    "ubatch": 512,
    "npredict": -1,
    "parallel": 1,
    "timeout": 1200,
    "kv_quant": "q8_0",
    "kv_preset": "",
    "reasoning_mode": "off",
    "load_mode": "none",
    "image_min_tokens": "",
    "image_max_tokens": "",
    "extra_args": "",
    "alias": "",
    "mmproj": "",
    "mmproj_auto": True,
    "no_mmproj_offload": False,
    # MTP 投机解码草稿令牌数上限（"" = 不启用）
    "spec_draft_n": "",
    # MoE 模型专用：把前 N 个专家层权重留在 CPU（0 = 全部上 GPU）
    "n_cpu_moe": 0,
}

# ── 运行时状态 ──
_cache_lock = threading.Condition()
_release_fetching = False
_cached_release: Optional[Dict[str, Any]] = None

_llama_proc: Optional[subprocess.Popen] = None
_llama_state: Dict[str, Any] = {
    "running": False, "pid": None, "port": None, "model": None,
    "launch_params": None,
    "starting": False,
    "launch_failed": None,
}
_state_lock = threading.Lock()

_launch_lock = threading.Lock()

_paused_launch: Optional[Dict[str, Any]] = None


def _stderr_log(m: str) -> None:
    sys.stderr.write(f"[llama] {m}\n")


def _notify_state_change():
    """启动/停止状态变化后通知前端刷新顶部胶囊（集成模式下显示 4 态）。"""
    try:
        js_pusher.push("llamaStateChanged")
    except Exception:
        pass


# ── Release 抓取与解析 ──

_FETCH_RETRIES = 3
_FETCH_BASE_DELAY = 1.0


def _fetch_json(url: str, timeout: float = 20.0,
                cancel_event: Optional[threading.Event] = None) -> Any:
    """抓取并解析 JSON；网络抖动时退避重试。"""
    last_err: Optional[Exception] = None
    for attempt in range(1, _FETCH_RETRIES + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise InstallCancelled()
        try:
            # 声明可接受 gzip：GitHub 的 JSON 压缩后体积约为原始的 1/10
            req = urllib.request.Request(url, headers={
                "User-Agent": "video-rename-gui",
                "Accept-Encoding": "gzip",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                stop_watch = None
                if cancel_event is not None:
                    stop_watch = start_cancel_watcher(cancel_event, resp.close)
                try:
                    raw = resp.read()
                    # urllib 不自动解压；magic 兜底覆盖代理强制压缩但不回传编码头的情形
                    if (resp.headers.get("Content-Encoding", "").lower() == "gzip"
                            or raw[:2] == b"\x1f\x8b"):
                        raw = gzip.decompress(raw)
                    return json.loads(raw.decode("utf-8"))
                finally:
                    if stop_watch is not None:
                        stop_watch()
        except InstallCancelled:
            raise
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt >= _FETCH_RETRIES:
                raise
            last_err = e
        except Exception as e:
            if cancel_event is not None and cancel_event.is_set():
                raise InstallCancelled()
            last_err = e
            if attempt >= _FETCH_RETRIES:
                raise
        if attempt < _FETCH_RETRIES:
            if cancel_event is not None and cancel_event.is_set():
                raise InstallCancelled()
            time.sleep(_FETCH_BASE_DELAY * attempt)
    raise last_err if last_err is not None else RuntimeError(f"抓取失败: {url}")


def _norm_proxy(proxy: str) -> str:
    """规范化加速代理站地址：去首尾空白/尾部斜杠；缺协议时补 https://。"""
    p = (proxy or "").strip().rstrip("/")
    if p and not re.match(r"^https?://", p):
        p = "https://" + p
    return p


def _proxy_url(proxy: str, url: str) -> str:
    """把 GitHub 下载地址前缀为加速代理站地址（gh-proxy 类代理：代理站 + 完整原地址）。"""
    p = _norm_proxy(proxy)
    if not p or not url.startswith("http"):
        return url
    return f"{p}/{url}"


# ── 预编译二进制资源正则（CUDA 版含独立 cudart；Vulkan/CPU 版单个压缩包）──
_PRE_RES = [
    ("cuda", re.compile(r"llama-.*-bin-win-cuda-(\d+\.\d+)-x64\.zip$")),
    ("vulkan", re.compile(r"llama-.*-bin-win-vulkan-x64\.zip$")),
    ("cpu", re.compile(r"llama-.*-bin-win-cpu-x64\.zip$")),
]
_CUDART_RE = re.compile(r"cudart-llama-bin-win-cuda-(\d+\.\d+)-x64\.zip$")

# release 缓存有效期（秒）：避免一直使用过期版本列表
_CACHE_TTL = 10 * 60
_cached_release_at = 0.0

# 启动健康检查超时（秒）：大模型加载可能更慢，但超时后会自动停止未就绪进程
_HEALTH_TIMEOUT_SEC = 90.0


def _ver_tuple(v: str) -> Tuple[int, ...]:
    """把 '12.4' 解析为 (12, 4)，用于精确版本比较（避免 float 误判 12.10→12.1）。"""
    try:
        return tuple(int(x) for x in str(v).split("."))
    except (ValueError, TypeError):
        return (0,)


def _build_key(b: Dict[str, Any]) -> str:
    """构建唯一标识，用作前端选项值。cuda 带版本号，其余用类型名。"""
    if b.get("type") == "cuda":
        return f"cuda-{b.get('ver', '')}"
    return b.get("type", "unknown")


def _parse_build_sel(sel: str, builds: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """把前端传回的 build_sel 解析为构建字典（如 "cuda-12.8"）；manual/空返回 None。"""
    if not sel or sel == "manual":
        return None
    return next((b for b in builds if _build_key(b) == sel), None)


def _parse_release_assets(data: Any) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """解析单个 release 的 win 构建资产；无可用构建返回 None。"""
    if not isinstance(data, dict):
        return None
    assets = data.get("assets", []) or []
    pairs: Dict[str, Dict[str, Any]] = {}

    def _new_pair(t: str, ver: str) -> Dict[str, Any]:
        return {"type": t, "ver": ver, "precompiled_url": "", "cudart_url": "",
                "precompiled_size": 0, "cudart_size": 0}

    for a in assets:
        name = a.get("name", "")
        url = a.get("browser_download_url", "")
        size = int(a.get("size") or 0)  # 供下载后完整性校验
        for t, rgx in _PRE_RES:
            m = rgx.search(name)
            if m:
                ver = m.group(1) if m.groups() else ""
                key = f"{t}-{ver}" if ver else t
                pairs.setdefault(key, _new_pair(t, ver))
                pairs[key]["precompiled_url"] = url
                pairs[key]["precompiled_size"] = size
                break
        m = _CUDART_RE.search(name)
        if m:
            ver = m.group(1)
            key = f"cuda-{ver}"
            pairs.setdefault(key, _new_pair("cuda", ver))
            pairs[key]["cudart_url"] = url
            pairs[key]["cudart_size"] = size

    order = {"cuda": 0, "vulkan": 1, "cpu": 2}
    builds = []
    for info in pairs.values():
        if not info["precompiled_url"]:
            continue
        b = {"type": info["type"], "ver": info["ver"],
             "precompiled_url": info["precompiled_url"], "cudart_url": info.get("cudart_url", ""),
             "precompiled_size": info.get("precompiled_size", 0),
             "cudart_size": info.get("cudart_size", 0)}
        b["key"] = _build_key(b)
        builds.append(b)

    if not builds:
        return None

    def _sort_key(b: Dict[str, Any]) -> Tuple[int, int, int]:
        """排序键：类型优先（CUDA 前），CUDA 版本高者优先（按段比较，避免 12.10→12.1 误判）。"""
        vt = _ver_tuple(b["ver"]) if b.get("ver") else (0,)
        return (order.get(b["type"], 9),
                -vt[0] if vt else 0,
                -vt[1] if len(vt) > 1 else 0)

    builds.sort(key=_sort_key)
    return (str(data.get("tag_name", "")), builds)


def get_latest_release_assets(force: bool = False,
                              stop_event: Optional[threading.Event] = None) -> Dict[str, Any]:
    """拉取最新可用 release 并解析各平台的预编译 + cudart 资源。"""
    global _cached_release, _cached_release_at, _release_fetching
    now = time.time()
    with _cache_lock:
        if not force and _cached_release is not None and (now - _cached_release_at) < _CACHE_TTL:
            return _cached_release

        while _release_fetching:
            if stop_event is not None and stop_event.is_set():
                raise InstallCancelled()
            _cache_lock.wait(0.25)
            if _cached_release is not None:
                return _cached_release
        _release_fetching = True

    try:
        releases = _fetch_json(GITHUB_API, cancel_event=stop_event)
        if not isinstance(releases, list):
            raise RuntimeError("GitHub 返回格式异常，未获取到 release 列表")
        found = None
        for rel in releases:
            found = _parse_release_assets(rel)
            if found is not None:
                break
        if found is None:
            raise RuntimeError("最近 release 中没有可用 win 构建资产")
        tag, builds = found

        gpu = detect_gpu()
        cuda_max = gpu.get("cuda_max", "")
        recommended_key = None
        if cuda_max:
            try:
                cm = _ver_tuple(cuda_max)
                for b in builds:
                    if b["type"] == "cuda" and _ver_tuple(b["ver"]) <= cm:
                        recommended_key = b["key"]
                        break
            except (ValueError, TypeError):
                pass
        if recommended_key is None and builds and gpu.get("has_nvidia"):
            recommended_key = builds[0]["key"]

        result: Dict[str, Any] = {
            "ok": True,
            "tag": tag,
            "builds": builds,
            "recommended_key": recommended_key,
            "gpu": gpu,
        }
        with _cache_lock:
            _cached_release = result
            _cached_release_at = now
        return result
    finally:
        with _cache_lock:
            _release_fetching = False
            _cache_lock.notify_all()


# ── 状态 / 模型扫描 ──

def _is_mmproj_name(name: str) -> bool:
    return "mmproj" in name.lower()


def _is_mtp_name(name: str) -> bool:
    """mtp 开头的 gguf 是投机解码草稿模型，不作为主模型列出。"""
    return name.lower().startswith("mtp")


def _find_draft_model(model_path: Path) -> Optional[Path]:
    """投机解码草稿模型：主模型目录子树内 mtp 开头的 .gguf，多个时取体积最小，没有返回 None。"""
    try:
        candidates = [(p.stat().st_size, str(p), p)
                      for p in model_path.parent.rglob("mtp*.gguf") if p.is_file()]
        return min(candidates)[2] if candidates else None
    except OSError:
        return None


def scan_mmprojs(model_path: Path) -> List[str]:
    """列出模型同目录（含子目录）下的 mmproj 投影文件。"""
    d = Path(model_path).parent
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.rglob("*.gguf") if _is_mmproj_name(p.name))


def _pick_mmproj(model_path: Path, mmprojs: List[str]) -> str:
    if not mmprojs:
        return ""
    if len(mmprojs) == 1:
        return mmprojs[0]
    m_tokens = set(re.split(r"[^a-zA-Z0-9]+", Path(model_path).stem.lower()))
    def _score(p: str) -> int:
        p_tokens = set(re.split(r"[^a-zA-Z0-9]+", Path(p).stem.lower()))
        return len(m_tokens & p_tokens)
    return max(sorted(mmprojs), key=_score)


_GGUF_INT_TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 10: "Q", 11: "q"}


def _read_gguf_meta(path: Path, max_bytes: int = 8 << 20) -> Dict[str, Any]:
    """读取 GGUF 文件头部的 KV 元数据（大词表模型头部可达数 MB，避免截断漏读）。"""
    try:
        with open(path, "rb") as f:
            head = f.read(max_bytes)
    except OSError:
        return {}
    if len(head) < 28 or head[:4] != b"GGUF":
        return {}
    # GGUF 规范版本 1/2/3 为小端；其余（极少见）按大端处理
    endian = "<"
    try:
        ver, = struct.unpack_from("<I", head, 4)
    except struct.error:
        return {}
    if ver not in (1, 2, 3):
        endian = ">"
    out: Dict[str, Any] = {}
    try:
        pos = 8  # magic(4) + version(4) 之后：tensor_count + metadata_kv_count
        _, kv_count = struct.unpack_from(endian + "QQ", head, pos)
        pos += 16
    except struct.error:
        return {}
    for _ in range(kv_count):
        # 顺序读完全部 KV（键数有限，均在头部；不读张量区）
        try:
            n, = struct.unpack_from(endian + "Q", head, pos)
            pos += 8
            key = head[pos:pos + n].decode("utf-8", "replace")
            pos += n
            t, = struct.unpack_from(endian + "I", head, pos)
            pos += 4
            if t in _GGUF_INT_TYPES:
                fmt = endian + _GGUF_INT_TYPES[t]
                val, = struct.unpack_from(fmt, head, pos)
                pos += struct.calcsize(fmt)
                out[key] = val
            elif t == 6:  # float32
                val, = struct.unpack_from(endian + "f", head, pos)
                pos += 4
                out[key] = val
            elif t == 7:  # bool
                out[key] = bool(head[pos])
                pos += 1
            elif t == 8:  # string
                n2, = struct.unpack_from(endian + "Q", head, pos)
                pos += 8
                out[key] = head[pos:pos + n2].decode("utf-8", "replace")
                pos += n2
            elif t == 9:  # array
                at, = struct.unpack_from(endian + "I", head, pos)
                pos += 4
                cnt, = struct.unpack_from(endian + "Q", head, pos)
                pos += 8
                arr = []
                for _ in range(cnt):
                    if at in _GGUF_INT_TYPES:
                        fmt = endian + _GGUF_INT_TYPES[at]
                        v, = struct.unpack_from(fmt, head, pos)
                        pos += struct.calcsize(fmt)
                        arr.append(v)
                    elif at == 8:
                        n3, = struct.unpack_from(endian + "Q", head, pos)
                        pos += 8
                        arr.append(head[pos:pos + n3].decode("utf-8", "replace"))
                        pos += n3
                    else:
                        break
                out[key] = arr
            elif t == 12:  # float64
                val, = struct.unpack_from(endian + "d", head, pos)
                pos += 8
                out[key] = val
            else:
                break  # 未知类型：放弃继续解析
        except (struct.error, IndexError, OverflowError):
            return out
    return out


_scan_cache_lock = threading.Lock()

_fp_cache: Dict[str, Tuple[str, float]] = {}
_FP_TTL_SEC = 60.0
_last_scan: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}


def _dir_fingerprint(base: Path, force: bool = False) -> str:
    """模型目录指纹：所有 *.gguf 的（相对路径, 大小, mtime_ns）。"""
    key = str(base)
    now = time.monotonic()
    if not force:
        hit = _fp_cache.get(key)
        if hit and now - hit[1] < _FP_TTL_SEC:
            return hit[0]
    entries = []
    for p in base.rglob("*.gguf"):
        try:
            st = p.stat()
            entries.append((str(p.relative_to(base)), st.st_size, st.st_mtime_ns))
        except OSError:
            entries.append((str(p.relative_to(base)), -1, -1))
    digest = hashlib.sha1()
    for e in sorted(entries):
        digest.update(repr(e).encode("utf-8", "surrogatepass"))
    fp = digest.hexdigest()
    _fp_cache[key] = (fp, now)
    return fp


def _load_scan_cache() -> Dict[str, Any]:
    """读取模型扫描缓存（mtime 缓存）；缺失 / 损坏 / 非 dict 一律按无缓存处理。

    结构：{"dirs": {models_dir: {"fingerprint": str, "models": [...]}}}。
    """
    try:
        from .workspace_paths import LLAMA_SCAN_CACHE_FILE
        from .workspace_store import read_json_cached
        return read_json_cached(LLAMA_SCAN_CACHE_FILE)
    except Exception:
        return {}


def _save_scan_cache(models_dir: str, fingerprint: str,
                     models: List[Dict[str, Any]]) -> None:
    """写模型扫描缓存（按目录分键，互不覆盖）；写失败静默（下次全量扫描兜底）。"""
    try:
        from .workspace_paths import LLAMA_SCAN_CACHE_FILE
        from .workspace_store import write_json
        data = _load_scan_cache()
        dirs = data.setdefault("dirs", {})
        dirs[models_dir] = {"fingerprint": fingerprint, "models": models}
        write_json(LLAMA_SCAN_CACHE_FILE, data)
    except Exception:
        pass


def _scan_full(base: Path) -> List[Dict[str, Any]]:
    """全量扫描：过滤主模型，逐个读 GGUF 头并聚合同目录 mmproj。"""
    all_gguf = sorted(base.rglob("*.gguf"))
    models = [p for p in all_gguf
              if not _is_mmproj_name(p.name) and not _is_mtp_name(p.name)
              and _is_first_shard(p.name)]

    mmprojs_all = sorted(str(p) for p in all_gguf if _is_mmproj_name(p.name))
    out = []
    for p in models:
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
        except OSError:
            size_mb = 0
        info = _extract_model_info(_read_gguf_meta(p))
        pdir = str(p.parent)
        mmprojs = [m for m in mmprojs_all if _path_in_dir(m, pdir)]
        out.append({
            "name": p.name, "path": str(p), "size_mb": round(size_mb, 1),
            "mmprojs": mmprojs,
            "pick_mmproj": _pick_mmproj(p, mmprojs),
            "moe": info["moe"],
            "expert_count": info["experts"],
            "ctx": info["ctx"],
            "layers": info["layers"],
        })
    return out


def scan_models(models_dir: Optional[Path] = None,
                force: bool = False) -> List[Dict[str, str]]:
    """扫描模型文件夹下的 .gguf 模型（排除 mmproj 与分片模型的非首片）。"""
    base = models_dir or get_models_dir()
    if not base.is_dir():
        return []
    with _scan_cache_lock:
        fp = _dir_fingerprint(base, force=force)
        hit = _last_scan.get(str(base))
        if hit and hit[0] == fp:
            return hit[1]
        cached = _load_scan_cache().get("dirs", {}).get(str(base))
        if cached and cached.get("fingerprint") == fp:
            models = cached.get("models") or []
            _last_scan[str(base)] = (fp, models)
            return models
        out = _scan_full(base)
        _save_scan_cache(str(base), fp, out)
        _last_scan[str(base)] = (fp, out)
        return out


# ── per-model 参数记忆 ──
_GLOBAL_ONLY_KEYS = frozenset({
    "model", "auto_run", "show_logs", "models_dir", "last_model",
    "integrate", "enabled",
})


def model_config_updates(params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """从提交的完整参数中拆分 per-model 更新。"""
    model = str(params.get("model") or "")
    if not model:
        return "", {}
    updates: Dict[str, Any] = {}
    for k, v in params.items():
        if k == "model":
            continue
        if k == "n_cpu_moe":
            try:
                n = int(v or 0)
            except (ValueError, TypeError):
                n = 0
            updates[k] = n if n > 0 else None
        else:
            updates[k] = v
    return model, updates


def apply_model_configs(params: Dict[str, Any]) -> bool:
    """把提交参数合并进指定模型的 per-model 配置。"""
    from .config_store import update_llama_model_configs
    model, updates = model_config_updates(params)
    if not model or not updates:
        return False

    def _mutate(mcfgs):
        entry = mcfgs.setdefault(model, {})
        for k, v in updates.items():
            if v is None:
                entry.pop(k, None)
            else:
                entry[k] = v
        return mcfgs

    update_llama_model_configs(_mutate)
    return True


def ensure_model_configs(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """为扫描到的新模型初始化 per-model 配置。"""
    from .config_store import (load_llama_config, load_llama_model_configs,
                               update_llama_model_configs)
    by_path = {str(m["path"]): m for m in models if m.get("path")}
    if not by_path:
        return {}
    mcfgs = load_llama_model_configs()
    missing = [p for p in by_path if p not in mcfgs]
    if missing:
        llama = load_llama_config() or {}

        def _mutate(inner):
            for p in missing:
                entry = {k: v for k, v in llama.items()
                         if k not in _GLOBAL_ONLY_KEYS}
                entry["ctx"] = 8192  # 上下文窗口初始默认
                entry.pop("n_cpu_moe", None)
                inner[p] = entry
            return inner

        update_llama_model_configs(_mutate)
        mcfgs = load_llama_model_configs()
    return mcfgs


def _path_in_dir(p: str, d: str) -> bool:
    """p（绝对路径）是否位于目录 d 下（分隔符双向归一，杜绝前缀误判）。"""
    if not d:
        return False
    p = str(p).replace("\\", "/")
    d = str(d).replace("\\", "/").rstrip("/")
    return p == d or p.startswith(d + "/")


def _abs_dir(d: str) -> str:
    """目录配置归一为绝对路径；空值 = 默认 models/（与 llama.cpp 同级）。"""
    d = str(d or "").strip()
    if not d:
        return str(MODELS_DIR)
    p = Path(d)
    return str(p if p.is_absolute() else APP_ROOT / p)


def purge_stale_after_dir_change(old_dir: str, new_dir: str) -> bool:
    """模型目录变更后的高层清理：独立文件失效条目 + last_model。返回是否发生清理。"""
    from .config_store import (load_llama_config, load_llama_model_configs,
                               update_llama_config, update_llama_model_configs)
    if str(old_dir or "") == str(new_dir or ""):
        return False
    nd = _abs_dir(new_dir)
    mcfgs = load_llama_model_configs()
    stale = [p for p in mcfgs if not _path_in_dir(p, nd)]
    lm = str(load_llama_config().get("last_model") or "")
    lm_stale = bool(lm) and not _path_in_dir(lm, nd)
    if not stale and not lm_stale:
        return False

    if stale:
        def _mutate(inner):
            for p in stale:
                inner.pop(p, None)
            return inner

        update_llama_model_configs(_mutate)
    if lm_stale:
        def _mutate_cfg(c):
            c.pop("last_model", None)
            return c

        update_llama_config(_mutate_cfg)
    return True


def get_status() -> Dict[str, Any]:
    """返回 llama.cpp 安装/运行状态。"""
    dir_exists = LLAMA_DIR.is_dir()
    ready = (LLAMA_DIR / EXE_NAME).is_file()
    with _state_lock:
        running = bool(_llama_state["running"]) and _llama_proc is not None and _llama_proc.poll() is None
        state = dict(_llama_state)
        state["running"] = running
    return {
        "dir_exists": dir_exists,
        "ready": ready,
        "models_dir": str(get_models_dir()),
        "default_models_dir": str(MODELS_DIR),
        "models": scan_models(),
        "running": running,
        "starting": bool(state.get("starting")),      # 正在启动（Popen 后、健康检查未通过）
        "launch_failed": state.get("launch_failed"),  # 最近一次启动失败原因；None = 无
        "pid": state.get("pid"),
        "port": state.get("port"),
        "model": state.get("model"),
    }


# ── 安装 / 卸载 ──

_DOWNLOAD_TIMEOUT = 120
_DL_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


class InstallCancelled(Exception):
    """安装被用户取消（stop_install 置位停止事件后由检查点抛出）。

    与网络错误区分：外层据此返回「安装已取消」，并清理未完成的下载文件。
    """


def _download(url: str, dest: Path, log_fn, label: str,
              expected_size: int = 0,
              cancel_event: Optional[threading.Event] = None) -> None:
    """流式下载：单连接 + Range 续传，失败自动重试，按 expected_size 校验完整性。"""
    if expected_size and dest.exists() and dest.stat().st_size == expected_size:
        log_fn(f"  [{label}] 已下载完整，跳过")
        return

    attempts = 3  # 重试次数：失败靠原地重试
    last_err: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise InstallCancelled()
        try:
            _download_single(url, dest, log_fn, label, expected_size, cancel_event)
            return
        except Exception as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 416:
                # 请求的 Range 超出文件大小 = 服务器侧已完整
                if not expected_size or (dest.exists() and dest.stat().st_size == expected_size):
                    log_fn(f"  [{label}] 已下载完整，跳过")
                    return
                # 416 但大小不符：本地文件损坏，删除后从头重试
                log_fn(f"  [{label}] 本地文件大小异常，删除重下")
                dest.unlink(missing_ok=True)
                continue
            # 取消监视线程关闭响应触发的异常：按取消处理，不重试
            if cancel_event is not None and cancel_event.is_set():
                raise InstallCancelled()
            last_err = e
            if attempt < attempts:
                if cancel_event is not None and cancel_event.is_set():
                    raise InstallCancelled()
                log_fn(f"  [{label}] 下载中断（{e}），第 {attempt}/{attempts} 次尝试失败，{attempt}s 后重试…")
                time.sleep(attempt)
    raise IOError(f"下载失败（已重试 {attempts} 次，最后错误: {last_err}）")


def _download_single(url: str, dest: Path, log_fn, label: str,
                     expected_size: int = 0,
                     cancel_event: Optional[threading.Event] = None) -> None:
    """单连接顺序下载（唯一下载路径）：206 续传 / 200 从头 / Range 被忽略时覆盖重下。"""
    headers = {"User-Agent": _DL_UA}
    resumed = dest.stat().st_size if dest.exists() else 0
    if resumed > 0:
        headers["Range"] = f"bytes={resumed}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:
        status = getattr(resp, "status", 200)
        if status == 206:
            total = resumed + int(resp.headers.get("Content-Length", "0") or 0)
            log_fn(f"  继续下载 [{label}]（已下载 {resumed // 1024 // 1024} MB，从断点继续）")
        else:
            resumed = 0  # 服务器忽略 Range：从头下载
            total = int(resp.headers.get("Content-Length", "0") or 0)
            log_fn(f"  开始下载 [{label}] → {dest.name}（共 {total // 1024 // 1024} MB）")
        # 取消监视线程：置位时立即关闭响应，中断阻塞的 socket 读（网络停滞也能秒停）
        stop_watch = None
        if cancel_event is not None:
            stop_watch = start_cancel_watcher(cancel_event, resp.close)
        try:
            downloaded = resumed
            session_base = resumed      # 本次会话起点（速度只算本轮新下载）
            last_pct = -1.0
            t0 = time.time()
            with open(dest, "ab" if status == 206 else "wb") as f:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise InstallCancelled()
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total
                        if pct - last_pct >= 0.1 or downloaded >= total:
                            last_pct = pct
                            elapsed = max(time.time() - t0, 0.001)
                            speed = (downloaded - session_base) / 1024 / 1024 / elapsed
                            log_fn(f"  [{label}] 进度 {downloaded // 1024 // 1024}/{total // 1024 // 1024} MB"
                                   f"（{int(pct * 100)}%，{speed:.1f} MB/s）")
                    else:
                        log_fn(f"  [{label}] 已下载 {downloaded // 1024 // 1024} MB")
        finally:
            if stop_watch is not None:
                stop_watch()
    # 完整性校验（元数据大小来自 GitHub API）
    final = dest.stat().st_size if dest.exists() else 0
    if expected_size and final != expected_size:
        raise IOError(f"文件不完整：本地 {final} 字节，预期 {expected_size} 字节")
    log_fn(f"  [完成] [{label}] 下载完成（{final // 1024 // 1024} MB）")


def _log_network_fallback(log_fn) -> None:
    """下载失败后的手动安装指引日志。"""
    log_fn("网络原因导致失败时，请稍后再试；也可手动下载解压安装：")
    log_fn("打开 https://github.com/ggml-org/llama.cpp/releases 下载对应 zip"
           "（cuda 预编译包，内含 llama-server.exe）")
    log_fn(f"解压后放入 {LLAMA_DIR} 文件夹，重新打开设置刷新状态即可（即「手动安装」模式）。")


def _normalize_layout(target: Path) -> bool:
    """确保 llama-server.exe 位于 target 根目录；若被包了一层子目录则上提。"""
    if (target / EXE_NAME).is_file():
        return True
    nested = None
    for p in target.rglob(EXE_NAME):
        nested = p.parent
        break
    if nested and nested != target:
        for item in list(nested.iterdir()):
            dest = target / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest, ignore_errors=True)
                else:
                    dest.unlink()
            shutil.move(str(item), str(target / item.name))
        # 清理空目录
        for d in sorted([x for x in nested.rglob("*") if x.is_dir()], key=lambda x: -len(str(x))):
            try:
                d.rmdir()
            except OSError:
                pass
        try:
            nested.rmdir()
        except OSError:
            pass
    return (target / EXE_NAME).is_file()


def _is_first_shard(name: str) -> bool:
    """分片模型（形如 model-00001-of-00003.gguf）仅第 1 片含元数据，扫描时跳过其余片。"""
    m = re.search(r"-(\d+)-of-\d+\.gguf$", name)
    if not m:
        return True
    return int(m.group(1)) == 1


def _extract_model_info(meta: Dict[str, Any]) -> Dict[str, Any]:
    """从 GGUF 元数据提取 ctx/层数/专家数（键名带架构前缀，读不到返回 0）。"""
    arch = meta.get("general.architecture")
    if isinstance(arch, list):
        arch = arch[0] if arch else ""
    prefix = (str(arch) + ".") if arch else ""

    def g(*keys):
        for k in keys:
            if k in meta:
                v = meta[k]
                if isinstance(v, list):
                    continue
                try:
                    return int(v)
                except (ValueError, TypeError):
                    return 0
        return 0

    ctx = g(prefix + "context_length", "llama.context_length")
    layers = g(prefix + "block_count", "llama.block_count")
    experts = g(prefix + "expert_count", "llama.expert_count")
    return {"ctx": ctx, "layers": layers, "experts": experts, "moe": experts > 1}


def install(build_sel: str, log_fn=None, proxy: str = "",
            stop_event: Optional[threading.Event] = None) -> Dict[str, Any]:
    """安装 llama.cpp：支持 manual / cuda-<ver> / vulkan / cpu。"""
    log_fn = log_fn or _stderr_log
    proxy = _norm_proxy(proxy)

    def _check_cancel() -> None:
        if stop_event is not None and stop_event.is_set():
            raise InstallCancelled()

    def _auto_enable() -> None:
        from .config_store import update_llama_config
        update_llama_config(lambda c: c.update(enabled=True, integrate=True) or c)

    if not build_sel:
        return {"ok": False, "error": "未指定构建（manual / cuda-<ver>）"}

    if build_sel == "manual":
        # 手动安装模式：仅创建文件夹，提示用户自行放入编译版本
        LLAMA_DIR.mkdir(parents=True, exist_ok=True)
        models_dir = get_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        log_fn("已创建 llama.cpp 文件夹（手动安装模式）。")
        log_fn(f"请将对应的编译版本解压进：{LLAMA_DIR}（需包含 llama-server.exe 及其依赖文件）")
        log_fn(f"模型文件 (.gguf) 请放入：{models_dir}")
        _auto_enable()
        return {
            "ok": True,
            "manual": True,
            "message": "已创建 llama.cpp 文件夹（手动安装）。请将对应的编译版本解压放入该目录、把 .gguf 模型放入模型文件夹后刷新状态即可。",
            "dir": str(LLAMA_DIR),
            "models_dir": str(models_dir),
        }

    # 仍在运行时拒绝覆盖（避免占用中的 exe 触发 PermissionError）
    if _llama_proc is not None and _llama_proc.poll() is None:
        return {"ok": False, "error": f"llama-server 正在运行（PID {_llama_proc.pid}），请先停止服务再安装/更新"}

    try:
        _check_cancel()
        release = get_latest_release_assets(stop_event=stop_event)
    except InstallCancelled:
        return {"ok": False, "cancelled": True, "error": "安装已取消"}
    except Exception as e:
        _log_network_fallback(log_fn)
        return {"ok": False, "error": f"获取 release 失败: {e}"}
    if not release.get("ok"):
        return {"ok": False, "error": release.get("error", "获取 release 失败")}
    build = _parse_build_sel(build_sel, release["builds"])
    if build is None:
        return {"ok": False, "error": f"未找到构建: {build_sel}"}

    LLAMA_DIR.mkdir(parents=True, exist_ok=True)

    key = build["key"]
    pre_zip = LLAMA_DIR / f"_pre_{key}.zip"
    cu_zip = LLAMA_DIR / f"_cudart_{key}.zip"
    dl_errors: Dict[str, Exception] = {}
    cancelled: List[InstallCancelled] = []
    try:
        tag = build["type"] + ((" " + build["ver"]) if build["ver"] else "")
        log_fn(f"开始安装 llama.cpp（{tag}）…")
        _check_cancel()
        if proxy:
            log_fn(f"下载将使用加速代理：{proxy}（构建列表仍从官方获取）")
        if build.get("cudart_url"):
            log_fn("将下载「预编译」与「cudart」两个压缩包…")
        else:
            log_fn("将下载「预编译」压缩包（单个文件）…")

        def _dl_safe(url, dest, label, slot, size):
            try:
                _download(_proxy_url(proxy, url), dest, log_fn, label,
                          expected_size=size, cancel_event=stop_event)
            except InstallCancelled as e:
                # 用户取消：不按网络错误处理（压缩包随后由 finally 统一清理）
                cancelled.append(e)
            except Exception as e:  # 异常记入 dl_errors 回传主线程
                dl_errors[slot] = e

        t_cu = None
        if build.get("cudart_url"):
            t_cu = threading.Thread(target=_dl_safe, args=(
                build["cudart_url"], cu_zip, "cudart", "cu",
                build.get("cudart_size", 0)))
            t_cu.start()
            if stop_event is not None:
                stop_event.wait(5)
            else:
                time.sleep(5)
        t_pre = threading.Thread(target=_dl_safe, args=(
            build["precompiled_url"], pre_zip, "预编译", "pre",
            build.get("precompiled_size", 0)))
        t_pre.start()
        if t_cu:
            t_cu.join()
        t_pre.join()
        if cancelled:
            raise cancelled[0]
        if dl_errors:
            errs = "；".join(dict.fromkeys(str(v) for v in dl_errors.values()))
            _log_network_fallback(log_fn)
            if proxy:
                # 代理只作用于下载：超时/失败时大概率是代理站本身失效
                log_fn(f"提示：当前加速代理站 {proxy} 可能已失效或暂时不稳定（下载超时多为代理失效）。")
                log_fn("可用浏览器/搜索引擎搜索「GitHub 加速代理」查找最新可用地址，")
                log_fn("回到安装弹窗的「加速代理地址」中换一个再试；也可改用「GitHub 直连」或「手动安装」。")
            return {"ok": False, "error": f"下载失败: {errs}"}

        _check_cancel()
        log_fn(("两个压缩包下载完成，开始解压…" if build.get("cudart_url")
                else "压缩包下载完成，开始解压…"))
        log_fn("解压预编译二进制…")
        with zipfile.ZipFile(pre_zip) as z:
            z.extractall(LLAMA_DIR)
        if build.get("cudart_url"):
            _check_cancel()
            log_fn("解压 cudart 运行时…")
            with zipfile.ZipFile(cu_zip) as z:
                z.extractall(LLAMA_DIR)
        else:
            log_fn("（该版本无独立 cudart 包，跳过）")

        if not _normalize_layout(LLAMA_DIR):
            return {"ok": False, "error": "解压后未找到 llama-server.exe，目录结构可能异常"}
        models_dir = get_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        log_fn(f"[完成] 安装完成：llama.cpp（{tag}）已就绪，模型请放入 {models_dir}")
        _auto_enable()
        return {"ok": True,
                "message": f"llama.cpp（{tag}）安装完成",
                "dir": str(LLAMA_DIR),
                "models_dir": str(models_dir)}
    except InstallCancelled:
        log_fn("安装已取消。")
        return {"ok": False, "cancelled": True, "error": "安装已取消"}
    except Exception as e:
        return {"ok": False, "error": f"安装失败: {e}"}
    finally:
        pre_zip.unlink(missing_ok=True)
        cu_zip.unlink(missing_ok=True)


def remove(log_fn=None) -> Dict[str, Any]:
    """卸载：先停止运行中的服务，再删除 llama.cpp 目录；删除失败如实返回。"""
    log_fn = log_fn or _stderr_log
    try:
        stop(log_fn=log_fn)
    except Exception as e:
        log_fn(f"停止服务时出错（继续尝试删除）: {e}")

    if LLAMA_DIR.is_dir():
        last_err: Optional[Exception] = None
        for _attempt in range(3):
            try:
                shutil.rmtree(LLAMA_DIR)
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(0.5)
        if last_err is not None:
            leftover = [str(p) for p in LLAMA_DIR.rglob("*") if p.is_file()]
            detail = f"（{len(leftover)} 个文件可能被占用: {leftover[0]}{'…' if len(leftover) > 1 else ''}）" if leftover else ""
            return {"ok": False, "error": f"删除失败: {last_err}{detail}"}
    if LLAMA_DIR.is_dir():
        return {"ok": False, "error": "目录仍然存在，可能文件被占用，无法卸载"}
    log_fn("已卸载 llama.cpp。")
    try:
        from .config_store import update_llama_config
        update_llama_config(lambda c: c.update(enabled=False, integrate=False) or c)
    except Exception as e:
        log_fn(f"关闭 llama 开关时出错: {e}")
    return {"ok": True, "message": "已卸载 llama.cpp"}


# ── 启动 / 停止 ──

def _model_layers(model_path: Path) -> int:
    """从 GGUF 头部读模型层数；读取失败返回 0。"""
    try:
        return int(_extract_model_info(_read_gguf_meta(model_path))["layers"])
    except Exception:
        return 0


def _port_in_use(host: str, port: int) -> bool:
    """端口占用预检：connect 成功 = 已有监听者。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _local_host(host: str) -> str:
    """监听地址归一为本机可探测地址（0.0.0.0/:: → 127.0.0.1）。"""
    return "127.0.0.1" if host in ("0.0.0.0", "::", "") else host


def _to_int(v: Any, default: int) -> int:
    """宽松转 int；失败返回 default。"""
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _build_args(model_path: Path, p: Dict[str, Any]) -> List[str]:
    def num(key, default):
        return _to_int(p.get(key, default), default)

    # ngl 恰好等于层数时视为「全量卸载」，自动改写为 999：
    # llama.cpp 的卸载单位是「N 个块 + 输出层」共 N+1 个、从后往前分（i_gpu_start = N+1-ngl），
    # ngl = 层数时恰好把块 0 留在 CPU；tensor 并行下该 CPU 块每步串行执行，吞吐显著下降
    ngl = num("ngl", DEFAULTS["ngl"])
    if ngl > 0:
        layers = _model_layers(model_path)
        if layers > 0 and ngl == layers:
            ngl = 999

    args: List[str] = [
        "-m", str(model_path),
        "--host", str(p.get("host", DEFAULTS["host"])),
        "--port", str(num("port", DEFAULTS["port"])),
        "-t", str(num("threads", DEFAULTS["threads"])),
        "-tb", str(num("threads_batch", DEFAULTS["threads_batch"])),
        "-ngl", str(ngl),
        "-c", str(num("ctx", DEFAULTS["ctx"])),
        "-b", str(num("batch", DEFAULTS["batch"])),
        "-ub", str(num("ubatch", DEFAULTS["ubatch"])),
        "-n", str(num("npredict", DEFAULTS["npredict"])),
        "-np", str(num("parallel", DEFAULTS["parallel"])),
        "--timeout", str(num("timeout", DEFAULTS["timeout"])),
    ]
    # 缓存预设（下拉）：统一 KV 缓存 / 禁用提示缓存与检查点 / 两者组合；空 = 默认不传参
    kv_preset = str(p.get("kv_preset", DEFAULTS["kv_preset"]) or "").strip()
    if kv_preset == "kv_unified":
        args.append("--kv-unified")
    elif kv_preset == "disable_reuse":
        args += ["--cache-ram", "0", "--ctx-checkpoints", "0"]
    elif kv_preset == "kv_unified_disable_reuse":
        args += ["--kv-unified", "--cache-ram", "0", "--ctx-checkpoints", "0"]
    # MoE 模型：把前 N 个专家层权重留在 CPU（0 = 不启用）
    n_cpu_moe = num("n_cpu_moe", 0)
    if n_cpu_moe > 0:
        args += ["--n-cpu-moe", str(n_cpu_moe)]
    # 加载模式：空 = 自动（不传参，llama.cpp 自行决定）；非法值忽略
    lm = str(p.get("load_mode", DEFAULTS["load_mode"]) or "").strip().lower()
    if lm in ("mmap", "mmap+mlock", "none", "mlock", "dio"):
        args += ["--load-mode", lm]
    # 视觉 token 预算：空 = 自动（不传参）；仅接受纯数字
    imin = str(p.get("image_min_tokens", DEFAULTS["image_min_tokens"]) or "").strip()
    if imin.isdigit():
        args += ["--image-min-tokens", imin]
    imax = str(p.get("image_max_tokens", DEFAULTS["image_max_tokens"]) or "").strip()
    if imax.isdigit():
        args += ["--image-max-tokens", imax]
    # 推理控制：off 关闭思考（默认，防复读）；auto/on 显式开关；思考深度仅文档支持的 low/medium/xhigh
    mode = str(p.get("reasoning_mode", DEFAULTS["reasoning_mode"]) or "off").strip().lower()
    if mode == "auto":
        args += ["--reasoning", "auto"]
    elif mode == "on":
        args += ["--reasoning", "on"]
    elif mode in ("low", "medium", "xhigh"):
        args += ["--reasoning-effort", mode]
    else:
        args += ["--reasoning", "off", "--reasoning-budget", "0"]
    # KV 缓存量化；空 = 不启用（不传参）；小写传参
    kv = str(p.get("kv_quant", DEFAULTS["kv_quant"]) or "").strip().lower()
    if kv:
        args += ["--cache-type-k", kv, "--cache-type-v", kv]
    # mmproj：前端显式指定或 launch() 自动检测；空 = 不使用；权重放 CPU 时附加 --no-mmproj-offload
    mmproj = p.get("mmproj", "") or ""
    if mmproj:
        args += ["--mmproj", str(mmproj)]
        if p.get("no_mmproj_offload", DEFAULTS["no_mmproj_offload"]):
            args.append("--no-mmproj-offload")
    # MTP 投机解码：不启用不传参；数字 = 草稿令牌数上限；
    # 主模型同目录有 mtp 开头的草稿模型时附加 --model-draft
    spec = str(p.get("spec_draft_n", DEFAULTS["spec_draft_n"]) or "").strip()
    if spec in ("1", "2", "3", "4", "5"):
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", spec]
        draft = _find_draft_model(model_path)
        if draft:
            args += ["--model-draft", str(draft)]
    extra = p.get("extra_args", "") or ""
    # 前端传 argv 列表（支持嵌套：每组 = 一次输入拆出的多个 argv）；兜底：非列表（手改配置）按空格拆分
    if isinstance(extra, (list, tuple)):
        for e in extra:
            if isinstance(e, (list, tuple)):
                args += [str(x).strip() for x in e if str(x).strip()]
            else:
                e = str(e).strip()
                if e:
                    args.append(e)
    elif str(extra).strip():
        args += str(extra).split()
    # 别名 -a：空 = 不传；argv 列表无 shell 解析，值内空格天然安全，不引号包裹
    alias = str(p.get("alias", "") or "").strip()
    if alias:
        args += ["-a", alias]
    return args


def _reader_thread(proc: subprocess.Popen, log_fn, show_logs: bool = True,
                   read_pipe: bool = True) -> None:
    try:
        if read_pipe:
            # 输出被接管：逐行转发日志（show_logs=False 时静默，仅保留停止提示）
            assert proc.stdout is not None
            for line in proc.stdout:
                if line.strip() and show_logs:
                    log_fn(line.rstrip())
    except Exception:
        pass
    finally:
        try:
            if read_pipe:
                proc.wait(timeout=5)
            else:
                proc.wait()
        except Exception:
            pass
        with _state_lock:
            if _llama_proc is proc:
                _llama_state["running"] = False
                _llama_state["starting"] = False
                _notify_state_change()
        try:
            unregister_subprocess(proc)
        except Exception:
            pass


def _terminate_process(proc: subprocess.Popen, wait: float = 5.0) -> None:
    """终止子进程：先温和 terminate，wait 内未退出则强制 kill。"""
    proc.terminate()
    try:
        proc.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            raise TimeoutError("进程在强制终止后仍未退出")


def _wait_for_health(host: str, port: int, timeout: float, log_fn, proc: subprocess.Popen) -> bool:
    """轮询 /health，确认 llama-server 真正就绪；进程退出立即返回 False。"""
    url = f"http://{host}:{port}/health"
    req = urllib.request.Request(url, headers={"User-Agent": "video-rename-gui"})
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            log_fn("进程在启动过程中异常退出。")
            return False
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def launch(model_path: str, params: Dict[str, Any], log_fn=None) -> Dict[str, Any]:
    """启动 llama-server。model_path 为空时自动从 models/ 选取（仅一个则用，多个则报错）。"""
    log_fn = log_fn or _stderr_log

    exe = LLAMA_DIR / EXE_NAME
    if not exe.is_file():
        return {"ok": False, "error": "llama-server.exe 未找到，请先安装 llama.cpp"}

    global _llama_proc
    with _launch_lock:
        if _llama_proc is not None and _llama_proc.poll() is None:
            return {"ok": False, "error": f"llama-server 已在运行（PID {_llama_proc.pid}），请先停止再启动"}

        # 解析模型路径
        chosen = model_path or ""
        if not chosen:
            models = scan_models(force=True)
            if len(models) == 0:
                return {"ok": False, "error": "models/ 中未找到 .gguf 模型，请先放入模型文件"}
            if len(models) == 1:
                chosen = models[0]["path"]
            else:
                return {"ok": False, "error": "检测到多个模型，请先在参数面板选择具体模型"}
        mp = Path(chosen)
        if not mp.is_file():
            return {"ok": False, "error": f"模型文件不存在: {chosen}"}

        merged = dict(DEFAULTS)
        merged.update({k: v for k, v in (params or {}).items() if v is not None})

        show_logs = bool((params or {}).get("show_logs"))
        use_pipe = show_logs
        if not use_pipe:
            try:
                _has_console = sys.stdout is not None and sys.stdout.fileno() >= 0
            except Exception:
                _has_console = False
            if not _has_console:
                use_pipe = True
                log_fn("当前进程无有效控制台句柄（pythonw 启动？），llama-server 输出无法落到终端，已改由程序内部丢弃")

        # mmproj：前端「多模态」选择框显式指定即加载；未指定时按 mmproj_auto 自动检测
        if not merged.get("mmproj") and merged.get("mmproj_auto", DEFAULTS["mmproj_auto"]):
            auto = _pick_mmproj(mp, scan_mmprojs(mp))
            if auto:
                merged["mmproj"] = auto

        args = _build_args(mp, merged)
        cmd = [str(exe)] + args

        # 端口占用预检：被占立即报错，避免启动后健康检查白等超时、误判模型加载失败
        chk_port = _to_int(merged.get("port", DEFAULTS["port"]), _to_int(DEFAULTS["port"], 0))
        if chk_port > 0 and _port_in_use(_local_host(str(merged.get("host", DEFAULTS["host"]))), chk_port):
            return {"ok": False,
                    "error": f"端口 {chk_port} 已被占用，请更换端口或停止占用它的进程"}

        try:
            creationflags = SUBPROCESS_KWARGS.get("creationflags", 0) if use_pipe else 0
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE if use_pipe else None,
                stderr=subprocess.STDOUT if use_pipe else None,
                text=True, encoding="utf-8", errors="replace",
                cwd=str(LLAMA_DIR), creationflags=creationflags,
            )
        except Exception as e:
            with _state_lock:
                _llama_state["starting"] = False
                _llama_state["launch_failed"] = f"启动失败: {e}"
            _notify_state_change()
            return {"ok": False, "error": f"启动失败: {e}"}

        _llama_proc = proc
        register_subprocess(proc)
        with _state_lock:
            _llama_state["running"] = True
            _llama_state["starting"] = True   # 进程已拉起，健康检查通过前保持「正在启动」
            _llama_state["launch_failed"] = None
            _llama_state["pid"] = proc.pid
            _llama_state["port"] = merged.get("port", DEFAULTS["port"])
            _llama_state["model"] = str(mp)
            _llama_state["launch_params"] = merged  # 供 pause_for_task 让出后恢复
        _notify_state_change()

    log_fn(f"正在加载模型: {mp.name}…")
    threading.Thread(target=_reader_thread,
                     args=(proc, log_fn, show_logs, use_pipe),
                     daemon=True).start()
    if show_logs:
        log_fn(f"已启动 llama-server（PID {proc.pid}），监听 http://{merged.get('host')}:{merged.get('port')}")
        log_fn(f"模型: {mp.name}")

    host = merged.get("host", DEFAULTS["host"])
    port = merged.get("port", DEFAULTS["port"])
    if not _wait_for_health(_local_host(host), port, timeout=_HEALTH_TIMEOUT_SEC, log_fn=log_fn, proc=proc):
        alive = proc.poll() is None
        if alive:
            # 超时但进程仍存活：必须回收并清状态，否则残留进程与「启动失败」矛盾、无法重启动
            log_fn("健康检查超时，正在停止未就绪的 llama-server…")
            try:
                _terminate_process(proc)
            except Exception as e:
                log_fn(f"停止未就绪进程失败: {e}")
            finally:
                try:
                    unregister_subprocess(proc)
                except Exception:
                    pass
                with _state_lock:
                    # 仅当自己仍是当前进程时才清状态/句柄，避免误清并发启动的新进程
                    if _llama_proc is proc:
                        _llama_state["running"] = False
                        _llama_state["starting"] = False
                        _llama_state["launch_failed"] = (
                            f"服务启动后未在 {int(_HEALTH_TIMEOUT_SEC)} 秒内就绪"
                            "（模型加载可能过慢或显存不足），已自动停止该进程")
                        _llama_proc = None
            _notify_state_change()
            return {"ok": False, "error": (
                f"服务启动后未在 {int(_HEALTH_TIMEOUT_SEC)} 秒内就绪"
                "（模型加载可能过慢或显存不足），已自动停止该进程")}
        manual_stop = False
        with _state_lock:
            # 启动窗口内退出：_reader_thread 可能已清理，这里兜底补记失败原因
            if _llama_proc is proc:
                _llama_state["running"] = False
                _llama_state["starting"] = False
                _llama_state["launch_failed"] = "llama-server 进程已退出（模型加载失败或显存不足），请查看日志"
            else:
                manual_stop = True   # stop() 已接管：用户在启动过程中主动停止
        _notify_state_change()
        if manual_stop:
            return {"ok": False, "cancelled": True, "error": "启动已手动停止"}
        return {"ok": False, "error": "llama-server 进程已退出（模型加载失败或显存不足），请查看日志"}
    log_fn("模型加载完成，服务已就绪。")
    with _state_lock:
        _llama_state["starting"] = False   # 健康检查通过 → 「运行中」
    _notify_state_change()
    return {"ok": True, "pid": proc.pid, "port": port, "model": str(mp)}


def stop(log_fn=None, grace: float = 5.0) -> Dict[str, Any]:
    """停止运行中的 llama-server（grace 秒内未退出则强杀）。"""
    global _llama_proc
    log_fn = log_fn or _stderr_log
    proc = _llama_proc
    if proc is None or proc.poll() is not None:
        with _state_lock:
            _llama_state["running"] = False
            _llama_state["starting"] = False
            _llama_state["launch_failed"] = None
            _llama_proc = None
        _notify_state_change()
        return {"ok": True, "message": "没有运行中的服务"}
    try:
        _terminate_process(proc, wait=grace)
    except Exception as e:
        return {"ok": False, "error": f"停止失败: {e}"}
    finally:
        try:
            unregister_subprocess(proc)
        except Exception:
            pass
        with _state_lock:
            _llama_state["running"] = False
            _llama_state["starting"] = False
            _llama_state["launch_failed"] = None
            _llama_proc = None
    _notify_state_change()
    log_fn("本地推理服务已停止。")
    return {"ok": True, "message": "已停止本地推理服务"}


def pause_for_task(log_fn=None) -> Dict[str, Any]:
    """让出显存：其他高显存任务（PixAI Tagger / Whisper 推理）开始前调用。"""
    global _paused_launch
    log_fn = log_fn or _stderr_log
    with _state_lock:
        proc = _llama_proc
        running = proc is not None and proc.poll() is None
        if running:
            _paused_launch = {
                "model": str(_llama_state.get("model") or ""),
                "params": dict(_llama_state.get("launch_params") or {}),
            }
    if not running:
        return {"ok": True, "was_running": False}
    log_fn("检测到本地推理服务在运行，为高显存任务让出显存，正在停止…")
    r = stop(log_fn=log_fn)
    if not r.get("ok"):
        # 停止失败：不留下待恢复状态，避免任务结束后误重启
        with _state_lock:
            _paused_launch = None
        return {"ok": False, "was_running": True, "error": r.get("error")}
    return {"ok": True, "was_running": True}


def resume_after_task(log_fn=None) -> Dict[str, Any]:
    """高显存任务完成后调用：若此前让出过显存，则按暂存的参数重新加载 llama-server。

    期间服务已被手动启动时跳过（不重复启动）。返回 dict 带 restarted 标志。
    """
    global _paused_launch
    log_fn = log_fn or _stderr_log
    with _state_lock:
        saved = _paused_launch
        _paused_launch = None
    if not saved:
        return {"ok": True, "restarted": False}
    if _llama_proc is not None and _llama_proc.poll() is None:
        log_fn("本地推理服务已在运行，跳过恢复。")
        return {"ok": True, "restarted": False}
    log_fn("任务处理完成，正在重新加载本地推理服务…")
    r = launch(saved.get("model") or "", saved.get("params") or {}, log_fn=log_fn)
    if r.get("ok"):
        r["restarted"] = True
    return r
