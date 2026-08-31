"""discovery.py — 多源视频发现、已处理过滤与缩略图/探针。"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .workspace_paths import PROBE_CACHE_FILE, stable_id, thumb_path
from .workspace_store import load_history, load_manifest, read_json, write_json
from .constants import (
    PROBE_POOL_WORKERS,
    PROBE_TIMEOUT,
    THUMB_LRU_MAX,
    THUMB_MAX_SIDE,
    THUMB_TIMEOUT,
)
from .env import FFMPEG_EXE, FFPROBE_EXE
from batch_rename.env import SUBPROCESS_KWARGS
from batch_rename.utils import path_stat
from batch_rename.collector import VideoCollector, is_mpeg_ts, is_video_file
from batch_rename.dedup import DUPLICATES_DIR


def _norm(p: str) -> str:
    """路径归一化（Windows 大小写/斜杠不敏感）。"""
    return os.path.normcase(os.path.normpath(p))


# ── 视频发现 ──


def collect_all() -> List[str]:
    """合并扫描 manifest 中所有根 + 零散文件。"""
    m = load_manifest()
    roots = [r for r in m.get("roots", []) if os.path.exists(r)]
    adhoc = [f for f in m.get("adhoc_files", []) if os.path.isfile(f)]
    videos = VideoCollector.collect(roots)
    seen = {_norm(v) for v in videos}
    for f in adhoc:
        n = _norm(f)
        if n not in seen and is_video_file(f):
            videos.append(f)
            seen.add(n)
    return videos


def collect_failed() -> List[str]:
    """递归扫描所有根文件夹下的 _failed 子目录。"""
    m = load_manifest()
    result: List[str] = []
    for root in m.get("roots", []):
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # 剪掉 _duplicates 目录，避免无谓遍历（去重移走的视频不算失败）
            dirnames[:] = [d for d in dirnames if d.lower() != DUPLICATES_DIR]
            if os.path.basename(dirpath).lower() == "_failed":
                for f in filenames:
                    full = os.path.join(dirpath, f)
                    if is_video_file(full):
                        result.append(full)
                dirnames[:] = []  # 不深入 _failed 的子目录
    return result


# ── 已处理识别 ──


def _is_processed(path: str, history_paths: Set[str]) -> bool:
    """判断视频是否已处理：history 中有记录（且 status 为 ok/skipped）。"""
    return _norm(path) in history_paths


def _build_history_path_index() -> Tuple[Set[str], Set[str], Dict[str, Dict[str, Any]]]:
    """返回 (done_paths, failed_paths, by_path)。"""
    h = load_history()
    done = set()
    failed = set()
    by_path: Dict[str, Dict[str, Any]] = {}
    for e in h.get("entries", []):
        status = e.get("status")
        new_p, orig_p = e.get("new_path"), e.get("original_path")
        # by_path 同时映射新名/原名 → 条目（供已处理视频查 id / 标题）
        for p in (new_p, orig_p):
            if p:
                by_path[_norm(p)] = e
        if status in ("ok", "skipped"):
            # done 只记 new_path：用户手动改回原名后，原名不在 done → 回到待处理重新处理
            if new_p:
                done.add(_norm(new_p))
            if status == "skipped" and orig_p:
                done.add(_norm(orig_p))
        elif status in ("error", "frame_error"):
            for p in (new_p, orig_p):
                if p:
                    failed.add(_norm(p))
    return done, failed, by_path


# ── 缩略图内存缓存 ──

_thumb_lru: OrderedDict[str, str] = OrderedDict()  # vid -> data_url
_thumb_lru_lock = threading.Lock()


def _thumb_lru_put(vid: str, url: str) -> None:
    with _thumb_lru_lock:
        _thumb_lru[vid] = url
        _thumb_lru.move_to_end(vid)
        while len(_thumb_lru) > THUMB_LRU_MAX:
            _thumb_lru.popitem(last=False)


# ── ffprobe / 缩略图 ──

# 限制同时运行的 ffmpeg/ffprobe 进程数，避免前端批量并发请求时进程风暴
_ffmpeg_semaphore = threading.Semaphore(PROBE_POOL_WORKERS)

# vid 级缩略图生成去重：同一 vid 并发请求时只生成一次
_thumb_inflight: Set[str] = set()
_thumb_inflight_lock = threading.Lock()


_probe_mem: Dict[str, Dict[str, Any]] = {}
_probe_loaded = False
# 探针缓存攒批落盘：miss 只进内存，批量/防抖后统一写一次（避免每次 miss 全文件读-改-写）
_probe_dirty = False
# _probe_mem 的唯一保护锁：pywebview 多线程并发 probe_video / prune 与 flush 遍历之间互斥，
# 落盘写快照（dict 拷贝）而非原 dict，避免 json.dump 遍历期间 dict 变更
_probe_lock = threading.Lock()
_probe_last_flush = 0.0


def probe_video(path: str) -> Dict[str, Any]:
    """探针元数据子集：先读缓存（mtime+size 校验），未命中真探针并写回。"""
    global _probe_loaded, _probe_dirty
    key = _norm(path)
    try:
        st = path_stat(path)
        mtime, size = int(st.st_mtime), st.st_size
    except OSError:
        mtime, size = 0, 0
    with _probe_lock:
        if not _probe_loaded:
            _probe_mem.update(read_json(PROBE_CACHE_FILE, {}))
            _probe_loaded = True
        ent = _probe_mem.get(key)
        if ent and ent.get("mtime") == mtime and ent.get("size") == size and ent.get("duration"):
            return {k: ent.get(k) for k in ("duration", "size", "resolution", "codec",
                                            "audio_codec", "has_audio")}
    info = _ffprobe_video(path)
    if info:
        entry = {"mtime": mtime, "size": size, **info}
        with _probe_lock:
            _probe_mem[key] = entry
            _probe_dirty = True
    return info


def flush_probe_cache(force: bool = False, debounce: float = 1.5) -> None:
    global _probe_dirty, _probe_last_flush
    now = time.time()
    with _probe_lock:
        if not _probe_dirty:
            return
        if not force and now - _probe_last_flush < debounce:
            return
        snapshot = dict(_probe_mem)
        _probe_dirty = False
        _probe_last_flush = now
    write_json(PROBE_CACHE_FILE, snapshot)


def prune_probe_cache(paths) -> None:
    if not _probe_loaded:
        return
    keep = {_norm(p) for p in paths}
    with _probe_lock:
        stale = [k for k in _probe_mem if k not in keep]
        for k in stale:
            del _probe_mem[k]
        if stale:
            _probe_dirty = True
    flush_probe_cache(force=True)


def _ffprobe_video(path: str) -> Dict[str, Any]:
    """ffprobe 获取视频元数据子集。"""
    with _ffmpeg_semaphore:
        try:
            cmd = [
                FFPROBE_EXE, "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT,
                encoding="utf-8", errors="replace", **SUBPROCESS_KWARGS,
            )
            if result.returncode != 0:
                return {}
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            streams = data.get("streams", [])
            vs = next((s for s in streams if s.get("codec_type") == "video"), None)
            astream = next((s for s in streams if s.get("codec_type") == "audio"), None)
            try:
                duration = int(float(fmt.get("duration", 0)))
            except (TypeError, ValueError):
                duration = 0
            try:
                size = int(fmt.get("size", 0))
            except (TypeError, ValueError):
                size = 0
            resolution = ""
            codec = ""
            if vs and vs.get("width"):
                resolution = f"{vs['width']}x{vs['height']}"
            if vs and vs.get("codec_name"):
                codec = vs["codec_name"]
            audio_codec = astream.get("codec_name", "") if astream else ""
            has_audio = astream is not None
            return {"duration": duration, "size": size, "resolution": resolution, "codec": codec,
                    "audio_codec": audio_codec, "has_audio": has_audio}
        except Exception:
            return {}


def generate_thumbnail(path: str, vid: str, max_side: int = THUMB_MAX_SIDE) -> Optional[str]:
    """生成缩略图：优先读内存 LRU → 磁盘缓存 → ffmpeg 抽帧。"""
    # 第 1 级：内存 LRU
    with _thumb_lru_lock:
        cached = _thumb_lru.get(vid)
        if cached is not None:
            _thumb_lru.move_to_end(vid)
            return cached

    # 第 2 级：磁盘缓存
    cache = thumb_path(vid)
    if cache.exists():
        try:
            with open(cache, "rb") as f:
                data = f.read()
            if data[:2] == b'\xff\xd8':  # JPEG 魔数校验
                url = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
                _thumb_lru_put(vid, url)
                return url
            cache.unlink(missing_ok=True)
        except OSError:
            pass

    with _thumb_inflight_lock:
        if vid in _thumb_inflight:
            return None  # 前端下次轮询时会命中缓存
        _thumb_inflight.add(vid)

    # 第 3 级：ffmpeg 抽帧
    try:
        def _extract(ss: str):
            # mpegts 输入 seek 不准，改输出 seek
            seek_args = ["-i", path, "-ss", ss] if is_mpeg_ts(path) else ["-ss", ss, "-noaccurate_seek", "-i", path]
            cmd = [
                FFMPEG_EXE, "-y", "-nostats", *seek_args,
                "-vframes", "1", "-q:v", "5",
                "-vf", f"scale={max_side}:-1",
                "-f", "image2pipe", "-vcodec", "mjpeg", "-",
            ]
            return subprocess.run(
                cmd, capture_output=True, timeout=THUMB_TIMEOUT, **SUBPROCESS_KWARGS,
            )

        try:
            duration = int(probe_video(path).get("duration") or 0)
        except Exception:
            duration = 0
        if is_mpeg_ts(path) or duration < 2:
            attempts = ["1", "0"]
        else:
            attempts = [str(duration // 2), "1", "0"]

        with _ffmpeg_semaphore:
            for ss in attempts:
                result = _extract(ss)
                if result.returncode == 0 and result.stdout and result.stdout[:2] == b'\xff\xd8':
                    break
        if result.returncode == 0 and result.stdout and result.stdout[:2] == b'\xff\xd8':
            # 原子写入磁盘缓存（tmp + os.replace，避免并发写出截断 JPEG）
            try:
                tmp = cache.with_suffix(".jpg.tmp")
                with open(tmp, "wb") as f:
                    f.write(result.stdout)
                os.replace(tmp, cache)
            except OSError:
                pass
            url = f"data:image/jpeg;base64,{base64.b64encode(result.stdout).decode()}"
            _thumb_lru_put(vid, url)
            return url
    except Exception:
        pass
    finally:
        with _thumb_inflight_lock:
            _thumb_inflight.discard(vid)
    return None


# ── 扫描入口 ──


def make_entry(path: str, info: Optional[Dict[str, Any]] = None,
               status: str = "pending", extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = Path(path)
    info = info or {}
    # 文件系统元数据：mtime 供前端按修改时间排序；size 探针值缺失时用 stat 兜底
    try:
        st = p.stat()
        mtime = int(st.st_mtime)
        fs_size = st.st_size
    except OSError:
        mtime, fs_size = 0, 0
    entry = {
        "id": stable_id(path),
        "path": str(path),
        "name": p.name,
        "dir": str(p.parent),
        "size": info.get("size", 0) or fs_size,
        "mtime": mtime,
        "duration": info.get("duration", 0),
        "resolution": info.get("resolution", ""),
        "status": status,
    }
    if extra:
        entry.update(extra)
    return entry


def scan_all() -> Dict[str, Any]:
    """全量扫描：合并根 + 零散 + 失败目录。"""
    done_paths, failed_paths, history_by_path = _build_history_path_index()

    pending: List[Dict[str, Any]] = []
    processed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    failed_norm: Set[str] = set()  # 已收录的失败视频归一化路径，用于 O(1) 去重

    all_videos = collect_all()
    for vp in all_videos:
        norm = _norm(vp)
        info = {}  # 不在主线程同步探针；交给前端按需请求
        if _is_processed(vp, done_paths):
            # 用 history 中的稳定 id（原路径哈希），保持改名后命中缓存
            hist = history_by_path.get(norm)
            vid = hist.get("id") if hist else stable_id(vp)
            entry = make_entry(vp, info, "processed",
                                extra={"id": vid})
            if hist:
                entry["title"] = hist.get("title", "")
                entry["plot"] = hist.get("plot", "")
                entry["tags"] = hist.get("tags", [])
                entry["original_name"] = hist.get("original_name", "")
                entry["processed_at"] = hist.get("processed_at", 0)
                # 处理时已探针的元数据直接回填：免去前端重新探针，也供按时长排序
                hinfo = hist.get("info") or {}
                if hinfo.get("duration"):
                    entry["duration"] = hinfo["duration"]
                if hinfo.get("resolution"):
                    entry["resolution"] = hinfo["resolution"]
                if hinfo.get("codec"):
                    entry["codec"] = hinfo["codec"]
                if hinfo.get("size"):
                    entry["size"] = hinfo["size"]
            processed.append(entry)
        elif norm in failed_paths:
            entry = make_entry(vp, info, "failed")
            failed.append(entry)
            failed_norm.add(norm)
        else:
            pending.append(make_entry(vp, info, "pending"))

    failed_videos = collect_failed()
    for vp in failed_videos:
        norm = _norm(vp)
        if norm in failed_norm:
            continue
        failed.append(make_entry(vp, {}, "failed"))
        failed_norm.add(norm)

    m = load_manifest()
    return {
        "pending": pending,
        "processed": processed,
        "failed": failed,
        "roots": m.get("roots", []),
        "adhoc_files": m.get("adhoc_files", []),
        "adhoc_count": len(m.get("adhoc_files", [])),
    }

