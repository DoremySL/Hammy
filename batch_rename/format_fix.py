from __future__ import annotations

import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .collector import is_mpeg_ts, mpegts_packet_size
from .dedup import DUPLICATES_DIR
from .dependencies import ffmpeg_tools
from .env import SUBPROCESS_KWARGS, logger
from .failed import move_with_companion
from .naming import resolve_collision
from .utils import path_exists, path_stat, rename_file, to_long_path

_SKIP_DIRS = frozenset({"_failed", DUPLICATES_DIR})

_PROBE_TIMEOUT_SEC = 30.0
_VALIDATE_TIMEOUT_SEC = 30.0
_REMUX_TIMEOUT_SEC = 3600.0

_FMT_EXT_MAP = {
    "avi": ".avi",
    "asf": ".wmv",
    "flv": ".flv",
    "mpegts": ".ts",
    "mpeg": ".mpg",
    "vob": ".vob",
    "mpegvideo": ".mpg",
}
_MOV_FAMILY = "mov,mp4,m4a,3gp,3g2,mj2"
_MATROSKA_FAMILY = "matroska,webm"
_RM_FAMILY = "rm"
_RMVB_CODECS = frozenset({"rv30", "rv40"})
_MP4_LIKE_EXTS = (".mp4", ".mov", ".m4v", ".3gp")

# 流式封装（抽帧 seek 不友好）：待处理时转 MP4 的目标格式
STREAM_EXTS = (".ts", ".mts", ".m2ts")


def _ensure_tools() -> None:
    """延迟定位 ffmpeg/ffprobe：GUI 启动早期不会走 ensure_dependencies()。"""
    if not ffmpeg_tools.ffmpeg or not ffmpeg_tools.ffprobe:
        ffmpeg_tools.locate()


_cancel = threading.Event()
_procs: Set[subprocess.Popen] = set()
_proc_lock = threading.Lock()


def request_cancel() -> None:
    _cancel.set()
    with _proc_lock:
        procs = list(_procs)
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass


def reset_cancel() -> None:
    _cancel.clear()


def _run(cmd: List[str], timeout: float) -> Optional[subprocess.CompletedProcess]:
    if _cancel.is_set():
        return None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding="utf-8", errors="replace", **SUBPROCESS_KWARGS)
    except OSError:
        return None
    with _proc_lock:
        _procs.add(proc)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            stdout, stderr = "", ""
    except OSError:
        return None
    finally:
        with _proc_lock:
            _procs.discard(proc)
    if _cancel.is_set():
        return None
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _probe_json(path: str) -> Optional[Dict[str, Any]]:
    r = _run([ffmpeg_tools.ffprobe, "-v", "quiet", "-print_format", "json",
              "-show_format", "-show_streams", to_long_path(path)],
             _PROBE_TIMEOUT_SEC)
    if r is None or r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def _matroska_doctype(path: str) -> str:
    try:
        with open(to_long_path(path), "rb") as f:
            head = f.read(64)
    except OSError:
        return ""
    if b"webm" in head:
        return "webm"
    if b"matroska" in head:
        return "matroska"
    return ""


def _map_extension(fmt_name: str, major_brand: str, path: str,
                   codec: str = "") -> Optional[str]:
    fmt_name = (fmt_name or "").strip()
    if fmt_name == _MOV_FAMILY:
        brand = (major_brand or "").strip().upper()
        if brand.startswith("QT"):
            return ".mov"
        if brand.startswith("M4V"):
            return ".m4v"
        if brand.startswith("3GP"):
            return ".3gp"
        return ".mp4"
    if fmt_name == _MATROSKA_FAMILY:
        return ".webm" if _matroska_doctype(path) == "webm" else ".mkv"
    if fmt_name == _RM_FAMILY:
        return ".rmvb" if codec in _RMVB_CODECS else ".rm"
    if fmt_name == "mpegts":
        return ".m2ts" if mpegts_packet_size(path) == 192 else ".ts"
    return _FMT_EXT_MAP.get(fmt_name)


def _sniff_mp4(path: str) -> bool:
    try:
        with open(to_long_path(path), "rb") as f:
            head = f.read(12)
    except OSError:
        return False
    return len(head) >= 8 and head[4:8] == b"ftyp"


def _mp4_complete(path: str) -> bool:
    try:
        long = to_long_path(path)
        fsize = os.path.getsize(long)
        with open(long, "rb") as f:
            pos = 0
            while pos < fsize:
                f.seek(pos)
                head = f.read(8)
                if len(head) < 8:
                    return False
                box_size = int.from_bytes(head[:4], "big")
                if box_size == 1:
                    largesize = f.read(8)
                    if len(largesize) < 8:
                        return False
                    box_size = int.from_bytes(largesize, "big")
                elif box_size == 0:
                    return True
                if box_size < 8:
                    return True
                pos += box_size
            return pos == fsize
    except OSError:
        return False


def _decodable(path: str) -> bool:
    r = _run([ffmpeg_tools.ffmpeg, "-v", "error", "-nostats",
              "-i", to_long_path(path), "-map", "0:v:0", "-frames:v", "1",
              "-f", "rawvideo", "pipe:"], _VALIDATE_TIMEOUT_SEC)
    return r is not None and r.returncode == 0


def detect_video(path: str) -> Dict[str, Any]:
    _ensure_tools()
    p = Path(path)
    out: Dict[str, Any] = {"path": str(p), "name": p.name, "dir": str(p.parent),
                           "size": 0, "ok": False, "ext": None, "reason": "",
                           "format_name": "", "codecs": ""}
    try:
        out["size"] = path_stat(path).st_size
    except OSError:
        out["reason"] = "文件不可读"
        return out
    data = _probe_json(path)
    if not data or not data.get("format"):
        if _sniff_mp4(path) and not _mp4_complete(path):
            out["reason"] = "mp4 结构不完整（疑似未下载完）"
        else:
            out["reason"] = "无法识别"
        return out
    fmt = data["format"]
    out["format_name"] = fmt.get("format_name", "")
    brand = (fmt.get("tags") or {}).get("major_brand", "")
    codecs = [s.get("codec_name", "") for s in data.get("streams", [])
              if s.get("codec_type") == "video"]
    out["codecs"] = ",".join(codecs)
    if not codecs:
        out["reason"] = "非视频（无视频流）"
        return out
    ext = _map_extension(out["format_name"], brand, path, codecs[0])
    if not ext:
        out["reason"] = f"不支持的封装格式 {out['format_name']}"
        return out
    if ext in _MP4_LIKE_EXTS and not _mp4_complete(path):
        out["reason"] = "mp4 结构不完整（疑似未下载完）"
        return out
    if not _decodable(path):
        out["reason"] = "无法解码（编码不受支持或内容损坏）"
        return out
    out["ok"] = True
    out["ext"] = ext
    return out


def scan_extensionless(roots: List[str]) -> List[Dict[str, Any]]:
    _ensure_tools()
    candidates: List[str] = []
    seen = set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(str(root)):
            if os.path.basename(dirpath).lower() in _SKIP_DIRS:
                dirnames[:] = []
                continue
            dirnames[:] = sorted(d for d in dirnames if d.lower() not in _SKIP_DIRS)
            for f in sorted(filenames):
                if f.startswith("."):
                    continue
                full = os.path.join(dirpath, f)
                if Path(full).suffix:
                    continue
                norm = os.path.normcase(os.path.normpath(full))
                if norm in seen:
                    continue
                seen.add(norm)
                candidates.append(full)
    if not candidates:
        return []
    workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(detect_video, candidates))


def add_extension(path: str, ext: str) -> Tuple[Optional[str], str]:
    p = Path(path)
    if not p.name or not p.stem:
        return None, "路径无效"
    dest, status = resolve_collision(p.parent, p.stem, ext, to_long_path(path))
    if status != "ok":
        return None, "目标重名无法避让"
    ok, err = rename_file(path, str(dest))
    if not ok:
        return None, "重命名失败"
    return str(dest), ""


def _ffmpeg_reason(stderr: str) -> str:
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    keys = ("matches no streams", "could not find", "unknown encoder",
            "invalid", "unsupported", "not found", "error")
    generic = ""
    for ln in lines:
        low = ln.lower()
        if "error opening output files" in low:
            generic = generic or ln
            continue
        if any(k in low for k in keys):
            return ln[-160:]
    if generic:
        return generic[-160:]
    return lines[-1][-160:] if lines else "ffmpeg 转换失败"


def _remux(src: str, dest_dir: Path, stem: str, container: str) -> Dict[str, Any]:
    ext = ".mp4" if container == "mp4" else ".mkv"
    target, status = resolve_collision(dest_dir, stem, ext, to_long_path(src))
    if status != "ok":
        return {"ok": False, "reason": "目标重名无法避让"}
    tmp = target.with_name(target.name + ".part")
    out: Dict[str, Any] = {"ok": False, "reason": "", "out": str(target),
                           "container": container}
    try:
        cmd = [ffmpeg_tools.ffmpeg, "-y", "-nostats", "-fflags", "+genpts",
               "-i", to_long_path(src), "-map", "0:v", "-map", "0:a?",
               "-c", "copy"]
        if container == "mp4":
            cmd += ["-movflags", "+faststart"]
        cmd += ["-f", "mp4" if container == "mp4" else "matroska",
                to_long_path(str(tmp))]
        r = _run(cmd, _REMUX_TIMEOUT_SEC)
        if r is None and _cancel.is_set():
            out["reason"] = "已取消"
            return out
        if r is None or r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            out["reason"] = _ffmpeg_reason(r.stderr) if r is not None else "ffmpeg 转换失败"
            return out
        info = _probe_json(str(tmp))
        if not info or not any(s.get("codec_type") == "video"
                               for s in info.get("streams", [])):
            out["reason"] = "转换产物校验失败"
            return out
        if not rename_file(str(tmp), str(target))[0]:
            out["reason"] = "输出文件写入失败"
            return out
        out["ok"] = True
        return out
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def convert_ts(ts_path: str) -> Dict[str, Any]:
    src = Path(ts_path)
    out: Dict[str, Any] = {"path": ts_path, "name": src.name,
                           "ok": False, "reason": ""}
    if src.parent.name.lower() in _SKIP_DIRS:
        out["reason"] = "文件位于排除目录内"
        return out
    if not path_exists(ts_path):
        out["skipped"] = True
        out["reason"] = "文件不存在（可能已被处理）"
        return out
    if _cancel.is_set():
        out["skipped"] = True
        out["reason"] = "已取消"
        return out
    if not is_mpeg_ts(ts_path):
        out["skipped"] = True
        out["reason"] = "内容不是 MPEG-TS"
        return out
    _ensure_tools()
    info = _probe_json(ts_path)
    if info is not None and not any(s.get("codec_type") == "video"
                                    for s in info.get("streams", [])):
        out["skipped"] = True
        out["reason"] = "无视频流，跳过"
        return out
    mp4 = _remux(ts_path, src.parent, src.stem, "mp4")
    if mp4["ok"]:
        result = mp4
    else:
        if _cancel.is_set():
            out["reason"] = "已取消"
            return out
        mkv = _remux(ts_path, src.parent, src.stem, "mkv")
        if not mkv["ok"]:
            out["reason"] = f"{mp4['reason']}；{mkv['reason']}"
            logger.warning(f"格式修复: TS 转换失败 {ts_path}: {out['reason']}")
            return out
        result = mkv
    out.update(ok=True, out=result["out"], container=result["container"])
    archived, dest, err = move_with_companion(ts_path, DUPLICATES_DIR)
    if archived:
        logger.info(f"格式修复: 已转换 {out['out']}，原 TS 已归档 "
                    f"{dest.parent.name}/{dest.name}")
    else:
        out["reason"] = f"原文件归档失败: {err}"
        logger.warning(f"格式修复: 已转换 {out['out']}，但原 TS 归档失败: {err}")
    return out
