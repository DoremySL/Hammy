"""nfo_import.py — 添加源时扫描带 hammy 标记的 NFO，重建 history 与 NFO 缓存。"""
from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from xml.etree import ElementTree as ET

from .workspace_paths import NFO_DIR, stable_id
from .workspace_store import append_history_entry, load_history
from batch_rename.collector import is_video_file
from batch_rename.utils import to_long_path

MARKER_TAG = "hammy"
MARKER_VERSION = "1"
_SKIP_DIRS = frozenset({"_failed", "_duplicates"})
_THUMB_TIME_RE = re.compile(r"\d{1,3}:\d{2}:\d{2}")


def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))


def _collect_nfo_files(path: str) -> List[str]:
    """收集路径涉及的 NFO：目录递归（跳过 _failed/_duplicates）；文件需为 .nfo 或视频（查同 stem 兄弟）。"""
    p = Path(path)
    if p.is_dir():
        out: List[str] = []
        for dirpath, dirnames, filenames in os.walk(path):
            if os.path.basename(dirpath).lower() in _SKIP_DIRS:
                dirnames[:] = []
                continue
            dirnames[:] = sorted(d for d in dirnames if d.lower() not in _SKIP_DIRS)
            for f in sorted(filenames):
                if f.lower().endswith(".nfo"):
                    out.append(os.path.join(dirpath, f))
        return out
    if p.is_file():
        if p.suffix.lower() == ".nfo":
            return [str(p)]
        sibling = p.with_suffix(".nfo")
        if sibling.is_file():
            return [str(sibling)]
    return []


def _stem_video_map(directory: str) -> Dict[str, List[str]]:
    """目录内视频文件按 stem（小写）分组。"""
    mapping: Dict[str, List[str]] = {}
    try:
        names = os.listdir(directory)
    except OSError:
        return mapping
    for name in sorted(names):
        full = os.path.join(directory, name)
        if not os.path.isfile(full) or not is_video_file(full):
            continue
        mapping.setdefault(Path(name).stem.lower(), []).append(full)
    return mapping


def _parse_marker(nfo_path: str) -> Optional[Any]:
    """解析 hammy 标记块，返回 (标记字段, NFO root)；无标记/解析失败/版本不符返回 None。"""
    try:
        with open(to_long_path(nfo_path), "rb") as f:
            data = f.read()
    except OSError:
        return None
    if b"hammy" not in data:
        return None
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None
    marker_el = root.find(MARKER_TAG)
    if marker_el is None or (marker_el.get("version") or "") != MARKER_VERSION:
        return None
    status = (marker_el.findtext("status") or "").strip().lower()
    if status not in ("ok", "skipped"):
        return None
    try:
        processed_at = float(marker_el.findtext("processed_at") or "")
    except ValueError:
        processed_at = time.time()
    thumb_time = (marker_el.findtext("thumb_time") or "").strip()
    if not _THUMB_TIME_RE.fullmatch(thumb_time):
        thumb_time = ""
    return {"status": status, "processed_at": processed_at, "thumb_time": thumb_time}, root


def _build_entry(video_path: str, marker: Dict[str, Any], root: ET.Element) -> Dict[str, Any]:
    """由 NFO 标准元素 + 标记块重建 history 条目。"""
    original_name = (root.findtext("originaltitle") or "").strip()
    width = (root.findtext("fileinfo/streamdetails/video/width") or "").strip()
    height = (root.findtext("fileinfo/streamdetails/video/height") or "").strip()
    try:
        duration = int(float(root.findtext("fileinfo/streamdetails/video/durationinseconds") or 0))
    except ValueError:
        duration = 0
    try:
        size = os.path.getsize(to_long_path(video_path))
    except OSError:
        size = 0
    entry: Dict[str, Any] = {
        "id": stable_id(video_path),
        "original_path": str(Path(video_path).parent / original_name) if original_name else video_path,
        "new_path": video_path,
        "original_name": original_name,
        "new_name": Path(video_path).name,
        "status": marker["status"],
        "title": (root.findtext("title") or "").strip(),
        "plot": (root.findtext("plot") or "").strip(),
        "tags": [t.text.strip() for t in root.findall("tag") if t.text],
        "info": {
            "duration": duration,
            "size": size,
            "resolution": f"{width}x{height}" if width and height else "",
            "codec": (root.findtext("fileinfo/streamdetails/video/codec") or "").strip(),
        },
        "processed_at": marker["processed_at"],
    }
    if marker["thumb_time"]:
        entry["thumb_time"] = marker["thumb_time"]
    return entry


def _cache_nfo(nfo_path: str, vid: str) -> None:
    try:
        NFO_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(to_long_path(nfo_path), to_long_path(str(NFO_DIR / f"{vid}.nfo")))
    except OSError:
        pass


def import_from_sources(paths: List[str]) -> Dict[str, int]:
    """扫描源路径，导入带标记的 NFO。返回计数。"""
    counts = {"imported": 0, "skipped_dup": 0, "skipped_no_video": 0, "skipped_bad": 0}
    known_ids: Set[str] = set()
    known_new_paths: Set[str] = set()
    for e in load_history().get("entries", []):
        if e.get("id"):
            known_ids.add(e["id"])
        if e.get("new_path"):
            known_new_paths.add(_norm(e["new_path"]))

    stem_maps: Dict[str, Dict[str, List[str]]] = {}
    for path in paths:
        for nfo_path in _collect_nfo_files(path):
            parsed = _parse_marker(nfo_path)
            if parsed is None:
                counts["skipped_bad"] += 1
                continue
            marker, root = parsed
            directory = str(Path(nfo_path).parent)
            if directory not in stem_maps:
                stem_maps[directory] = _stem_video_map(directory)
            candidates = stem_maps[directory].get(Path(nfo_path).stem.lower(), [])
            if len(candidates) != 1:
                counts["skipped_no_video"] += 1
                continue
            video_path = candidates[0]
            vid = stable_id(video_path)
            if vid in known_ids or _norm(video_path) in known_new_paths:
                counts["skipped_dup"] += 1
                continue
            append_history_entry(_build_entry(video_path, marker, root))
            _cache_nfo(nfo_path, vid)
            known_ids.add(vid)
            known_new_paths.add(_norm(video_path))
            counts["imported"] += 1
    return counts
