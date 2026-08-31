"""workspace_service.py — 业务逻辑：缓存路径解析、还原、对账、清理。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .workspace_paths import NFO_DIR, THUMB_DIR
from .workspace_store import (
    _lock,
    load_history,
    remove_history_by_id,
    save_history,
    save_manifest,
    get_history_by_id,
)
from batch_rename.utils import path_exists, to_long_path


# ── 缓存路径解析 ──


def find_cached_nfo(vid: str) -> Optional[Path]:
    """查找缓存的 NFO 文件（<stable_id>.nfo，处理时由管线直接以此名写入）。"""
    p = NFO_DIR / f"{vid}.nfo"
    return p if p.exists() else None


# ── 还原功能 ──


def restore_video(vid: str) -> Dict[str, Any]:
    """还原一个视频：把现名 rename 回原名 + 删除视频目录里已导出的 NFO。"""
    with _lock:
        entry = get_history_by_id(vid)
        if not entry:
            return {"ok": False, "message": "未找到还原记录"}

        original_path = entry.get("original_path", "")
        new_path = entry.get("new_path", "")
        status = entry.get("status", "")

        if status not in ("ok", "skipped"):
            return {"ok": False, "message": f"该视频状态为 {status}，无法还原"}

        if not original_path or not new_path or original_path == new_path:
            # 文件名未变化，无需重命名，但删除 history 使其回到待处理
            remove_history_by_id(vid)
            return {"ok": True, "message": "文件名未变化，已从已处理中移除"}

        if not path_exists(new_path):
            return {"ok": False, "message": f"新文件已不存在: {new_path}"}

        if path_exists(original_path):
            return {"ok": False, "message": f"原文件名已存在，跳过避免覆盖: {original_path}"}

        try:
            os.replace(to_long_path(new_path), to_long_path(original_path))
        except OSError as e:
            new_drive = os.path.splitdrive(new_path)[0]
            orig_drive = os.path.splitdrive(original_path)[0]
            if new_drive != orig_drive:
                return {
                    "ok": False,
                    "message": (
                        f"跨磁盘卷无法原子重命名（{new_drive} → {orig_drive}）。"
                        f"请手动移动文件：\n  {new_path}\n  → {original_path}"
                    ),
                }
            return {"ok": False, "message": f"重命名失败: {e}"}

        # 字幕等附件跟随还原（.srt/.zh.srt 改回原名；NFO 由下方删除逻辑处理）
        try:
            from batch_rename.naming import _rename_companion_files
            _rename_companion_files(
                Path(new_path), Path(original_path),
                suffixes=(".srt", ".zh.srt"),
            )
        except Exception:
            pass

        # 删除视频目录里已导出的 NFO（若有）
        exported_nfo = Path(new_path).with_suffix(".nfo")
        if path_exists(exported_nfo):
            try:
                os.remove(to_long_path(str(exported_nfo)))
            except OSError:
                pass

        remove_history_by_id(vid)

        return {
            "ok": True,
            "message": f"已还原: {Path(new_path).name} → {Path(original_path).name}",
            "original_path": original_path,
            "new_path": new_path,
        }


def restore_videos(vids: List[str]) -> Dict[str, Any]:
    """批量还原多个视频的原文件名。"""
    ok = 0
    failed = 0
    msgs: List[str] = []
    for vid in vids:
        r = restore_video(vid)
        if r.get("ok"):
            ok += 1
        else:
            failed += 1
            if r.get("message"):
                msgs.append(r["message"])
    return {"ok": failed == 0, "ok_count": ok, "failed_count": failed, "messages": msgs[:10]}


# ── Reconcile：history 与磁盘对账 ──


def reconcile_history() -> Dict[str, int]:
    """统计 history 中文件在磁盘上的实际状态。返回 {total, alive, missing}。"""
    h = load_history()
    entries = h.get("entries", [])
    total = len(entries)
    alive = 0
    for e in entries:
        np = e.get("new_path") or e.get("original_path") or ""
        if np and os.path.exists(np):
            alive += 1
    return {"total": total, "alive": alive, "missing": total - alive}


def prune_missing_history() -> Dict[str, int]:
    """清理 history 中磁盘上已不存在的记录 + 孤立缓存。返回 {removed, thumbs, nfos}。"""
    with _lock:
        h = load_history()
        entries = h.get("entries", [])
        keep: List[Dict[str, Any]] = []
        removed_entries: List[Dict[str, Any]] = []
        for e in entries:
            np = e.get("new_path") or e.get("original_path") or ""
            if np and os.path.exists(np):
                keep.append(e)
            else:
                removed_entries.append(e)
        h["entries"] = keep
        save_history(h)

        removed_thumbs = 0
        removed_nfos = 0
        for e in removed_entries:
            vid = e.get("id") or ""
            if vid:
                tp = THUMB_DIR / f"{vid}.jpg"
                if tp.exists():
                    try:
                        tp.unlink()
                        removed_thumbs += 1
                    except OSError:
                        pass
            if vid:
                nf = NFO_DIR / f"{vid}.nfo"
                if nf.exists():
                    try:
                        nf.unlink()
                        removed_nfos += 1
                    except OSError:
                        pass
        return {"removed": len(removed_entries), "thumbs": removed_thumbs, "nfos": removed_nfos}


# ── 清除全部缓存 ──


def _clear_dir(d: Path) -> int:
    """删除目录下所有文件，返回成功删除的数量。"""
    if not d.exists():
        return 0
    count = 0
    for f in d.iterdir():
        if f.is_file():
            try:
                f.unlink()
                count += 1
            except OSError:
                pass
    return count


def clear_workspace_cache(clear_history: bool = True,
                           clear_thumbs: bool = True,
                           clear_nfo: bool = True,
                           clear_manifest: bool = False) -> Dict[str, Any]:
    """清除工作区缓存。视频本体不受影响。"""
    cleared = {"history": 0, "thumbs": 0, "nfo": 0, "manifest": False}
    with _lock:
        if clear_history:
            save_history({"entries": []})
            cleared["history"] = 1
        if clear_thumbs:
            cleared["thumbs"] = _clear_dir(THUMB_DIR)
        if clear_nfo:
            cleared["nfo"] = _clear_dir(NFO_DIR)
        if clear_manifest:
            save_manifest({"roots": [], "adhoc_files": []})
            cleared["manifest"] = True
    return {"ok": True, "cleared": cleared}
