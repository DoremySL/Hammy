"""workspace_store.py — JSON 原子读写 + 线程安全 + History 缓存/批量落盘 + Manifest CRUD。"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .workspace_paths import (
    HISTORY_FILE,
    MANIFEST_FILE,
    NFO_DIR,
    THUMB_DIR,
    WORKSPACE_DIR,
)

_lock = threading.RLock()

# update_json 的 mutator 返回此哨兵 → 跳过写入，函数直接返回当前值
# （用于「读 + 按需补全」场景：文件完整时保持纯读，不产生任何磁盘写）
NO_WRITE = object()

# ── History 进程内缓存（按 mtime 失效）+ 批量落盘 ──
_hist_cache: Optional[Dict[str, Any]] = None
_hist_mtime: Optional[float] = None
_batch_mode: bool = False
_batch_dirty: bool = False


def _backup_corrupt_json(path: Path) -> None:
    """JSON 解析失败时把原文件备份为 <name>.json.bak（保留最新一份）再回退默认。"""
    try:
        if path.exists() and path.stat().st_size > 0:
            shutil.copy2(str(path), str(path) + ".bak")
            sys.stderr.write(
                f"[workspace_store] {path.name} 内容损坏，已备份为 {path.name}.bak 并回退默认值\n")
    except OSError:
        pass


# ── 初始化 ──


def ensure_workspace() -> None:
    """首次运行时创建 _workspace 目录结构。"""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    NFO_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_FILE.exists():
        write_json(MANIFEST_FILE, {"roots": [], "adhoc_files": []})
    if not HISTORY_FILE.exists():
        write_json(HISTORY_FILE, {"entries": []})


# ── JSON 读写（线程安全） ──


def read_json(path: Path, default: Any) -> Any:
    """读取 JSON；文件缺失/损坏/不可读时返回 default。

    写侧为 tmp + os.replace 原子替换，读到的是完整新旧之一，无需加锁。
    """
    try:
        if not path.exists():
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        _backup_corrupt_json(path)
        return default
    except OSError:
        return default


# read_json_cached 的进程内缓存：{路径: ((mtime_ns, size), 解析结果)}
_json_read_cache: Dict[str, Tuple[Any, Any]] = {}


def read_json_cached(path: Path) -> Dict[str, Any]:
    """read_json 的 mtime 缓存版（dict 专用）：高频轮询读免重复读盘。

    （mtime_ns, size）未变时返回缓存快照（顶层拷贝，调用方按只读使用）；
    update_json 写入会主动失效，绕过它直改磁盘则靠 mtime 变化自然失效。
    """
    key = str(path)
    try:
        st = path.stat()
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        sig = None
    hit = _json_read_cache.get(key)
    if hit is not None and hit[0] == sig:
        return dict(hit[1])
    data = read_json(path, {})
    data = data if isinstance(data, dict) else {}
    _json_read_cache[key] = (sig, data)
    return dict(data)


def write_json(path: Path, data: Any) -> None:
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


def update_json(path: Path, mutator: Callable[[Any], Any],
                default_factory: Optional[Callable[[], Any]] = None) -> Any:
    """原子读-改-写（进程内由 RLock 串行化；跨进程由单实例守卫保证无并发）。"""
    with _lock:
        # 读取（内联而非调用 read_json，避免重复解析缓存读）
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    current = json.load(f)
                if not isinstance(current, dict):
                    _backup_corrupt_json(path)
            else:
                current = default_factory() if default_factory else None
        except json.JSONDecodeError:
            _backup_corrupt_json(path)  # 损坏先留档再回退默认
            current = default_factory() if default_factory else None
        except OSError:
            current = default_factory() if default_factory else None
        result = mutator(current)
        if result is NO_WRITE:
            return current
        # 写入（内联，避免 write_json 重复获取锁的开销）
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        _json_read_cache.pop(str(path), None)  # 写后失效，read_json_cached 下次重读
        return result


# ── Manifest：用户添加的源 ──


def _default_manifest() -> Dict[str, Any]:
    return {"roots": [], "adhoc_files": []}


def load_manifest() -> Dict[str, Any]:
    return read_json(MANIFEST_FILE, _default_manifest())


def save_manifest(data: Dict[str, Any]) -> None:
    write_json(MANIFEST_FILE, data)


def _update_manifest(mutator: Callable[[Dict[str, Any]], Any]) -> Any:
    """在单个临界区内读-改-写 manifest（非 dict 时回落默认结构）。"""
    def _wrapped(current):
        if not isinstance(current, dict):
            current = _default_manifest()
        return mutator(current)
    return update_json(MANIFEST_FILE, _wrapped, default_factory=_default_manifest)


def add_root(path: str) -> Dict[str, Any]:
    """追加一个根（文件夹或文件）。返回更新后的 manifest。"""
    norm = os.path.normcase(os.path.normpath(path))

    def _mutate(m):
        if norm not in {os.path.normcase(os.path.normpath(r)) for r in m["roots"]}:
            m["roots"].append(path)
        return m

    return _update_manifest(_mutate)


def add_adhoc_files(paths: List[str]) -> Dict[str, Any]:
    """追加零散文件。"""
    def _mutate(m):
        existing = {os.path.normcase(os.path.normpath(p)) for p in m["adhoc_files"]}
        for p in paths:
            n = os.path.normcase(os.path.normpath(p))
            if n not in existing:
                m["adhoc_files"].append(p)
                existing.add(n)
        return m

    return _update_manifest(_mutate)


def _remove_from_list(lst: list, path: str) -> list:
    target = os.path.normcase(os.path.normpath(path))
    return [r for r in lst if os.path.normcase(os.path.normpath(r)) != target]


def remove_root(path: str) -> Dict[str, Any]:
    def _mutate(m):
        m["roots"] = _remove_from_list(m["roots"], path)
        return m

    return _update_manifest(_mutate)


def remove_adhoc(path: str) -> Dict[str, Any]:
    def _mutate(m):
        m["adhoc_files"] = _remove_from_list(m["adhoc_files"], path)
        return m

    return _update_manifest(_mutate)


def update_adhoc_path(old_path: str, new_path: str) -> bool:
    """把 manifest 中 adhoc_files 的 old_path 替换为 new_path（含去重），返回是否发生替换。"""
    target = os.path.normcase(os.path.normpath(old_path))
    changed = {"v": False}

    def _mutate(m):
        new_list = []
        for p in m["adhoc_files"]:
            if os.path.normcase(os.path.normpath(p)) == target:
                new_list.append(new_path)
                changed["v"] = True
            else:
                new_list.append(p)
        if not changed["v"]:
            return NO_WRITE
        seen = set()
        deduped = []
        for p in new_list:
            n = os.path.normcase(os.path.normpath(p))
            if n not in seen:
                seen.add(n)
                deduped.append(p)
        m["adhoc_files"] = deduped
        return m

    _update_manifest(_mutate)
    return changed["v"]


def clear_sources() -> Dict[str, Any]:
    save_manifest({"roots": [], "adhoc_files": []})
    return load_manifest()


# ── History：每个视频的改名记录 + 标签 ──


def _current_history_mtime() -> Optional[float]:
    try:
        return HISTORY_FILE.stat().st_mtime
    except OSError:
        return None


def _load_history_locked() -> Dict[str, Any]:
    """读取 history 到进程内缓存；磁盘 mtime 未变则直接复用缓存。"""
    global _hist_cache, _hist_mtime
    mtime = _current_history_mtime()
    if _hist_cache is not None and _hist_mtime == mtime:
        return _hist_cache
    data = read_json(HISTORY_FILE, {"entries": []})
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        data = {"entries": []}
    _hist_cache = data
    _hist_mtime = mtime
    return data


def load_history() -> Dict[str, Any]:
    with _lock:
        return _load_history_locked()


def save_history(data: Dict[str, Any]) -> None:
    global _hist_cache, _hist_mtime
    with _lock:
        write_json(HISTORY_FILE, data)
        _hist_cache = data
        _hist_mtime = _current_history_mtime()


def begin_batch() -> None:
    """进入批量处理模式：append_history_entry 只更新内存缓存，延迟到 flush_batch 落盘。"""
    global _batch_mode, _batch_dirty
    with _lock:
        _load_history_locked()  # 预热缓存
        _batch_mode = True
        _batch_dirty = False


def flush_batch(keep_mode: bool = False) -> None:
    """结束批量模式并把累积的 history 变更一次性原子落盘。"""
    global _batch_mode, _batch_dirty
    with _lock:
        dirty = _batch_dirty
        if not keep_mode:
            _batch_mode = False
        _batch_dirty = False
        if dirty and _hist_cache is not None:
            save_history(_hist_cache)


def append_history_entry(entry: Dict[str, Any]) -> None:
    """追加一条处理记录（去重：同 id+original_path 覆盖）。"""
    global _batch_dirty
    with _lock:
        h = _load_history_locked()
        entries = h.get("entries", [])
        key = (entry.get("id"), entry.get("original_path"))
        entries = [e for e in entries
                   if (e.get("id"), e.get("original_path")) != key]
        entries.append(entry)
        h["entries"] = entries
        if _batch_mode:
            _batch_dirty = True
        else:
            save_history(h)


def get_history_by_id(vid: str) -> Optional[Dict[str, Any]]:
    h = load_history()
    for e in h.get("entries", []):
        if e.get("id") == vid:
            return e
    return None


def remove_history_by_id(vid: str) -> None:
    """删除指定 id 的 history 记录（还原后调用）。"""
    with _lock:
        h = _load_history_locked()
        h["entries"] = [e for e in h.get("entries", []) if e.get("id") != vid]
        save_history(h)
