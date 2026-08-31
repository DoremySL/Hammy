"""简易视频去重：大小分组 + 头尾部分哈希 → 移入 _duplicates/ 子目录。"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Tuple

from .env import logger
from .failed import move_with_companion
from .utils import path_stat, to_long_path

# 头 / 尾各读取的字节数
_CHUNK = 1024 * 1024

# 去重移动目标子目录名（与 _failed 平级，扫描时一并跳过）
DUPLICATES_DIR = "_duplicates"


# ═══ 部分哈希 ══════════════════════════════════════════════════
def partial_hash(path: str) -> Optional[str]:
    """读取文件头尾各 _CHUNK 字节计算 MD5，失败返回 None。"""
    try:
        size = path_stat(path).st_size
    except OSError:
        return None
    h = hashlib.md5()
    try:
        with open(to_long_path(path), "rb") as f:
            if size <= _CHUNK * 2:
                h.update(f.read())  # 小文件：全量
            else:
                h.update(f.read(_CHUNK))          # 头
                f.seek(size - _CHUNK)
                h.update(f.read(_CHUNK))          # 尾
    except OSError:
        return None
    return h.hexdigest()


# ═══ 查找重复 ══════════════════════════════════════════════════
def find_duplicates(paths: List[str]) -> List[Dict]:
    """在给定的视频路径列表中查找内容重复的分组。"""
    # 1) 按文件大小分组（大小不同必然不重复）
    by_size: Dict[int, List[str]] = {}
    for p in paths:
        try:
            size = path_stat(p).st_size
        except OSError:
            continue
        by_size.setdefault(size, []).append(p)

    groups: List[Dict] = []
    for size, same_size in by_size.items():
        if len(same_size) < 2:
            continue
        # 2) 同大小内再按部分哈希分组
        by_hash: Dict[str, List[str]] = {}
        for p in same_size:
            digest = partial_hash(p)
            if digest is None:
                continue  # 无法读取 → 跳过，不参与判重
            by_hash.setdefault(digest, []).append(p)

        for digest, dup in by_hash.items():
            if len(dup) < 2:
                continue
            # 字典序排序，保留第一个
            dup_sorted = sorted(dup, key=lambda x: os.path.normcase(os.path.normpath(x)))
            groups.append({
                "keep": dup_sorted[0],
                "remove": dup_sorted[1:],
                "size": size,
            })
    return groups


# ═══ 执行移动 ══════════════════════════════════════════════════
def move_to_duplicates(remove_paths: List[str]) -> Tuple[List[str], List[str]]:
    """将判定为重复的文件移入其所在目录的 _duplicates/ 子目录。"""
    moved: List[str] = []
    errors: List[str] = []
    for full_path in remove_paths:
        ok, dest, err = move_with_companion(full_path, DUPLICATES_DIR)
        if not ok:
            logger.warning(f"无法移动重复文件 {full_path}: {err}")
            errors.append(full_path)
            continue
        moved.append(full_path)
        logger.info(f"已移除重复文件: {dest.parent.name}/{dest.name}")
    return moved, errors
