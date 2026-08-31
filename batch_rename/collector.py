"""视频文件发现。"""
import os
from pathlib import Path
from typing import List

from .env import VIDEO_EXTS_CERTAIN, VIDEO_EXTS_AMBIGUOUS
from .dedup import DUPLICATES_DIR
from .utils import to_long_path


# 扫描时跳过的特殊子目录（_failed 由失败收尾创建，_duplicates 由去重创建）
_SKIP_DIRS = frozenset({"_failed", DUPLICATES_DIR})


# ── MPEG-TS 魔数嗅探常量 ──
_TS_SYNC_BYTE = 0x47
_TS_PACKET_SIZE = 188
_TS_SYNC_CHECKS = 4  # 校验前 4 个包的 sync byte


def is_mpeg_ts(path: str) -> bool:
    """读文件头判断是否为 MPEG-TS 视频。"""
    try:
        with open(to_long_path(path), "rb") as f:
            header = f.read(_TS_PACKET_SIZE * _TS_SYNC_CHECKS)
    except OSError:
        return False
    if len(header) < _TS_PACKET_SIZE:
        return False
    return all(
        header[i * _TS_PACKET_SIZE] == _TS_SYNC_BYTE
        for i in range(_TS_SYNC_CHECKS)
    )


def is_video_file(path: str) -> bool:
    """统一判定某路径是否为可收集的视频文件。"""
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTS_CERTAIN:
        return True
    if ext in VIDEO_EXTS_AMBIGUOUS:
        return is_mpeg_ts(path)
    return False


class VideoCollector:

    @staticmethod
    def collect(paths: List[str]) -> List[str]:
        videos: List[str] = []
        seen: set = set()  # 去重：避免传入重叠文件夹时同一文件被收集多次
        for path_str in paths:
            p = Path(path_str)
            if p.is_file() and is_video_file(str(p)):
                norm = os.path.normcase(os.path.normpath(str(p)))
                if norm not in seen:
                    seen.add(norm)
                    videos.append(str(p))
            elif p.is_dir():
                for root, dirs, files in os.walk(str(p)):
                    # 防御：用户直接传入 _failed / _duplicates 时不继续递归
                    if os.path.basename(root).lower() in _SKIP_DIRS:
                        dirs[:] = []
                        continue
                    # 正常遍历时过滤掉特殊子目录，阻止 walk 进入
                    dirs[:] = sorted(d for d in dirs if d.lower() not in _SKIP_DIRS)
                    for f in sorted(files):
                        full = os.path.join(root, f)
                        if is_video_file(full):
                            norm = os.path.normcase(os.path.normpath(full))
                            if norm not in seen:
                                seen.add(norm)
                                videos.append(full)
        return videos
