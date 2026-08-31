"""失败文件移动收尾。"""
import os
from pathlib import Path
from typing import List, Optional, Tuple

from .env import logger
from .utils import to_long_path, path_exists, rename_file
from .naming import resolve_collision


def move_with_companion(full_path: str, subdir: str) -> Tuple[bool, Optional[Path], str]:
    """把文件移入自身目录的 <subdir>/ 子目录，并带上同名 .nfo。
    返回 (ok, dest, err)。
    """
    try:
        if not path_exists(full_path):
            return False, None, "文件不存在"
        p = Path(full_path)
        dest_dir = p.parent / subdir
        os.makedirs(to_long_path(str(dest_dir)), exist_ok=True)
        dest, status = resolve_collision(dest_dir, p.stem, p.suffix, to_long_path(full_path))
        if status == "skipped":
            return True, dest, ""
        if status == "error":
            return False, None, "路径冲突"
        if not rename_file(full_path, str(dest))[0]:
            return False, None, "系统重命名失败"
        src_nfo = p.with_suffix(".nfo")
        if path_exists(str(src_nfo)):
            try:
                rename_file(str(src_nfo), str(dest.with_suffix(".nfo")))
            except Exception as e:
                logger.debug(f"同名 NFO 移动失败（忽略）: {e}")
        return True, dest, ""
    except Exception as e:
        return False, None, str(e)


def move_to_failed(error_files: List[Tuple[str, str]]) -> None:
    """将处理失败的文件移入 _failed 子目录。"""
    for full_path, fname in error_files:
        ok, dest, err = move_with_companion(full_path, "_failed")
        if not ok:
            logger.warning(f"移动失败文件失败: {fname}" + (f": {err}" if err else ""))
            continue
        logger.info(f"已将失败文件移至: {dest.parent.name}/{dest.name}")
