"""工具函数：文件 I/O、终端显示。"""
import os
import platform
from pathlib import Path
from typing import Tuple, Union

from .env import LONG_PATH_PREFIX, SEP_CHAR, SEP_CHAR_SOFT, SEP_WIDTH, SEP_INDENT, logger
from .types import SummaryStats


# ═══ 文件操作 ═══════════════════════════════════════════════════
def to_long_path(path: str) -> str:
    """Windows 长路径前缀，突破 260 字符限制。"""
    if platform.system() == "Windows":
        if path.startswith(LONG_PATH_PREFIX):
            return path
        abs_path = os.path.abspath(path)
        if abs_path.startswith("\\\\"):
            return "\\\\?\\UNC\\" + abs_path[2:]
        return LONG_PATH_PREFIX + abs_path
    return os.path.abspath(path)


def path_exists(path: Union[str, Path]) -> bool:
    return os.path.exists(to_long_path(str(path)))


def path_stat(path: Union[str, Path]) -> os.stat_result:
    return os.stat(to_long_path(str(path)))


def rename_file(src: str, dst: str) -> Tuple[bool, str]:
    try:
        os.replace(to_long_path(src), to_long_path(dst))
        return True, ""
    except OSError as e:
        logger.error(f"系统重命名失败: {e}")
        return False, str(e)


# ═══ 安全类型转换 ═════════════════════════════════════════════
def safe_int(v, default=0):
    """容错 int 转换：兼容 None/空串/非法字符串/缺失字段。失败返回 default。
    需要区分"值为 0"与"字段缺失"时显式传 default=None。"""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def safe_float(v, default=0.0):
    """容错 float 转换：兼容 None/空串/非法字符串/缺失字段。失败返回 default。
    需要区分"值为 0.0"与"字段缺失"时显式传 default=None。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ═══ 文本 / 错误检测 ═══════════════════════════════════════════
def sep(soft: bool = False) -> str:
    ch = SEP_CHAR_SOFT if soft else SEP_CHAR
    return SEP_INDENT + (ch * SEP_WIDTH)


def retry_hint(retries: int, success: bool = False) -> str:
    if retries <= 0:
        return ""
    return f" (重试 {retries} 次后成功)" if success else f" (重试 {retries} 次)"


def print_summary(summary: SummaryStats) -> None:
    """输出批处理结果汇总报告。"""
    logger.info("")
    logger.info(sep())
    time_str = f" (耗时 {summary.elapsed:.0f}秒)" if summary.elapsed else ""
    logger.info(f"{SEP_INDENT}批量重命名完成{time_str}")
    logger.info(sep(soft=True))
    logger.info(f"{SEP_INDENT}成功: {summary.ok}")
    if summary.skipped > 0:
        logger.info(f"{SEP_INDENT}跳过: {summary.skipped}")
    if summary.error > 0:
        logger.info(f"{SEP_INDENT}失败: {summary.error}")
    if getattr(summary, "cancelled", 0) > 0:
        logger.info(f"{SEP_INDENT}未处理（已取消）: {summary.cancelled}")
    if summary.processed_count > 0 and summary.elapsed:
        logger.info(f"{SEP_INDENT}平均: {summary.elapsed / summary.processed_count:.1f} 秒/视频")
    logger.info(sep())
