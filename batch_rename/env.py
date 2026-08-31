"""环境：常量、日志。"""
import sys
import logging
import subprocess

# ── 路径与文件系统常量 ──
LONG_PATH_PREFIX = "\\\\?\\"
# 视频扩展名分级：CERTAIN 直接收集；AMBIGUOUS 需魔数嗅探确认
VIDEO_EXTS_CERTAIN = frozenset(
    ".mp4 .mov .avi .mkv .m4v .wmv .flv .webm .mts .m2ts .mpg .mpeg .vob .3gp .3g2".split()
)
VIDEO_EXTS_AMBIGUOUS = frozenset({".ts"})

# ── 子进程平台参数（Windows 下隐藏控制台窗口，避免终端闪烁）──
SUBPROCESS_KWARGS = {}
if sys.platform == "win32":
    SUBPROCESS_KWARGS["creationflags"] = subprocess.CREATE_NO_WINDOW

# ── 终端显示 ──
SEP_CHAR = "━"
SEP_CHAR_SOFT = "─"
SEP_WIDTH = 74
SEP_INDENT = " "

# ── 全局日志器 ──
# 只 getLogger 不做配置，handler/level 由入口调用方负责
logger = logging.getLogger("BatchRename")
