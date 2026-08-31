"""运行时依赖引导：ffmpeg/ffprobe 定位 + openai 库延迟绑定。"""
import os
import sys
import shutil
import platform
from pathlib import Path
from typing import Tuple


class DependencyError(RuntimeError):
    """运行时依赖缺失（ffmpeg/openai 等）。"""


class AIClientBindings:
    """openai 库的延迟绑定容器。"""
    OpenAI = None
    APIConnectionError = None
    APITimeoutError = None
    RateLimitError = None
    InternalServerError = None
    APIStatusError = None

    def load(self) -> None:
        """导入 openai 库并绑定错误类。失败则抛出 DependencyError。"""
        try:
            from openai import (
                OpenAI,
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                InternalServerError,
                APIStatusError,
            )
            self.OpenAI = OpenAI
            self.APIConnectionError = APIConnectionError
            self.APITimeoutError = APITimeoutError
            self.RateLimitError = RateLimitError
            self.InternalServerError = InternalServerError
            self.APIStatusError = APIStatusError
        except ImportError:
            raise DependencyError(
                f"未检测到 openai 库。\n"
                f"请安装：{sys.executable} -m pip install 'openai>=1.0'"
            )

    @property
    def retryable_errors(self) -> Tuple[type, ...]:
        """可重试的 openai 异常元组，未加载时抛 DependencyError。"""
        if self.APIConnectionError is None:
            raise DependencyError("openai 依赖未加载，请先调用 ensure_dependencies()")
        return (self.APIConnectionError, self.APITimeoutError,
                self.RateLimitError, self.InternalServerError)


class FFmpegTools:
    """ffmpeg/ffprobe 可执行文件路径。"""
    ffmpeg: str = ""
    ffprobe: str = ""

    def locate(self) -> None:
        """定位 ffmpeg/ffprobe，失败则抛出 DependencyError。"""
        self.ffmpeg = find_tool("ffmpeg")
        self.ffprobe = find_tool("ffprobe")
        missing = [n for n, p in (("ffmpeg", self.ffmpeg), ("ffprobe", self.ffprobe)) if not p]
        if missing:
            raise DependencyError(
                f"找不到: {', '.join(missing)}\n"
                "请安装 ffmpeg 并确保在系统 PATH 中或放在脚本目录的 ffmpeg/ 下。"
            )


bindings = AIClientBindings()
ffmpeg_tools = FFmpegTools()


def find_tool(name: str) -> str:
    """定位可执行文件：脚本目录 ffmpeg/ → PATH → 常见安装路径。"""
    ext = ".exe" if platform.system() == "Windows" else ""
    local = Path(__file__).parent.parent / "ffmpeg" / (name + ext)
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if found:
        return found
    if platform.system() == "Windows":
        # WinGet Links 目录（旧版 winget v1.7 及以前）
        winget_links = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"
        if winget_links.exists():
            p = winget_links / (name + ext)
            if p.exists():
                return str(p)
        # 常见安装路径
        for base in (
            r"C:\ffmpeg\bin",
            r"C:\Program Files\ffmpeg\bin",
            r"C:\Program Files (x86)\ffmpeg\bin",
        ):
            p = Path(base) / (name + ext)
            if p.exists():
                return str(p)
        # winget 安装后的默认路径（用户级）
        for user_base in os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramFiles", ""):
            if not user_base:
                continue
            for sub in ("ffmpeg", "Gyan.FFmpeg", "Gyan.FFmpeg.Shared"):
                p = Path(user_base) / sub / "bin" / (name + ext)
                if p.exists():
                    return str(p)
    else:
        for base in ("/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"):
            p = Path(base) / name
            if p.exists():
                return str(p)
    return ""


def ensure_dependencies() -> None:
    """确保 ffmpeg/ffprobe 与 openai 库可用。"""
    ffmpeg_tools.locate()
    bindings.load()
