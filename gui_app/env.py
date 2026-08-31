"""env.py — GUI 层的环境常量与路径。"""
import os
import sys
from pathlib import Path

# 应用根目录（gui_app 的上一级）
APP_ROOT = Path(__file__).resolve().parent.parent

def _resolve_ffmpeg(name: str) -> str:
    """解析 ffmpeg/ffprobe 路径：优先目录内 ffmpeg\\，其次系统 PATH 及常见安装路径。"""
    from batch_rename.dependencies import find_tool
    return find_tool(name)


# 便携 Python（优先）；缺失时回退到系统 PATH / 当前解释器，使“不依赖便携 Python”
# 的系统 Python 模式也能为 whisper/llama 建立虚拟环境。
def _resolve_portable_python_exe():
    portable = APP_ROOT / "python" / "python.exe"
    if portable.is_file():
        return str(portable)
    import shutil
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            return found
    return sys.executable


PYTHON_EXE = _resolve_portable_python_exe()
PYTHONW_EXE = (
    str(APP_ROOT / "python" / "pythonw.exe")
    if (APP_ROOT / "python" / "pythonw.exe").is_file()
    else ""
)
# ffmpeg / ffprobe：优先目录内 ffmpeg\，其次系统 PATH（及常见安装路径）
FFMPEG_EXE = _resolve_ffmpeg("ffmpeg")
FFPROBE_EXE = _resolve_ffmpeg("ffprobe")

# 从引擎层导入，避免重复定义
from batch_rename.env import VIDEO_EXTS_CERTAIN, VIDEO_EXTS_AMBIGUOUS  # noqa: E402

# 文件对话框过滤器（从引擎层扩展名集合单一来源生成，避免多处硬编码不一致）
VIDEO_EXTS = VIDEO_EXTS_CERTAIN | VIDEO_EXTS_AMBIGUOUS
VIDEO_FILTER_GLOB = ";".join(f"*{ext}" for ext in sorted(VIDEO_EXTS))
VIDEO_FILTER_WINFORMS = f"视频文件 ({VIDEO_FILTER_GLOB})"
VIDEO_FILTER_NATIVE = f"视频文件\0{VIDEO_FILTER_GLOB}\0所有文件\0*.*\0\0"

# 把 APP_ROOT 加入 sys.path，便于 import batch_rename
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _resolve_python() -> str:
    """解析可用的 Python 解释器路径：优先便携 Python，其次系统 PATH（python/python3），
    最后回退到当前进程 sys.executable（保证总有值，bat 与依赖提示据此给出安装命令）。
    """
    if os.path.isfile(PYTHON_EXE):
        return PYTHON_EXE
    if os.path.isfile(PYTHONW_EXE):
        return PYTHONW_EXE
    import shutil
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            return found
    return sys.executable


def _python_available() -> bool:
    """Python 是否可用：便携 Python 存在，或系统 PATH 中能找到 python/python3。"""
    if os.path.isfile(PYTHON_EXE) or os.path.isfile(PYTHONW_EXE):
        return True
    import shutil
    return any(shutil.which(n) for n in ("python", "python3"))


def check_dependencies() -> dict:
    """启动期依赖检测。
        返回 {ok: bool, missing: [str], python: bool, ffmpeg: bool, ffprobe: bool, openai: bool, webview: bool}。
        不做任何网络访问或系统写入。
    """
    result = {
        "ok": True,
        "missing": [],
        "python": False,
        "ffmpeg": False,
        "ffprobe": False,
        "openai": False,
        "webview": False,
        "ffmpeg_path": FFMPEG_EXE,
        "ffprobe_path": FFPROBE_EXE,
    }

    # Python 解释器：便携 Python 存在，或系统 PATH 中可用（python/python3）即视为可用
    result["python"] = _python_available()
    result["python_path"] = _resolve_python()

    # ffmpeg / ffprobe（FFMPEG_EXE/FFPROBE_EXE 已按 目录内 → 系统 PATH 解析）
    if os.path.isfile(FFMPEG_EXE):
        result["ffmpeg"] = True
    else:
        result["missing"].append("ffmpeg.exe 未找到（目录内 ffmpeg\\ 或系统 PATH 均无）")
    if os.path.isfile(FFPROBE_EXE):
        result["ffprobe"] = True
    else:
        result["missing"].append("ffprobe.exe 未找到（目录内 ffmpeg\\ 或系统 PATH 均无）")

    try:
        __import__("openai")
        result["openai"] = True
    except ImportError:
        result["missing"].append("openai 库未安装（pip install openai）")

    try:
        __import__("webview")
        result["webview"] = True
    except ImportError:
        result["missing"].append("pywebview 库未安装（pip install pywebview）")

    result["ok"] = not result["missing"]
    return result


def check_all_startup_deps() -> dict:
    """启动期全量依赖检测（基础 + WebView2 + .NET）。"""
    deps = check_dependencies()
    if sys.platform == "win32":
        wv2 = check_webview2_runtime()
        deps["webview2"] = wv2
        if not wv2:
            deps["missing"].append(
                "WebView2 Runtime 未安装（请从 microsoft.com 下载安装）"
            )
            deps["ok"] = False
        # pythonnet 统一走 netfx 默认运行时（Windows 自带 .NET Framework 4.8），
        # 便携与系统 Python 一致，不需要 .NET Desktop Runtime。
        dotnet = check_dotnet_framework()
        deps["dotnet"] = dotnet
        if not dotnet["installed"]:
            deps["missing"].append(
                ".NET Framework 4.8 不可用（pythonnet + WinForms 需要）"
            )
            deps["ok"] = False
    return deps


def check_webview2_runtime() -> bool:
    """Windows 下检测 WebView2 Runtime 是否可用（pywebview 在 Win 上需要）。"""
    if sys.platform != "win32":
        return True
    try:
        import winreg  # type: ignore
        for hive, path in (
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"),
            (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"),
        ):
            try:
                key = winreg.OpenKey(hive, path)
                winreg.CloseKey(key)
                return True
            except OSError:
                continue
        return False
    except Exception:
        # 检测失败不阻断启动，让 webview.start 报具体错
        return True


def check_dotnet_framework() -> dict:
    """检测 Windows 自带的 .NET Framework 4.x（pythonnet netfx 运行时需要）。"""
    result = {"installed": False, "versions": [], "latest": None, "mode": "netfx"}
    if sys.platform != "win32":
        result["installed"] = True
        return result
    try:
        import winreg  # type: ignore
        # .NET Framework 4.x 的安装版本注册在 HKLM\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full
        key_path = r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full"
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                key = winreg.OpenKey(hive, key_path)
                try:
                    release, _ = winreg.QueryValueEx(key, "Release")
                finally:
                    winreg.CloseKey(key)
                break
            except OSError:
                continue
        else:
            release = 0
        # Release 值与版本对照（>=528040 为 4.8）
        if release:
            result["installed"] = True
            result["versions"] = [f".NET Framework 4.x (Release {release})"]
            result["latest"] = "4.8" if release >= 528040 else "4.x"
    except Exception as e:
        # 检测手段失败不等于依赖缺失，放行并记录（绿色/受限环境）
        sys.stderr.write(f"[env] .NET Framework 检测异常（放行）: {e}\n")
        result["installed"] = True
    return result


def make_logger(prefix: str):
    """生成『callback 优先、缺省写 stderr（带模块前缀）』的日志函数。"""
    def _log(msg: str, callback=None) -> None:
        if callback:
            callback(msg)
        else:
            sys.stderr.write(f"[{prefix}] {msg}\n")
    return _log
