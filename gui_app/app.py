"""app.py — pywebview 主入口。"""
from __future__ import annotations

import atexit
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path

# ── 确保可以 import gui_app 子模块 ──
SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


os.environ.pop("PYTHONNET_RUNTIME", None)

# 记录 pythonnet 是否成功初始化，供 mainthread.py 选择对话框实现（WinForms STA vs 原生 ctypes）
try:
    from gui_app.mainthread import set_pythonnet_status
    __import__("clr")  # 能 import 说明 pythonnet 已就绪
    set_pythonnet_status(True)
except Exception:
    try:
        from gui_app.mainthread import set_pythonnet_status
        set_pythonnet_status(False)
    except Exception:
        pass


# ── 全局崩溃日志 ──
CRASH_LOG = Path(tempfile.gettempdir()) / "video_rename_gui_crash.log"
CRASH_LOG_PREV = Path(tempfile.gettempdir()) / "video_rename_gui_crash.prev.log"


def _crash_log(text: str) -> None:
    try:
        CRASH_LOG.write_text(text, encoding="utf-8", errors="replace")
    except Exception:
        pass


def _msgbox(msg: str, title: str = "错误", flags: int = 0x10) -> int:
    """弹出 Win32 MessageBox；失败时输出到 stderr。"""
    try:
        import ctypes
        return ctypes.windll.user32.MessageBoxW(0, msg, title, flags)
    except Exception:
        print(f"[{title}] {msg}", file=sys.stderr)
        return 0


def _setup_crash_handlers() -> None:
    """注册全局异常钩子。"""
    try:
        def _report_main(exc_type, exc_value, exc_tb):
            tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            _crash_log(tb_text)
            _msgbox(f"程序出错:\n{tb_text[-1200:]}", "错误")

        def _report_thread(args):
            tb_text = "".join(traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback))
            _crash_log(tb_text)
            sys.stderr.write(f"[crash] 后台线程异常 ({args.thread.name}): {args.exc_value}\n")

        sys.excepthook = _report_main
        threading.excepthook = _report_thread
    except Exception:
        pass


_setup_crash_handlers()


# ── 导入 webview ──
try:
    import webview
except ImportError:
    from gui_app.env import _resolve_python
    _msgbox(
        "缺少 pywebview 库。\n\n"
        "请用当前生效的 Python 解释器安装:\n"
        f'"{_resolve_python()}" -m pip install pywebview',
        "依赖缺失",
    )
    sys.exit(1)


from gui_app.api import Api
from gui_app.env import check_all_startup_deps
from gui_app.workspace_store import clear_sources, ensure_workspace
from gui_app import ui_loader

# ── 配置 ──
WINDOW_TITLE = "Hammy"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 860

HTML_PATH = SCRIPT_DIR / "ui" / "index.html"


# ── 窗口尺寸记忆 ──


def _load_window_geometry() -> dict:
    """从 _workspace/config.json 读取窗口几何。"""
    try:
        from gui_app import config_store
        cfg = config_store.load_config()
        return cfg.get("window_geometry") or {}
    except Exception:
        return {}


def _save_window_geometry(geo: dict) -> None:
    """保存窗口几何到 _workspace/config.json（原子读-改-写，不与设置保存互覆盖）。"""
    try:
        from gui_app import config_store
        config_store.update_config(lambda cfg: cfg.update(window_geometry=geo) or cfg)
    except Exception as e:
        sys.stderr.write(f"[app] 窗口几何保存失败: {e}\n")


def _is_visible_position(x: int, y: int, w: int, h: int) -> bool:
    """判断窗口左上角是否落在可见屏幕区域内。"""
    # 最小化哨兵（-32000）直接判定不可见
    if x <= -30000 or y <= -30000:
        return False
    try:
        import ctypes
        # 虚拟屏幕（覆盖所有显示器）的范围
        vx = ctypes.windll.user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
        vy = ctypes.windll.user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
        vw = ctypes.windll.user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
        vh = ctypes.windll.user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
        # 窗口需至少有一部分落在虚拟屏幕范围内
        return (vx - w < x < vx + vw) and (vy - h < y < vy + vh)
    except Exception:
        # 获取屏幕范围失败时不拦截，沿用保存的坐标
        return True


# ── 启动期依赖检测 ──


def _check_startup_deps() -> dict:
    """依赖检测：返回 {ok, missing: [str], ...}。"""
    return check_all_startup_deps()


def _show_dep_error(deps: dict) -> None:
    missing = deps.get("missing", [])
    msg = "启动前缺少以下依赖：\n\n"
    for m in missing:
        msg += f"  • {m}\n"

    needs_ffmpeg = any("ffmpeg" in m for m in missing)
    needs_libs = any(
        ("openai" in m or "pywebview" in m) and "未安装" in m for m in missing
    )
    needs_wv2 = any("WebView2" in m for m in missing)
    needs_dotnet = any(".NET" in m for m in missing)

    msg += "\n请按以下步骤处理：\n"
    step = 1
    if needs_ffmpeg:
        msg += f"  {step}. 把 ffmpeg.exe / ffprobe.exe 放到 {APP_ROOT}\\ffmpeg\\ 或加入系统 PATH\n"
        step += 1
    if needs_libs:
        # 用当前生效的 Python 解释器给出安装命令（portable 或系统 python 均适用）
        py = deps.get("python_path") or sys.executable
        msg += f"  {step}. 安装 Python 依赖库：\n     \"{py}\" -m pip install openai pywebview\n"
        step += 1
    if needs_wv2:
        msg += f"  {step}. WebView2 Runtime: https://developer.microsoft.com/microsoft-edge/webview2/\n"
        step += 1
    if needs_dotnet:
        msg += f"  {step}. .NET Framework 4.8: 请启用 Windows 功能或安装 .NET Framework（pythonnet 需要）\n"
        step += 1
    _msgbox(msg, "依赖缺失")


# ── 单实例守卫 ──

_SINGLE_INSTANCE_MUTEX = None  # 持有句柄防 GC；不 CloseHandle，进程退出时由 OS 回收


def _ensure_single_instance() -> None:
    """单实例守卫：已有实例运行时激活其窗口并退出本进程。"""
    global _SINGLE_INSTANCE_MUTEX
    if sys.platform != "win32":
        return
    try:
        import ctypes
        import ctypes.wintypes
        import hashlib

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.restype = ctypes.c_void_p
        k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        key = hashlib.sha1(
            os.path.normcase(str(APP_ROOT)).encode("utf-8", "surrogatepass")
        ).hexdigest()[:8]
        handle = k32.CreateMutexW(None, False, f"Local\\Hammy_{key}")
        if not handle:
            return  # 创建失败：放行，不阻断启动
        if ctypes.get_last_error() != 183:  # ERROR_ALREADY_EXISTS
            _SINGLE_INSTANCE_MUTEX = handle  # 首个实例：持有句柄到进程退出
            return
        # 已有实例在运行：关闭本进程拿到的句柄，尝试激活已有窗口后退出
        k32.CloseHandle(handle)
        try:
            u32 = ctypes.WinDLL("user32", use_last_error=True)
            u32.FindWindowW.restype = ctypes.wintypes.HWND
            u32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
            hwnd = u32.FindWindowW(None, WINDOW_TITLE)
            if hwnd:
                if u32.IsIconic(hwnd):
                    u32.ShowWindow(hwnd, 9)  # SW_RESTORE：从最小化还原
                u32.SetForegroundWindow(hwnd)
        except Exception:
            pass
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        sys.stderr.write(f"[single-instance] 守卫异常，放行启动: {e}\n")


# ── 主入口 ──


def _init_workspace() -> str:
    """依赖检测 + workspace 初始化 + 构建 HTML。返回 html_content。"""
    # 保留上次崩溃日志（重命名为 .prev），而非无条件删除
    try:
        if CRASH_LOG.exists():
            CRASH_LOG.replace(CRASH_LOG_PREV)
    except Exception:
        pass

    deps = _check_startup_deps()
    if not deps["ok"]:
        _show_dep_error(deps)
        sys.exit(1)

    ensure_workspace()
    # 启动时清除上次添加的文件夹/文件记录（NFO/缩略图保留，可手动清理）
    try:
        clear_sources()
    except Exception:
        pass

    if not HTML_PATH.exists():
        _msgbox(f"界面文件未找到:\n{HTML_PATH}", "文件缺失")
        sys.exit(1)
    return ui_loader.build_html()


def _create_window(api: "Api", html_content: str):
    """创建 pywebview 窗口并从 config 恢复几何。返回 window。"""
    geo = _load_window_geometry()
    init_w = geo.get("width") or WINDOW_WIDTH
    init_h = geo.get("height") or WINDOW_HEIGHT
    init_x = geo.get("x")
    init_y = geo.get("y")
    # 坐标不可见（最小化哨兵/拔副屏残留）则放弃保存值，走默认居中
    if init_x is not None and init_y is not None and not _is_visible_position(
            init_x, init_y, init_w, init_h):
        init_x = init_y = None
    window_kwargs = dict(
        title=WINDOW_TITLE,
        html=html_content,
        width=init_w,
        height=init_h,
        js_api=api,
        min_size=(900, 600),
        resizable=True,
        easy_drag=False,
    )
    if init_x is not None and init_y is not None:
        window_kwargs["x"] = init_x
        window_kwargs["y"] = init_y
    window = webview.create_window(**window_kwargs)
    api.set_window(window)
    return window


def _bind_window_events(window, api: "Api"):
    """绑定所有窗口事件（resized/moved/closing/closed）。

    返回 _geo_flush_loop 供后台线程使用。
    """
    # ── 几何保存节流机制 ──
    _geo_save_lock = threading.Lock()
    _geo_save_pending = {"data": None}
    _geo_stop = threading.Event()  # 统一生命周期：关窗时停止几何保存线程

    def _schedule_save_geo(data):
        # Workaround: Windows 最小化时 WinForms 上报 (-32000, -32000) 哨兵坐标，
        # 跳过保存避免下次窗口恢复到屏外
        if data.get("x", 0) <= -30000 or data.get("y", 0) <= -30000:
            return
        with _geo_save_lock:
            _geo_save_pending["data"] = data

    def _geo_flush_loop():
        while not _geo_stop.is_set():
            _geo_stop.wait(1.0)  # 每秒检查一次，节流；stop 时立即退出
            with _geo_save_lock:
                data = _geo_save_pending["data"]
                _geo_save_pending["data"] = None
            if data:
                _save_window_geometry(data)

    # ── resized / moved 事件保存几何 ──
    try:
        def _capture_geo():
            _schedule_save_geo({
                "width": window.width,
                "height": window.height,
                "x": window.x,
                "y": window.y,
            })

        def _on_resized(*_args):
            try:
                _capture_geo()
            except Exception:
                pass

        def _on_moved(*_args):
            try:
                _capture_geo()
            except Exception:
                pass
        window.events.resized += _on_resized
        window.events.moved += _on_moved
    except Exception as e:
        sys.stderr.write(f"[events] resized/moved hook 注册失败: {e}\n")

    # ── 关窗/退出时清理 ffmpeg 子进程（幂等：atexit + closed 只执行一次）──
    _cleaned_up = threading.Event()

    def _cleanup_subprocesses():
        """统一清理：停止运行中的任务 + 等待 history 落盘 + 终止 ffmpeg 子进程。"""
        if _cleaned_up.is_set():
            return
        _cleaned_up.set()
        try:
            api.stop()  # 触发 stop_event
        except Exception:
            pass
        # 等待 runner 线程落盘 history：每 0.5s 轮询，最长 30s，超时则告警而非静默丢失。
        try:
            import time as _time
            runner = getattr(api, "_runner", None)
            if runner and runner._thread and runner._thread.is_alive():
                deadline = _time.monotonic() + 30
                flushed = getattr(runner, "_history_flushed", None)
                while runner._thread.is_alive() and _time.monotonic() < deadline:
                    runner._thread.join(timeout=0.5)
                    if flushed is not None and flushed.is_set():
                        break
                if runner._thread.is_alive():
                    msg = ("[cleanup] runner 未在 30s 内完成 history 落盘，"
                           "最后一批处理记录可能丢失\n")
                    _crash_log(msg)
                    sys.stderr.write(msg)
        except Exception:
            pass
        try:
            from batch_rename.subprocess_registry import terminate_all_subprocesses
            terminate_all_subprocesses()
        except Exception as e:
            sys.stderr.write(f"[cleanup] terminate_all_subprocesses failed: {e}\n")
        _geo_stop.set()  # 停止几何保存线程
        # stop 后补写一次 pending：geo 线程是 daemon，进程退出时可能来不及落盘。
        try:
            with _geo_save_lock:
                data = _geo_save_pending["data"]
                _geo_save_pending["data"] = None
            if data:
                _save_window_geometry(data)
        except Exception:
            pass

    # atexit 兜底（崩溃/强杀时也尽量清理）
    atexit.register(_cleanup_subprocesses)

    # ── 处理中关窗加确认 ──
    def _on_closing():
        """关窗前确认（处理进行中时）。返回 False 取消关闭。"""
        try:
            runner = getattr(api, "_runner", None)
            if runner and runner.is_running:
                # 弹原生确认框（前端 confirm 在 closing 事件里不可靠）
                MB_YESNO = 0x04
                MB_ICONQUESTION = 0x20
                MB_DEFBUTTON2 = 0x100
                result = _msgbox(
                    "处理仍在进行，关闭将中断当前任务。\n\n确定关闭吗？",
                    "确认关闭",
                    MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON2,
                )
                # IDYES=6, IDNO=7
                if result != 6:
                    return False  # 取消关闭
        except Exception:
            pass
        return True  # 允许关闭

    try:
        window.events.closing += _on_closing
    except Exception as e:
        sys.stderr.write(f"[events] closing hook 注册失败: {e}\n")

    try:
        window.events.closed += _cleanup_subprocesses
    except Exception as e:
        sys.stderr.write(f"[events] closed hook 注册失败: {e}\n")

    return _geo_flush_loop


def _start_background_loops(geo_flush_loop, api=None) -> callable:
    """构建后台循环，返回 webview.start 的 func 回调。

    """

    def _on_started():
        from gui_app.js_push import js_pusher
        import time as _time

        def _flush_loop():
            while True:
                js_pusher.flush()
                _time.sleep(0.06)

        # flush 线程必须等页面加载完成（前端 core.js 已定义 window.__ui）再启动：
        # 页面未就绪时 evaluate_js 会报错，连续失败会触发 js_push.py 的 5s 暂停
        # 并清空队列，导致启动期推送事件（llama 状态等）丢失
        _flush_started = threading.Event()

        def _start_flush():
            if _flush_started.is_set():
                return
            _flush_started.set()
            t1 = threading.Thread(target=_flush_loop, name="js-flush", daemon=True)
            t1.start()

        window = getattr(api, "_window", None) if api is not None else None
        if window is not None:
            try:
                window.events.loaded += lambda *_a: _start_flush()
                # 兜底：若订阅前 loaded 已触发（pywebview 事件不重放），延迟后直接
                # 启动；期间推送在队列中正常积压，flush 启动后统一发出，不丢事件
                def _flush_fallback():
                    _time.sleep(15)
                    _start_flush()
                threading.Thread(
                    target=_flush_fallback, name="js-flush-fallback", daemon=True
                ).start()
            except Exception as e:
                sys.stderr.write(f"[events] loaded hook 注册失败，直接启动 flush: {e}\n")
                _start_flush()
        else:
            _start_flush()

        t2 = threading.Thread(target=geo_flush_loop, name="geo-save", daemon=True)
        t2.start()

        # 本地推理自动运行（扩展功能，未安装/未启用时在端点内直接跳过）
        if api is not None and hasattr(api, "auto_run_llama"):
            def _auto_run():
                try:
                    api.auto_run_llama()
                except Exception as e:
                    sys.stderr.write(f"[auto_run] 启动本地推理失败: {e}\n")
            threading.Thread(target=_auto_run, name="llama-auto-run", daemon=True).start()

    return _on_started


def main() -> None:
    _ensure_single_instance()  # 必须先于 _init_workspace（避免清空运行中实例的 manifest / 双进程写配置）
    html_content = _init_workspace()

    # 尽早创建作业对象（kill-on-close）：此后注册的所有子进程（ffmpeg、
    # llama-server 等）都会挂入其中，程序以任意方式退出时由内核自动终止
    try:
        from batch_rename.job_object import init_job_object
        init_job_object()
    except Exception as e:
        sys.stderr.write(f"[cleanup] job_object 初始化失败: {e}\n")

    api = Api()
    window = _create_window(api, html_content)

    geo_flush_loop = _bind_window_events(window, api)
    on_started = _start_background_loops(geo_flush_loop, api)

    try:
        # Windows 上使用 winforms backend（pywebview 6.x 唯一支持），
        # 页面渲染由 WebView2 Runtime 提供
        webview.start(func=on_started, private_mode=False)
    except Exception as e:
        _crash_log(f"webview.start error: {traceback.format_exc()}")
        _msgbox(
            f"启动窗口失败:\n{e}\n\n"
            "可能原因：\n"
            "1. WebView2 Runtime 未安装或版本过旧（请从 microsoft.com 下载）\n"
            "2. pythonnet 初始化失败（需要 Windows 自带 .NET Framework 4.8）\n"
            "3. 被系统安全策略阻止加载.NET（解压前尝试右键属性先解锁）\n\n"
            "请安装对应运行时后重试。",
            "启动错误",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
