"""tray.py — 最小化隐藏到系统托盘（pystray）。

pystray/PIL 未安装时 start_tray 返回 False，窗口保持普通最小化行为。
有 hammy.ico / icon.ico 时用作托盘图标，否则程序生成占位图标。
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

_WINDOW_TITLE = "Hammy"  # 与 app.WINDOW_TITLE 一致（避免循环导入）

_icon = None
_poll_stop: threading.Event | None = None
_u32 = None


def _user32():
    global _u32
    if _u32 is None:
        import ctypes
        u32 = ctypes.WinDLL("user32")
        u32.FindWindowW.restype = ctypes.c_void_p
        u32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        u32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        u32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
        _u32 = u32
    return _u32


def _find_hwnd() -> int | None:
    try:
        return _user32().FindWindowW(None, _WINDOW_TITLE) or None
    except Exception:
        return None


def _load_icon_image():
    """托盘图标：优先 gui_app/ui 下的 hammy.ico / icon.ico，否则生成占位图。"""
    from PIL import Image, ImageDraw, ImageFont

    uidir = Path(__file__).resolve().parent / "ui"
    for name in ("hammy.ico", "icon.ico"):
        p = uidir / name
        if p.is_file():
            try:
                return Image.open(p).convert("RGBA")
            except Exception:
                break

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([2, 2, 62, 62], radius=14, fill=(45, 90, 160, 255))
    try:
        font = ImageFont.truetype("arialbd.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), "H", font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((32 - tw / 2 - bbox[0], 32 - th / 2 - bbox[1]), "H",
           font=font, fill=(255, 255, 255, 255))
    return img


def start_tray(window) -> bool:
    """启动托盘。成功返回 True；pystray/PIL 不可用返回 False（普通最小化）。"""
    global _icon, _poll_stop
    if _icon is not None:
        return True
    try:
        import pystray
        img = _load_icon_image()
    except Exception as e:
        sys.stderr.write(f"[tray] pystray 不可用，托盘未启用: {e}\n")
        return False

    u32 = _user32()

    def _show_window():
        try:
            window.show()
        except Exception:
            pass
        hwnd = _find_hwnd()
        if hwnd:
            u32.ShowWindow(hwnd, 9)  # SW_RESTORE：同时覆盖最小化/隐藏两种状态
            u32.SetForegroundWindow(hwnd)

    def _hide_window():
        try:
            window.hide()
        except Exception:
            pass
        hwnd = _find_hwnd()
        if hwnd:
            u32.ShowWindow(hwnd, 0)  # SW_HIDE

    def _on_minimize(*_a):
        try:
            _hide_window()
        except Exception:
            pass

    def _on_show(_icon=None, _item=None):
        _show_window()

    def _on_quit(_icon=None, _item=None):
        stop_tray()
        try:
            window.destroy()
        except Exception as e:
            sys.stderr.write(f"[tray] destroy 失败: {e}\n")

    def _poll_loop(stop: threading.Event):
        # minimized 事件不可用时的兜底：轮询 IsIconic
        while not stop.is_set():
            stop.wait(1.0)
            hwnd = _find_hwnd()
            if hwnd and u32.IsWindowVisible(hwnd) and u32.IsIconic(hwnd):
                _hide_window()

    menu = pystray.Menu(
        pystray.MenuItem("显示", _on_show, default=True),  # default: 双击左键触发
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("退出", _on_quit),
    )
    _icon = pystray.Icon(_WINDOW_TITLE, img, _WINDOW_TITLE, menu)

    has_minimized_event = False
    try:
        window.events.minimized += _on_minimize
        has_minimized_event = True
    except Exception:
        pass
    try:
        window.events.closed += lambda *_a: stop_tray()
    except Exception:
        pass

    threading.Thread(target=_icon.run, name="tray-icon", daemon=True).start()
    if not has_minimized_event:
        _poll_stop = threading.Event()
        threading.Thread(target=_poll_loop, args=(_poll_stop,),
                         name="tray-poll", daemon=True).start()
    return True


def stop_tray():
    """移除托盘图标并停止轮询（幂等）。"""
    global _icon, _poll_stop
    if _poll_stop is not None:
        _poll_stop.set()
        _poll_stop = None
    icon, _icon = _icon, None
    if icon is not None:
        try:
            icon.stop()
        except Exception:
            pass
