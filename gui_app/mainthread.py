"""mainthread.py — 把阻塞 UI 调用（文件对话框）封送到 WinForms 主线程。

"""
from __future__ import annotations

import sys
from typing import Callable, TypeVar

T = TypeVar("T")

# pythonnet 是否成功初始化（由 app.py 启动时设置）
_pythonnet_ok = False


def set_pythonnet_status(ok: bool) -> None:
    """由 app.py 在 _init_pythonnet 后调用，记录 pythonnet 是否可用。"""
    global _pythonnet_ok
    _pythonnet_ok = ok


def _get_main_form():
    """获取主窗体（pywebview 创建的第一个 Form）。"""
    if not _pythonnet_ok:
        return None
    try:
        from System.Windows.Forms import Application
        forms = Application.OpenForms
        if forms is None or forms.Count == 0:
            return None
        return forms[0]
    except Exception as e:
        sys.stderr.write(f"[mainthread] _get_main_form failed: {e}\n")
        return None


def run_on_ui_thread(fn: Callable[[], T]) -> T:
    """在 WinForms 主线程上同步执行 fn，返回其结果；异常向上抛。

    若无法获取主窗体，直接在当前线程调用 fn（退化为非线程封送）。
    """
    form = _get_main_form()
    if form is None:
        return fn()

    box: dict = {}

    def _run():
        try:
            box["v"] = fn()
        except Exception as e:
            box["e"] = e

    try:
        from System import Action
        form.Invoke(Action(_run))
    except Exception as e:
        sys.stderr.write(f"[mainthread] Invoke failed: {e}\n")
        return fn()

    if "e" in box:
        raise box["e"]
    return box.get("v")
