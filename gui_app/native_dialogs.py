"""native_dialogs.py — Windows 原生文件/文件夹对话框（ctypes 实现）。

- 文件选择: comdlg32.GetOpenFileNameW (支持多选)
- 文件夹选择: shell32.SHBrowseForFolderW
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import sys
from typing import List

if sys.platform != "win32":
    raise ImportError("native_dialogs only supports Windows")

from .env import VIDEO_FILTER_NATIVE  # noqa: E402

# ── 文件选择对话框 ──


class _OPENFILENAMEW(ctypes.Structure):
    _fields_ = [
        ("lStructSize", wt.DWORD),
        ("hwndOwner", wt.HWND),
        ("hInstance", wt.HINSTANCE),
        ("lpstrFilter", wt.LPCWSTR),
        ("lpstrCustomFilter", wt.LPWSTR),
        ("nMaxCustFilter", wt.DWORD),
        ("nFilterIndex", wt.DWORD),
        ("lpstrFile", wt.LPWSTR),
        ("nMaxFile", wt.DWORD),
        ("lpstrFileTitle", wt.LPWSTR),
        ("nMaxFileTitle", wt.DWORD),
        ("lpstrInitialDir", wt.LPCWSTR),
        ("lpstrTitle", wt.LPCWSTR),
        ("Flags", wt.DWORD),
        ("nFileOffset", wt.WORD),
        ("nFileExtension", wt.WORD),
        ("lpstrDefExt", wt.LPCWSTR),
        ("lCustData", wt.LPARAM),
        ("lpfnHook", ctypes.c_void_p),
        ("lpTemplateName", wt.LPCWSTR),
        ("pvReserved", ctypes.c_void_p),
        ("dwReserved", wt.DWORD),
        ("FlagsEx", wt.DWORD),
    ]


# Flags
_OFN_ALLOWMULTISELECT = 0x00000200
_OFN_EXPLORER = 0x00080000
_OFN_FILEMUSTEXIST = 0x00001000
_OFN_PATHMUSTEXIST = 0x00000800
_OFN_NOCHANGEDIR = 0x00000008

_comdlg32 = ctypes.windll.comdlg32


def pick_files(
    title: str = "选择文件",
    filter_str: str = VIDEO_FILTER_NATIVE,
    initial_dir: str = "",
    parent_hwnd=None,
) -> List[str]:
    """打开文件选择对话框，返回选中文件路径列表。"""
    buf_size = 32768
    buf = ctypes.create_unicode_buffer(buf_size)

    ofn = _OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(_OPENFILENAMEW)
    ofn.hwndOwner = parent_hwnd
    ofn.lpstrFilter = filter_str
    ofn.nFilterIndex = 1
    ofn.lpstrFile = ctypes.cast(buf, wt.LPWSTR)
    ofn.nMaxFile = buf_size
    ofn.lpstrTitle = title
    ofn.Flags = (
        _OFN_ALLOWMULTISELECT | _OFN_EXPLORER |
        _OFN_FILEMUSTEXIST | _OFN_PATHMUSTEXIST | _OFN_NOCHANGEDIR
    )
    if initial_dir and os.path.isdir(initial_dir):
        ofn.lpstrInitialDir = initial_dir
    ofn.lpstrDefExt = "mp4"

    if not _comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
        return []

    # 解析多选结果：
    # 单选: "C:\\path\\file.mp4\0"
    # 多选: "C:\\dir\0file1.mp4\0file2.mp4\0\0"
    raw = buf[:]
    parts = raw.split("\0")
    while parts and parts[-1] == "":
        parts.pop()

    if not parts:
        return []

    if len(parts) == 1:
        return [parts[0]]

    # 多选：第一部分是目录，其余是文件名
    directory = parts[0]
    return [os.path.join(directory, f) for f in parts[1:]]


# ── 文件夹选择对话框 ──


class _BROWSEINFOW(ctypes.Structure):
    _fields_ = [
        ("hwndOwner", wt.HWND),
        ("pidlRoot", ctypes.c_void_p),
        ("pszDisplayName", wt.LPWSTR),
        ("lpszTitle", wt.LPCWSTR),
        ("ulFlags", ctypes.c_uint),
        ("lpfn", ctypes.c_void_p),
        ("lParam", wt.LPARAM),
        ("iImage", ctypes.c_int),
    ]


_BIF_RETURNONLYFSDIRS = 0x00000001
_BIF_NEWDIALOGSTYLE = 0x00000040

_shell32 = ctypes.windll.shell32
_ole32 = ctypes.windll.ole32

_shell32.SHBrowseForFolderW.restype = ctypes.c_void_p
_shell32.SHBrowseForFolderW.argtypes = [ctypes.POINTER(_BROWSEINFOW)]
_shell32.SHGetPathFromIDListW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
_shell32.SHGetPathFromIDListW.restype = wt.BOOL
_ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]


def pick_folder(
    title: str = "选择文件夹",
    parent_hwnd=None,
) -> str:
    """打开文件夹选择对话框，返回选中路径（取消返回空字符串）。"""
    # CoInitializeEx 明确指定 STA 模式；若线程已初始化 COM（RPC_E_CHANGED_MODE）则忽略
    hr = _ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED
    need_uninit = (hr == 0)  # S_OK 才需要 Uninit
    try:
        display_buf = ctypes.create_unicode_buffer(260)

        bi = _BROWSEINFOW()
        bi.hwndOwner = parent_hwnd
        bi.pidlRoot = None
        bi.pszDisplayName = ctypes.cast(display_buf, wt.LPWSTR)
        bi.lpszTitle = title
        bi.ulFlags = _BIF_RETURNONLYFSDIRS | _BIF_NEWDIALOGSTYLE
        bi.lpfn = None
        bi.lParam = 0

        pidl = _shell32.SHBrowseForFolderW(ctypes.byref(bi))
        if not pidl:
            return ""

        # 32768 支持长路径（本应用支持 \\?\ 前缀，260 MAX_PATH 不够）
        path_buf = ctypes.create_unicode_buffer(32768)
        result = _shell32.SHGetPathFromIDListW(pidl, path_buf)
        _ole32.CoTaskMemFree(pidl)

        if result:
            return path_buf.value
        return ""
    finally:
        if need_uninit:
            _ole32.CoUninitialize()
