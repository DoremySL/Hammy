"""SystemMixin — 文件/文件夹选择对话框、资源管理器联动、workspace 维护。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..workspace_paths import NFO_DIR, THUMB_DIR
from ..workspace_store import load_history
from ..workspace_service import clear_workspace_cache, prune_missing_history, reconcile_history
from ..env import APP_ROOT, VIDEO_FILTER_WINFORMS


class SystemMixin:
    """系统工具与 workspace 维护相关 API。"""

    # ── 对话框目录记忆 ──

    def _initial_dir(self) -> str:
        """返回有效的对话框初始目录。"""
        last = getattr(self, "_last_dir", "") or ""
        if last:
            try:
                if Path(last).is_dir():
                    return last
            except Exception:
                pass
        # 兜底用程序所在目录（避免打开用户主目录/盘符根，误选会导致全盘扫描）
        return str(APP_ROOT)

    def _remember_dir(self, paths: List[str], is_folder: bool = False) -> None:
        """记住所选路径所在目录，作为下次对话框的初始位置。"""
        try:
            if not paths:
                return
            p = Path(paths[0])
            target = p if (is_folder and p.is_dir()) else p.parent
            if target.is_dir():
                self._last_dir = str(target)
        except Exception:
            pass

    # ── 文件选择对话框 ──

    def _pick_dialog(self, dialog_type, native_fallback_fn, is_folder: bool = False, **dialog_kwargs) -> list:
        """通用文件/文件夹选择对话框。WinForms 优先，异常时 fallback 到原生对话框。"""
        from ..mainthread import run_on_ui_thread
        init_dir = self._initial_dir()

        def _do():
            return self._window.create_file_dialog(
                dialog_type, directory=init_dir, allow_multiple=True, **dialog_kwargs)

        try:
            result = run_on_ui_thread(_do)
        except Exception as e:
            import sys
            sys.stderr.write(f"[api] dialog WinForms failed: {e}\n")
            return native_fallback_fn()
        if not result:
            return []
        if isinstance(result, str):
            result = [result]
        else:
            result = list(result)
        self._remember_dir(result, is_folder=is_folder)
        return result

    def pick_files(self) -> List[str]:
        """打开文件选择对话框。优先 WinForms 封送；仅异常时才用原生对话框兜底。"""
        import webview

        # pywebview 6.x 新枚举，旧常量已弃用
        dlg = getattr(webview, "FileDialog", None)
        open_type = dlg.OPEN if dlg else webview.OPEN_DIALOG
        return self._pick_dialog(
            open_type,
            self._pick_files_native,
            file_types=(VIDEO_FILTER_WINFORMS,),
        )

    def _pick_files_native(self) -> List[str]:
        """Win32 原生文件选择后备方案（ctypes，零依赖）。"""
        try:
            from ..native_dialogs import pick_files as native_pick
            return native_pick(title="选择视频文件")
        except Exception as e:
            sys.stderr.write(f"[api] pick_files native failed: {e}\n")
            return []

    def pick_folders(self) -> List[str]:
        """打开文件夹选择对话框。优先 WinForms 封送；仅异常时才用原生对话框兜底。"""
        import webview

        # pywebview 6.x 新枚举（FOLDER_DIALOG 已弃用）
        dlg = getattr(webview, "FileDialog", None)
        folder_type = dlg.FOLDER if dlg else webview.FOLDER_DIALOG
        return self._pick_dialog(folder_type, self._pick_folders_native, is_folder=True)

    def _pick_folders_native(self) -> List[str]:
        """Win32 原生文件夹选择后备方案（ctypes，零依赖）。"""
        try:
            from ..native_dialogs import pick_folder as native_pick
            folder = native_pick(title="选择文件夹")
            return [folder] if folder else []
        except Exception as e:
            sys.stderr.write(f"[api] pick_folders native failed: {e}\n")
            return []

    # ── 系统工具 ──

    def open_in_explorer(self, path: str) -> Dict[str, Any]:
        """在资源管理器中打开文件位置并选中。"""
        import subprocess
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            subprocess.Popen(["explorer", "/select,", path], **kw)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def open_bing_search(self, query: str) -> Dict[str, Any]:
        """用系统默认浏览器打开 Bing 搜索页（待处理详情页 IP 标签点击查询用）。"""
        import webbrowser
        from urllib.parse import quote
        try:
            webbrowser.open(f"https://bing.com/search?q={quote(query)}")
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": f"打开浏览器失败: {e}"}

    # ── 原生标题栏配色（随应用主题）──

    def set_titlebar_dark(self, dark: bool) -> Dict[str, Any]:
        """同步原生标题栏明暗，与应用内部主题一致，避免割裂感。"""
        if sys.platform != "win32":
            return {"ok": True, "skipped": True}
        try:
            import ctypes
            from ctypes import wintypes
            window = self._window
            # pywebview winforms 后端：window.native 即 BrowserForm，.Handle 为 HWND
            form = getattr(window, "native", None) if window else None
            if form is None:
                return {"ok": False, "error": "window native form unavailable"}
            hwnd = int(form.Handle.ToInt64())
            value = 1 if dark else 0
            DwmSetWindowAttribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
            DwmSetWindowAttribute.argtypes = [
                wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
            ]
            DwmSetWindowAttribute.restype = ctypes.c_long  # HRESULT
            hr = DwmSetWindowAttribute(
                hwnd, 20, ctypes.byref(ctypes.c_int(value)), 4)
            # 属性 20 在 Win10 < 1809 不存在（返回失败码），回退到旧的 19
            if hr != 0:  # S_OK == 0
                DwmSetWindowAttribute(
                    hwnd, 19, ctypes.byref(ctypes.c_int(value)), 4)
            return {"ok": True}
        except Exception as e:
            # 标题栏配色是锦上添花，任何异常都不应阻断主题切换
            return {"ok": False, "error": str(e)}

    # ── workspace 维护 ──

    def get_workspace_stats(self) -> Dict[str, Any]:
        """返回 workspace 统计：history 条数、缓存大小。"""
        h = load_history()
        entries = h.get("entries", [])

        def _dir_stats(d) -> Tuple[int, int]:
            """单次遍历目录，返回 (文件数, 总字节数)。"""
            count = 0
            size = 0
            try:
                for f in d.iterdir():
                    try:
                        if f.is_file():
                            count += 1
                            size += f.stat().st_size
                    except OSError:
                        pass
            except OSError:
                pass
            return count, size

        thumb_count, thumb_size = _dir_stats(THUMB_DIR)
        nfo_count, nfo_size = _dir_stats(NFO_DIR)
        return {
            "history_count": len(entries),
            "thumb_count": thumb_count,
            "nfo_count": nfo_count,
            "thumb_size_mb": round(thumb_size / (1024 * 1024), 2),
            "nfo_size_mb": round(nfo_size / (1024 * 1024), 2),
            "reconcile": reconcile_history(),
        }

    def prune_history(self) -> Dict[str, Any]:
        """清理 history 中磁盘上已不存在的记录，同时清除对应的缩略图/NFO 缓存。"""
        result = prune_missing_history()
        return {"ok": True, **result}

    def clear_workspace(self, clear_history: bool = True,
                        clear_thumbs: bool = True,
                        clear_nfo: bool = True,
                        clear_manifest: bool = False) -> Dict[str, Any]:
        """清除全部缓存（history + thumbnails + nfo），可选清源清单。默认不动 manifest 与视频本体。"""
        return clear_workspace_cache(
            clear_history=clear_history,
            clear_thumbs=clear_thumbs,
            clear_nfo=clear_nfo,
            clear_manifest=clear_manifest,
        )
