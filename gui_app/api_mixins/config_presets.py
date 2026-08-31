"""ConfigPresetMixin — 配置读写、提示词预设 CRUD、优先标签管理/导入导出。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .. import config_store, prompts


class ConfigPresetMixin:
    """配置与提示词预设相关 API。"""

    # ── 配置 ──

    def get_config(self) -> Dict[str, Any]:
        return config_store.load_config()

    def save_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return config_store.save_config(data)

    # ── 提示词预设 ──

    def list_presets(self) -> List[Dict[str, Any]]:
        return prompts.list_presets()

    def get_active_preset(self) -> Dict[str, Any]:
        return prompts.get_active()

    def get_preset(self, pid: str) -> Dict[str, Any]:
        p = prompts.get_preset(pid)
        return p if p else {"error": "预设不存在"}

    def save_preset(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        return prompts.save_preset(preset)

    def delete_preset(self, pid: str) -> Dict[str, Any]:
        return prompts.delete_preset(pid)

    def set_active_preset(self, pid: str) -> Dict[str, Any]:
        return prompts.set_active(pid)

    def preview_prompt(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        return {"prompt": prompts.preview_prompt(fields)}

    # ── 优先标签 ──

    def get_priority_tags(self) -> Dict[str, Any]:
        return prompts.load_priority_tags()

    def save_priority_tags(self, enabled: Any, items: Any) -> Dict[str, Any]:
        return prompts.save_priority_tags(enabled, items)

    def preview_priority_tags(self, enabled: Any, items: Any) -> Dict[str, Any]:
        return {"section": prompts.build_priority_tags_section(enabled, items)}

    def import_priority_tags(self) -> Dict[str, Any]:
        """弹出文件选择对话框导入优先标签 JSON（不落地，返回给前端载入页面）。"""
        import webview
        from ..mainthread import run_on_ui_thread

        dlg = getattr(webview, "FileDialog", None)
        open_type = dlg.OPEN if dlg else webview.OPEN_DIALOG

        def _do():
            return self._window.create_file_dialog(
                open_type,
                directory=self._initial_dir(),
                allow_multiple=False,
                file_types=("JSON 文件 (*.json)",),
            )

        try:
            result = run_on_ui_thread(_do)
        except Exception as e:
            return {"ok": False, "error": f"对话框失败: {e}"}
        if not result:
            return {"ok": False, "cancelled": True}
        path = result[0] if isinstance(result, (list, tuple)) else result
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            return {"ok": False, "error": f"读取/解析失败: {e}"}
        # 兼容 {enabled, items} 与纯数组两种根结构
        if isinstance(data, list):
            enabled, items = True, data
        elif isinstance(data, dict):
            enabled, items = bool(data.get("enabled", True)), data.get("items", [])
        else:
            return {"ok": False, "error": "JSON 结构不符合预期"}
        items = prompts.normalize_priority_items(items)
        if not items:
            return {"ok": False, "error": "文件中没有有效标签"}
        self._remember_dir([path])
        return {"ok": True, "enabled": enabled, "items": items}

    def export_priority_tags(self, enabled: Any, items: Any) -> Dict[str, Any]:
        """弹出保存对话框，把当前优先标签导出为 JSON 文件。"""
        import webview
        from ..mainthread import run_on_ui_thread

        dlg = getattr(webview, "FileDialog", None)
        save_type = dlg.SAVE if dlg else webview.SAVE_DIALOG

        def _do():
            return self._window.create_file_dialog(
                save_type,
                directory=self._initial_dir(),
                save_filename="priority_tags.json",
                file_types=("JSON 文件 (*.json)",),
            )

        try:
            result = run_on_ui_thread(_do)
        except Exception as e:
            return {"ok": False, "error": f"对话框失败: {e}"}
        if not result:
            return {"ok": False, "cancelled": True}
        path = result[0] if isinstance(result, (list, tuple)) else result
        data = {
            "enabled": bool(enabled),
            "items": prompts.normalize_priority_items(items),
        }
        try:
            Path(path).write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            return {"ok": False, "error": f"写入失败: {e}"}
        self._remember_dir([path])
        return {"ok": True, "path": path}
