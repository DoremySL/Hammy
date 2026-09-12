"""ConfigPresetMixin — 配置读写、提示词预设 CRUD、标签检索管理/导入导出。"""
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

    def set_prompt_use_example(self, enabled: bool) -> Dict[str, Any]:
        """设置拼接提示词时是否附带示例段（默认关闭；模板中保存的示例内容不受影响）。"""
        config_store.update_config(
            lambda cfg: cfg.update(prompt_use_example=bool(enabled)) or cfg)
        return {"ok": True, "use_example": bool(enabled)}

    def set_prompt_thumb_optimize(self, enabled: bool) -> Dict[str, Any]:
        """设置是否启用缩略图优化（注入 thumb_time 字段并据其重生成封面）。"""
        config_store.update_config(
            lambda cfg: cfg.setdefault("video", {}).update(
                frame_time_tags=2 if enabled else 0) or cfg)
        return {"ok": True, "thumb_optimize": bool(enabled)}

    def preview_prompt(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        # 预览跟随当前设置：时间标签=添加并用于优化缩略图时，展示注入的 thumb_time 字段；
        # 示例段是否拼接跟随「拼接示例」开关
        cfg = config_store.load_config()
        video = cfg.get("video") or {}
        with_thumb = int(video.get("frame_time_tags") or 0) == 2
        use_example = bool(cfg.get("prompt_use_example", False))
        return {"prompt": prompts.preview_prompt(
            fields, with_thumb_time=with_thumb, use_example=use_example)}

    # ── 标签检索 ──

    def get_priority_tags(self) -> Dict[str, Any]:
        return prompts.load_priority_tags()

    def save_priority_tags(self, items: Any, mode: Any,
                           disabled_groups: Any = None) -> Dict[str, Any]:
        return prompts.save_priority_tags(items, mode, disabled_groups)

    def import_priority_tags(self) -> Dict[str, Any]:
        """弹出文件选择对话框导入标签检索 JSON（不落地，返回给前端载入页面）。"""
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
        # 兼容 {items} 与纯数组两种根结构（旧文件的 mode/disabled_groups 忽略）
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("items", [])
        else:
            return {"ok": False, "error": "JSON 结构不符合预期"}
        items = prompts.normalize_priority_items(items)
        if not items:
            return {"ok": False, "error": "文件中没有有效标签"}
        self._remember_dir([path])
        return {"ok": True, "items": items}

    def export_priority_tags(self, items: Any) -> Dict[str, Any]:
        """弹出保存对话框，把当前标签检索导出为 JSON 文件。"""
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
        data = prompts.normalize_priority_items(items)
        try:
            Path(path).write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            return {"ok": False, "error": f"写入失败: {e}"}
        self._remember_dir([path])
        return {"ok": True, "path": path}
