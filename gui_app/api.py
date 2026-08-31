"""api.py — JS <-> Python 桥接层（pywebview）。"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from . import runner
from .api_mixins import (
    ConfigPresetMixin,
    ExperimentalMixin,
    MediaMixin,
    ModelDownloadMixin,
    ProcessingMixin,
    SourcesMixin,
    SystemMixin,
)
from .env import check_all_startup_deps
from .js_push import js_pusher


class Api(
    SourcesMixin,
    ProcessingMixin,
    MediaMixin,
    ConfigPresetMixin,
    SystemMixin,
    ExperimentalMixin,
    ModelDownloadMixin,
):
    """暴露给前端 JS 的 API 对象（Mixin 组合）。"""

    def __init__(self):
        self._window = None
        self._runner: Optional[runner.PipelineRunner] = None
        self._progress_re = re.compile(r"^\s*\[(\d+)/(\d+)\]")
        self._last_dir: str = ""

    def set_window(self, window) -> None:
        self._window = window
        js_pusher.set_window(window)

    # ── 启动期检测 ──

    def check_startup(self) -> Dict[str, Any]:
        """启动期依赖检测：返回所有依赖状态（与 app.py 启动期同口径）。"""
        return check_all_startup_deps()
