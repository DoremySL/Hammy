"""GUI 托管的结构化配置；主配置唯一来源 _workspace/config.json（由 GUI 读写）。"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .workspace_paths import (
    CONFIG_FILE,
    LLAMA_CONFIG_FILE,
    LLAMA_MODEL_CONFIGS_FILE,
    TAGGER_CONFIG_FILE,
    TRANSCRIBE_CONFIG_FILE,
)
from .workspace_store import NO_WRITE, read_json_cached, update_json


# ── 默认配置 ──


def _default_config() -> Dict[str, Any]:
    return {
        "ai": {
            "model": "model",
            "base_url": "http://localhost:8080/v1",
            "api_key": "not-needed",
            "max_tokens": 3000,
            "temperature": 0.6,
            "top_p": 0.8,
            "retry_times": 2,
            "ai_timeout": 300,
            "ai_workers": 4,
            "enforce_json_mode": True,
        },
        "video": {
            "sampling_points": 7,
            "frames_per_point": 3,
            "frame_max_side": 672,
            "frame_time_tags": 2,
            "frame_extract_mode": "auto",
        },
        "naming": {
            "naming_mode": "date",
        },

        "active_prompt_id": "default",
        "theme": "",
        "nfo_auto_export": False,
        "animation_enabled": True,
        "experimental": {},
        "debug_log": False,
    }


# ── 读写 ──


def _fill_defaults(cfg: Dict[str, Any]) -> bool:

    changed = False
    defaults = _default_config()
    for sec, fields in defaults.items():
        if isinstance(fields, dict):
            if not isinstance(cfg.get(sec), dict):
                cfg[sec] = {}
                changed = True
            for k, v in fields.items():
                if k not in cfg[sec]:
                    cfg[sec][k] = v
                    changed = True
        else:
            if sec not in cfg:
                cfg[sec] = fields
                changed = True
    return changed


def load_config() -> Dict[str, Any]:
    """读取 config.json；不存在/损坏时原子写入内置默认配置（损坏先备份 .bak）。"""
    def _ensure(current):
        if not isinstance(current, dict):
            return _default_config()
        if _fill_defaults(current):
            return current
        return NO_WRITE

    return update_json(CONFIG_FILE, _ensure)


def save_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """保存 config.json（深度合并）。"""
    def _merge(current: Dict[str, Any]) -> Dict[str, Any]:
        for sec, fields in data.items():
            if isinstance(fields, dict):
                if not isinstance(current.get(sec), dict):
                    current[sec] = {}
                current[sec].update(fields)
            else:
                current[sec] = fields
        return current

    update_config(_merge)
    return {"ok": True}


def update_config(mutator: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
    """原子读-改-写 config.json（mutator 可原地修改后返回 None）。"""
    def _wrapped(current):
        if not isinstance(current, dict):
            current = _default_config()
        _fill_defaults(current)
        result = mutator(current)
        return current if result is None else result

    return update_json(CONFIG_FILE, _wrapped)


# ── 试验模块配置（独立文件 _workspace/<模块>/config.json） ──


def _load_module_config(path, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """读取模块配置并合并默认值（仅保留 schema 内的键）。"""
    data = read_json_cached(path)
    merged = dict(defaults)
    merged.update({k: v for k, v in data.items() if k in merged})
    return merged


def _update_module_config(path, defaults: Dict[str, Any],
                          mutator: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """原子读-改-写模块配置（mutator 就地修改后可返回 None）。"""
    def _wrapped(current):
        current = current if isinstance(current, dict) else {}
        merged = dict(defaults)
        merged.update(current)
        result = mutator(merged)
        return merged if result is None else result

    return update_json(path, _wrapped, default_factory=dict)


def _tagger_defaults() -> Dict[str, Any]:
    """tagger 模块配置默认值（文件缺失/损坏时的兜底）。"""
    return {"enabled": False, "integration": False, "threshold": 0.75,
            "general_count": 15, "precision": ""}


def _transcribe_defaults() -> Dict[str, Any]:
    """语音转写模块配置默认值。"""
    return {"enabled": False, "model": "whisper-turbo", "inject_model": "sensevoice",
            "workers": 2, "max_chars": 2000, "inject_timestamps": False,
            "integration": True}


def load_tagger_config() -> Dict[str, Any]:
    """读取 Hammy-tagger 模块配置（缺失/损坏回退默认值）。"""
    return _load_module_config(TAGGER_CONFIG_FILE, _tagger_defaults())


def update_tagger_config(
    mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """原子读-改-写 tagger 模块配置。"""
    return _update_module_config(TAGGER_CONFIG_FILE, _tagger_defaults(), mutator)


def load_transcribe_config() -> Dict[str, Any]:
    """读取语音转写模块配置。"""
    return _load_module_config(TRANSCRIBE_CONFIG_FILE, _transcribe_defaults())


def update_transcribe_config(
    mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """原子读-改-写语音转写模块配置。"""
    return _update_module_config(TRANSCRIBE_CONFIG_FILE, _transcribe_defaults(), mutator)


def load_llama_config() -> Dict[str, Any]:
    """读取 llama 全局运行参数与偏好（mtime 缓存，只读快照，写走 update_*）。"""
    return read_json_cached(LLAMA_CONFIG_FILE)


def update_llama_config(
    mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """原子读-改-写 llama 全局配置（与 config.json 同一 RLock 互斥；mutator 可原地修改后返回 None）。"""
    def _wrapped(current):
        result = mutator(current)
        return current if result is None else result

    return update_json(LLAMA_CONFIG_FILE, _wrapped, default_factory=dict)


def load_llama_model_configs() -> Dict[str, Any]:
    """读取每个模型一套运行参数：{模型路径: {ctx, ngl, ...}}（mtime 缓存，只读快照）。"""
    return read_json_cached(LLAMA_MODEL_CONFIGS_FILE)


def update_llama_model_configs(
    mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """原子读-改-写 per-model 配置（与 config.json 同一 RLock 互斥；mutator 可原地修改后返回 None）。"""
    def _wrapped(current):
        result = mutator(current)
        return current if result is None else result

    return update_json(LLAMA_MODEL_CONFIGS_FILE, _wrapped, default_factory=dict)
