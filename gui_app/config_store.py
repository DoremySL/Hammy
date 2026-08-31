"""GUI 托管的结构化配置；主配置唯一来源 _workspace/config.json（由 GUI 读写）。"""
from __future__ import annotations

from typing import Any, Callable, Dict

from .workspace_paths import (
    CONFIG_FILE,
    LLAMA_CONFIG_FILE,
    LLAMA_MODEL_CONFIGS_FILE,
    PIXAI_CONFIG_FILE,
    WHISPER_CONFIG_FILE,
)
from .workspace_store import NO_WRITE, read_json, read_json_cached, update_json


# ── 默认配置 ──


def _default_config() -> Dict[str, Any]:
    return {
        "ai": {
            "model": "model",
            "base_url": "http://localhost:8080/v1",
            "api_key": "not-needed",
            "max_tokens": 500,
            "temperature": 0.6,
            "top_p": 0.8,
            "retry_times": 2,
            "ai_timeout": 300,
            "ai_workers": 4,
            "enforce_json_mode": True,
        },
        "video": {
            "sampling_points": 5,
            "frames_per_point": 3,
            "frame_max_side": 640,
        },
        "naming": {
            "include_date": True,
            "include_original": False,
        },

        "active_prompt_id": "default",
        "theme": "",
        "nfo_auto_export": False,
        "force_animation": True,
        "experimental": {},
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

    cfg = update_json(CONFIG_FILE, _ensure)
    # 组装 experimental 段（模块配置合成，仅内存不落盘）
    _assemble_experimental(cfg)
    return cfg


def save_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """保存 config.json（深度合并）。"""
    exp = data.get("experimental")
    if isinstance(exp, dict):
        pixai = {_PIXAI_KEY_MAP_REV[k]: v for k, v in exp.items() if k in _PIXAI_KEY_MAP_REV}
        whisper = {_WHISPER_KEY_MAP_REV[k]: v for k, v in exp.items() if k in _WHISPER_KEY_MAP_REV}
        if pixai:
            update_pixai_config(lambda c: c.update(pixai) or c)
        if whisper:
            update_whisper_config(lambda c: c.update(whisper) or c)
        # 剩余非模块键（如有）仍走 config.json
        rest = {k: v for k, v in exp.items()
                if k not in _PIXAI_KEY_MAP_REV and k not in _WHISPER_KEY_MAP_REV
                and k not in _LLAMA_KEYS}
        if rest:
            data = {**data, "experimental": rest}
        else:
            data = {k: v for k, v in data.items() if k != "experimental"}

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
    """原子读-改-写 config.json。"""
    def _wrapped(current):
        if not isinstance(current, dict):
            current = _default_config()
        _fill_defaults(current)
        return mutator(current)

    return update_json(CONFIG_FILE, _wrapped)


# ── 试验模块配置（独立文件 _workspace/<模块>/config.json） ──

# 模块文件内键名 → experimental 段键名（load_config 组装用）
_PIXAI_KEY_MAP = {
    "enabled": "pixai_tagger_enabled",
    "classify": "pixai_classify",
    "frames": "pixai_frames",
    "short_side": "pixai_short_side",
    "crop_square": "pixai_crop_square",
    "crop_portrait": "pixai_crop_portrait",
    "threshold": "pixai_threshold",
}
_WHISPER_KEY_MAP = {
    "enabled": "whisper_enabled",
    "model": "whisper_model",
    "vad": "whisper_vad",
    "language": "whisper_language",
    "max_chars": "whisper_max_chars",
    "inject_timestamps": "whisper_inject_timestamps",
    "batch": "whisper_batch",
    "workers": "whisper_workers",
}
# 反向映射：experimental 段键名 → 模块文件键名（save_config 拆键路由用）
_PIXAI_KEY_MAP_REV = {v: k for k, v in _PIXAI_KEY_MAP.items()}
_WHISPER_KEY_MAP_REV = {v: k for k, v in _WHISPER_KEY_MAP.items()}
# llama 开关在 experimental 段的键名：不经 save_config，只走
# set_llama_enabled / set_llama_config，提交时直接丢弃
_LLAMA_KEYS = frozenset({"llama_enabled", "llama_integrate"})


def _pixai_defaults() -> Dict[str, Any]:
    """pixai 模块配置默认值（文件缺失/损坏时的兜底）。"""
    return {"enabled": False, "classify": True, "frames": 15, "short_side": 448,
            "crop_square": False, "crop_portrait": False, "threshold": 0.9}


def _whisper_defaults() -> Dict[str, Any]:
    """whisper 模块配置默认值（文件缺失/损坏时的兜底）。"""
    return {"enabled": False, "model": "v3-turbo", "vad": True, "language": "",
            "max_chars": 800, "inject_timestamps": False, "batch": False,
            "workers": 4}


def load_pixai_config() -> Dict[str, Any]:
    """读取 pixai-tagger 模块配置。文件不存在/损坏时返回默认值（enabled=False）。"""
    data = read_json(PIXAI_CONFIG_FILE, {})
    merged = dict(_pixai_defaults())
    if isinstance(data, dict):
        merged.update(data)
    return merged


def update_pixai_config(
    mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """原子读-改-写 pixai 模块配置（与 config.json 同一 RLock 互斥）。

    mutator 接收已补全默认值的配置 dict，就地修改后返回。
    """
    def _wrapped(current):
        current = current if isinstance(current, dict) else {}
        merged = dict(_pixai_defaults())
        merged.update(current)
        return mutator(merged)

    return update_json(PIXAI_CONFIG_FILE, _wrapped, default_factory=dict)


def load_whisper_config() -> Dict[str, Any]:
    """读取 faster-whisper 模块配置。文件不存在/损坏时返回默认值（enabled=False）。"""
    data = read_json(WHISPER_CONFIG_FILE, {})
    merged = dict(_whisper_defaults())
    if isinstance(data, dict):
        merged.update(data)
    return merged


def update_whisper_config(
    mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """原子读-改-写 whisper 模块配置。"""
    def _wrapped(current):
        current = current if isinstance(current, dict) else {}
        merged = dict(_whisper_defaults())
        merged.update(current)
        return mutator(merged)

    return update_json(WHISPER_CONFIG_FILE, _wrapped, default_factory=dict)


def load_llama_config() -> Dict[str, Any]:
    """读取 llama 全局运行参数与偏好（mtime 缓存，只读快照，写走 update_*）。"""
    return read_json_cached(LLAMA_CONFIG_FILE)


def update_llama_config(
    mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """原子读-改-写 llama 全局配置（与 config.json 同一 RLock 互斥）。"""
    return update_json(LLAMA_CONFIG_FILE, mutator, default_factory=dict)


def load_llama_model_configs() -> Dict[str, Any]:
    """读取每个模型一套运行参数：{模型路径: {ctx, ngl, ...}}（mtime 缓存，只读快照）。"""
    return read_json_cached(LLAMA_MODEL_CONFIGS_FILE)


def update_llama_model_configs(
    mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """原子读-改-写 per-model 配置（与 config.json 同一 RLock 互斥）。"""
    return update_json(
        LLAMA_MODEL_CONFIGS_FILE, mutator, default_factory=dict
    )


def _assemble_experimental(cfg: Dict[str, Any]) -> None:
    """把三个模块文件夹的配置合成 experimental 段（前端读取结构不变）。"""
    exp = cfg.setdefault("experimental", {})
    for k, v in load_pixai_config().items():
        exp[_PIXAI_KEY_MAP.get(k, k)] = v
    for k, v in load_whisper_config().items():
        exp[_WHISPER_KEY_MAP.get(k, k)] = v
    llama = load_llama_config()
    exp["llama_enabled"] = bool(llama.get("enabled", False))
    exp["llama_integrate"] = bool(llama.get("integrate", False))
