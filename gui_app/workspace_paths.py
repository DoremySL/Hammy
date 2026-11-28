"""工作区路径常量与稳定 ID。"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

# ── 应用根目录（gui_app 的上一级）──
APP_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = APP_ROOT / "_workspace"
THUMB_DIR = WORKSPACE_DIR / "thumbnails"
NFO_DIR = WORKSPACE_DIR / "nfo"
MANIFEST_FILE = WORKSPACE_DIR / "manifest.json"
HISTORY_FILE = WORKSPACE_DIR / "history.json"
PROMPTS_FILE = WORKSPACE_DIR / "prompts.json"
CONFIG_FILE = WORKSPACE_DIR / "config.json"
PROBE_CACHE_FILE = WORKSPACE_DIR / "probe_cache.json"
SIMILAR_CACHE_FILE = WORKSPACE_DIR / "similar_cache.json"
# ── 扩展模块数据目录（卸载时整目录删除）──
# llama.cpp
LLAMA_CONFIG_FILE = WORKSPACE_DIR / "llama" / "config.json"
LLAMA_MODEL_CONFIGS_FILE = WORKSPACE_DIR / "llama" / "model_configs.json"
LLAMA_SCAN_CACHE_FILE = WORKSPACE_DIR / "llama" / "scan_cache.json"
# Hammy-tagger
TAGGER_CONFIG_FILE = WORKSPACE_DIR / "tagger" / "config.json"
TAGGER_TAGS_FILE = WORKSPACE_DIR / "tagger" / "tags.json"
# 语音转写
TRANSCRIBE_CONFIG_FILE = WORKSPACE_DIR / "transcribe" / "config.json"
TRANSCRIBE_TRANSCRIPTS_FILE = WORKSPACE_DIR / "transcribe" / "transcripts.json"
TRANSCRIBE_SRT_DIR = WORKSPACE_DIR / "transcribe" / "srt"
TERM_LIBRARY_FILE = WORKSPACE_DIR / "term_library.json"


def stable_id(path: str) -> str:
    """对路径做稳定哈希，作为缩略图/NFO 的文件名。改名后仍可命中。"""
    norm = os.path.normcase(os.path.normpath(path))
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:16]


def thumb_path(vid: str) -> Path:
    return THUMB_DIR / f"{vid}.jpg"


def nfo_path(vid: str) -> Path:
    return NFO_DIR / f"{vid}.nfo"
