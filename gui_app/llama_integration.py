"""本地推理集成（可插拔）：开启后用本地 llama-server 的 OpenAI 兼容接口覆盖远程 API 参数，
只提供运行时覆盖值，磁盘上用户参数不动；需同时满足 llama 启用 + 集成开关 + 已安装才生效。"""

from typing import Any, Dict, Optional

DEFAULT_MODEL = "model"        # 与 config_store ai 段默认一致（llama-server 接受任意 model 名）
DEFAULT_API_KEY = "not-needed"  # 与 config_store ai 段默认一致（本地服务不校验密钥）


def is_active(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """集成是否生效：llama 模块启用 + 集成开关开启 + llama.cpp 已安装。"""
    if cfg is None:
        from .config_store import load_config
        cfg = load_config()
    exp = cfg.get("experimental")
    exp = exp if isinstance(exp, dict) else {}
    if not exp.get("llama_enabled", False):
        return False
    if not exp.get("llama_integrate", False):
        return False
    # 未安装（llama-server.exe 缺失）时自动回退原逻辑，保证可插拔
    try:
        from .llama_cpp import LLAMA_DIR, EXE_NAME
        from pathlib import Path
        return (Path(LLAMA_DIR) / EXE_NAME).is_file()
    except Exception:
        return False


def ai_override(cfg: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """集成激活时返回 {model, base_url, api_key, ai_workers} 覆盖值；未激活返回 None。

    ai_workers 由本地推理的「并发线程 -np」（parallel）接管，请求并发不超过
    llama-server 的并行槽数。
    """
    if not is_active(cfg):
        return None
    from .config_store import load_llama_config, load_llama_model_configs
    llama = load_llama_config() or {}
    model_cfg = load_llama_model_configs().get(str(llama.get("last_model") or ""), {}) or {}
    host = model_cfg.get("host") or llama.get("host") or "127.0.0.1"
    port = model_cfg.get("port") or llama.get("port") or 8080
    try:
        workers = max(1, int(llama.get("parallel") or 1))
    except (TypeError, ValueError):
        workers = 1
    return {
        "model": DEFAULT_MODEL,
        "api_key": DEFAULT_API_KEY,
        "base_url": f"http://{host}:{port}/v1",
        "ai_workers": workers,
    }
