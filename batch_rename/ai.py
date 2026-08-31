"""AI 分析与 JSON 提取。"""
import json
import random
import re
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, NamedTuple

from .env import logger
from .dependencies import bindings, DependencyError
from .config import Config, OpenAIClient
from .types import Frame


# ── AI 上下文溢出错误检测关键词 ──
_CONTEXT_ERROR_KEYWORDS = (
    "context_length_exceeded", "context length", "max_length",
    "maximum context length", "maximum token", "message too long",
    "reduce the length", "too many tokens",
)


def is_context_error(err_msg: str) -> bool:
    """判断错误信息是否为上下文窗口溢出。"""
    lower = err_msg.lower()
    return any(k in lower for k in _CONTEXT_ERROR_KEYWORDS)


# ── 可立即重试的 AI 解析异常 ──
class AIFormatError(ValueError):
    """AI 输出 JSON 格式错误（可立即重试，无需退避）。"""


class AITitleEmptyError(ValueError):
    """AI 返回的 title 为空（可立即重试，无需退避）。"""


def _strip_trailing_commas(s: str) -> str:
    """删除紧邻 } 或 ] 的尾随逗号。"""
    out: List[str] = []
    in_string = False
    escape_next = False
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if escape_next:
            escape_next = False
            out.append(ch)
            i += 1
            continue
        if in_string:
            if ch == '\\':
                escape_next = True
            elif ch == '"':
                in_string = False
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == ',':
            j = i + 1
            while j < n and s[j] in ' \t\r\n':
                j += 1
            if j < n and s[j] in '}]':
                i += 1
                continue
        out.append(ch)
        i += 1
    return ''.join(out)


def _extract_first_json(text: str, require_key: Optional[str] = "title") -> Optional[Dict[str, Any]]:
    """提取首个完整合法的 JSON 对象。"""
    search_from = 0
    text_len = len(text)
    while search_from < text_len:
        start = text.find('{', search_from)
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, text_len):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if in_string and ch == '\\':
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    if require_key is None:
                        try:
                            return json.loads(candidate, strict=False)
                        except json.JSONDecodeError:
                            search_from = i + 1
                            break
                    try:
                        obj = json.loads(candidate, strict=False)
                    except json.JSONDecodeError:
                        try:
                            obj = json.loads(_strip_trailing_commas(candidate), strict=False)
                        except json.JSONDecodeError:
                            search_from = i + 1
                            break
                    if isinstance(obj, dict) and any(k.lower() == require_key for k in obj):
                        return obj
                    search_from = i + 1
                    break
        else:
            # 该 { 无配对，从下一个 { 继续
            search_from = start + 1
            continue
    return None


class AnalyzeResult(NamedTuple):
    """analyze_frames 的返回值。"""
    title: str
    plot: str
    tags: List[str]
    retries: int
    err_msg: str
    # error_kind: "" 成功 / format / empty / api / server / retryable / cancel / other
    error_kind: str = ""


def _fmt_hms(seconds: float) -> str:
    """秒 → HH:MM:SS（四舍五入到秒，小时位补零）。"""
    total = max(0, int(seconds + 0.5))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_messages(frames: List[Frame], config: Config, video_path: str,
                    duration: float, container_title: str = "",
                    extra_meta: str = "") -> list:
    """构建 API 请求消息（system + 关键帧图片 + 辅助信息 + prompt）。"""
    vid_name = Path(video_path).name
    dur_str = _fmt_hms(duration) if duration > 0 else "未知"
    meta_parts = [f"- 原始文件名: {vid_name}", f"- 视频总时长: {dur_str}"]
    if container_title:
        meta_parts.append(f"- 容器标题: {container_title}（可能含有准确且重要的描述，可优先参考）")
    meta_info = "\n\n[辅助参考信息]\n" + "\n".join(meta_parts)
    if extra_meta:
        meta_info += "\n\n" + extra_meta

    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})

    user_content = []
    for frame in frames:
        if config.frame_time_tags:
            user_content.append({"type": "text", "text": _fmt_hms(frame.ts)})
        user_content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{frame.b64}"}})
    user_content.append({"type": "text", "text": meta_info})
    user_content.append({"type": "text", "text": config.prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def _parse_ai_response(raw: str) -> Dict[str, Any]:
    """从响应提取并解析 JSON。"""
    raw = raw.lstrip("\ufeff")
    try:
        data = json.loads(raw, strict=False)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        return data
    data = _extract_first_json(raw, require_key="title")
    if data is None:
        logger.debug(f"原始内容: {raw[:200]}...")
        raise AIFormatError("AI 输出格式错误 (未检测到包含 title 的完整JSON结构)")
    return data


def _get_ci(data: Dict[str, Any], key: str) -> Any:
    """大小写不敏感取键。"""
    key = key.lower()
    for k, v in data.items():
        if k.lower() == key:
            return v
    return None


def _clean_tags(tags: Any) -> List[str]:
    """去重标签列表（字符串输入切分兜底）。"""
    if isinstance(tags, str):
        tags = re.split(r"[,、，]", tags)
    elif not isinstance(tags, list):
        return []
    seen: set = set()
    unique: list = []
    for t in tags:
        if not t:
            continue
        s = str(t).strip()
        s_lower = s.lower()
        if s_lower not in seen:
            seen.add(s_lower)
            unique.append(s)
    return unique


def _call_and_parse(client: OpenAIClient, model: str, messages: list,
                    config: Config, stop_event: threading.Event) -> AnalyzeResult:
    """单次调用 AI 并解析响应。"""
    kwargs = {
        "model": model,
        "messages": messages,
        "timeout": config.ai_timeout,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    if config.enforce_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    box: dict = {}

    def _do_call():
        try:
            box["resp"] = client.chat.completions.create(**kwargs)
        except Exception as e:
            box["err"] = e

    t = threading.Thread(target=_do_call, daemon=True)
    t.start()
    while t.is_alive():
        if stop_event.wait(0.2):
            return AnalyzeResult("", "", [], 0, "已取消", "cancel")
        t.join(timeout=0.2)
    if "err" in box:
        raise box["err"]
    resp = box.get("resp")

    if not (resp.choices and resp.choices[0].message):
        raise ValueError("AI 返回空内容")
    raw = (resp.choices[0].message.content or "").strip()
    data = _parse_ai_response(raw)
    title = str(data.get("title") or _get_ci(data, "title") or "").strip('"\'').strip()
    plot = str(data.get("plot") or _get_ci(data, "plot") or "").strip()
    tags = _clean_tags(data.get("tags") or _get_ci(data, "tags") or [])
    if not title:
        raise AITitleEmptyError("AI 返回的 title 为空")
    return AnalyzeResult(title, plot, tags, 0, "")


def _api_status_cls():
    """返回 bindings.APIStatusError 类，依赖未加载时抛 DependencyError。"""
    cls = bindings.APIStatusError
    if cls is None:
        raise DependencyError("openai 依赖未加载，请先调用 ensure_dependencies()")
    return cls


def _error_kind_of(e: Exception) -> str:
    """对异常进行错误分类，返回 error_kind 字符串。"""
    if isinstance(e, _api_status_cls()):
        sc = getattr(e, "status_code", None)
        # 429/5xx 可重试，其余 4xx 终止
        if sc == 429 or (isinstance(sc, int) and 500 <= sc < 600):
            return "retryable"
        return "server"
    if isinstance(e, bindings.retryable_errors):
        return "retryable"
    if isinstance(e, ValueError):
        return "empty"
    return "other"


_KIND_PREFIX = {
    "server": "服务器错误",
    "retryable": "可重试",
    "empty": "空内容",
    "other": "未知错误",
}

# 退避策略：线性递增封顶 + 随机抖动
_BACKOFF_BASE_SEC = 2.0
_BACKOFF_MAX_SEC = 30.0


def _retry_after_secs(e: Exception) -> Optional[float]:
    """从 429 响应的 Retry-After 头提取服务端建议等待秒数，无则返回 None。"""
    try:
        resp = getattr(e, "response", None)
        if resp is None:
            return None
        ra = resp.headers.get("Retry-After")
        return float(ra) if ra else None
    except (TypeError, ValueError):
        return None


def _backoff_secs(e: Exception, attempt: int) -> float:
    """计算退避等待秒数。"""
    ra = _retry_after_secs(e)
    if ra is not None and ra > 0:
        return min(ra, _BACKOFF_MAX_SEC)
    return min(_BACKOFF_BASE_SEC * (attempt + 1), _BACKOFF_MAX_SEC) + random.uniform(0, 1)


def _handle_retryable_error(e: Exception, attempt: int, max_attempts: int,
                            stop_event: threading.Event) -> Optional[AnalyzeResult]:
    """处理一次可重试异常：返回 None 表示退避后继续，返回结果表示终止。"""
    label = f"{attempt + 1}/{max_attempts}"

    # APIStatusError：限流(429)与 5xx 可重试；其余（如 4xx 配置错误）直接终止不重试
    if isinstance(e, _api_status_cls()):
        sc = e.status_code
        if not (sc == 429 or (isinstance(sc, int) and 500 <= sc < 600)):
            logger.error(f"API 错误: {e}")
            return AnalyzeResult("", "", [], attempt, f"API错误: {e}", "api")

    kind = _error_kind_of(e)
    logger.warning(f"{_KIND_PREFIX[kind]} ({label}): {e}")
    if stop_event.wait(_backoff_secs(e, attempt)):
        return AnalyzeResult("", "", [], attempt, "已取消", "cancel")
    return None


def analyze_frames(
    client: OpenAIClient, model: str, frames: List[Frame], config: Config,
    stop_event: threading.Event, video_path: str, duration: float,
    container_title: str = "", extra_meta: str = "",
) -> AnalyzeResult:
    """调用 AI 分析关键帧，返回 AnalyzeResult。"""
    if not frames:
        logger.warning("无有效帧可发送")
        return AnalyzeResult("", "", [], 0, "无有效帧", "other")

    messages = _build_messages(frames, config, video_path, duration,
                               container_title, extra_meta)
    max_attempts = config.retry_times + 1
    last_error = "未知错误"
    last_error_kind = "other"

    for attempt in range(max_attempts):
        if stop_event.is_set():
            return AnalyzeResult("", "", [], attempt, "已取消", "cancel")
        try:
            result = _call_and_parse(client, model, messages, config, stop_event)
            return result._replace(retries=attempt)
        except (AIFormatError, AITitleEmptyError) as e:
            # 格式错误 / title 空：立即重试，不退避
            last_error = str(e)
            last_error_kind = "format"
            logger.warning(f"{e} ({attempt + 1}/{max_attempts})，立即重试...")
            continue
        except Exception as e:
            result = _handle_retryable_error(e, attempt, max_attempts, stop_event)
            if result is not None:
                return result
            last_error = str(e)
            last_error_kind = _error_kind_of(e)
            continue

    return AnalyzeResult("", "", [], config.retry_times, last_error, last_error_kind)
