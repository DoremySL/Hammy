"""srt_translate.py — SRT 字幕翻译模块（独立模块，方便测试）。"""
from __future__ import annotations

import math
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional

from .env import make_logger

# ── 常量 ──
_BATCH_SIZE = 64         # 初始每批段落数
_RETRY_SIZES = {16, 8}   # 这些批次大小有 1 次重试机会，其余失败直接拆分

# 翻译任务专用请求参数（不读取设置页的 ai_timeout / max_tokens /
# temperature / top_p / response_format —— 那些面向视频重命名场景）：
_REQUEST_TIMEOUT = 60    # 单次请求超时（秒）：整批 64 行翻译响应可能较慢

_SYSTEM_PROMPT = """你是一个字幕翻译专家。用户会提供带编号的字幕文本，你必须逐行翻译为简体中文（必须使用简体字，禁止使用繁体字）。
规则：
1. 忠实准确地进行翻译，允许使用直白或粗俗的描述，不应随意删减或篡改，不应回避或淡化，以保持作品的原貌。
2. 每行输入对应一行输出，保持编号和行数完全一致
3. 将翻译结果包裹在 <textarea> 标签内输出
4. 标签内只输出“编号. 翻译”格式，不要添加解释或注释
5. 即使某行很短或是语气词，也必须翻译，不得跳过"""


class TranslationCancelled(Exception):
    """翻译被用户取消（stop_event 置位），已完成分片的结果仍保留在内存。"""


_default_log = make_logger("srt_translate")


def _parse_srt(srt_content: str) -> List[dict]:
    """解析 SRT 内容为段落列表 [{idx, time, text}]。"""
    segments = []
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        idx_line = lines[0].strip()
        time_line = lines[1].strip()
        text_lines = lines[2:]
        if not idx_line.isdigit() or '-->' not in time_line:
            continue
        segments.append({
            'idx': int(idx_line),
            'time': time_line,
            'text': '\n'.join(text_lines),
        })
    return segments


def _build_srt(segments: List[dict]) -> str:
    """从段落列表重建 SRT 内容。"""
    parts = []
    for i, seg in enumerate(segments, 1):
        parts.append(f"{i}")
        parts.append(seg['time'])
        parts.append(seg['text'])
        parts.append('')
    return '\n'.join(parts)


# ── OpenAI SDK 调用（与项目其他 AI 调用统一）─────────────────
_client_cache: dict = {}
_client_cache_lock = threading.Lock()


def _get_client(api_key: str, base_url: str):
    """获取（并缓存）OpenAI client，与项目其他 AI 调用统一走 openai SDK。"""
    key = (api_key, base_url)
    with _client_cache_lock:
        client = _client_cache.get(key)
        if client is None:
            from batch_rename.dependencies import bindings, DependencyError
            if bindings.OpenAI is None:
                try:
                    bindings.load()
                except DependencyError as e:
                    raise RuntimeError(f"openai 依赖未加载，无法翻译字幕: {e}")
            client = bindings.OpenAI(
                api_key=api_key or "not-needed",
                base_url=base_url,
            )
            _client_cache[key] = client
        return client


def _call_ai(prompt: str, api_key: str, base_url: str, model: str) -> str:
    """调用 OpenAI 兼容 API（同步，SDK 实现），参数用本模块专用常量。"""
    client = _get_client(api_key, base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
        timeout=_REQUEST_TIMEOUT,
    )
    if not (resp.choices and resp.choices[0].message):
        raise ValueError("AI 返回空内容")
    return (resp.choices[0].message.content or "").strip()


def _describe_error(e: Exception) -> tuple:
    """将调用异常转为 (人类可读描述, fatal)。"""
    try:
        from batch_rename.dependencies import bindings, DependencyError
    except Exception:
        return str(e) or type(e).__name__, False

    if bindings.APIStatusError is None:
        try:
            bindings.load()
        except DependencyError as de:
            return f"openai 依赖未加载: {de}", True
    try:
        # APITimeoutError 是 APIConnectionError 的子类，先判超时
        if isinstance(e, bindings.APITimeoutError):
            return f"请求超时（{_REQUEST_TIMEOUT}s）", False
        if isinstance(e, bindings.APIConnectionError):
            return "无法连接 AI 服务，请检查 base_url 与网络", False
        if isinstance(e, bindings.APIStatusError):
            sc = getattr(e, "status_code", None)
            if sc in (401, 403):
                return f"认证失败 ({sc})，请检查 api_key", True
            if sc == 404:
                return f"端点不存在 ({sc})，请检查 base_url", True
            if sc == 429:
                return "限流 (429)", False
            if isinstance(sc, int) and 500 <= sc < 600:
                return f"服务器错误 ({sc})", False
            return f"API 错误 ({sc})", True
        if isinstance(e, DependencyError):
            return f"openai 依赖未加载: {e}", True
    except Exception:
        pass
    return str(e) or type(e).__name__, False


# ── 核心翻译逻辑 ──

def translate_srt_file(src_path: str, dest_path: str,
                       api_key: str, base_url: str, model: str,
                       workers: int = 4,
                       log_fn: Optional[Callable[[str], None]] = None,
                       stop_event: Optional[object] = None) -> None:
    """翻译 SRT 文件为简体中文并保存。
        Args:
        src_path: 源 SRT 文件路径
        dest_path: 目标 SRT 文件路径（.zh.srt）
        api_key: AI API 密钥
        base_url: AI API 基础地址
        model: 模型名称
        workers: 并发数（将字幕拆成多少份并行处理）
        log_fn: 警告日志回调（如小批次翻译失败保留原文）
        stop_event: threading.Event；置位后在下一批翻译前抛 TranslationCancelled
        （进行中的单批请求完成后生效）
    """
    log_fn = log_fn or _default_log

    def _check_stop() -> None:
        if stop_event is not None and stop_event.is_set():
            raise TranslationCancelled()

    _check_stop()
    srt_content = Path(src_path).read_text(encoding='utf-8')
    segments = _parse_srt(srt_content)
    if not segments:
        raise ValueError("SRT 文件无有效段落")

    # 将段落拆成 workers 份（每份内部再按 _BATCH_SIZE 分批调用）
    n = len(segments)
    chunk_count = min(workers, max(1, math.ceil(n / _BATCH_SIZE)))
    chunk_size = math.ceil(n / chunk_count)
    chunks = [segments[i:i + chunk_size] for i in range(0, n, chunk_size)]

    def _translate_chunk(chunk: List[dict]) -> None:
        """翻译一个分片内的所有批次（带校验、重试；每批前检查取消）。"""
        for i in range(0, len(chunk), _BATCH_SIZE):
            _check_stop()
            _translate_batch_with_retry(chunk[i:i + _BATCH_SIZE],
                                        api_key, base_url, model, log_fn)

    if chunk_count <= 1:
        _translate_chunk(segments)
    else:
        with ThreadPoolExecutor(max_workers=chunk_count) as pool:
            futures = [pool.submit(_translate_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                future.result()  # 抛出异常（含 TranslationCancelled）则向上传播

    result = _build_srt(segments)
    Path(dest_path).write_text(result, encoding='utf-8')


def _translate_batch_with_retry(
    batch: List[dict],
    api_key: str, base_url: str, model: str,
    log_fn: Callable[[str], None],
) -> None:
    """翻译单个批次，带行数校验 + 分级重试/拆分。"""
    size = len(batch)
    if size == 0:
        return

    has_retry = size in _RETRY_SIZES
    max_attempts = 2 if has_retry else 1

    for attempt in range(max_attempts):
        try:
            prompt = _build_prompt(batch)
            raw_response = _call_ai(prompt, api_key, base_url, model)
            trans_map = _extract_translation(raw_response)

            if len(trans_map) == size:
                _apply_map(batch, trans_map)
                return

            if attempt < max_attempts - 1:
                continue

        except Exception as e:
            desc, fatal = _describe_error(e)
            log_fn(f"  批次请求失败（{size}段）: {desc}")
            if fatal:
                # 认证/端点/配置类错误：重试与拆分均无意义，直接保留原文
                log_fn(f"  片段 {batch[0]['idx']}~{batch[-1]['idx']} 因 {desc} 保留原文")
                return
            if attempt < max_attempts - 1:
                continue

    # 所有尝试失败：拆分或放弃
    if size > 8:
        mid = size // 2
        _translate_batch_with_retry(batch[:mid], api_key, base_url, model, log_fn)
        _translate_batch_with_retry(batch[mid:], api_key, base_url, model, log_fn)
    else:
        log_fn(f"  片段 {batch[0]['idx']}~{batch[-1]['idx']} 翻译失败（重试耗尽），保留原文")


# ── 提示词构建 & 响应解析 ──

def _build_prompt(batch: List[dict]) -> str:
    """构建翻译提示词（textarea 包裹要求）。"""
    parts = [
        "###请翻译以下内容（保持编号对齐，结果用 <textarea> 包裹）",
        "<textarea>",
    ]
    for j, seg in enumerate(batch):
        parts.append(f"{j+1}. {seg['text']}")
    parts.append("</textarea>")
    return '\n'.join(parts)


def _extract_translation(raw_response: str) -> dict:
    """从 AI 响应中提取翻译结果。
        Returns:
        {编号(int): 翻译文本(str)}
    """
    # 优先提取最后一个 <textarea> 内容
    textarea_matches = re.findall(r'<textarea[^>]*>(.*?)</textarea>', raw_response, re.DOTALL)
    content = textarea_matches[-1].strip() if textarea_matches else raw_response

    trans_map = {}
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)[.\u3001\uff0e)]\s*(.+)$', line)
        if m:
            trans_map[int(m.group(1))] = m.group(2).strip()

    # 回退：无编号时按行序生成
    if not trans_map:
        plain_lines = [l.strip() for l in content.splitlines() if l.strip()]
        trans_map = {i + 1: line for i, line in enumerate(plain_lines)}

    return trans_map


def _apply_map(batch: List[dict], trans_map: dict) -> None:
    """将翻译映射应用到批次段落。"""
    for j, seg in enumerate(batch):
        key = j + 1
        if key in trans_map:
            seg['text'] = trans_map[key]
