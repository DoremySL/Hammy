"""标签检索召回：词面三级匹配（关键词+关联词）+ 可选向量融合。"""
from __future__ import annotations

import re
import threading
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

KIND_SCORE = {"exact": 1.0, "compact": 0.9, "token": 0.7}

_SEP_RE = re.compile(r"[\s_\-]+")
_LATIN_RE = re.compile(r"[a-z0-9]+")
_RELATED_SPLIT_RE = re.compile(r"[,，、;；]+")

_FULL_HEADER = "【标签检索】\n为视频生成 tags 时，若画面内容匹配，请优先采用以下指定标签："
_CAND_HEADER = ("【标签检索·召回候选】\n以下标签来自标签库检索，可能包含与视频无关的条目，"
                "仅将确实匹配的并入 tags：")

def _norm(s: Any) -> str:
    """规范化：全角→半角、小写、分隔符（空格/_/-）统一为单空格。"""
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    return _SEP_RE.sub(" ", s).strip()

def _compact(s: Any) -> str:
    """紧凑规范化：在 _norm 基础上再去掉全部分隔符（office_lady / office-lady / office lady 互通）。"""
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    return _SEP_RE.sub("", s)

def split_related(related: Any) -> List[str]:
    """关联词字符串 → 关联词列表（按中英文逗号/顿号/分号切分）。"""
    return [t for t in (x.strip() for x in _RELATED_SPLIT_RE.split(str(related or ""))) if t]

def build_section(items: List[Dict[str, str]], header: str) -> str:
    """条目列表 → 提示词段落（- 关键词：描述；不含关联词）。"""
    lines = [header]
    for it in items:
        desc = it.get("description", "")
        lines.append(f"- {it['keyword']}：{desc}" if desc else f"- {it['keyword']}")
    return "\n".join(lines)

class _Entry:
    """单个标签的预计算匹配索引：关键词与各关联词各为一组匹配目标。"""
    __slots__ = ("idx", "item", "targets")

    def __init__(self, idx: int, item: Dict[str, str]):
        self.idx = idx
        self.item = item
        self.targets: List[Tuple[str, str, frozenset]] = []
        for text in [item["keyword"], *split_related(item.get("related", ""))]:
            norm = _norm(text)
            if not norm:
                continue
            compact = _compact(text)
            latin_tokens = frozenset(_LATIN_RE.findall(norm))
            self.targets.append((norm, compact, latin_tokens))

class TagRecall:
    """标签检索提供方：显式模式路由 + 词面召回（+ 可选向量召回合并）。"""

    def __init__(self, items: Any, top_k: int = 40, dense=None,
                 vec_threshold: float = 0.45, vec_top_n: int = 20,
                 mode: str = "off"):
        """mode：off=不注入 / full=单轮全量注入 / rag=两轮召回；非法值按 off。"""
        norm_items: List[Dict[str, str]] = []
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                kw = str(it.get("keyword", "") or "").strip()
                if not kw:
                    continue
                norm_items.append({"keyword": kw,
                                   "description": str(it.get("description", "") or "").strip(),
                                   "related": str(it.get("related", "") or "").strip()})

        self.total_items = len(norm_items)
        self.items = norm_items
        self.top_k = max(1, int(top_k or 40))
        self.dense = dense
        self.vec_threshold = max(0.0, min(1.0, float(vec_threshold if vec_threshold is not None else 0.45)))
        self.vec_top_n = max(1, int(vec_top_n or 0))
        self._dense_built = False
        self._dense_lock = threading.Lock()

        # 空库一律不注入；模式显式给定，不做条目数自动降级
        m = str(mode or "").strip().lower()
        self.mode = m if m in ("off", "full", "rag") and self.total_items > 0 else "off"

        self.full_section = build_section(norm_items, _FULL_HEADER) if self.mode == "full" else ""
        self._entries = ([_Entry(i, it) for i, it in enumerate(norm_items)]
                         if self.mode == "rag" else [])

    def recall(self, parts: Dict[str, str]) -> List[Dict[str, str]]:
        """词面召回并取向量命中，返回去重后的候选条目；仅 rag 模式有效。"""
        if self.mode != "rag" or not self._entries:
            return []
        hay_n = _norm("\n".join(v for v in parts.values() if v))
        hay_c = _compact(hay_n)
        hay_tokens = frozenset(_LATIN_RE.findall(hay_n))

        hits = []
        for e in self._entries:
            kind = self._match(e, hay_n, hay_c, hay_tokens)
            if kind is not None:
                hits.append((KIND_SCORE[kind], e.idx, e))
        hits.sort(key=lambda h: (-h[0], h[1]))

        top = hits[: self.top_k]
        candidates = [e.item for _, _, e in top]
        taken = {idx for _, idx, _ in top}

        if self.dense is not None:
            try:
                self._ensure_dense_built()
                vec = self.dense.search(parts, top_k=self.vec_top_n)
            except Exception:
                vec = []  # 向量层失败：降级为纯词面
            for m in vec[: self.vec_top_n]:
                if float(m.get("score", 0.0)) < self.vec_threshold:
                    continue
                idx = int(m.get("idx", -1))
                if 0 <= idx < len(self.items) and idx not in taken:
                    taken.add(idx)
                    candidates.append(self.items[idx])
        return candidates

    def _ensure_dense_built(self) -> None:
        """惰性构建向量索引一次（items 顺序即 idx，词面/向量两路共用）。"""
        if self._dense_built:
            return
        with self._dense_lock:
            if self._dense_built:
                return
            try:
                self.dense.build_index(self.items)
                self._dense_built = True
            except Exception:
                self.dense = None
                raise

    def warmup_dense(self) -> None:
        """预热：提前构建向量索引（管线启动前调用，避免首个视频承担构建耗时）。"""
        if self.dense is not None and self.mode == "rag":
            self._ensure_dense_built()

    @staticmethod
    def _match(e: _Entry, hay_n: str, hay_c: str, hay_tokens: frozenset) -> Optional[str]:
        """任一匹配目标（关键词或关联词）三级命中即召回：精确 → 紧凑 → 分词。"""
        for norm, compact, latin_tokens in e.targets:
            if norm in hay_n:
                return "exact"
            if compact and compact in hay_c:
                return "compact"
            if latin_tokens and latin_tokens <= hay_tokens:
                return "token"
        return None

    def build_candidates_section(self, candidates: List[Dict[str, str]]) -> str:
        """候选条目 → 第二轮提示词段落。"""
        return build_section(candidates, _CAND_HEADER)
