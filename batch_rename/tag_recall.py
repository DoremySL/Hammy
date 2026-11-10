"""标签检索：on/enhanced 共用首轮字面召回（关键词+关联词三级匹配）；
enhanced 追加第二轮纯向量召回（dense 不可用时自动等同 on）。"""
from __future__ import annotations

import re
import threading
import unicodedata
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

KIND_SCORE = {"exact": 1.0, "compact": 0.9, "token": 0.7}

_SEP_RE = re.compile(r"[\s_\-]+")
_LATIN_RE = re.compile(r"[a-z0-9]+")

_FIELD_SEP = "\x00"

# 全分组共用注入头
_LITERAL_HEADER = ("【标签检索】\n以下是根据视频信息（文件名等）从标签库检索到的候选标签，"
                   "若画面内容匹配，请优先采用；角色/IP 等名称请尽量采用标签中的中文名写法：")
_CAND_HEADER = ("【标签检索·召回候选】\n以下是根据首轮分析结果与视频信息从标签库检索到的候选标签，"
                "可能包含与画面无关的条目：")

def _norm(s: Any) -> str:
    """规范化：全角→半角、小写、分隔符（空格/_/-）统一为单空格。"""
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    return _SEP_RE.sub(" ", s).strip()

def _compact(s: Any) -> str:
    """紧凑规范化：在 _norm 基础上再去掉全部分隔符（office_lady / office-lady / office lady 互通）。"""
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    return _SEP_RE.sub("", s)

def split_related(related: Any) -> List[str]:
    """关联词取值（单值字段，不再按逗号切分）。"""
    t = str(related or "").strip()
    return [t] if t else []

def build_section(items: List[Dict[str, str]], header: str) -> str:
    """条目列表 → 提示词段落（- 关键词：描述；不含关联词）。"""
    lines = [header]
    for it in items:
        desc = it.get("description", "")
        lines.append(f"- {it['keyword']}：{desc}" if desc else f"- {it['keyword']}")
    return "\n".join(lines)

class _Target(NamedTuple):
    norm: str
    compact: str
    latin_tokens: frozenset
    exact_re: Optional[re.Pattern]
    compact_re: Optional[re.Pattern]

class _Entry:
    """单个标签的预计算匹配索引：关键词与各关联词各为一组匹配目标。"""
    __slots__ = ("idx", "item", "targets")

    def __init__(self, idx: int, item: Dict[str, str]):
        self.idx = idx
        self.item = item
        self.targets: List[_Target] = []
        for text in [item["keyword"], *split_related(item.get("related", ""))]:
            norm = _norm(text)
            if not norm:
                continue
            compact = _compact(text)
            latin_tokens = frozenset(_LATIN_RE.findall(norm))
            exact_re = compact_re = None
            if latin_tokens:
                exact_re = re.compile(rf"(?<![a-z0-9]){re.escape(norm)}(?![a-z0-9])")
                flex = r"[\s_-]*".join(map(re.escape, compact))
                compact_re = re.compile(rf"(?<![a-z0-9]){flex}(?![a-z0-9])")
            self.targets.append(_Target(norm, compact, latin_tokens, exact_re, compact_re))

class TagRecall:
    """标签检索提供方：首轮字面召回注入（on/enhanced）+ enhanced 第二轮纯向量召回。"""

    def __init__(self, items: Any, top_k: int = 40, dense=None,
                 vec_threshold: float = 0.45, vec_top_n: int = 20,
                 mode: str = "off"):
        """mode：off=不注入 / on=首轮字面召回注入（单轮）/
        enhanced=字面注入 + 第二轮向量召回修订（dense 不可用时等同 on）；非法值按 off。"""
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
        self.mode = m if m in ("off", "on", "enhanced") and self.total_items > 0 else "off"

        self._entries = ([_Entry(i, it) for i, it in enumerate(norm_items)]
                         if self.mode in ("on", "enhanced") else [])

    @property
    def has_dense(self) -> bool:
        """enhanced 档两轮生效的前提：向量后端存在（构建失败会被置 None）。"""
        return self.mode == "enhanced" and self.dense is not None

    def recall_literal(self, parts: Dict[str, str]) -> List[Dict[str, str]]:
        """首轮字面召回（关键词+关联词三级匹配，top_k 截断）；仅 on/enhanced 有效。"""
        if self.mode not in ("on", "enhanced") or not self._entries:
            return []
        hay_n = _norm(_FIELD_SEP.join(v for v in parts.values() if v))
        hay_c = _compact(hay_n)
        hay_tokens = frozenset(_LATIN_RE.findall(hay_n))

        hits = []
        for e in self._entries:
            kind = self._match(e, hay_n, hay_c, hay_tokens)
            if kind is not None:
                hits.append((KIND_SCORE[kind], e.idx, e))
        hits.sort(key=lambda h: (-h[0], h[1]))
        return [e.item for _, _, e in hits[: self.top_k]]

    def literal_injection(self, parts: Dict[str, str]) -> Tuple[str, List[str]]:
        """首轮字面召回 → (注入段落, 命中关键词列表)；无命中返回 ("", [])。"""
        cands = self.recall_literal(parts)
        if not cands:
            return "", []
        return (build_section(cands, _LITERAL_HEADER),
                [it["keyword"] for it in cands])

    def recall_dense(self, parts: Dict[str, str]) -> List[Dict[str, str]]:
        """第二轮纯向量召回；仅 enhanced 且向量后端可用，任何失败返回 []（跳过第二轮）。"""
        if not self.has_dense:
            return []
        try:
            self._ensure_dense_built()
            vec = self.dense.search(parts, top_k=self.vec_top_n)
        except Exception:
            return []  # 向量层失败：跳过第二轮，沿用首轮结果
        out = []
        for m in vec[: self.vec_top_n]:
            if float(m.get("score", 0.0)) < self.vec_threshold:
                continue
            idx = int(m.get("idx", -1))
            if 0 <= idx < len(self.items):
                out.append(self.items[idx])
        return out

    def _ensure_dense_built(self) -> None:
        """惰性构建向量索引一次（items 顺序即 idx，与标签库共用）。"""
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
        if self.dense is not None and self.mode == "enhanced":
            self._ensure_dense_built()

    @staticmethod
    def _match(e: _Entry, hay_n: str, hay_c: str, hay_tokens: frozenset) -> Optional[str]:
        """任一匹配目标（关键词或关联词）三级命中即召回：精确 → 紧凑 → 分词。"""
        for t in e.targets:
            if t.norm in hay_n and (t.exact_re is None or t.exact_re.search(hay_n)):
                return "exact"
            if t.compact and t.compact in hay_c:
                if t.compact_re is None or t.compact_re.search(hay_n):
                    return "compact"
            if t.latin_tokens and t.latin_tokens <= hay_tokens:
                return "token"
        return None

    def build_candidates_section(self, candidates: List[Dict[str, str]]) -> str:
        """候选条目 → 第二轮提示词段落。"""
        return build_section(candidates, _CAND_HEADER)
