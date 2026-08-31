"""同内容不同版本视频比对核心：抽帧指纹 + 平移对齐 + 头尾多余量推断。"""
from __future__ import annotations

import threading
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .dependencies import ffmpeg_tools
from .video import run_subprocess_with_cancel

# ── 指纹 ──
WINDOW_POS = (0.02, 0.25, 0.50, 0.75, 0.92)  # 窗口位置（占时长比例）
PHASE1_WINDOW = 2  # 阶段 1 粗判窗
WINDOW_SEC = 30.0
SHORT_VIDEO_SEC = 150.0   # 短视频时长阈值
_FRAME_BYTES = 72         # 9x8 灰度
_WINDOW_TIMEOUT = 60.0
_FULL_TIMEOUT = 300.0

# ── 判定 ──
DUR_TOL = 15.0            # 时长预筛容差（秒）
ASPECT_TOL = 0.02         # 宽高比预筛容差
DHASH_MAX_DIST = 10       # 单帧汉明距离阈值（/64）
WINDOW_MIN_MATCH = 8      # 单窗匹配帧下限
MIN_MATCHED_WINDOWS = 3   # 匹配窗数下限
SHORT_MIN_MATCH = 30      # 短视频单段匹配帧下限
DELTA_RANGE = 15.0        # 平移试错范围（秒）
PRE_FILTER_DIST = 20      # 阶段 1 预筛阈值


@dataclass(frozen=True)
class ScanMode:
    """扫描模式参数。"""
    dur_tol: float
    delta_range: float
    window_sec: float
    window_min_match: int
    use_keyframes: bool = False
    windows: Tuple[int, ...] = (0, 1, 2, 3, 4)
    short_sec: float = SHORT_VIDEO_SEC
    tiered_dur_tol: bool = False
    kf_unique_div: int = 3     # 关键帧回退分母

    def dur_tol_for(self, duration: float) -> float:
        """按时长返回预筛容差。"""
        if not self.tiered_dur_tol:
            return self.dur_tol
        if duration < 60.0:
            return 0.0
        if duration <= 180.0:
            return 1.0
        return 2.0


# 快速 / 常规 / 极慢
FAST = ScanMode(2.0, 2.0, 15.0, 6, True, (1, 2, 3), 90.0,
                tiered_dur_tol=True, kf_unique_div=6)
NORMAL = ScanMode(15.0, 15.0, 30.0, 8, True)
EXTREME = ScanMode(35.0, 35.0, 60.0, 8)
SCAN_MODES = {"fast": FAST, "normal": NORMAL, "extreme": EXTREME}


def dhash_gray(b: bytes) -> int:
    """9x8 灰度行内相邻列差 → 64 位哈希。"""
    h = 0
    for row in range(8):
        o = row * 9
        for col in range(8):
            if b[o + col] > b[o + col + 1]:
                h |= 1 << (row * 8 + col)
    return h


def _is_flat(b: bytes) -> bool:
    return max(b) - min(b) <= 4


def _frames_from_raw(raw: bytes, t0: float) -> List[Tuple[float, int]]:
    out = []
    n = len(raw) // _FRAME_BYTES
    for k in range(n):
        b = raw[k * _FRAME_BYTES:(k + 1) * _FRAME_BYTES]
        if _is_flat(b):
            continue
        out.append((t0 + k, dhash_gray(b)))
    return out


def _raw_unique_count(raw: bytes) -> int:
    """去重后的不同内容帧数。"""
    n = 0
    prev = None
    for k in range(len(raw) // _FRAME_BYTES):
        b = raw[k * _FRAME_BYTES:(k + 1) * _FRAME_BYTES]
        if b != prev:
            n += 1
            prev = b
    return n


def fingerprint_windows(path: str, duration: float,
                        is_ts: bool, idxs: Sequence[int],
                        stop_event: threading.Event,
                        mode: ScanMode = NORMAL) -> Dict[int, List[Tuple[float, int]]]:
    """抽取指纹帧 [(t, hash)]。"""
    if not ffmpeg_tools.ffmpeg:
        ffmpeg_tools.locate()
    vf = "fps=1,scale=9:8,format=gray"
    raw_out = ["-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]

    def run(cmd: List[str], t0: float, timeout: float) -> List[Tuple[float, int]]:
        stdout, _ = run_subprocess_with_cancel(cmd, timeout, stop_event)
        return _frames_from_raw(stdout or b"", t0)

    def run_kf(cmd: List[str], t0: float, timeout: float, min_unique: int) -> List[Tuple[float, int]]:
        raw = run_subprocess_with_cancel(
            [ffmpeg_tools.ffmpeg, "-y", "-skip_frame", "nokey", *cmd], timeout, stop_event)[0] or b""
        frames = _frames_from_raw(raw, t0)
        if _raw_unique_count(raw) < min_unique:
            raw2 = run_subprocess_with_cancel(
                [ffmpeg_tools.ffmpeg, "-y", *cmd], timeout, stop_event)[0] or b""
            if _raw_unique_count(raw2) > _raw_unique_count(raw):
                frames = _frames_from_raw(raw2, t0)
        return frames

    # mpegts / 短视频：整段单次解码
    if is_ts or duration <= mode.short_sec:
        cmd = ["-i", path, "-vf", vf, *raw_out]
        if mode.use_keyframes:
            return {0: run_kf(cmd, 0.0, _FULL_TIMEOUT,
                              max(4, int(duration) // mode.kf_unique_div))}
        return {0: run([ffmpeg_tools.ffmpeg, "-y", *cmd], 0.0, _FULL_TIMEOUT)}
    result: Dict[int, List[Tuple[float, int]]] = {}
    min_unique = max(4, int(mode.window_sec) // mode.kf_unique_div)
    for i in idxs:
        if stop_event.is_set():
            break
        pos = WINDOW_POS[i] * duration
        cmd = ["-ss", f"{pos:.3f}", "-i", path, "-t", f"{mode.window_sec:.3f}",
               "-vf", vf, *raw_out]
        if mode.use_keyframes:
            result[i] = run_kf(cmd, pos, _WINDOW_TIMEOUT, min_unique)
        else:
            result[i] = run([ffmpeg_tools.ffmpeg, "-y", *cmd], pos, _WINDOW_TIMEOUT)
    return result


# ── 对齐 ──

def _match_count(bin_a: List[Tuple[float, int]], bin_b: List[Tuple[float, int]],
                 delta: float, cap: int = 10 ** 9) -> int:
    """按平移量 delta 统计匹配帧数。"""
    if not bin_a or not bin_b:
        return 0
    tb = [f[0] for f in bin_b]
    count = 0
    for t, h in bin_a:
        lo = bisect_left(tb, t + delta - 0.5)
        hi = bisect_right(tb, t + delta + 0.5)
        for j in range(lo, hi):
            if (h ^ bin_b[j][1]).bit_count() <= DHASH_MAX_DIST:
                count += 1
                break
        if count >= cap:
            break
    return count


def _bin_by_time(frames: List[Tuple[float, int]], n: int) -> List[List[Tuple[float, int]]]:
    t0, t1 = frames[0][0], frames[-1][0]
    if t1 <= t0:
        return [frames]
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n)]
    step = (t1 - t0) / n
    for f in frames:
        idx = min(n - 1, int((f[0] - t0) / step))
        bins[idx].append(f)
    return bins


def _bin_by_windows(frames: List[Tuple[float, int]], duration: float,
                    window_sec: float = WINDOW_SEC,
                    idxs: Sequence[int] = (0, 1, 2, 3, 4)) -> List[List[Tuple[float, int]]]:
    """单段指纹按窗口位置切 bin。"""
    bins: List[List[Tuple[float, int]]] = []
    for i in idxs:
        p = WINDOW_POS[i] * duration
        bins.append([f for f in frames if p - 0.5 <= f[0] < p + window_sec])
    return bins


def pre_dist(fa: List[Tuple[float, int]], fb: List[Tuple[float, int]]) -> int:
    """阶段 1 帧级粗筛距离。"""
    def mid3(f: List[Tuple[float, int]]) -> List[int]:
        k = max(0, len(f) // 2 - 1)
        return [h for _, h in f[k:k + 3]]
    return min((x ^ y).bit_count() for x in mid3(fa) for y in mid3(fb))


@dataclass(frozen=True)
class AlignResult:
    """对齐结果：平移量 δ、匹配窗统计与推断的头尾多余量。"""
    delta: float
    matched_bins: int
    n_bins: int
    total: int
    head_diff: float
    tail_diff: float


def align(segs_a: List[List[Tuple[float, int]]], segs_b: List[List[Tuple[float, int]]],
          delta_dur: float, duration_a: float, duration_b: float = 0.0,
          quick: bool = False, mode: ScanMode = NORMAL) -> Optional[AlignResult]:
    """平移试错对齐。"""
    short = duration_a <= mode.short_sec
    segs_a = [s for s in segs_a if s]
    segs_b = [s for s in segs_b if s]
    if not segs_a or not segs_b:
        return None
    if quick:
        bins_a, bins_b = [segs_a[0]], [segs_b[0]]
    elif len(segs_a) > 1 and len(segs_b) > 1:
        bins_a, bins_b = segs_a, segs_b
    elif len(segs_a) == 1 and len(segs_b) == 1:
        if short:
            bins_a, bins_b = segs_a, segs_b
        else:
            bins_a = _bin_by_time(segs_a[0], 5)
            bins_b = _bin_by_time(segs_b[0], 5)
    elif len(segs_a) > 1:
        bins_a = segs_a
        bins_b = _bin_by_windows(segs_b[0], duration_b or duration_a,
                                 mode.window_sec, mode.windows)
    else:
        bins_b = segs_b
        bins_a = _bin_by_windows(segs_a[0], duration_a, mode.window_sec, mode.windows)
    if not bins_a or not bins_b or len(bins_a) != len(bins_b):
        return None
    n_bins = len(bins_a)

    if mode.delta_range <= 5:
        probe_deltas = list(range(-int(mode.delta_range), int(mode.delta_range) + 1))
    else:
        probe_deltas = [0] + [d * s for d in range(5, int(mode.delta_range) + 1, 5)
                              for s in (1, -1)]
    if not any(_match_count(bins_a[0], bins_b[0], d, cap=2) >= 2 for d in probe_deltas):
        return None

    deltas = [float(d) for d in range(int(-mode.delta_range), int(mode.delta_range) + 1)]
    best_delta, best_total, best_bins = 0.0, -1, []
    for d in deltas:
        per = [_match_count(a, b, d) for a, b in zip(bins_a, bins_b)]
        total = sum(per)
        if total > best_total:
            best_delta, best_total, best_bins = d, total, per
    if best_total <= 0:
        return None
    single = n_bins == 1
    if single and short:
        avail = min(len(bins_a[0]), len(bins_b[0]))
        thr = min(SHORT_MIN_MATCH, max(1, avail // 2))
    else:
        thr = mode.window_min_match
    matched = sum(1 for c in best_bins if c >= thr)
    ok = (best_bins[0] >= thr) if (quick or single) else matched >= MIN_MATCHED_WINDOWS
    if not ok:
        return None
    return AlignResult(best_delta, matched, n_bins, best_total,
                       best_delta, delta_dur - best_delta)


# ── 聚组与多余量 ──

def infer_extras(members: Sequence[str],
                 edges: List[Tuple[str, str, float, float]]) -> Dict[str, float]:
    """推断每成员头尾多余量。"""
    adj: Dict[str, List[Tuple[str, float, float]]] = {m: [] for m in members}
    for a, b, hd, td in edges:
        adj[a].append((b, hd, td))
        adj[b].append((a, -hd, -td))
    head: Dict[str, float] = {}
    root = members[0]
    head[root] = 0.0
    queue = [root]
    while queue:
        cur = queue.pop()
        for nxt, hd, td in adj[cur]:
            if nxt not in head:
                head[nxt] = head[cur] + hd + td
                queue.append(nxt)
    vals = {m: head.get(m, 0.0) for m in members}
    base = min(vals.values())
    return {m: v - base for m, v in vals.items()}


def _res_height(res: str) -> int:
    try:
        return int(res.split("x")[1])
    except (IndexError, ValueError):
        return 0


def aspect_ratio(res: str) -> float:
    try:
        w, h = res.split("x")
        return float(w) / float(h) if float(h) else 0.0
    except (ValueError, AttributeError):
        return 0.0


# 编码优先级
_CODEC_PRIORITY = {"hevc": 4, "av1": 3, "vp9": 2, "h264": 1}


def cluster_groups(metas: Dict[str, Dict], accepted: List[Tuple[str, str, AlignResult]]):
    """union-find 聚组，返回 [{paths, keep, extras, edges}]。"""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in accepted:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: Dict[str, List[str]] = {}
    edges_by: Dict[str, List[Tuple[str, str, AlignResult]]] = {}
    for a, b, r in accepted:
        if find(a) != find(b):
            continue
        root = find(a)
        groups.setdefault(root, [])
        edges_by.setdefault(root, []).append((a, b, r))
        for m in (a, b):
            if m not in groups[root]:
                groups[root].append(m)

    out = []
    for root, members in groups.items():
        edges = [(a, b, r.head_diff, r.tail_diff) for a, b, r in edges_by[root]]
        extras = infer_extras(members, edges)
        best = max(members, key=lambda m: (_res_height(metas[m].get("resolution", "")),
                                           -extras[m],
                                           _CODEC_PRIORITY.get(metas[m].get("codec", ""), 0),
                                           metas[m].get("size", 0)))
        out.append({"paths": members, "keep": best, "extras": extras,
                    "edges": edges_by[root]})
    return out
