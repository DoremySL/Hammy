"""公用帧预处理：黑帧/纯色帧判定、四边黑边裁剪、感知签名、同点位近似去重、JPEG 重编码。

抽帧后统一执行：LLM、Tagger 与缩略图刷新共用同一批存活帧。
语义与 tagger 仓库 hammy_tagger.trim_bars 一致，对已裁剪帧幂等。
"""
import base64
import io
from typing import Dict, List, Optional, Tuple

from PIL import Image

from .env import logger
from .phash import dhash_gray, ncc
from .types import Frame

BAR_PIXEL = 20       # 像素值低于此视为黑
BLACK_FRAME = 0.99   # 全帧黑像素占比 ≥ 此值 → 黑帧
FLAT_SPAN = 2        # 纯色判定：峰值灰度左右各取几个桶
FLAT_COVER = 0.99    # 纯色判定：这些桶的像素占比 ≥ 此值 → 纯色帧
BAR_RATIO = 0.99     # 行/列黑像素占比 ≥ 此值计为黑边
BAR_MAX_TRIM = 0.45  # 每条边最多裁掉的比例
JPEG_QUALITY = 88    # 重编码质量（无裁剪时原图直接复用，不经此路径）

# ── 同点位相似去重阈值（bench 目检定标）──
_DEDUP_DHASH_MAX = 20   # dhash 粗筛阈值（/64）
_DEDUP_NCC_MIN = 0.80   # 32x32 灰度相关性阈值

_LUT: Optional[List[int]] = None


def _bin_lut() -> List[int]:
    """黑像素→255 二值化 LUT（懒构建，进程级复用）。"""
    global _LUT
    if _LUT is None:
        _LUT = [255 if v < BAR_PIXEL else 0 for v in range(256)]
    return _LUT


def preprocess_frame_signature(jpeg: bytes) -> Optional[Tuple[bytes, int, bytes]]:
    """JPEG → (预处理后 JPEG, dhash, 32x32 灰度签名)；黑帧/纯色帧/坏帧返回 None。"""
    try:
        im = Image.open(io.BytesIO(jpeg))
        im.load()
    except Exception:
        return None
    g = im.convert("L")
    w, h = g.size
    if w <= 0 or h <= 0:
        return None
    hist = g.histogram()
    if sum(hist[:BAR_PIXEL]) >= w * h * BLACK_FRAME:
        return None
    peak = max(range(256), key=hist.__getitem__)
    lo, hi = max(0, peak - FLAT_SPAN), min(256, peak + FLAT_SPAN + 1)
    if sum(hist[lo:hi]) >= w * h * FLAT_COVER:
        return None
    bw = g.point(_bin_lut())
    data = bw.tobytes()
    rows_dark = [data.count(255, y * w, (y + 1) * w) for y in range(h)]
    # ROTATE_270(CW)：转置后第 r 行 = 原图第 r 列
    data_t = bw.transpose(Image.ROTATE_270).tobytes()
    cols_dark = [data_t.count(255, x * h, (x + 1) * h) for x in range(w)]
    top = 0
    while top < h * BAR_MAX_TRIM and rows_dark[top] >= w * BAR_RATIO:
        top += 1
    bottom = h
    while h - bottom < h * BAR_MAX_TRIM and rows_dark[bottom - 1] >= w * BAR_RATIO:
        bottom -= 1
    left = 0
    while left < w * BAR_MAX_TRIM and cols_dark[left] >= h * BAR_RATIO:
        left += 1
    right = w
    while w - right < w * BAR_MAX_TRIM and cols_dark[right - 1] >= h * BAR_RATIO:
        right -= 1
    if (left, top, right, bottom) == (0, 0, w, h):
        content, out = g, jpeg  # 无裁剪：原图直接复用，省一次重编码
    else:
        content = g.crop((left, top, right, bottom))
        buf = io.BytesIO()
        try:
            im.crop((left, top, right, bottom)).save(buf, "JPEG", quality=JPEG_QUALITY)
        except Exception:
            return None
        out = buf.getvalue()
    dhash = dhash_gray(content.resize((9, 8), Image.BILINEAR).tobytes())
    sig = content.resize((32, 32), Image.BILINEAR).tobytes()
    return out, dhash, sig


def _dedup_point_frames(items: List[Tuple[Frame, int, bytes]]) -> List[Frame]:
    """同点位近似帧去重：dhash 粗筛 + 灰度相关性确认，保留先出现的帧。

    仅限同点位：不同点位画面相同是有效信息（场景未变化/重演），不做合并。"""
    kept: List[Tuple[Frame, int, bytes]] = []
    by_point: Dict[int, List[Tuple[int, bytes]]] = {}
    for frame, dh, sig in items:
        prev = by_point.get(frame.point_index, [])
        if any((dh ^ h).bit_count() <= _DEDUP_DHASH_MAX
               and ncc(sig, s) >= _DEDUP_NCC_MIN for h, s in prev):
            continue
        kept.append((frame, dh, sig))
        by_point.setdefault(frame.point_index, []).append((dh, sig))
    if len(kept) != len(items):
        logger.debug(f"同点位相似去重: {len(items)} -> {len(kept)}")
    return [f for f, _, _ in kept]


def preprocess_frames(frames: List[Frame]) -> List[Frame]:
    """抽帧后统一清理：黑帧/纯色帧丢弃、黑边裁剪、同点位近似去重。"""
    items: List[Tuple[Frame, int, bytes]] = []
    for f in frames:
        result = preprocess_frame_signature(base64.b64decode(f.b64))
        if result is not None:
            done, dh, sig = result
            items.append((Frame(f.ts, base64.b64encode(done).decode(), f.point_index),
                          dh, sig))
    return _dedup_point_frames(items)
