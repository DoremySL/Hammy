"""公用帧预处理：黑帧判定、四边黑边裁剪、JPEG 重编码（纯 PIL，无 NumPy）。

抽帧后统一执行：LLM、Tagger 与缩略图刷新共用同一批存活帧。
语义与 tagger 仓库 hammy_tagger.trim_bars 一致，对已裁剪帧幂等。
"""
import io
from typing import List, Optional

from PIL import Image

BAR_PIXEL = 20       # 像素值低于此视为黑
BLACK_FRAME = 0.99   # 全帧黑像素占比 ≥ 此值 → 黑帧
BAR_RATIO = 0.99     # 行/列黑像素占比 ≥ 此值计为黑边
BAR_MAX_TRIM = 0.45  # 每条边最多裁掉的比例
JPEG_QUALITY = 88    # 重编码质量（无裁剪时原图直接复用，不经此路径）

_LUT: Optional[List[int]] = None


def _bin_lut() -> List[int]:
    """黑像素→255 二值化 LUT（懒构建，进程级复用）。"""
    global _LUT
    if _LUT is None:
        _LUT = [255 if v < BAR_PIXEL else 0 for v in range(256)]
    return _LUT


def preprocess_frame(jpeg: bytes) -> Optional[bytes]:
    """单帧 JPEG bytes → 预处理后 JPEG bytes；黑帧/坏帧返回 None。"""
    try:
        im = Image.open(io.BytesIO(jpeg))
        im.load()
    except Exception:
        return None
    g = im.convert("L")
    w, h = g.size
    if w <= 0 or h <= 0:
        return None
    if sum(g.histogram()[:BAR_PIXEL]) >= w * h * BLACK_FRAME:
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
        return jpeg  # 无裁剪：原图直接复用，省一次重编码
    buf = io.BytesIO()
    try:
        im.crop((left, top, right, bottom)).save(buf, "JPEG", quality=JPEG_QUALITY)
    except Exception:
        return None
    return buf.getvalue()


def preprocess_frames(jpegs: List[bytes]) -> List[bytes]:
    """批量预处理，返回存活帧列表（顺序保留，黑帧/坏帧丢弃）。"""
    out: List[bytes] = []
    for j in jpegs:
        r = preprocess_frame(j)
        if r is not None:
            out.append(r)
    return out
