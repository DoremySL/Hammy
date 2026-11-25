"""感知哈希与灰度相关性。"""


def dhash_gray(b: bytes) -> int:
    """9x8 灰度行内相邻列差 → 64 位哈希。"""
    h = 0
    for row in range(8):
        o = row * 9
        for col in range(8):
            if b[o + col] > b[o + col + 1]:
                h |= 1 << (row * 8 + col)
    return h


def ncc(a: bytes, b: bytes) -> float:
    """两张 32x32 灰度图的皮尔逊相关系数。"""
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = da = db = 0.0
    for x, y in zip(a, b):
        dx, dy = x - ma, y - mb
        num += dx * dy
        da += dx * dx
        db += dy * dy
    if da <= 0 or db <= 0:
        return 1.0 if a == b else 0.0
    return num / (da * db) ** 0.5
