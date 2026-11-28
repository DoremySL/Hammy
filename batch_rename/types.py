"""管线共享数据类型。"""
from dataclasses import dataclass
from typing import List, NamedTuple


class RenameRecord(NamedTuple):
    """单条重命名结果记录。"""
    status: str       # "ok" | "skipped" | "error"
    old_name: str
    new_name: str
    # 附加信息：error 时存错误描述
    message: str = ""


class Frame(NamedTuple):
    """单帧：相对秒时间戳 + JPEG base64 + 点位索引。"""
    ts: float
    b64: str
    point_index: int


@dataclass
class SummaryStats:
    """批处理结果统计，由管线填充后传给报告函数。"""
    results: List[RenameRecord]
    ok: int = 0
    skipped: int = 0
    error: int = 0
    cancelled: int = 0           # 停止/熔断时已入队但未处理的视频数
    elapsed: float = 0.0
    processed_count: int = 0   # 实际参与处理的视频数
