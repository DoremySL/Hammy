"""PixAI Tagger 专用抽帧：复用 batch_rename/video.py 的探测与选帧，帧以 base64 驻留内存不落盘。"""
from __future__ import annotations

import base64
import threading
from typing import List, Optional

# 复用引擎层实现（与 gui_app/env.py 同款"GUI 层依赖引擎层"模式）
from batch_rename.dependencies import ffmpeg_tools
from batch_rename.env import logger
from batch_rename.video import (probe_video, select_sampled_timestamps,
                                _extract_single_frame)

# ── 常量 ──
FRAME_COUNT = 15
SHORT_SIDE = 448

# video.py 的复用函数要求 stop_event 非 None（直接调用 .is_set()）；
# 本模块允许 None，用共享空事件兜底（永不会被置位）
_NO_STOP = threading.Event()


def _build_vf(width: int, height: int,
              short_side: int = SHORT_SIDE, crop_square: bool = False,
              crop_portrait: bool = False) -> str:
    """根据视频原始宽高构建 ffmpeg -vf 滤镜链。"""
    if width > height:
        if crop_square:
            return f"scale=-2:{short_side},crop={short_side}:{short_side}"
        return f"scale=-2:{short_side}"
    if width < height and crop_portrait:
        return f"scale={short_side}:-2,crop={short_side}:{short_side}:0:(ih-{short_side})*0.2"
    return f"scale={short_side}:-2"


def extract_frames_for_tagger(
    video_path: str,
    stop_event: Optional[threading.Event] = None,
    frame_count: int = FRAME_COUNT,
    short_side: int = SHORT_SIDE,
    crop_square: bool = False,
    crop_portrait: bool = False,
) -> List[str]:
    """为 PixAI Tagger 从视频中抽取固定帧。
        Args:
        video_path: 视频文件路径
        stop_event: 取消事件（可为 None）
        frame_count: 抽取帧数
        short_side: 短边像素
        crop_square: 横屏是否中心裁剪为正方形
        crop_portrait: 竖屏是否按「剩余空间 20% 起点」裁剪为正方形
        Returns:
        抽取成功的帧列表（JPEG base64 字符串，驻留内存）。
    """
    if stop_event is None:
        stop_event = _NO_STOP
    if stop_event.is_set():
        return []
    if not ffmpeg_tools.ffprobe or not ffmpeg_tools.ffmpeg:
        logger.error("[pixai_frames] ffmpeg/ffprobe 不可用")
        return []

    # 探测元数据（含流信息/分辨率）+ 关键帧时间戳：按帧数开等距点位，每点位取最近关键帧
    info, key_ts = probe_video(video_path, stop_event, frame_count, 1)
    if stop_event.is_set():
        return []
    duration = info.get("duration", 0.0)
    video_meta = info.get("video") or {}
    width = video_meta.get("width") or 0
    height = video_meta.get("height") or 0

    # 分辨率未知时使用保守滤镜（短边缩放不裁剪）
    if width > 0 and height > 0:
        vf = _build_vf(width, height, short_side, crop_square, crop_portrait)
    else:
        vf = f"scale={short_side}:-2"

    timestamps = select_sampled_timestamps(duration, key_ts, frame_count, 1)

    frames: List[str] = []
    for ts in timestamps:
        if stop_event.is_set():
            break
        jpeg = _extract_single_frame(video_path, ts, vf, stop_event)
        if jpeg:
            frames.append(base64.b64encode(jpeg).decode())
    return frames
