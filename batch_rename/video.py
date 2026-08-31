"""视频探测、关键帧抽取。"""
import json
import base64
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from .env import logger, SUBPROCESS_KWARGS
from .subprocess_registry import register_subprocess, unregister_subprocess
from .dependencies import ffmpeg_tools
from .config import Config
from .utils import safe_int, safe_float
from .types import Frame


# ── 抽帧策略常量 ──
_SHORT_VIDEO_THRESHOLD = 10.0   # 无关键帧的短视频走"头中尾"三帧兜底
_MIN_KEYFRAME_GAP_SEC = 1.0    # 关键帧最小间隔（稀疏化）

# ── 子进程超时（秒）──
_PROBE_META_TIMEOUT_SEC = 15.0    # ffprobe 元数据 JSON 探测
_PROBE_KEYFRAMES_TIMEOUT_SEC = 60.0  # 关键帧窗口采样探测超时
_PROBE_WINDOW_HALF_SEC = 5.0     # 采样窗口基础半径（秒），实际取 max(此值, 2×每点帧数)
EXTRACT_FRAME_TIMEOUT_SEC = 30.0    # ffmpeg 单帧抽取
_EXTRACT_TS_TIMEOUT_SEC = 300.0    # mpegts 整片顺序抽帧


def run_subprocess_with_cancel(cmd: list, timeout: float, stop_event: threading.Event) -> Tuple[Optional[bytes], Optional[bytes]]:
    """运行子进程，支持通过 stop_event 提前取消。取消或超时时返回 (None, None)"""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **SUBPROCESS_KWARGS)
    register_subprocess(p)
    try:
        deadline = time.time() + timeout
        while True:
            if stop_event.is_set() or time.time() > deadline:
                # 取消或超时：终止进程并丢弃部分输出
                p.terminate()
                try:
                    p.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()
                    try:
                        p.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                return None, None
            # 短轮询：每 0.1 秒检查一次 stop_event
            try:
                return p.communicate(timeout=0.1)
            except subprocess.TimeoutExpired:
                continue
    finally:
        unregister_subprocess(p)


def _parse_frame_rate(rate_str: str) -> float:
    """将 ffprobe 返回的分数形式帧率转为浮点数，如 "30000/1001" → 29.97。"""
    if not rate_str:
        return 0.0
    if '/' in rate_str:
        num, _, den = rate_str.partition('/')
        d = safe_float(den)
        return safe_float(num) / d if d else 0.0
    return safe_float(rate_str)


def _parse_format(fmt: Dict[str, Any]) -> Dict[str, Any]:
    """解析 ffprobe format 部分，返回含 format 字段 + 空 video/audio 的 meta dict。"""
    meta: Dict[str, Any] = {
        "duration": 0.0, "creation_time": "",
        "size": 0, "bit_rate": 0, "format_name": "",
        "encoder": "", "container_title": "",
        "video": {}, "audio": [],
    }
    meta["duration"] = safe_float(fmt.get("duration"))
    meta["size"] = safe_int(fmt.get("size"))
    meta["bit_rate"] = safe_int(fmt.get("bit_rate"))
    meta["format_name"] = fmt.get("format_name", "")

    tags = fmt.get("tags", {})
    ct = tags.get("creation_time", "")
    # 过滤 "0000-00-00..." 占位日期
    meta["creation_time"] = ct if ct and not ct.startswith("0000") else ""
    meta["encoder"] = tags.get("encoder", "")
    meta["container_title"] = tags.get("title", "")
    return meta


def _parse_streams(data: Dict[str, Any]) -> Dict[str, Any]:
    """解析 ffprobe streams，缺失字段用 None 表示。"""
    video: Dict[str, Any] = {}
    audio_list: List[Dict[str, Any]] = []

    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video" and not video:
            fr = _parse_frame_rate(stream.get("r_frame_rate", ""))
            video = {
                "codec": stream.get("codec_name"),
                "width": safe_int(stream.get("width"), None),
                "height": safe_int(stream.get("height"), None),
                "frame_rate": fr if fr > 0 else None,
                "bit_rate": safe_int(stream.get("bit_rate"), None),
                "pix_fmt": stream.get("pix_fmt"),
                "profile": stream.get("profile"),
                "field_order": stream.get("field_order"),
                "duration": safe_float(stream.get("duration"), None),
                "display_aspect_ratio": stream.get("display_aspect_ratio"),
                "color_space": stream.get("color_space"),
                "color_range": stream.get("color_range"),
                "color_transfer": stream.get("color_transfer"),
                "color_primaries": stream.get("color_primaries"),
                "level": safe_int(stream.get("level"), None),
            }
        elif codec_type == "audio":
            audio_list.append({
                "codec": stream.get("codec_name"),
                "channels": safe_int(stream.get("channels"), None),
                "sample_rate": safe_int(stream.get("sample_rate"), None),
                "channel_layout": stream.get("channel_layout"),
                "language": stream.get("tags", {}).get("language"),
                "bit_rate": safe_int(stream.get("bit_rate"), None),
            })

    return {"video": video, "audio": audio_list}


def _parse_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """从 ffprobe JSON 中提取格式与流信息，返回统一的 metadata dict。"""
    meta = _parse_format(data.get("format", {}))
    streams = _parse_streams(data)
    meta["video"] = streams["video"]
    meta["audio"] = streams["audio"]
    return meta


def _kf_from_csv_line(raw: bytes) -> Optional[float]:
    """解析关键帧 CSV 行 "pts_time,flags" → 关键帧时间戳；非关键帧/无效行返回 None。"""
    parts = raw.decode("utf-8", errors="replace").strip().split(",")
    if len(parts) < 2 or "K" not in parts[1]:
        return None
    pts = parts[0].strip()
    if not pts or pts == "N/A":
        return None
    try:
        v = float(pts)
    except ValueError:
        return None
    return v if v >= 0 else None


def _probe_half(per_point: int) -> float:
    """采样窗口半径（秒）：随每点帧数扩大，保证窗口内关键帧填得满每点帧数。"""
    return max(_PROBE_WINDOW_HALF_SEC, 2.0 * per_point)


def _probe_window_positions(duration: float, points: int) -> List[float]:
    """点位均匀分布：points 个位置（单点位取中点）。"""
    if points > 1:
        return [duration * i / (points - 1) for i in range(points)]
    return [duration / 2]


def _probe_keyframes_intervals(video_path: str, duration: float, points: int,
                               per_point: int,
                               stop_event: threading.Event) -> List[float]:
    """按点位窗口采样关键帧时间戳，重叠窗口合并后一次读取，去重排序返回。"""
    if duration <= 0:
        return []
    half = _probe_half(per_point)
    windows = sorted((max(0.0, t - half), min(duration, t + half))
                     for t in _probe_window_positions(duration, points))
    merged: List[List[float]] = []
    for lo, hi in windows:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    cmd = [
        ffmpeg_tools.ffprobe, "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
        "-of", "csv=p=0",
        "-read_intervals", ",".join(f"{lo:.3f}%+{hi - lo:.3f}" for lo, hi in merged),
        video_path,
    ]
    stdout, _ = run_subprocess_with_cancel(cmd, _PROBE_KEYFRAMES_TIMEOUT_SEC, stop_event)
    key_ts = set()
    for raw in (stdout or b"").splitlines():
        v = _kf_from_csv_line(raw)
        if v is not None:
            key_ts.add(v)
    return sorted(key_ts)


def probe_video(video_path: str, stop_event: threading.Event,
                points: int = 5, per_point: int = 3) -> Tuple[Dict[str, Any], List[float]]:
    """分两次 ffprobe 调用：
    1. JSON 获取格式/流元数据（全量条目，NFO 与 ts 配帧均依赖流信息）
    2. 按点位窗口采样关键帧时间戳"""
    meta: Dict[str, Any] = {}
    key_ts: List[float] = []

    try:
        entries = ("format=duration,size,bit_rate,format_name:"
                   "format_tags=creation_time,title,encoder:"
                   "stream=codec_name,codec_type,width,height,r_frame_rate,"
                   "bit_rate,pix_fmt,profile,field_order,duration,"
                   "display_aspect_ratio,color_space,color_range,color_transfer,"
                   "color_primaries,level,"
                   "sample_rate,channels,channel_layout:"
                   "stream_tags=language")
        # 不限定 -select_streams，以获取音频流信息
        meta_cmd = [
            ffmpeg_tools.ffprobe, "-v", "quiet", "-print_format", "json",
            "-show_entries", entries, video_path,
        ]
        stdout, _ = run_subprocess_with_cancel(meta_cmd, _PROBE_META_TIMEOUT_SEC, stop_event)
        if stdout:
            data = json.loads(stdout)
            meta = _parse_metadata(data)

        if stop_event.is_set():
            return meta, []

        key_ts = _probe_keyframes_intervals(video_path, meta.get("duration") or 0.0,
                                            points, per_point, stop_event)
    except Exception as e:
        logger.debug(f"ffprobe 解析失败 ({Path(video_path).name}): {e}", exc_info=True)

    return meta, key_ts


def _extract_single_frame(video_path: str, ts: float, vf: str,
                          stop_event: threading.Event) -> Optional[bytes]:
    """按时间戳抽取单帧：ffmpeg 输出 JPEG 经 stdout 管道返回，失败/取消返回 None。"""
    if stop_event.is_set():
        return None
    cmd = [
        ffmpeg_tools.ffmpeg, "-y",
        "-ss", f"{ts:.3f}",
        "-i", video_path,
        "-vframes", "1",
        "-vf", vf,
        "-q:v", "2",
        "-f", "image2", "pipe:1",
    ]
    stdout, stderr = run_subprocess_with_cancel(cmd, EXTRACT_FRAME_TIMEOUT_SEC, stop_event)
    if stdout and stdout[:2] == b"\xff\xd8":
        return stdout
    if stdout is not None:
        if stderr:
            logger.debug(f"ffmpeg 抽帧返回异常 ({Path(video_path).name}@{ts:.3f}s): "
                         f"{stderr.decode('utf-8', 'ignore')[:200]}")
        else:
            logger.debug(f"ffmpeg 抽帧输出非 JPEG 或过小 ({Path(video_path).name}@{ts:.3f}s, "
                         f"size={len(stdout)} bytes, header={stdout[:4].hex()})")
    return None


def _extract_frame(video_path: str, ts: float, config: Config, stop_event: threading.Event) -> Optional[Frame]:
    if stop_event.is_set():
        return None
    try:
        vf = f"scale=min({config.frame_max_side}\\,iw):min({config.frame_max_side}\\,ih):force_original_aspect_ratio=decrease"
        jpeg = _extract_single_frame(video_path, ts, vf, stop_event)
        if jpeg:
            return Frame(ts, base64.b64encode(jpeg).decode())
        return None
    except Exception as e:
        logger.debug(f"ffmpeg 抽帧异常 ({Path(video_path).name}@{ts:.3f}s): {e}", exc_info=True)
        return None


def _split_jpegs(data: bytes) -> List[bytes]:
    """按 SOI/EOI 魔数切分 image2pipe 串联输出的 JPEG。"""
    out: List[bytes] = []
    i = 0
    while True:
        start = data.find(b"\xff\xd8", i)
        if start < 0:
            break
        end = data.find(b"\xff\xd9", start)
        if end < 0:
            break
        end += 2
        out.append(data[start:end])
        i = end
    return out


def _extract_frames_ts(video_path: str, targets: List[float], config: Config,
                       stop_event: threading.Event,
                       frame_rate: float = 0.0) -> List[Optional[Frame]]:
    """mpegts 专用抽帧：不 seek 单次顺序解码，select 放行目标帧。"""
    if stop_event.is_set():
        return [None] * len(targets)
    # 帧率未知时窗口只需覆盖浮点误差（目标即关键帧精确 pts），过宽会在高帧率下罩住两帧
    win = 0.5 / frame_rate if frame_rate > 1 else 0.005
    wins = "+".join(f"between(t\\,{t - 0.001:.3f}\\,{t + win:.3f})" for t in targets)
    vf = (f"select={wins},"
          f"scale=min({config.frame_max_side}\\,iw):min({config.frame_max_side}\\,ih):force_original_aspect_ratio=decrease")
    cmd = [
        ffmpeg_tools.ffmpeg, "-y", "-i", video_path, "-copyts",
        "-vf", vf, "-fps_mode", "passthrough",
        "-q:v", "2", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    ]
    stdout, _ = run_subprocess_with_cancel(cmd, _EXTRACT_TS_TIMEOUT_SEC, stop_event)
    if not stdout:
        return [None] * len(targets)
    jpegs = _split_jpegs(stdout)
    if len(jpegs) != len(targets):
        logger.debug(f"ts 抽帧配帧不齐 ({Path(video_path).name}): "
                     f"jpegs={len(jpegs)} targets={len(targets)}")
        return [None] * len(targets)
    return [Frame(t, base64.b64encode(j).decode()) for t, j in zip(targets, jpegs)]


def _sparsify_keyframes(key_ts: List[float], min_gap: float) -> List[float]:
    """只保留与上一保留项时间差大于 min_gap 的关键帧。"""
    out: List[float] = []
    for ts in key_ts:
        if not out or ts - out[-1] > min_gap:
            out.append(ts)
    return out


def _anchor_keyframe_indices(n_keyframes: int, points: int) -> List[int]:
    """关键帧充足时，均匀采样出 points 个关键帧索引。"""
    step = (n_keyframes - 1) / (points - 1) if points > 1 else 0
    return [int(round(i * step)) for i in range(points)]


def select_timestamps(duration: float, key_ts: List[float], points: int) -> List[float]:
    """时间戳选取策略：关键帧均匀采样 → 短视频头中尾 → 时长均分 → 兜底 0。"""
    if key_ts:
        if len(key_ts) >= points:
            return [key_ts[i] for i in _anchor_keyframe_indices(len(key_ts), points)]
        return key_ts
    if 0 < duration < _SHORT_VIDEO_THRESHOLD:
        return [0.0, duration / 2.0, max(0.0, duration - 0.1)]
    if duration > 0:
        actual_n = min(points, max(1, int(duration)))
        return [duration * (i + 1) / (actual_n + 1) for i in range(actual_n)]
    return [0.0]


def select_sampled_timestamps(duration: float, key_ts: List[float],
                              points: int, per_point: int) -> List[float]:
    """采样模式选帧：与探测窗口对齐，每个点位取 per_point 个关键帧（窗口内均匀采样）。

    不按索引取全局锚点——关键帧在窗口间分布不均时锚点会被密集窗口挤占，
    按点位分组才能保证点位 × 每点帧数语义。
    """
    if not key_ts or duration <= 0:
        return select_timestamps(duration, key_ts, points)
    half = _probe_half(per_point)
    positions = _probe_window_positions(duration, points)
    key_ts = _sparsify_keyframes(key_ts, _MIN_KEYFRAME_GAP_SEC)
    out: List[float] = []
    seen: set = set()
    for p in positions:
        near = [t for t in key_ts if abs(t - p) <= half]
        if not near:
            continue
        if len(near) <= per_point:
            chosen = near
        elif per_point == 1:
            chosen = [min(near, key=lambda t: abs(t - p))]
        else:
            chosen = [near[i] for i in _anchor_keyframe_indices(len(near), per_point)]
        for t in chosen:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out or select_timestamps(duration, key_ts, points)


def probe_and_extract_keyframes(video_path: str, config: Config, stop_event: threading.Event) -> Tuple[Dict[str, Any], List[Frame]]:
    """探测视频元数据 + 关键帧 → 选取时间戳 → 逐帧抽取为 Frame（时间戳+base64）。"""
    if stop_event.is_set():
        return {}, []
    if not ffmpeg_tools.ffprobe or not ffmpeg_tools.ffmpeg:
        # 依赖未加载/缺失时明确报错，而不是让每个文件都静默"抽帧失败"
        logger.error("ffmpeg/ffprobe 不可用（依赖未加载或未安装），无法处理视频。"
                     "请先调用 ensure_dependencies() 或安装 ffmpeg。")
        return {}, []
    points = max(1, config.sampling_points)
    per_point = max(1, config.frames_per_point)
    info, key_ts = probe_video(video_path, stop_event, points, per_point)
    duration = info.get("duration", 0.0)
    if stop_event.is_set():
        return info, []

    timestamps = select_sampled_timestamps(duration, key_ts, points, per_point)
    frames: List[Frame] = []
    if "mpegts" in (info.get("format_name") or ""):
        fr = (info.get("video") or {}).get("frame_rate") or 0.0
        for f in _extract_frames_ts(video_path, timestamps, config, stop_event, fr):
            if f:
                frames.append(f)
    else:
        for ts in timestamps:
            frame = _extract_frame(video_path, ts, config, stop_event)
            if frame:
                frames.append(frame)
    return info, frames
