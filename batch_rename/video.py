"""视频探测、关键帧抽取。"""
import bisect
import json
import base64
import os
import subprocess
import tempfile
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from .env import logger, SUBPROCESS_KWARGS
from .subprocess_registry import register_subprocess, unregister_subprocess
from .dependencies import ffmpeg_tools
from .config import Config
from .preprocess import preprocess_frames
from .utils import safe_int, safe_float
from .types import Frame


# ── 抽帧策略常量 ──
_KF_PAIR_FRAMES = 8            # 目标后此帧数内还有关键帧时 concat 段会多吐帧（B 帧 dts 滞后），跳过 concat
_PER_FRAME_WORKERS = 4         # 逐帧抽取的并发进程数

# ── 子进程超时（秒）──
_PROBE_META_TIMEOUT_SEC = 15.0    # ffprobe 元数据 JSON 探测
_PROBE_KEYFRAMES_TIMEOUT_SEC = 60.0  # 关键帧窗口采样探测超时
_EXTRACT_FRAME_TIMEOUT_SEC = 30.0    # ffmpeg 单帧抽取
_EXTRACT_CONCAT_TIMEOUT_SEC = 120.0  # ffmpeg concat 多段一次出全部帧

# ── 点位分区半径（秒）──
_PROBE_WINDOW_HALF_MIN = 5.0    # 基础半径
_PROBE_WINDOW_HALF_MAX = 60.0    # 半径上限


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
        "duration": 0.0, "start_time": 0.0, "creation_time": "",
        "size": 0, "bit_rate": 0, "format_name": "",
        "encoder": "", "container_title": "",
        "video": {}, "audio": [],
    }
    meta["duration"] = safe_float(fmt.get("duration"))
    # 流时间基（探测边界换算用）
    meta["start_time"] = max(0.0, safe_float(fmt.get("start_time")))
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


def _point_positions(duration: float, points: int) -> List[float]:
    """点位均匀分布：points 个位置（单点位取中点）。"""
    if points > 1:
        return [duration * i / (points - 1) for i in range(points)]
    return [duration / 2]


def _point_regions(duration: float, points: int) -> List[Tuple[int, float, float, float]]:
    """点位互斥分区：返回 (索引, 位置, 起点, 终点)，相邻分区只共享边界。"""
    positions = _point_positions(duration, points)
    spacing = duration / (points - 1) if points > 1 else duration
    radius = min(spacing / 2.0, _PROBE_WINDOW_HALF_MAX,
                 max(_PROBE_WINDOW_HALF_MIN, spacing / 6))
    return [(i, p, max(0.0, p - radius), min(duration, p + radius))
            for i, p in enumerate(positions)]


def _probe_csv(video_path: str, spec: str,
               stop_event: threading.Event) -> Optional[bytes]:
    """按 read_intervals 读取 packet CSV；失败/取消返回 None。"""
    cmd = [
        ffmpeg_tools.ffprobe, "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
        "-of", "csv=p=0",
        "-read_intervals", spec,
        video_path,
    ]
    stdout, _ = run_subprocess_with_cancel(cmd, _PROBE_KEYFRAMES_TIMEOUT_SEC, stop_event)
    return stdout


def _probe_interval_anchors(video_path: str, merged: List[List[float]],
                            stop_event: threading.Event) -> Optional[List[float]]:
    """极短读取定位各区间实际起读的关键帧锚点（read_intervals 时长以回退
    关键帧为锚）；行无法按区间对齐返回 None。"""
    spec = ",".join(f"{lo:.3f}%+0.001" for lo, _ in merged)
    stdout = _probe_csv(video_path, spec, stop_event)
    pts: List[float] = []
    for ln in (stdout or b"").splitlines():
        if not ln.strip():
            continue
        try:
            pts.append(float(ln.decode("utf-8", "replace").split(",")[0]))
        except (ValueError, IndexError):
            return None
    out: List[float] = []
    i = 0
    for lo, _ in merged:
        if i >= len(pts):
            return None
        out.append(pts[i])      # 每区间至少一行
        i += 1
        while i < len(pts) and pts[i] <= lo + 0.05:   # 同簇重复 pts 归入本区间
            i += 1
    if i != len(pts):
        return None
    return out


def _probe_keyframes_intervals(video_path: str, duration: float, points: int,
                               stop_event: threading.Event,
                               start_time: float = 0.0) -> List[float]:
    """按点位分区采样关键帧时间戳（内容基准），相邻分区合并后读取，去重排序返回。"""
    if duration <= 0:
        return []
    windows = [(lo + start_time, hi + start_time)
               for _, _, lo, hi in _point_regions(duration, points)]
    merged: List[List[float]] = []
    for lo, hi in windows:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    if len(merged) == 1:
        # 单区间整段直读
        spec = f"{merged[0][0]:.3f}%+{merged[0][1] - merged[0][0]:.3f}"
    else:
        # 先定位锚点再从锚点精读
        anchors = _probe_interval_anchors(video_path, merged, stop_event)
        if anchors is not None:
            spec = ",".join(f"{k:.3f}%+{max(0.06, hi - k + 0.05):.3f}"
                            for k, (_, hi) in zip(anchors, merged))
        else:
            # 锚点定位失败兜底：时长翻倍
            spec = ",".join(f"{lo:.3f}%+{(hi - lo) * 2:.3f}" for lo, hi in merged)
    stdout = _probe_csv(video_path, spec, stop_event)
    key_ts = set()
    for raw in (stdout or b"").splitlines():
        v = _kf_from_csv_line(raw)
        if v is not None:
            key_ts.add(v)
    return sorted(t - start_time for t in key_ts)


def probe_video(video_path: str, stop_event: threading.Event,
                points: int) -> Tuple[Dict[str, Any], List[float]]:
    """分两次 ffprobe 调用：
    1. JSON 获取格式/流元数据（全量条目，NFO 依赖流信息）
    2. 按点位分区采样关键帧时间戳（内容基准）"""
    meta: Dict[str, Any] = {}
    key_ts: List[float] = []

    try:
        entries = ("format=duration,start_time,size,bit_rate,format_name:"
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
                                            points, stop_event,
                                            meta.get("start_time") or 0.0)
    except Exception as e:
        logger.debug(f"ffprobe 解析失败 ({Path(video_path).name}): {e}", exc_info=True)

    return meta, key_ts


def _extract_single_frame(video_path: str, ts: float, vf: str,
                          stop_event: threading.Event) -> Optional[bytes]:
    """按内容时间戳抽取单帧：ffmpeg 输出 JPEG 经 stdout 管道返回，失败/取消返回 None。"""
    if stop_event.is_set():
        return None
    cmd = [
        ffmpeg_tools.ffmpeg, "-y",
        "-ss", f"{ts:.6f}",
        "-i", video_path,
        "-vframes", "1",
        "-vf", vf,
        "-q:v", "2",
        "-f", "image2", "pipe:1",
    ]
    stdout, stderr = run_subprocess_with_cancel(cmd, _EXTRACT_FRAME_TIMEOUT_SEC, stop_event)
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


def _default_frame_vf(config: Config) -> str:
    """默认抽帧滤镜：等比缩到长边不超过 frame_max_side。"""
    return (f"scale=min({config.frame_max_side}\\,iw):min({config.frame_max_side}\\,ih)"
            f":force_original_aspect_ratio=decrease")


def _extract_frame(video_path: str, ts: float, point_index: int, config: Config,
                   stop_event: threading.Event,
                   frame_vf: Optional[str] = None) -> Optional[Frame]:
    jpeg = _extract_single_frame(video_path, ts, frame_vf or _default_frame_vf(config),
                                 stop_event)
    if jpeg:
        return Frame(ts, base64.b64encode(jpeg).decode(), point_index)
    return None


def _extract_per_frame(video_path: str, samples: List[Tuple[float, int]], config: Config,
                       stop_event: threading.Event,
                       frame_vf: Optional[str] = None) -> List[Frame]:
    """逐帧抽取兜底：并发单帧进程，结果按样本顺序保留；单帧失败/取消跳过。"""
    with futures.ThreadPoolExecutor(max_workers=_PER_FRAME_WORKERS) as ex:
        results = list(ex.map(
            lambda s: _extract_frame(video_path, s[0], s[1], config, stop_event,
                                     frame_vf=frame_vf),
            samples))
    return [f for f in results if f]


def _frame_tau(frame_rate: float) -> float:
    """帧间隔估计；帧率未知取 0.033s。"""
    return 1.0 / frame_rate if frame_rate > 1 else 0.033


def _concat_vulnerable(key_ts: List[float], timestamps: List[float],
                        frame_rate: float) -> bool:
    """目标关键帧后极近处还有关键帧（编码器成对插入）时，concat 的 outpoint 按 dts
    截断会漏进相邻关键帧导致配帧不齐，此类目标直接走逐帧。"""
    limit = _KF_PAIR_FRAMES * _frame_tau(frame_rate)
    for t in timestamps:
        i = bisect.bisect_right(key_ts, t)
        if i < len(key_ts) and key_ts[i] - t <= limit:
            return True
    return False


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


def _pair_samples(video_path: str, samples: List[Tuple[float, int]],
                 jpegs: List[bytes], kind: str) -> Optional[List[Frame]]:
    """样本与 JPEG 按序配对成 Frame；数量不齐返回 None（调用方回退逐帧）。"""
    if len(jpegs) != len(samples):
        logger.debug(f"{kind} 抽帧配帧不齐，回退逐帧 ({Path(video_path).name}): "
                     f"jpegs={len(jpegs)} targets={len(samples)}")
        return None
    return [Frame(t, base64.b64encode(j).decode(), pi)
            for (t, pi), j in zip(samples, jpegs)]


def _write_ffconcat_script(video_path: str, segments) -> str:
    """segments: [(inpoint, outpoint)] → ffconcat 脚本。"""
    # 路径内的 ' 以 '\'' 拼接转义
    safe_path = video_path.replace("'", "'\\''")
    lines = ["ffconcat version 1.0"]
    for lo, hi in segments:
        lines.append(f"file '{safe_path}'")
        lines.append(f"inpoint {lo:.6f}")
        lines.append(f"outpoint {hi:.6f}")
    fd, path = tempfile.mkstemp(suffix=".ffconcat")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        os.unlink(path)
        raise
    return path


def _extract_frames_concat(video_path: str, samples: List[Tuple[float, int]], config: Config,
                           stop_event: threading.Event, frame_rate: float,
                           frame_vf: Optional[str] = None,
                           start_time: float = 0.0) -> Optional[List[Frame]]:
    """关键帧点位单进程多段抽取：出帧数与样本一一配对成功才交付，否则 None 回退逐帧。"""
    if stop_event.is_set():
        return None
    # ffconcat in/outpoint 按流 pts 解释，内容时间需换算
    epochs = [(t + start_time, t + start_time + 0.5 * _frame_tau(frame_rate))
              for t, _ in samples]
    script = _write_ffconcat_script(video_path, epochs)
    try:
        cmd = [
            ffmpeg_tools.ffmpeg, "-y",
            "-threads", "1",  # 解码单线程：多线程与 concat 段丢弃有竞态，偶发中途崩溃
            "-skip_frame", "nokey",
            "-f", "concat", "-safe", "0", "-i", script,
            "-fps_mode", "passthrough",
            "-vf", frame_vf or _default_frame_vf(config),
            "-q:v", "2", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ]
        stdout, _ = run_subprocess_with_cancel(cmd, _EXTRACT_CONCAT_TIMEOUT_SEC, stop_event)
        if not stdout:
            return None
        return _pair_samples(video_path, samples, _split_jpegs(stdout), "concat")
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass


def _anchor_keyframe_indices(n_keyframes: int, points: int) -> List[int]:
    """关键帧充足时，均匀采样出 points 个关键帧索引。"""
    step = (n_keyframes - 1) / (points - 1) if points > 1 else 0
    return [int(round(i * step)) for i in range(points)]


def _fill_positions(p: float, lo: float, hi: float, per_point: int,
                    end: float) -> List[float]:
    """分区内无关键帧时的普通帧位置：单点取点位，多点取分区内均匀位。"""
    if per_point == 1:
        return [min(p, end)]
    step = (hi - lo) / (per_point + 1)
    return [min(lo + step * (i + 1), end) for i in range(per_point)]


def select_point_samples(duration: float, key_ts: List[float],
                         points: int, per_point: int,
                         frame_rate: float = 0.0) -> List[Tuple[float, int]]:
    """按点位分区选帧：分区内优先关键帧，无关键帧补普通帧位置。

    返回 (时间戳, 点位索引)，时间升序、点位索引单调不减；key_ts 需升序传入。
    duration 缺失时：有关键帧均匀采样，否则取首帧。
    """
    if duration <= 0:
        if not key_ts:
            return [(0.0, 0)]
        if len(key_ts) < points:
            return [(t, i) for i, t in enumerate(key_ts)]
        return [(key_ts[i], r)
                for r, i in enumerate(_anchor_keyframe_indices(len(key_ts), points))]
    # 片尾内收一个帧间隔，防越过末帧
    end = max(0.0, duration - _frame_tau(frame_rate))
    regions = _point_regions(duration, points)
    out: List[Tuple[float, int]] = []
    seen: set = set()
    for pi, p, lo, hi in regions:
        # 分区只在边界可能重合；已被前一点位取走的边界帧不重复使用
        near = [t for t in key_ts if lo <= t <= hi and t not in seen]
        if not near:
            chosen = _fill_positions(p, lo, hi, per_point, end)
        elif len(near) <= per_point:
            chosen = near
        elif per_point == 1:
            chosen = [min(near, key=lambda t: abs(t - p))]
        else:
            chosen = [near[i] for i in _anchor_keyframe_indices(len(near), per_point)]
        seen.update(chosen)
        out.extend((t, pi) for t in chosen)
    return out


def probe_and_extract_keyframes(video_path: str, config: Config, stop_event: threading.Event,
                                frame_vf: Optional[str] = None) -> Tuple[Dict[str, Any], List[Frame]]:
    """探测视频元数据 + 关键帧 → 选点 → 抽取为 Frame（时间戳+base64+点位索引）。

    Frame.ts 与 key_ts 均为内容相对时间；frame_vf 传入时替换默认缩放滤镜；
    抽帧后执行公用预处理。
    """
    if stop_event.is_set():
        return {}, []
    if not ffmpeg_tools.ffprobe or not ffmpeg_tools.ffmpeg:
        # 依赖未加载/缺失时明确报错，而不是让每个文件都静默"抽帧失败"
        logger.error("ffmpeg/ffprobe 不可用（依赖未加载或未安装），无法处理视频。"
                     "请先调用 ensure_dependencies() 或安装 ffmpeg。")
        return {}, []
    points = config.sampling_points
    per_point = config.frames_per_point
    info, key_ts = probe_video(video_path, stop_event, points)
    # 视频流时长兜底
    duration = info.get("duration") or (info.get("video") or {}).get("duration") or 0.0
    start_time = info.get("start_time", 0.0)
    if stop_event.is_set():
        return info, []

    frame_rate = (info.get("video") or {}).get("frame_rate") or 0.0
    samples = select_point_samples(duration, key_ts, points, per_point, frame_rate)
    timestamps = [t for t, _ in samples]
    mode = config.frame_extract_mode
    key_set = set(key_ts)
    aligned = bool(key_ts) and all(t in key_set for t in timestamps)

    # ── 抽帧路由 ──
    if mode == "auto" and key_ts and aligned \
            and not _concat_vulnerable(key_ts, timestamps, frame_rate):
        # auto：关键帧点位单进程多段，配帧不齐回退逐帧
        frames = _extract_frames_concat(video_path, samples, config, stop_event,
                                        frame_rate, frame_vf=frame_vf,
                                        start_time=start_time) or []
        if not frames:
            frames = _extract_per_frame(video_path, samples, config, stop_event,
                                         frame_vf=frame_vf)
    else:
        # per_frame / 兜底点位 / 成对关键帧：逐帧
        frames = _extract_per_frame(video_path, samples, config, stop_event,
                                     frame_vf=frame_vf)

    if not stop_event.is_set():
        frames = preprocess_frames(frames)
    return info, frames
