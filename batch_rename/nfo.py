"""NFO 媒体信息文件生成。"""
import os
import re
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from xml.etree import ElementTree as ET
from xml.dom import minidom

from .utils import to_long_path
from .env import logger
from .naming import parse_creation_time


# XML 1.0 不允许的字符：U+0000-U+001F 中除 tab(0x09)、LF(0x0a)、CR(0x0d) 外
_ILLEGAL_XML_CHARS = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")

# 隔行扫描场序标识
_INTERLACE_ORDERS = frozenset({"tt", "bb", "tb", "bt", "tff", "bff"})


def _safe_xml_text(text: str) -> str:
    """清洗非法 XML 控制字符。"""
    if not text:
        return ""
    return _ILLEGAL_XML_CHARS.sub("", text)


# ── 流字段映射表：(源字段, XML标签, 格式化函数) ──
_VIDEO_FIELD_MAP = [
    ("codec", "codec", _safe_xml_text),
    ("width", "width", lambda v: str(v)),
    ("height", "height", lambda v: str(v)),
    ("frame_rate", "framerate", lambda v: f"{v:.3f}"),
    ("bit_rate", "bitrate", lambda v: str(v // 1000)),
    ("pix_fmt", "pixfmt", _safe_xml_text),
    ("profile", "profile", _safe_xml_text),
    ("duration", "durationinseconds", lambda v: str(int(v))),
    ("display_aspect_ratio", "aspect", _safe_xml_text),
    ("color_space", "colorspace", _safe_xml_text),
    ("color_range", "colorrange", _safe_xml_text),
    ("color_transfer", "colortransfer", _safe_xml_text),
    ("color_primaries", "colorprimaries", _safe_xml_text),
    ("level", "level", lambda v: str(v)),
]

_AUDIO_FIELD_MAP = [
    ("codec", "codec", _safe_xml_text),
    ("channels", "channels", lambda v: str(v)),
    ("sample_rate", "samplerate", lambda v: str(v)),
    ("channel_layout", "channellayout", _safe_xml_text),
    ("language", "language", _safe_xml_text),
    ("bit_rate", "bitrate", lambda v: str(v // 1000)),
]


def _add_stream_fields(parent: ET.Element, stream: Dict[str, Any], field_map) -> None:
    """按映射表生成子元素（值为 None 或空串时跳过，0 视为有效值）。"""
    for src_key, xml_tag, fmt in field_map:
        val = stream.get(src_key)
        if val is not None and val != "":
            ET.SubElement(parent, xml_tag).text = fmt(val)


def _build_nfo_xml(title: str, plot: str, tags: List[str], info: Dict[str, Any],
                    original_name: str) -> str:
    """使用 xml.etree.ElementTree 构建 NFO XML 内容并序列化为字符串。"""
    root = ET.Element("movie")

    ET.SubElement(root, "title").text = _safe_xml_text(title)
    ET.SubElement(root, "plot").text = _safe_xml_text(plot)

    duration = info.get("duration", 0.0)
    if duration > 0:
        ET.SubElement(root, "runtime").text = str(int(duration / 60))

    ET.SubElement(root, "originaltitle").text = _safe_xml_text(original_name)

    ct = info.get("creation_time", "")
    if ct:
        dt = parse_creation_time(ct)
        if dt:
            local_dt = dt.astimezone()
            ET.SubElement(root, "premiered").text = local_dt.strftime("%Y-%m-%d")
            ET.SubElement(root, "year").text = str(local_dt.year)

    video = info.get("video", {})
    audio_list = info.get("audio", [])

    if video:
        fileinfo = ET.SubElement(root, "fileinfo")
        streamdetails = ET.SubElement(fileinfo, "streamdetails")
        ve = ET.SubElement(streamdetails, "video")
        _add_stream_fields(ve, video, _VIDEO_FIELD_MAP)
        fo = video.get("field_order") or ""
        scantype = "Interlaced" if (fo and fo.lower() in _INTERLACE_ORDERS) else "Progressive"
        ET.SubElement(ve, "scantype").text = scantype

        for audio in audio_list:
            ae = ET.SubElement(streamdetails, "audio")
            _add_stream_fields(ae, audio, _AUDIO_FIELD_MAP)

    for tag in tags:
        ET.SubElement(root, "tag").text = _safe_xml_text(tag)

    rough = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="    ", encoding="utf-8").decode("utf-8")


def write_nfo(video_path: str, title: str, plot: str, tags: List[str],
              info: Dict[str, Any], original_name: str,
              target_dir: Optional[str] = None,
              nfo_name: Optional[str] = None) -> str:
    """写入 NFO 文件；target_dir 指定缓存目录时可通过 nfo_name 指定唯一文件名。
    返回最终路径，失败返回空串。
    """
    try:
        vp = Path(video_path)
        if target_dir:
            target_dir_path = Path(target_dir)
            target_dir_path.mkdir(parents=True, exist_ok=True)
            target = str(target_dir_path / (nfo_name or f"{vp.stem}.nfo"))
        else:
            target = str(vp.with_suffix(".nfo"))
        safe_target = to_long_path(target)
        content = _build_nfo_xml(title, plot, tags, info, original_name)
        # 原子写入：临时文件 + os.replace
        tmp = f"{safe_target}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, safe_target)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

        size_mb = info.get("size", 0) / (1024 * 1024)
        video = info.get("video", {})
        codec_info = f'{video.get("codec") or "?"}/{video.get("height") or "?"}p' if video else ""
        audio_count = len(info.get("audio", []))
        audio_info = f"{audio_count}音轨" if audio_count else ""
        size_info = f'{size_mb:.0f}MB' if size_mb > 0 else ""
        extra = " | ".join(filter(None, [codec_info, audio_info, size_info]))
        logger.debug(f"[NFO] 已生成: {Path(target).name}" + (f" ({extra})" if extra else ""))
        return target
    except Exception as e:
        logger.warning(f"[NFO] 生成失败: {e}")
        return ""
