"""文件名处理：清洗、日期提取、重命名、stem 构建。"""
import os
import re
import datetime
import platform
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from .env import logger
from .utils import to_long_path, path_exists, path_stat, rename_file

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .config import Config


_WIN_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_WIN_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
})

_MIN_VALID_YEAR = 2000          # 早于此的日期视为占位/无效
_MIN_VALID_TIMESTAMP = 100_000_000  # 1973-03-03，早于此视为无效
_MAX_COLLISION_SUFFIX = 100
_MAX_STEM_PART_CHARS = 50

_rename_lock = threading.Lock()

# ffprobe 返回的 creation_time 格式（ISO8601 / QuickTime 等）
_CREATION_TIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y:%m:%d %H:%M:%S",
)


def parse_creation_time(s: str) -> Optional[datetime.datetime]:
    """解析 creation_time，无效或早于 _MIN_VALID_YEAR 返回 None；无时区按 UTC。"""
    if not s:
        return None
    ct = s.replace("Z", "+00:00")
    for fmt in _CREATION_TIME_FORMATS:
        try:
            dt = datetime.datetime.strptime(ct, fmt)
            if dt.year >= _MIN_VALID_YEAR:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                return dt
        except ValueError:
            continue
    return None


def extract_date_str(video_path: str, creation_time: str) -> str:
    """日期来源优先级：视频元数据 creation_time → 文件时间戳 → 当前时间。"""
    dt = parse_creation_time(creation_time)
    if dt is not None:
        # creation_time 为 UTC，转本地时区
        return dt.astimezone().strftime("%Y%m%d-%H%M")
    try:
        st = path_stat(video_path)
        # 优先 st_birthtime（macOS/BSD），否则取较早的 mtime/ctime
        birth = getattr(st, "st_birthtime", None)
        if birth and birth > _MIN_VALID_TIMESTAMP:
            ts = birth
        else:
            ts = min(st.st_mtime, st.st_ctime)
        if ts > _MIN_VALID_TIMESTAMP:
            return datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d-%H%M")
    except Exception as e:
        logger.debug(f"读取文件时间戳失败 ({Path(video_path).name}): {e}")
    return datetime.datetime.now().strftime("%Y%m%d-%H%M")


def sanitize_filename(text: str, max_chars: Optional[int] = _MAX_STEM_PART_CHARS) -> str:
    """清洗文件名：非法字符替换、空白折叠、截断、保留名检查。"""
    text = _WIN_ILLEGAL.sub('_', text)
    text = re.sub(r'\s+', '_', text.strip())
    text = re.sub(r'_+', '_', text)
    text = text.strip('. _-')
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    text = text.rstrip('. _-')
    if platform.system() == "Windows":
        if text.upper() in _WIN_RESERVED_NAMES:
            text = text + "_"
    return text or "untitled"


def resolve_collision(dest_dir: Path, stem: str, suffix: str, src_long: str,
                      max_try: int = _MAX_COLLISION_SUFFIX) -> Tuple[Optional[Path], str]:
    """解析不冲突的目标路径，返回 (Path, 状态)：ok / skipped / error。"""
    target = dest_dir / f"{stem}{suffix}"
    if not path_exists(target):
        return target, "ok"
    # normcase：Windows 大小写不敏感比较
    if os.path.normcase(to_long_path(str(target))) == os.path.normcase(src_long):
        return target, "skipped"
    for counter in range(1, max_try + 1):
        candidate = dest_dir / f"{stem}_{counter}{suffix}"
        if not path_exists(candidate):
            return candidate, "ok"
    return None, "error"


def rename_video(video_path: str, new_stem: str, config: "Config",
                 err_out: Optional[List[str]] = None) -> Tuple[str, str]:
    """加锁重命名：目标已存在则加 _1/_2 后缀避让，同名跳过。"""
    p = Path(video_path)
    with _rename_lock:
        src_long = to_long_path(str(p))
        target, status = resolve_collision(p.parent, new_stem, p.suffix, src_long)
        if status == "skipped":
            logger.debug(f"[跳过] 文件名未变化: {p.name}")
            return str(target), "skipped"
        if status == "error":
            logger.error(f"重命名失败: {p.name}")
            return video_path, "error"

        try:
            orig_stat = path_stat(p)
        except Exception as e:
            logger.debug(f"读取原文件 stat 失败（将不保留时间戳）: {e}")
            orig_stat = None
        ok, err = rename_file(str(p), str(target))
        if not ok:
            logger.error(f"重命名失败: {p.name}")
            if err_out is not None:
                err_out.append(err)
            return video_path, "error"
        logger.debug(f"[改名] {p.name} -> {target.name}")
        if orig_stat:
            try:
                os.utime(to_long_path(str(target)), (orig_stat.st_atime, orig_stat.st_mtime))
            except Exception as e:
                logger.debug(f"保留时间戳失败: {e}")

        _rename_companion_files(p, target)

        return str(target), "ok"


_COMPANION_SUFFIXES = (".nfo", ".srt", ".zh.srt")


def _rename_companion_files(old_video: Path, new_video: Path,
                            suffixes: Tuple[str, ...] = _COMPANION_SUFFIXES) -> None:
    """同步重命名同目录下的伴随文件。"""
    old_stem = old_video.stem
    new_stem = new_video.stem
    parent = new_video.parent

    for suffix in suffixes:
        old_file = parent / f"{old_stem}{suffix}"
        if not old_file.is_file():
            continue
        new_file = parent / f"{new_stem}{suffix}"
        try:
            if new_file.exists():
                logger.debug(f"[伴随] 目标已存在，跳过: {new_file.name}")
                continue
            os.replace(to_long_path(str(old_file)), to_long_path(str(new_file)))
            logger.debug(f"[伴随] {old_file.name} -> {new_file.name}")
        except Exception as e:
            logger.debug(f"[伴随] 重命名失败 {old_file.name}: {e}")


def _expand_counter(s: str, counter: int) -> str:
    """把替换结果中的 ${...} 编号占位符展开为实际编号。"""
    def repl(m):
        spec = m.group(1).strip()
        padding, start, increment = 2, 1, 1
        if spec != "n":
            for kv in spec.split(";"):
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                k, v = k.strip(), v.strip()
                if k == "padding" and v.isdigit():
                    padding = int(v)
                elif k == "start" and v.lstrip("-").isdigit():
                    start = int(v)
                elif k == "increment" and v.lstrip("-").isdigit():
                    increment = int(v)
        return str(start + (counter - 1) * increment).zfill(padding)
    return re.sub(r"\$\{([^}]*)\}", repl, s)


_WORD_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


def _case_transform(s: str, case_mode: str) -> str:
    if case_mode == "upper":
        return s.upper()
    if case_mode == "lower":
        return s.lower()
    if case_mode in ("title", "capitalized"):
        lower_rest = case_mode == "title"
        def cap(m):
            w = m.group(0)
            return w[0].upper() + (w[1:].lower() if lower_rest else w[1:])
        return _WORD_RE.sub(cap, s)
    return s


def apply_manual_transform(stem: str, mode: str, text: str, text2: str = "",
                           use_regex: bool = False,
                           counter: Optional[int] = None,
                           match_all: bool = True,
                           case_mode: str = "") -> Tuple[str, str]:
    """手动变换文件名主干：prefix / suffix / remove / replace。
    返回 (新stem, 错误信息)。"""
    count = 1 if not match_all else 0   # 0=全部替换
    if mode == "prefix":
        raw = (text or "").lstrip() + stem
    elif mode == "suffix":
        raw = stem + (text or "").rstrip()
    elif mode == "remove":
        text = (text or "").strip()
        if not text:
            return stem, "请输入要删除的文本"
        if use_regex:
            try:
                raw = re.sub(text, "", stem, count=count)
            except re.error as e:
                return stem, f"正则表达式错误: {e}"
        else:
            raw = stem.replace(text, "", -1 if match_all else 1)
    elif mode == "replace":
        text = (text or "").strip()
        if not text:
            raw = stem   # 空查找 = 不替换，仅整体变换（大小写等）
        elif use_regex:
            try:
                raw = re.sub(text, text2 or "", stem, count=count)
            except re.error as e:
                return stem, f"正则表达式错误: {e}"
        else:
            raw = stem.replace(text, text2 or "", -1 if match_all else 1)
        if counter is not None:
            raw = _expand_counter(raw, counter)
    else:
        return stem, f"未知模式: {mode}"
    if not raw.strip():
        return stem, "重命名结果为空"
    # 清洗替换引入的非法字符与首尾空白/点
    new_stem = _WIN_ILLEGAL.sub('_', raw).strip().rstrip(' .')
    if not new_stem:
        return stem, "重命名结果为空"
    if case_mode:
        new_stem = _case_transform(new_stem, case_mode)
    if platform.system() == "Windows" and new_stem.upper() in _WIN_RESERVED_NAMES:
        return stem, "结果与系统保留名冲突"
    return new_stem, ""


_SUB_LANG_TAGS = ("简体", "繁体", "簡體", "繁體", "简", "繁", "chs", "cht", "sc", "tc",
                  "zh-hans", "zh-hant", "zh", "big5", "gb", "gbk", "gb2312",
                  "en", "eng", "ja", "jpn", "ko", "kor")
_SUB_LANG_RE = re.compile(r"\.(" + "|".join(
    sorted(_SUB_LANG_TAGS, key=len, reverse=True)) + r")$", re.IGNORECASE)
_NAT_SPLIT_RE = re.compile(r"(\d+)")


def _nat_sort_key(s: str) -> List[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in _NAT_SPLIT_RE.split(s)]


def strip_sub_lang(stem: str) -> Tuple[str, str]:
    """剥掉字幕名末尾的语言标记（可多层，如 a.zh.zh），返回 (剩余名, 标记串)。"""
    langs = []
    while True:
        m = _SUB_LANG_RE.search(stem)
        if not m:
            break
        langs.append(m.group(1))
        stem = stem[:m.start()]
    return stem, ".".join(reversed(langs))


def match_subtitle_files(video_paths: List[str], sub_paths: List[str]):
    """两级匹配字幕：L1 精确同名，L2 自然排序配对。返回 [(视频下标, [(字幕路径, 语言标记)])]。"""
    videos = [Path(p).stem for p in video_paths]
    subs = []
    for sp in sub_paths:
        stem = Path(sp).stem
        key, lang = strip_sub_lang(stem)
        subs.append((stem, key, lang, str(sp)))
    used_v = [False] * len(videos)
    used_s = [False] * len(subs)
    pairs: List[Tuple[int, List[Tuple[str, str]]]] = []
    # L1：去语言标记后与视频名完全一致
    for i, v in enumerate(videos):
        matched = [(s, j) for j, s in enumerate(subs) if not used_s[j] and s[1] == v]
        if not matched:
            continue
        matched.sort(key=lambda t: _nat_sort_key(t[0][0]))
        pairs.append((i, [(s[3], s[2]) for s, _ in matched]))
        used_v[i] = True
        for _, j in matched:
            used_s[j] = True
    # L2：剩余字幕按去标记键分组，与剩余视频顺序配对
    rest_v = [i for i in range(len(videos)) if not used_v[i]]
    rest_v.sort(key=lambda i: _nat_sort_key(videos[i]))
    rest_s = [s for j, s in enumerate(subs) if not used_s[j]]
    rest_s.sort(key=lambda s: (_nat_sort_key(s[1]), _nat_sort_key(s[0])))
    groups: List[Tuple[str, List[Tuple[str, str]]]] = []
    for s in rest_s:
        if groups and groups[-1][0] == s[1]:
            groups[-1][1].append((s[3], s[2]))
        else:
            groups.append((s[1], [(s[3], s[2])]))
    for i, (_, sl) in zip(rest_v, groups):
        pairs.append((i, sl))
    return pairs


def build_new_stem(vp: str, info: Dict[str, Any], title: str, config: "Config") -> str:
    """根据配置构建新文件名 stem（不含扩展名）。"""
    parts = []
    if config.include_date:
        parts.append(extract_date_str(vp, info.get("creation_time", "")))
    parts.append(sanitize_filename(title, _MAX_STEM_PART_CHARS))
    if config.include_original:
        parts.append(sanitize_filename(Path(vp).stem, _MAX_STEM_PART_CHARS))
    return "_".join(parts)
