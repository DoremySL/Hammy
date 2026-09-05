"""models_downloader.py — 模型搜索与下载：HF-Mirror / ModelScope 双源 + 通用直连下载。"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .installer import start_cancel_watcher

# ── 全局配置 ──
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
MS_ENDPOINT = os.environ.get("MODELSCOPE_ENDPOINT", "https://modelscope.cn").rstrip("/")
DEFAULT_TIMEOUT = 30          # 网络/单请求超时(秒)
DEFAULT_RETRIES = 3
DOWNLOAD_CHUNK = 1024 * 256
USER_AGENT = "video-rename-gui"

# 单文件分片下载（多连接并行，仅大文件启用）
CHUNK_MIN_SIZE = 200 * 1024 * 1024
CHUNK_WORKERS = 6
CHUNK_SEG_SIZE = 64 * 1024 * 1024
CHUNK_MAX_SEGS = 36                    # 段数上限（取并发数的整数倍）
CHUNK_TIMEOUT = 30                     # 分片连接超时(秒)

_O_BINARY = getattr(os, "O_BINARY", 0)


class HttpError(Exception):
    """HTTP 请求/解析失败（重试后仍失败）。"""


class DownloadCancelled(Exception):
    """下载被用户取消。"""


class ChunkedFallback(Exception):
    """分片下载不可用（服务端忽略 Range），回退单流。"""


class _SegmentIncomplete(Exception):
    """单段未下满即连接提前关闭，触发该段重试。"""


# ── 并发保护：同一时刻只允许一个下载任务 ──
_dl_lock = threading.Lock()
_dl_cancel = threading.Event()
_dl_active = False
_tree_cache: Dict[str, List[Dict[str, Any]]] = {}  # 仓库文件树会话内缓存


def get_cancel_event() -> threading.Event:
    return _dl_cancel


def begin_download() -> bool:
    """尝试占坑：True = 成功开始下载；False = 已有任务进行中。"""
    global _dl_active
    with _dl_lock:
        if _dl_active:
            return False
        _dl_active = True
        _dl_cancel.clear()
        return True


def end_download() -> None:
    global _dl_active
    with _dl_lock:
        _dl_active = False


def set_cancel() -> None:
    _dl_cancel.set()


# ============================== 工具函数 ==============================
def _human_size(num: float) -> str:
    """字节数 -> 人类可读 (B/KB/MB/GB)。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def split_repo_id(repo_id: str) -> Tuple[str, str]:
    """仓库 ID -> (author, repo_name)。"""
    if "/" in repo_id:
        author, _, name = repo_id.partition("/")
        return author, name
    return "", repo_id


def _safe_relpath(path: str) -> bool:
    """路径成分检查：拒绝空段 / . / .. / 盘符 / 反斜杠，防止越出目标目录。"""
    return all(p and p not in (".", "..") and ":" not in p and "\\" not in p
               for p in path.split("/"))


# ============================== HTTP 请求层 ==============================
def _headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    h = {"User-Agent": USER_AGENT}
    if extra:
        h.update(extra)
    return h


def _retry_request(req: urllib.request.Request,
                   timeout: int,
                   cancel_event: Optional[threading.Event],
                   out_headers: Optional[Dict[str, str]] = None):
    """带重试退避的 urlopen，返回响应对象（调用方负责 with 关闭）。"""
    last_err = None
    for attempt in range(1, DEFAULT_RETRIES + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelled()
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            if out_headers is not None:
                out_headers.update(dict(resp.headers))
            return resp
        except DownloadCancelled:
            raise
        except Exception as e:  # noqa
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelled()
            last_err = e
            time.sleep(1.0 * attempt)
    raise HttpError(f"请求失败({DEFAULT_RETRIES}次重试后): {getattr(req, 'full_url', req)}\n原因: {last_err}")


def get_json(url: str, params: Optional[Dict[str, Any]] = None,
             timeout: int = DEFAULT_TIMEOUT,
             cancel_event: Optional[threading.Event] = None,
             out_headers: Optional[Dict[str, str]] = None) -> Any:
    """GET 请求并解析 JSON。"""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    req = urllib.request.Request(url, headers=_headers())
    with _retry_request(req, timeout, cancel_event, out_headers) as r:
        return json.loads(r.read().decode("utf-8"))


def put_json(url: str, body: Dict[str, Any],
             timeout: int = DEFAULT_TIMEOUT,
             cancel_event: Optional[threading.Event] = None) -> Any:
    """PUT 请求（JSON body）并解析 JSON。"""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="PUT",
                                 headers=_headers({"Content-Type": "application/json"}))
    with _retry_request(req, timeout, cancel_event) as r:
        return json.loads(r.read().decode("utf-8"))


# ============================== 源适配层 ==============================
class HFSource:
    """hf-mirror.com（HuggingFace API 兼容）。"""
    id = "hf"
    endpoint = HF_ENDPOINT
    default_revision = "main"
    # 排序 token 直接透传 HF API
    sorts = ("trendingScore", "lastModified", "createdAt", "downloads", "likes")

    def search(self, keyword: str, sort: str = "trendingScore", limit: int = 20,
               cursor: str = "") -> Tuple[List[Dict[str, Any]], Optional[str]]:
        params = {
            "search": keyword,
            "sort": sort,
            "direction": -1,
            "limit": limit,
            # 接口仅在 sort=lastModified 时返回 lastModified，需显式展开
            "expand[]": ["lastModified", "downloads", "likes", "pipeline_tag", "gated"],
        }
        if cursor:
            params["cursor"] = cursor
        headers: Dict[str, str] = {}
        data = get_json(f"{self.endpoint}/api/models", params=params, out_headers=headers)
        if not isinstance(data, list):
            raise HttpError("搜索结果格式异常, 非数组")
        return data, self._next_cursor(headers)

    @staticmethod
    def _next_cursor(headers: Dict[str, str]) -> Optional[str]:
        """从 Link 响应头解析 rel="next" 的翻页游标（热门榜等排序不支持 offset，只能游标翻页）。"""
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
        if not m:
            return None
        q = urllib.parse.parse_qs(urllib.parse.urlparse(m.group(1)).query)
        return q.get("cursor", [None])[0]

    def list_files(self, repo_id: str, revision: str,
                   cancel_event: Optional[threading.Event]) -> List[Dict[str, Any]]:
        """递归获取仓库全部文件，归一化为 [{path, size, type}]。"""
        url = (f"{self.endpoint}/api/models/{urllib.parse.quote(repo_id)}"
               f"/tree/{urllib.parse.quote(revision)}")
        data = get_json(url, params={"recursive": "true"}, cancel_event=cancel_event)
        if not isinstance(data, list):
            raise HttpError("文件列表格式异常")
        return [{"path": f.get("path") or "",
                 "size": int(f.get("lfs", {}).get("size") if f.get("lfs") else f.get("size")) or 0,
                 "type": f.get("type") or "file"}
                for f in data if f.get("path")]

    def file_url(self, repo_id: str, filename: str, revision: str) -> str:
        """文件下载地址。实际文件经 302 跳转到 LFS 存储 / 直链。"""
        fname = filename.lstrip("/")
        return (f"{self.endpoint}/{urllib.parse.quote(repo_id)}/resolve/"
                f"{urllib.parse.quote(revision)}/{fname}")

    def repo_page_url(self, repo_id: str) -> str:
        return f"{self.endpoint}/{urllib.parse.quote(repo_id)}"


class MSSource:
    """modelscope.cn 魔搭（国内源，LFS 经阿里 CDN，支持 Range）。"""
    id = "ms"
    endpoint = MS_ENDPOINT
    default_revision = "master"
    sorts = ("trendingScore", "downloads", "likes")
    _SORT_MAP = {"trendingScore": "Default", "downloads": "DownloadsCount",
                 "likes": "StarsCount"}

    def search(self, keyword: str, sort: str = "trendingScore", limit: int = 20,
               cursor: str = "") -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """搜索模型（PUT dolphin 接口）。空关键词 = 浏览全站。

        魔搭无游标翻页，cursor 约定传页码字符串（"2"、"3"…），None 为第一页。
        """
        page = int(cursor) if str(cursor).strip().isdigit() else 1
        body = {
            "PageSize": limit,
            "PageNumber": page,
            "Name": keyword or "",
            "SortBy": self._SORT_MAP.get(sort, "Default"),
            "Target": "",
            "SingleCriterion": [],
        }
        data = put_json(f"{self.endpoint}/api/v1/dolphin/models", body)
        if not isinstance(data, dict) or data.get("Code") != 200:
            msg = data.get("Message") if isinstance(data, dict) else str(data)
            raise HttpError(f"搜索失败: {msg or '响应格式异常'}")
        model = (data.get("Data") or {}).get("Model") or {}
        total = int(model.get("TotalCount") or 0)
        entries = []
        for it in model.get("Models") or []:
            org = (it.get("Path") or "").strip()
            name = (it.get("Name") or "").strip()
            if not org or not name:
                continue
            entries.append({
                "id": f"{org}/{name}",
                "downloads": int(it.get("Downloads") or 0),
                "likes": int(it.get("Stars") or 0),
                "lastModified": _ms_ts_to_iso(it.get("LastUpdatedTime")),
            })
        has_next = page * limit < total
        return entries, (str(page + 1) if has_next else None)

    def list_files(self, repo_id: str, revision: str,
                   cancel_event: Optional[threading.Event]) -> List[Dict[str, Any]]:
        url = (f"{self.endpoint}/api/v1/models/{urllib.parse.quote(repo_id)}/repo/files")
        data = get_json(url, params={"Revision": revision, "Recursive": "true"},
                        cancel_event=cancel_event)
        if not isinstance(data, dict) or data.get("Code") != 200:
            raise HttpError("文件列表格式异常")
        files = (data.get("Data") or {}).get("Files") or []
        return [{"path": f.get("Path") or "",
                 "size": int(f.get("Size") or 0),
                 "type": "directory" if f.get("Type") == "tree" else "file"}
                for f in files if f.get("Path")]

    def file_url(self, repo_id: str, filename: str, revision: str) -> str:
        fname = filename.lstrip("/")
        return (f"{self.endpoint}/models/{urllib.parse.quote(repo_id)}/resolve/"
                f"{urllib.parse.quote(revision)}/{fname}")

    def repo_page_url(self, repo_id: str) -> str:
        return f"{self.endpoint}/models/{urllib.parse.quote(repo_id)}"


def _ms_ts_to_iso(ts: Any) -> Optional[str]:
    """魔搭 Unix 秒 -> ISO 字符串（与 HF lastModified 同构）。"""
    try:
        t = int(ts)
    except (TypeError, ValueError):
        return None
    if t <= 0:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(t))


_SOURCES = {"hf": HFSource(), "ms": MSSource()}


def get_source(source: str):
    s = _SOURCES.get((source or "").strip().lower())
    if s is None:
        raise HttpError(f"未知模型源: {source or '(空)'}")
    return s


def api_search_models(source: str, keyword: str = "", sort: str = "trendingScore",
                      limit: int = 20, cursor: str = "") -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """搜索模型仓库。返回 (结果列表, 下一页游标)，游标语义随源（hf token / ms 页码）。"""
    return get_source(source).search(keyword, sort=sort, limit=limit, cursor=cursor)


def api_list_files(source: str, repo_id: str, revision: str = "",
                   cancel_event: Optional[threading.Event] = None) -> List[Dict[str, Any]]:
    """递归获取仓库全部文件（归一化 [{path, size, type}]，会话内缓存）。"""
    s = get_source(source)
    rev = revision or s.default_revision
    key = f"{s.id}:{repo_id}@{rev}"
    cached = _tree_cache.get(key)
    if cached is not None:
        return cached
    files = s.list_files(repo_id, rev, cancel_event)
    _tree_cache[key] = files
    return files


def resolve_download_url(source: str, repo_id: str, filename: str,
                         revision: str = "") -> str:
    s = get_source(source)
    return s.file_url(repo_id, filename, revision or s.default_revision)


def repo_page_url(source: str, repo_id: str) -> str:
    return get_source(source).repo_page_url(repo_id)


# ============================== 文件分类 ==============================
def classify_files(files: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按文件名分类: mmproj（含 'mmproj' 字样，与 scan_mmprojs 判定一致）→ mmproj；"""
    out: Dict[str, List[Dict[str, Any]]] = {"gguf": [], "mmproj": [], "other": []}
    for f in files:
        if f.get("type") == "directory":
            continue
        name = f.get("path") or ""
        if not name:
            continue
        entry = {"name": name, "size": int(f.get("size") or 0)}
        if "mmproj" in Path(name).name.lower():
            out["mmproj"].append(entry)
        elif name.lower().endswith(".gguf"):
            out["gguf"].append(entry)
        else:
            out["other"].append(entry)
    for v in out.values():
        v.sort(key=lambda e: e["name"])
    return out


# ============================== 下载层 ==============================
def _delete_quiet(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _download_once(url: str, dest_path: str, filename: str,
                   log_fn: Callable[[str], None],
                   progress_cb: Optional[Callable[[Dict[str, Any]], None]],
                   cancel_event: Optional[threading.Event],
                   expected_size: int, timeout: int) -> None:
    """单次下载尝试：Range 续传(206)/从头(200)/已完整(416)，完成后校验大小。"""
    resumed = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    headers = _headers()
    if resumed > 0:
        headers["Range"] = f"bytes={resumed}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range 超出：本地已 >= 服务器大小
            if not expected_size or resumed == expected_size:
                log_fn(f"  已是完整文件，跳过: {filename}（{_human_size(resumed)}）")
                return
            _delete_quiet(dest_path)
            raise HttpError(f"本地文件大小异常: {filename}")
        raise
    with resp:
        status = getattr(resp, "status", 200)
        if status == 206:
            total = resumed + int(resp.headers.get("Content-Length", "0") or 0)
            mode = "ab"
        else:  # 200 或服务器忽略 Range
            resumed = 0
            total = int(resp.headers.get("Content-Length", "0") or 0)
            mode = "wb"
        if not total:
            total = expected_size  # 响应头缺失时用文件树大小兜底
        log_fn(f"  {'继续' if mode == 'ab' else '开始'}下载: {filename}"
               f"（共 {_human_size(total) if total else '?'}）")
        downloaded = resumed
        last_pct = -1.0
        last_report = 0
        stop_watch = None
        if cancel_event is not None:
            stop_watch = start_cancel_watcher(cancel_event, resp.close)
        try:
            with open(dest_path, mode) as fp:
                while True:
                    chunk = resp.read(DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    fp.write(chunk)
                    downloaded += len(chunk)
                    if cancel_event and cancel_event.is_set():
                        raise DownloadCancelled()
                    if progress_cb:
                        if total:
                            pct = downloaded / total
                            if pct - last_pct >= 0.01 or downloaded >= total:
                                last_pct = pct
                                progress_cb({"type": "progress", "file": filename,
                                             "done": downloaded, "total": total, "pct": pct})
                        elif downloaded - last_report >= 2 * 1024 * 1024:
                            last_report = downloaded
                            progress_cb({"type": "progress", "file": filename,
                                         "done": downloaded, "total": None, "pct": None})
        finally:
            if stop_watch is not None:
                stop_watch()
    final = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    expect = expected_size or total
    if expect and final != expect:
        _delete_quiet(dest_path)
        raise HttpError(f"校验失败: {filename}（本地 {final}，预期 {expect}）")
    if progress_cb:
        progress_cb({"type": "file_done", "file": filename, "size": final})
    log_fn(f"  下载完成: {filename}（{_human_size(final)}）")


def _download_file(url: str, dest_path: str, filename: str, log_fn: Callable[[str], None],
                   progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
                   cancel_event: Optional[threading.Event] = None,
                   expected_size: int = 0,
                   timeout: int = DEFAULT_TIMEOUT) -> None:
    """下载单个文件：已完整跳过，不完整续传，失败重试（保留断点），取消清理半成品。"""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    if cancel_event and cancel_event.is_set():
        raise DownloadCancelled()
    exist_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    if expected_size and exist_size == expected_size:
        log_fn(f"  已是完整文件，跳过: {filename}（{_human_size(exist_size)}）")
        if progress_cb:
            progress_cb({"type": "file_skip", "file": filename, "size": exist_size})
        return
    if expected_size and exist_size > expected_size:
        _delete_quiet(dest_path)
        log_fn(f"  本地文件大小异常，重新下载: {filename}")

    last_err = None
    for attempt in range(1, DEFAULT_RETRIES + 1):
        if cancel_event and cancel_event.is_set():
            raise DownloadCancelled()
        try:
            _download_once(url, dest_path, filename, log_fn, progress_cb,
                           cancel_event, expected_size, timeout)
            return
        except DownloadCancelled:
            _delete_quiet(dest_path)
            raise
        except Exception as e:  # noqa
            if cancel_event and cancel_event.is_set():
                _delete_quiet(dest_path)
                raise DownloadCancelled()
            last_err = e
        if attempt < DEFAULT_RETRIES:
            if cancel_event and cancel_event.is_set():
                raise DownloadCancelled()
            log_fn(f"  下载中断（{last_err}），{attempt}s 后重试"
                   f"（{attempt}/{DEFAULT_RETRIES - 1}）…")
            if cancel_event is not None:
                if cancel_event.wait(attempt):
                    raise DownloadCancelled()
            else:
                time.sleep(attempt)
    raise HttpError(f"下载失败({DEFAULT_RETRIES}次重试后): {filename}\n原因: {last_err}")


def _make_chunk_ranges(size: int) -> List[Tuple[int, int, int]]:
    """把文件切成大小均等的段，返回 [(start, end, idx), ...]。"""
    segs = min((size + CHUNK_SEG_SIZE - 1) // CHUNK_SEG_SIZE, CHUNK_MAX_SEGS)
    seg_len = (size + segs - 1) // segs
    ranges = []
    for i in range(segs):
        s = i * seg_len
        e = min(s + seg_len, size)
        if e > s:
            ranges.append((s, e, i))
    return ranges


class _ChunkProgress:
    """分片进度累加器：锁内累加并判定是否到达上报阈值，回调由调用方在锁外执行。"""

    def __init__(self, total: int, initial: int = 0) -> None:
        self._lock = threading.Lock()
        self._last_pct = -1.0
        self.total = total
        self.done = initial

    def add(self, n: int) -> bool:
        with self._lock:
            self.done += n
            pct = self.done / self.total
            report = pct - self._last_pct >= 0.01 or self.done >= self.total
            if report:
                self._last_pct = pct
            return report


def _delete_parts(part_paths: List[str]) -> None:
    for p in part_paths:
        _delete_quiet(p)


def _merge_prefix_into_dest(dest_path: str, ranges: List[Tuple[int, int, int]],
                            part_paths: List[str]) -> None:
    """把从头连续的前缀段合并进 dest（末段允许不完整），供单流从断点续传。"""
    n = 0
    for (s, e, i) in ranges:
        seg = e - s
        actual = os.path.getsize(part_paths[i]) if os.path.exists(part_paths[i]) else 0
        if actual == 0 or actual > seg:
            break
        n += 1
        if actual < seg:
            break
    if n == 0:
        return
    with open(dest_path, "wb") as out:
        for i in range(n):
            with open(part_paths[i], "rb") as pin:
                shutil.copyfileobj(pin, out)


def _download_file_chunked(url: str, dest_path: str, filename: str,
                           log_fn: Callable[[str], None],
                           progress_cb: Optional[Callable[[Dict[str, Any]], None]],
                           cancel_event: Optional[threading.Event],
                           expected_size: int = 0,
                           timeout: int = CHUNK_TIMEOUT) -> None:
    """单文件多连接分片下载：段级断点续传 + 失败降级单流。"""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    if not expected_size or expected_size < CHUNK_MIN_SIZE:
        raise HttpError(f"分片条件不足: {filename}")

    ranges = _make_chunk_ranges(expected_size)
    part_paths = [f"{dest_path}.part{i}" for (_s, _e, i) in ranges]
    log_fn(f"  分片下载: {filename}（{_human_size(expected_size)}，"
           f"{len(ranges)} 段/{CHUNK_WORKERS} 连接）")

    def worker(start: int, end: int, idx: int) -> None:
        part_path = part_paths[idx]
        seg_bytes = end - start
        last_err: Optional[Exception] = None
        for attempt in range(1, DEFAULT_RETRIES + 1):
            if cancel_event and cancel_event.is_set():
                raise DownloadCancelled()
            if abort.is_set():
                return
            existing = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            if existing > seg_bytes:
                _delete_quiet(part_path)
                existing = 0
            if existing >= seg_bytes:
                return
            try:
                wfd = os.open(part_path, os.O_RDWR | os.O_CREAT | _O_BINARY, 0o644)
                try:
                    if existing:
                        os.lseek(wfd, existing, os.SEEK_SET)
                    written = existing
                    headers = _headers()
                    headers["Range"] = f"bytes={start + existing}-{end - 1}"
                    req = urllib.request.Request(url, headers=headers)
                    box: dict = {}

                    def _do_urlopen():
                        try:
                            box["resp"] = urllib.request.urlopen(req, timeout=timeout)
                        except Exception as e:
                            box["err"] = e

                    t_urlopen = threading.Thread(target=_do_urlopen, daemon=True)
                    t_urlopen.start()
                    while t_urlopen.is_alive():
                        if cancel_event is not None and cancel_event.is_set():
                            raise DownloadCancelled()
                        if abort.is_set():
                            return
                        t_urlopen.join(timeout=0.2)
                    if "err" in box:
                        e = box["err"]
                        if isinstance(e, urllib.error.HTTPError):
                            if e.code == 416:
                                raise HttpError(f"分片下载失败: {filename}"
                                                f"（段 {idx} 返回 416，文件树大小与服务器不符）") from e
                            raise
                        raise e
                    resp = box.get("resp")
                    with resp:
                        if getattr(resp, "status", 200) == 200:
                            raise ChunkedFallback()
                        stop_watch = (start_cancel_watcher(cancel_event, resp.close)
                                      if cancel_event else None)
                        try:
                            while written < seg_bytes:
                                if abort.is_set():
                                    return
                                if cancel_event and cancel_event.is_set():
                                    raise DownloadCancelled()
                                chunk = resp.read(DOWNLOAD_CHUNK)
                                if not chunk:
                                    raise _SegmentIncomplete()
                                os.write(wfd, chunk)
                                written += len(chunk)
                                if progress.add(len(chunk)) and progress_cb:
                                    progress_cb({"type": "progress", "file": filename,
                                                 "done": progress.done,
                                                 "total": expected_size,
                                                 "pct": progress.done / expected_size})
                        finally:
                            if stop_watch is not None:
                                stop_watch()
                    return
                finally:
                    os.close(wfd)
            except (DownloadCancelled, ChunkedFallback, HttpError):
                raise
            except Exception as e:  # noqa
                if cancel_event and cancel_event.is_set():
                    raise DownloadCancelled()
                last_err = e
            if attempt < DEFAULT_RETRIES:
                if cancel_event is not None:
                    if cancel_event.wait(attempt):
                        raise DownloadCancelled()
                else:
                    time.sleep(attempt)
        raise HttpError(f"分片下载失败: {filename}（段 {idx}: {last_err}）")

    outer_tries = 2
    last_err: Optional[Exception] = None
    for outer in range(1, outer_tries + 1):
        abort = threading.Event()
        progress = _ChunkProgress(
            expected_size,
            sum(os.path.getsize(p) if os.path.exists(p) else 0 for p in part_paths))
        try:
            with ThreadPoolExecutor(max_workers=CHUNK_WORKERS) as ex:
                futures = [ex.submit(worker, s, e, i) for (s, e, i) in ranges]
                try:
                    for fu in as_completed(futures):
                        fu.result()
                except Exception:
                    abort.set()
                    raise
            for (s, e, i) in ranges:
                actual = os.path.getsize(part_paths[i]) if os.path.exists(part_paths[i]) else -1
                if actual != e - s:
                    raise HttpError(f"分片校验失败: {filename}"
                                    f"（段 {i} 本地 {actual}，预期 {e - s}）")
            with open(dest_path, "wb") as out:
                for (s, e, i) in ranges:
                    with open(part_paths[i], "rb") as pin:
                        shutil.copyfileobj(pin, out)
                    os.remove(part_paths[i])
            if os.path.getsize(dest_path) != expected_size:
                raise HttpError(f"分片校验失败: {filename}"
                                f"（本地 {os.path.getsize(dest_path)}，预期 {expected_size}）")
            if progress_cb:
                progress_cb({"type": "file_done", "file": filename, "size": expected_size})
            log_fn(f"  分片下载完成: {filename}（{_human_size(expected_size)}）")
            return
        except (ChunkedFallback, DownloadCancelled):
            _delete_parts(part_paths)
            raise
        except Exception as e:
            last_err = e
            if outer < outer_tries:
                log_fn(f"  分片失败，保留进度重试（{outer}/{outer_tries - 1}）: {e}")
                if cancel_event is not None:
                    if cancel_event.wait(outer):
                        raise DownloadCancelled()
                else:
                    time.sleep(outer)

    log_fn(f"  分片多次失败（{last_err}），降级单流: {filename}")
    try:
        _merge_prefix_into_dest(dest_path, ranges, part_paths)
        _download_file(url, dest_path, filename, log_fn, progress_cb, cancel_event,
                       expected_size=expected_size, timeout=timeout)
    finally:
        _delete_parts(part_paths)


def _fetch_sizes(source: str, repo_id: str, revision: str,
                 cancel_event: Optional[threading.Event]) -> Dict[str, int]:
    """拉取文件树建立 path→size 映射（已完整跳过与完成后校验用），失败降级为空。"""
    try:
        files = api_list_files(source, repo_id, revision=revision, cancel_event=cancel_event)
        return {f["path"]: f["size"] for f in files
                if f.get("type") != "directory" and f.get("path")}
    except DownloadCancelled:
        raise
    except Exception:
        return {}


def _disk_error(dest_dir: str, need: int) -> Optional[str]:
    """磁盘空间预检：空间不足返回错误文案，否则 None。"""
    if need <= 0:
        return None
    try:
        free = shutil.disk_usage(dest_dir).free
    except OSError:
        return None
    if free and free < need * 1.05:
        return f"磁盘空间不足：约需 {_human_size(need)}，剩余 {_human_size(free)}"
    return None


def _download_entries(entries: List[Dict[str, Any]], cleanup_dir: str,
                      log_fn: Callable[[str], None],
                      progress_cb: Optional[Callable[[Dict[str, Any]], None]],
                      cancel_event: Optional[threading.Event],
                      cleanup_on_cancel: bool) -> Dict[str, Any]:
    """逐条目下载（download_files / download_urls 共用）。

    entries = [{"url", "dest", "display", "size"}]；取消时 cleanup_on_cancel=True
    删除整个 cleanup_dir，否则只删未完成文件。progress 事件统一携带 idx/count。
    """
    partial_files: set = set()

    def _on_cancel() -> None:
        if cleanup_dir and cleanup_on_cancel:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
            log_fn("已取消，未完成的下载文件已清理（下次安装将重新下载）")
            return
        removed = 0
        for f in list(partial_files):
            try:
                os.remove(f)
                removed += 1
            except OSError:
                pass
        log_fn("已取消，未完成的文件已清理（下次重新下载）" if removed
               else "已取消，无未完成的文件")

    ok, failed, cancelled = 0, [], False
    count = len(entries)
    for idx, ent in enumerate(entries, 1):
        if cancel_event and cancel_event.is_set():
            cancelled = True
            _on_cancel()
            break
        url, dest, display, exp = ent["url"], ent["dest"], ent["display"], ent["size"]
        log_fn(f"  [{idx}/{count}] {display}")

        def file_cb(ev: Dict[str, Any], idx: int = idx, display: str = display) -> None:
            if progress_cb:
                progress_cb({**ev, "file": display, "idx": idx, "count": count})

        if progress_cb:
            progress_cb({"type": "file_start", "file": display, "idx": idx,
                         "count": count})
        partial_files.add(dest)
        try:
            if exp >= CHUNK_MIN_SIZE and not os.path.exists(dest):
                try:
                    _download_file_chunked(url, dest, display, log_fn, file_cb,
                                           cancel_event, expected_size=exp)
                except ChunkedFallback:
                    log_fn(f"  分片不可用（服务端忽略 Range），回退单流: {display}")
                    _download_file(url, dest, display, log_fn, file_cb, cancel_event,
                                   expected_size=exp)
            else:
                _download_file(url, dest, display, log_fn, file_cb, cancel_event,
                               expected_size=exp)
            ok += 1
            partial_files.discard(dest)
        except DownloadCancelled:
            cancelled = True
            _on_cancel()
            break
        except Exception as e:
            failed.append({"file": display, "error": str(e)[:200]})
            log_fn(f"  失败: {display} — {str(e)[:200]}")
            partial_files.discard(dest)

    log_fn(f"完成：成功 {ok} / 失败 {len(failed)}" + (" / 已取消" if cancelled else "")
           + f"  |  保存于 {os.path.abspath(cleanup_dir)}")
    return {"ok": not failed and not cancelled, "downloaded": ok,
            "failed": failed, "cancelled": cancelled, "dir": os.path.abspath(cleanup_dir)}


def download_files(source: str, repo_id: str, filenames: List[str], dest_root: str,
                   log_fn: Optional[Callable[[str], None]] = None,
                   progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
                   cancel_event: Optional[threading.Event] = None,
                   revision: str = "",
                   cleanup_on_cancel: bool = False) -> Dict[str, Any]:
    """从指定源按勾选文件下载，按 作者/仓库名 两级目录组织。"""
    log_fn = log_fn or (lambda s: None)
    s = get_source(source)
    rev = revision or s.default_revision
    if not _safe_relpath(repo_id):
        return {"ok": False, "error": f"仓库名不合法: {repo_id}"}
    author, repo_name = split_repo_id(repo_id)
    repo_dir = os.path.join(dest_root, author, repo_name)
    os.makedirs(repo_dir, exist_ok=True)
    log_fn(f"开始下载 {repo_id}（{s.id}）→ {os.path.abspath(repo_dir)}"
           f"（共 {len(filenames)} 个文件）")

    size_by_path = _fetch_sizes(s.id, repo_id, rev, cancel_event)
    if not size_by_path:
        log_fn("提示：未能获取文件大小信息，本次下载将跳过完整性校验")
    err = _disk_error(repo_dir, sum(size_by_path.get(f, 0) for f in filenames))
    if err:
        return {"ok": False, "error": err}

    entries, pre_failed = [], []
    for filename in filenames:
        if not _safe_relpath(filename):
            pre_failed.append({"file": filename, "error": "路径不合法"})
            continue
        entries.append({
            "url": s.file_url(repo_id, filename, rev),
            "dest": os.path.join(repo_dir, *filename.split("/")),
            "display": filename,
            "size": size_by_path.get(filename, 0),
        })
    r = _download_entries(entries, repo_dir, log_fn, progress_cb,
                          cancel_event, cleanup_on_cancel)
    if pre_failed:
        r["failed"] = pre_failed + r["failed"]
        r["ok"] = False
    return r


def download_urls(items: List[Dict[str, Any]], dest_root: str,
                  log_fn: Optional[Callable[[str], None]] = None,
                  progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
                  cancel_event: Optional[threading.Event] = None,
                  cleanup_on_cancel: bool = False) -> Dict[str, Any]:
    """直连 URL 下载（不经模型源 API）：items = [{"url", "filename", "size", "label"?}]。"""
    log_fn = log_fn or (lambda s: None)
    os.makedirs(dest_root, exist_ok=True)
    entries, pre_failed = [], []
    for it in items:
        fn = str(it.get("filename") or "")
        if not fn or not _safe_relpath(fn):
            pre_failed.append({"file": fn or "?", "error": "文件名不合法"})
            continue
        entries.append({
            "url": it["url"],
            "dest": os.path.join(dest_root, *fn.split("/")),
            "display": str(it.get("label") or fn),
            "size": int(it.get("size") or 0),
        })
    err = _disk_error(dest_root, sum(e["size"] for e in entries))
    if err:
        return {"ok": False, "error": err, "downloaded": 0, "failed": pre_failed,
                "cancelled": False, "dir": os.path.abspath(dest_root)}
    log_fn(f"开始直连下载 → {os.path.abspath(dest_root)}（共 {len(entries)} 个文件）")
    r = _download_entries(entries, dest_root, log_fn, progress_cb,
                          cancel_event, cleanup_on_cancel)
    if pre_failed:
        r["failed"] = pre_failed + r["failed"]
        r["ok"] = False
    return r
