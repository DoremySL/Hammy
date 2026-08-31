"""hf_downloader.py — 基于 HF-Mirror (hf-mirror.com) 的模型搜索与下载。"""
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


def get_json(url: str, params: Optional[Dict[str, Any]] = None,
             timeout: int = DEFAULT_TIMEOUT,
             cancel_event: Optional[threading.Event] = None,
             out_headers: Optional[Dict[str, str]] = None) -> Any:
    """GET 请求并解析 JSON（自动跟随重定向），失败重试 + 递增退避。"""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    last_err = None
    for attempt in range(1, DEFAULT_RETRIES + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelled()
        try:
            req = urllib.request.Request(url, headers=_headers())
            with urllib.request.urlopen(req, timeout=timeout) as r:
                stop_watch = None
                if cancel_event is not None:
                    stop_watch = start_cancel_watcher(cancel_event, r.close)
                try:
                    if out_headers is not None:
                        out_headers.update(dict(r.headers))
                    return json.loads(r.read().decode("utf-8"))
                finally:
                    if stop_watch is not None:
                        stop_watch()
        except DownloadCancelled:
            raise
        except Exception as e:  # noqa
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelled()
            last_err = e
            time.sleep(1.0 * attempt)
    raise HttpError(f"请求失败({DEFAULT_RETRIES}次重试后): {url}\n原因: {last_err}")


# ============================== HF API 层 ==============================
def _next_cursor(headers: Dict[str, str]) -> Optional[str]:
    """从 Link 响应头解析 rel="next" 的翻页游标（热门榜等排序不支持 offset，只能游标翻页）。"""
    m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
    if not m:
        return None
    q = urllib.parse.parse_qs(urllib.parse.urlparse(m.group(1)).query)
    return q.get("cursor", [None])[0]


def api_search_models(keyword: str, sort: str = "trendingScore",
                      limit: int = 20,
                      cursor: str = "") -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """搜索模型。对应 HF Hub API: GET /api/models

    sort 可选: trendingScore / downloads / likes / lastModified / createdAt
    返回 (结果列表, 下一页游标)，翻页游标取自 Link 响应头。
    """
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
    data = get_json(f"{HF_ENDPOINT}/api/models", params=params, out_headers=headers)
    if not isinstance(data, list):
        raise HttpError("搜索结果格式异常, 非数组")
    return data, _next_cursor(headers)


def api_list_files(repo_id: str, revision: str = "main",
                   cancel_event: Optional[threading.Event] = None) -> List[Dict[str, Any]]:
    """递归获取仓库全部文件(含子目录)。GET /api/models/{id}/tree/{rev}?recursive=true（会话内缓存）"""
    key = f"{repo_id}@{revision}"
    cached = _tree_cache.get(key)
    if cached is not None:
        return cached
    url = (f"{HF_ENDPOINT}/api/models/{urllib.parse.quote(repo_id)}"
           f"/tree/{urllib.parse.quote(revision)}")
    data = get_json(url, params={"recursive": "true"}, cancel_event=cancel_event)
    if not isinstance(data, list):
        raise HttpError("文件列表格式异常")
    _tree_cache[key] = data
    return data


def resolve_download_url(repo_id: str, filename: str, revision: str = "main") -> str:
    """构造文件下载地址。实际文件经 302 跳转到 LFS 存储 / 直链。"""
    fname = filename.lstrip("/")
    return (f"{HF_ENDPOINT}/{urllib.parse.quote(repo_id)}/resolve/"
            f"{urllib.parse.quote(revision)}/{fname}")


# ============================== 文件分类 ==============================
def _file_size(f: Dict[str, Any]) -> int:
    lfs = f.get("lfs")
    return int((lfs.get("size") if lfs else f.get("size")) or 0)


def classify_files(files: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按文件名分类: mmproj（含 'mmproj' 字样，与 scan_mmprojs 判定一致）→ mmproj；"""
    out: Dict[str, List[Dict[str, Any]]] = {"gguf": [], "mmproj": [], "other": []}
    for f in files:
        if f.get("type") == "directory":
            continue
        name = f.get("path", "")
        if not name:
            continue
        entry = {"name": name, "size": _file_size(f)}
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


def _fetch_sizes(repo_id: str, revision: str,
                 cancel_event: Optional[threading.Event]) -> Dict[str, int]:
    """拉取文件树建立 path→size 映射（已完整跳过与完成后校验用），失败降级为空。"""
    try:
        files = api_list_files(repo_id, revision, cancel_event=cancel_event)
        return {f["path"]: _file_size(f) for f in files
                if f.get("type") != "directory" and f.get("path")}
    except DownloadCancelled:
        raise
    except Exception:
        return {}


def download_files(repo_id: str, filenames: List[str], dest_root: str,
                   log_fn: Optional[Callable[[str], None]] = None,
                   progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
                   cancel_event: Optional[threading.Event] = None,
                   revision: str = "main",
                   cleanup_on_cancel: bool = False) -> Dict[str, Any]:
    """按勾选文件逐个下载，按 作者/仓库名 两级目录组织。
        progress 事件统一携带 idx/count，供前端计算多文件整体进度。
        返回 {"ok", "downloaded", "failed": [{"file","error"}], "cancelled", "dir"}。
    """
    log_fn = log_fn or (lambda s: None)
    if not _safe_relpath(repo_id):
        return {"ok": False, "error": f"仓库名不合法: {repo_id}"}
    author, repo_name = split_repo_id(repo_id)
    repo_dir = os.path.join(dest_root, author, repo_name)
    os.makedirs(repo_dir, exist_ok=True)
    log_fn(f"开始下载 {repo_id} → {os.path.abspath(repo_dir)}（共 {len(filenames)} 个文件）")

    size_by_path = _fetch_sizes(repo_id, revision, cancel_event)
    if not size_by_path:
        log_fn("提示：未能获取文件大小信息，本次下载将跳过完整性校验")
    need = sum(size_by_path.get(f, 0) for f in filenames)
    try:
        free = shutil.disk_usage(repo_dir).free
    except OSError:
        free = 0
    if need and free and free < need * 1.05:
        return {"ok": False, "error": f"磁盘空间不足：约需 {_human_size(need)}，剩余 {_human_size(free)}"}
    partial_files: set = set()  # 本次会话中开始下载但未完成的文件

    def _on_cancel() -> None:
        """取消收尾：cleanup_on_cancel=True 删整个目录；否则删未完成文件（保留完整文件）。"""
        if cleanup_on_cancel:
            shutil.rmtree(repo_dir, ignore_errors=True)
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
    count = len(filenames)
    for idx, filename in enumerate(filenames, 1):
        if cancel_event and cancel_event.is_set():
            cancelled = True
            _on_cancel()
            break
        if not _safe_relpath(filename):
            failed.append({"file": filename, "error": "路径不合法"})
            continue
        url = resolve_download_url(repo_id, filename, revision)
        dest = os.path.join(repo_dir, *filename.split("/"))
        log_fn(f"  [{idx}/{count}] {filename}")

        def file_cb(ev: Dict[str, Any], idx: int = idx) -> None:
            if progress_cb:
                progress_cb({**ev, "idx": idx, "count": count})

        if progress_cb:
            progress_cb({"type": "file_start", "file": filename, "idx": idx,
                         "count": count})
        exp = size_by_path.get(filename, 0)
        partial_files.add(dest)
        try:
            if exp >= CHUNK_MIN_SIZE and not os.path.exists(dest):
                try:
                    _download_file_chunked(url, dest, filename, log_fn, file_cb,
                                           cancel_event, expected_size=exp)
                except ChunkedFallback:
                    log_fn(f"  分片不可用（服务端忽略 Range），回退单流: {filename}")
                    _download_file(url, dest, filename, log_fn, file_cb, cancel_event,
                                   expected_size=exp)
            else:
                _download_file(url, dest, filename, log_fn, file_cb, cancel_event,
                               expected_size=exp)
            ok += 1
            partial_files.discard(dest)
        except DownloadCancelled:
            cancelled = True
            _on_cancel()
            break
        except Exception as e:
            failed.append({"file": filename, "error": str(e)[:200]})
            log_fn(f"  失败: {filename} — {str(e)[:200]}")
            partial_files.discard(dest)

    log_fn(f"完成：成功 {ok} / 失败 {len(failed)}" + (" / 已取消" if cancelled else "")
           + f"  |  保存于 {os.path.abspath(repo_dir)}")
    return {"ok": not failed and not cancelled, "downloaded": ok,
            "failed": failed, "cancelled": cancelled, "dir": os.path.abspath(repo_dir)}
