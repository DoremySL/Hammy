import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app import models_downloader as md


def _resp(status=200, body=b"x" * 100, cl=None):
    """构造伪 HTTP 响应：read() 依次返回 body 分块后结束。

    MagicMock + __enter__ 返回自身，模拟 urlopen 返回的上下文管理器
    （普通 Mock 无 __enter__，`with` 会抛 TypeError）。
    """
    r = mock.MagicMock()
    r.status = status
    r.headers = {"Content-Length": str(cl if cl is not None else len(body))}
    r.read.side_effect = [body, b""]
    r.__enter__.return_value = r
    r.__exit__.return_value = False
    return r


def _json_resp(data, headers=None):
    """构造返回 JSON 的伪响应（headers 默认空 dict，供 Link 游标解析用）。"""
    r = mock.MagicMock()
    r.headers = headers or {}
    r.read.return_value = json.dumps(data).encode("utf-8")
    r.__enter__.return_value = r
    r.__exit__.return_value = False
    return r


class TestApiLayer(unittest.TestCase):
    """get_json 重试 / api_search_models / api_list_files / URL 构造。"""

    def test_get_json_retry_then_success(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=[OSError("reset"), _json_resp([1, 2])]), \
             mock.patch("gui_app.models_downloader.time.sleep"):
            data = md.get_json("http://x/api", params={"a": "b"})
        self.assertEqual(data, [1, 2])

    def test_get_json_gives_up_after_retries(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=OSError("reset")) as m, \
             mock.patch("gui_app.models_downloader.time.sleep"):
            with self.assertRaises(md.HttpError):
                md.get_json("http://x/api")
        self.assertEqual(m.call_count, 3)

    def test_search_models_params(self):
        with mock.patch("urllib.request.urlopen", return_value=_json_resp([])), \
             mock.patch("urllib.request.Request", wraps=urllib.request.Request) as req:
            md.api_search_models("hf", "gguf", sort="downloads", limit=10)
        url = req.call_args.args[0]
        self.assertIn("/api/models", url)
        self.assertIn("search=gguf", url)
        self.assertIn("sort=downloads", url)
        self.assertIn("direction=-1", url)
        self.assertIn("limit=10", url)
        self.assertIn("lastModified", url)

    def test_search_models_cursor(self):
        """带 cursor 翻页：URL 带游标，从 Link 头解析下一页游标。"""
        link = ('<https://huggingface.co/api/models?search=gguf&sort=trendingScore'
                '&cursor=abc%3D%3D>; rel="next"')
        with mock.patch("urllib.request.urlopen",
                        return_value=_json_resp([{"id": "a/gguf"}], headers={"Link": link})), \
             mock.patch("urllib.request.Request", wraps=urllib.request.Request) as req:
            data, cur = md.api_search_models("hf", "gguf", cursor="xyz=")
        self.assertEqual([m["id"] for m in data], ["a/gguf"])
        self.assertEqual(cur, "abc==")
        url = req.call_args.args[0]
        self.assertIn("cursor=xyz%3D", url)
        self.assertNotIn("offset", url)

    def test_search_models_no_cursor_when_last_page(self):
        with mock.patch("urllib.request.urlopen",
                        return_value=_json_resp([], headers={"Link": '<http://x>; rel="prev"'})):
            data, cur = md.api_search_models("hf", "gguf")
        self.assertEqual(data, [])
        self.assertIsNone(cur)

    def test_list_files_recursive(self):
        with mock.patch("urllib.request.urlopen", return_value=_json_resp([])), \
             mock.patch("urllib.request.Request", wraps=urllib.request.Request) as req:
            md.api_list_files("hf", "unsloth/Qwen-GGUF")
        url = req.call_args.args[0]
        self.assertIn("/api/models/unsloth/Qwen-GGUF/tree/main", url)
        self.assertIn("recursive=true", url)

    def test_resolve_download_url(self):
        url = md.resolve_download_url("hf", "unsloth/Qwen-GGUF",
                                      "Qwen-GGUF-Q4_K_M.gguf")
        self.assertEqual(url, md.HF_ENDPOINT
                         + "/unsloth/Qwen-GGUF/resolve/main/Qwen-GGUF-Q4_K_M.gguf")


class TestModelScopeSource(unittest.TestCase):
    """魔搭源：搜索归一化 / 排序映射 / 页码游标 / 文件列表归一化 / URL 形态。"""

    def test_ms_search_normalize_and_cursor(self):
        data = {"Code": 200, "Data": {"Model": {
            "TotalCount": 25,
            "Models": [
                {"Path": "unsloth", "Name": "Qwen3-GGUF", "Downloads": 111,
                 "Stars": 7, "LastUpdatedTime": 1749401232},
                {"Path": "", "Name": "bad"},
                {},
            ],
        }}}
        with mock.patch("gui_app.models_downloader.put_json", return_value=data) as pj:
            entries, cur = md.api_search_models("ms", "qwen", sort="downloads", limit=20)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["id"], "unsloth/Qwen3-GGUF")
        self.assertEqual(e["downloads"], 111)
        self.assertEqual(e["likes"], 7)
        self.assertTrue(e["lastModified"].startswith("2025-06-08T"))
        self.assertTrue(e["lastModified"].endswith("Z"))
        self.assertEqual(cur, "2")
        body = pj.call_args.args[1]
        self.assertEqual(body["SortBy"], "DownloadsCount")
        self.assertEqual(body["PageSize"], 20)

    def test_ms_search_last_page_no_cursor(self):
        data = {"Code": 200, "Data": {"Model": {"TotalCount": 20, "Models": [
            {"Path": "a", "Name": "b"}]}}}
        with mock.patch("gui_app.models_downloader.put_json", return_value=data):
            entries, cur = md.api_search_models("ms", "q", limit=20)
        self.assertEqual(len(entries), 1)
        self.assertIsNone(cur)

    def test_ms_search_invalid_cursor_falls_back_page1(self):
        data = {"Code": 200, "Data": {"Model": {"TotalCount": 100, "Models": []}}}
        with mock.patch("gui_app.models_downloader.put_json", return_value=data) as pj:
            _, cur = md.api_search_models("ms", "q", limit=20, cursor="abc")
        self.assertEqual(pj.call_args.args[1]["PageNumber"], 1)
        self.assertEqual(cur, "2")

    def test_ms_search_unsupported_sort_falls_back_default(self):
        data = {"Code": 200, "Data": {"Model": {"TotalCount": 0, "Models": []}}}
        with mock.patch("gui_app.models_downloader.put_json", return_value=data) as pj:
            md.api_search_models("ms", "q", sort="lastModified")
        self.assertEqual(pj.call_args.args[1]["SortBy"], "Default")

    def test_ms_search_error_raises(self):
        with mock.patch("gui_app.models_downloader.put_json",
                        return_value={"Code": 10010202002, "Message": "参数错误",
                                      "Success": False}):
            with self.assertRaises(md.HttpError) as ctx:
                md.api_search_models("ms", "q")
        self.assertIn("搜索失败", str(ctx.exception))

    def test_ms_list_files_normalized_and_cached(self):
        data = {"Code": 200, "Data": {"Files": [
            {"Path": "sub", "Type": "tree", "Size": 0},
            {"Path": "m.gguf", "Type": "blob", "Size": 5},
        ]}}
        with mock.patch("gui_app.models_downloader.get_json",
                        return_value=data) as gj:
            files = md.api_list_files("ms", "org/repo")
            files_again = md.api_list_files("ms", "org/repo")
        self.assertEqual(files, [{"path": "sub", "size": 0, "type": "directory"},
                                 {"path": "m.gguf", "size": 5, "type": "file"}])
        self.assertEqual(files_again, files)
        self.assertEqual(gj.call_count, 1)  # 缓存命中不发请求
        url = gj.call_args.args[0]
        self.assertIn("/api/v1/models/org/repo/repo/files", url)

    def test_ms_urls_use_master(self):
        self.assertEqual(md.resolve_download_url("ms", "org/repo", "f.gguf"),
                         md.MS_ENDPOINT + "/models/org/repo/resolve/master/f.gguf")
        self.assertEqual(md.repo_page_url("ms", "org/repo"),
                         md.MS_ENDPOINT + "/models/org/repo")

    def test_ms_download_uses_master_url_and_layout(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = Path(td.name)
        with mock.patch("urllib.request.urlopen", return_value=_resp(200, b"x" * 10, cl=10)), \
             mock.patch("urllib.request.Request", wraps=urllib.request.Request) as req, \
             mock.patch("gui_app.models_downloader.time.sleep"), \
             mock.patch("gui_app.models_downloader._fetch_sizes",
                        return_value={"model.gguf": 10}):
            r = md.download_files("ms", "org/repo", ["model.gguf"], str(root))
        self.assertTrue(r["ok"])
        self.assertEqual((root / "org" / "repo" / "model.gguf").read_bytes(), b"x" * 10)
        self.assertIn(md.MS_ENDPOINT + "/models/org/repo/resolve/master/model.gguf",
                      req.call_args.args[0])

    def test_unknown_source_rejected(self):
        with self.assertRaises(md.HttpError):
            md.api_search_models("xx", "q")


class TestClassifyFiles(unittest.TestCase):
    """gguf / mmproj / other 分类（mmproj 优先于 .gguf 后缀）。"""

    def test_classify(self):
        files = [
            {"type": "file", "path": "README.md", "size": 100},
            {"type": "file", "path": "model-Q4_K_M.gguf", "size": 1 << 20},
            {"type": "file", "path": "mmproj-Qwen-VL-f16.gguf", "size": 500},
            {"type": "file", "path": "sub/dir/model.gguf", "size": 300},
            {"type": "directory", "path": "sub"},
        ]
        r = md.classify_files(files)
        self.assertEqual([f["name"] for f in r["gguf"]],
                         ["model-Q4_K_M.gguf", "sub/dir/model.gguf"])
        self.assertEqual([f["name"] for f in r["mmproj"]], ["mmproj-Qwen-VL-f16.gguf"])
        self.assertEqual([f["name"] for f in r["other"]], ["README.md"])
        self.assertEqual(r["gguf"][0]["size"], 1 << 20)

    def test_mmproj_any_case(self):
        r = md.classify_files([{"type": "file", "path": "MMPROJ-X.gguf"}])
        self.assertEqual([f["name"] for f in r["mmproj"]], ["MMPROJ-X.gguf"])
        self.assertEqual(r["gguf"], [])


class TestDownloadFile(unittest.TestCase):
    """_download_file：跳过完整 / 续传(206) / 从头(200) / 416 已完整 / 校验失败重下 / 取消清理 / 失败保留断点。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.dest = Path(self._td.name) / "model.gguf"
        self.logs = []
        self.events = []

    def _run(self, urlopen_side_effect, prefill=0, cancel=None, expected_size=0):
        if prefill:
            self.dest.write_bytes(b"A" * prefill)
        with mock.patch("urllib.request.urlopen",
                        side_effect=urlopen_side_effect) as m, \
             mock.patch("urllib.request.Request",
                        wraps=urllib.request.Request) as req, \
             mock.patch("gui_app.models_downloader.time.sleep"):
            md._download_file("http://x/model.gguf", str(self.dest),
                              "model.gguf", self.logs.append,
                              self.events.append, cancel,
                              expected_size=expected_size)
        return m, req

    def test_fresh_download(self):
        m, req = self._run([_resp(200, b"x" * 100, cl=100)], expected_size=100)
        self.assertEqual(self.dest.read_bytes(), b"x" * 100)
        self.assertNotIn("Range", req.call_args.kwargs["headers"])
        self.assertTrue(any("开始下载" in s for s in self.logs))
        self.assertTrue(any("下载完成" in s for s in self.logs))
        self.assertEqual(self.events[-1]["type"], "file_done")
        self.assertTrue(any(e["type"] == "progress" and e["pct"] == 1.0
                            for e in self.events))

    def test_complete_file_skipped(self):
        m, req = self._run([_resp(200, b"x" * 100, cl=100)], prefill=100, expected_size=100)
        self.assertEqual(self.dest.read_bytes(), b"A" * 100)  # 原样保留
        self.assertEqual(m.call_count, 0)  # 已完整，不发请求
        self.assertTrue(any("已是完整文件" in s for s in self.logs))
        self.assertEqual(self.events[0]["type"], "file_skip")

    def test_resume_206(self):
        m, req = self._run([_resp(206, b"B" * 50, cl=50)], prefill=50, expected_size=100)
        self.assertEqual(self.dest.read_bytes(), b"A" * 50 + b"B" * 50)
        self.assertEqual(req.call_args.kwargs["headers"]["Range"], "bytes=50-")
        self.assertTrue(any("继续下载" in s for s in self.logs))

    def test_range_ignored_restarts(self):
        m, req = self._run([_resp(200, b"C" * 100, cl=100)], prefill=50, expected_size=100)
        self.assertEqual(self.dest.read_bytes(), b"C" * 100)  # 从头覆盖

    def test_416_complete_when_size_unknown(self):
        err = urllib.error.HTTPError("http://x", 416, "Range Not Satisfiable",
                                     {}, io.BytesIO(b""))
        with mock.patch("urllib.request.urlopen", side_effect=err), \
             mock.patch("urllib.request.Request", wraps=urllib.request.Request), \
             mock.patch("gui_app.models_downloader.time.sleep"):
            md._download_file("http://x/model.gguf", str(self.dest),
                              "model.gguf", self.logs.append,
                              expected_size=0)
        self.assertTrue(any("已是完整文件" in s for s in self.logs))

    def test_416_size_mismatch_redownloads(self):
        self.dest.write_bytes(b"A" * 50)
        err = urllib.error.HTTPError("http://x", 416, "Range Not Satisfiable",
                                     {}, io.BytesIO(b""))
        with mock.patch("urllib.request.urlopen",
                        side_effect=[err, _resp(200, b"x" * 100, cl=100)]), \
             mock.patch("urllib.request.Request", wraps=urllib.request.Request), \
             mock.patch("gui_app.models_downloader.time.sleep"):
            md._download_file("http://x/model.gguf", str(self.dest),
                              "model.gguf", self.logs.append,
                              expected_size=100)
        self.assertEqual(self.dest.read_bytes(), b"x" * 100)

    def test_verify_fail_redownloads(self):
        m, req = self._run([_resp(200, b"x" * 90, cl=90), _resp(200, b"x" * 100, cl=100)],
                           expected_size=100)
        self.assertEqual(self.dest.read_bytes(), b"x" * 100)
        self.assertEqual(m.call_count, 2)
        self.assertTrue(any("校验失败" in s for s in self.logs))

    def test_failure_keeps_partial(self):
        def _mk():
            r = mock.MagicMock()
            r.status = 200
            r.headers = {"Content-Length": "100"}
            r.read.side_effect = [b"x" * 50, OSError("reset")]
            r.__enter__.return_value = r
            r.__exit__.return_value = False
            return r

        with mock.patch("urllib.request.urlopen", side_effect=[_mk(), _mk(), _mk()]), \
             mock.patch("urllib.request.Request", wraps=urllib.request.Request), \
             mock.patch("gui_app.models_downloader.time.sleep"):
            with self.assertRaises(md.HttpError):
                md._download_file("http://x/model.gguf", str(self.dest),
                                  "model.gguf", self.logs.append,
                                  expected_size=100)
        self.assertEqual(self.dest.read_bytes(), b"x" * 50)  # 半成品保留，供续传

    def test_cancel_midway_deletes_partial(self):
        ev = threading.Event()

        def _read(_n):
            ev.set()
            return b"z" * 256

        r = mock.MagicMock()
        r.status = 200
        r.headers = {"Content-Length": "1000000"}
        r.read.side_effect = _read
        r.__enter__.return_value = r
        r.__exit__.return_value = False

        with mock.patch("urllib.request.urlopen", return_value=r), \
             mock.patch("urllib.request.Request", wraps=urllib.request.Request), \
             mock.patch("gui_app.models_downloader.time.sleep"):
            with self.assertRaises(md.DownloadCancelled):
                md._download_file("http://x/model.gguf", str(self.dest),
                                  "model.gguf", self.logs.append,
                                  None, ev, expected_size=1000000)
        self.assertFalse(self.dest.exists())  # 取消删除半成品

    def test_give_up_after_attempts(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=OSError("reset")), \
             mock.patch("gui_app.models_downloader.time.sleep"):
            with self.assertRaises(md.HttpError):
                md._download_file("http://x/model.gguf", str(self.dest),
                                  "model.gguf", self.logs.append)
        retries = [s for s in self.logs if "重试" in s]
        self.assertEqual(len(retries), 2)


class TestChunkedDownload(unittest.TestCase):
    """_download_file_chunked：合并 / 段内续传 / 中断重试 / 忽略Range回退 /
    失败降级+前缀合并 / 取消清理 / 异常段重置 / 小文件走单流。"""

    SIZE = 2048   # 4 段 × 512B（setUp 中把阈值调小，避免真实 200MB）

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.dest = Path(self._td.name) / "model.gguf"
        self.logs = []
        self.events = []
        self.data = bytes(range(256)) * 8
        for name, val in [("CHUNK_MIN_SIZE", 1024), ("CHUNK_SEG_SIZE", 512),
                          ("CHUNK_WORKERS", 3)]:
            p = mock.patch.object(md, name, val)
            p.start()
            self.addCleanup(p.stop)

    def _server(self, breaker=None):
        """按请求 Range 头切片返回数据并记录 (start, end)。

        breaker(start) 命中时截断响应体一半，模拟连接中途断流。
        """
        calls = []

        def _open(req, timeout=None):
            m = re.match(r"bytes=(\d+)-(\d+)", req.headers.get("Range") or "")
            if m:
                s, e = int(m.group(1)), int(m.group(2))
                body, status = self.data[s:e + 1], 206
            else:
                s, e, body, status = 0, len(self.data) - 1, self.data, 200
            calls.append((s, e))
            if breaker and breaker(s):
                body = body[:len(body) // 2]
            return _resp(status, body, cl=len(body))

        _open.calls = calls
        return _open

    def _chunked(self, side_effect, cancel=None):
        with mock.patch("urllib.request.urlopen", side_effect=side_effect), \
             mock.patch("gui_app.models_downloader.time.sleep"):
            md._download_file_chunked(
                "http://x/model.gguf", str(self.dest), "model.gguf",
                self.logs.append, self.events.append, cancel,
                expected_size=self.SIZE)

    def _parts(self):
        return sorted(self.dest.parent.glob("*.part*"))

    def test_full_merge(self):
        server = self._server()
        self._chunked(server)
        self.assertEqual(self.dest.read_bytes(), self.data)
        self.assertEqual(self._parts(), [])
        self.assertEqual(len(server.calls), 4)
        self.assertEqual(self.events[-1]["type"], "file_done")
        self.assertTrue(any(e["type"] == "progress" and e["pct"] == 1.0
                            for e in self.events))
        self.assertTrue(any("分片下载完成" in s for s in self.logs))

    def test_resume_from_existing_parts(self):
        Path(str(self.dest) + ".part0").write_bytes(self.data[:256])   # 半段
        Path(str(self.dest) + ".part1").write_bytes(self.data[512:1024])  # 完整段
        server = self._server()
        self._chunked(server)
        self.assertEqual(self.dest.read_bytes(), self.data)
        self.assertIn((256, 511), server.calls)     # 半段从断点续传
        self.assertNotIn((0, 511), server.calls)    # 不整段重下
        self.assertNotIn((512, 1023), server.calls)  # 完整段跳过

    def test_interrupted_segment_retries_from_offset(self):
        broken = set()

        def breaker(s):
            if s == 1024 and s not in broken:
                broken.add(s)
                return True
            return False

        server = self._server(breaker)
        self._chunked(server)
        self.assertEqual(self.dest.read_bytes(), self.data)
        self.assertIn((1024, 1535), server.calls)   # 首次请求（被截断 256B）
        self.assertIn((1280, 1535), server.calls)   # 重试从 1024+256 续传

    def test_range_ignored_raises_fallback(self):
        def _open(req, timeout=None):
            return _resp(200, self.data, cl=self.SIZE)

        with self.assertRaises(md.ChunkedFallback):
            self._chunked(_open)
        self.assertEqual(self._parts(), [])
        self.assertFalse(self.dest.exists())

    def test_download_files_range_ignored_falls_back(self):
        def _open(req, timeout=None):
            return _resp(200, self.data, cl=self.SIZE)

        with mock.patch("urllib.request.urlopen", side_effect=_open), \
             mock.patch("gui_app.models_downloader.time.sleep"), \
             mock.patch("gui_app.models_downloader._fetch_sizes",
                        return_value={"model.gguf": self.SIZE}):
            r = md.download_files(
                "hf", "unsloth/Qwen-GGUF", ["model.gguf"], str(self.dest.parent),
                log_fn=self.logs.append)
        self.assertTrue(r["ok"])
        self.assertEqual((self.dest.parent / "unsloth" / "Qwen-GGUF" / "model.gguf")
                         .read_bytes(), self.data)
        self.assertTrue(any("回退单流" in s for s in self.logs))

    def test_failure_degrades_with_prefix_merged(self):
        Path(str(self.dest) + ".part0").write_bytes(self.data[:512])
        with mock.patch("urllib.request.urlopen", side_effect=OSError("reset")), \
             mock.patch("gui_app.models_downloader.time.sleep"), \
             mock.patch.object(md, "_download_file") as df:
            md._download_file_chunked(
                "http://x/model.gguf", str(self.dest), "model.gguf",
                self.logs.append, self.events.append, None,
                expected_size=self.SIZE)
        df.assert_called_once()
        self.assertEqual(df.call_args.kwargs["expected_size"], self.SIZE)
        self.assertEqual(self.dest.read_bytes(), self.data[:512])  # 前缀段并入 dest
        self.assertEqual(self._parts(), [])
        self.assertTrue(any("降级单流" in s for s in self.logs))

    def test_cancel_cleans_parts(self):
        ev = threading.Event()

        def _read(_n):
            if not ev.is_set():
                ev.set()
                return b"z" * 256
            raise OSError("closed")

        def _open(req, timeout=None):
            r = mock.MagicMock()
            r.status = 206
            r.headers = {"Content-Length": "512"}
            r.read.side_effect = _read
            r.__enter__.return_value = r
            r.__exit__.return_value = False
            return r

        with mock.patch("urllib.request.urlopen", side_effect=_open), \
             mock.patch("gui_app.models_downloader.time.sleep"):
            with self.assertRaises(md.DownloadCancelled):
                md._download_file_chunked(
                    "http://x/model.gguf", str(self.dest), "model.gguf",
                    self.logs.append, self.events.append, ev,
                    expected_size=self.SIZE)
        self.assertEqual(self._parts(), [])
        self.assertFalse(self.dest.exists())

    def test_cancel_during_urlopen_returns_quickly(self):
        """urlopen 阻塞期间 cancel：worker 应在 1s 内退出并抛 DownloadCancelled。"""
        blocker = threading.Event()
        started = threading.Event()

        def blocking_open(req, timeout=None):
            started.set()
            blocker.wait(timeout=10)
            raise OSError("should be cancelled before urlopen returns")

        ev = threading.Event()
        result_box = {}

        def run():
            try:
                with mock.patch("urllib.request.urlopen", side_effect=blocking_open), \
                     mock.patch("gui_app.models_downloader.time.sleep"):
                    md._download_file_chunked(
                        "http://x/model.gguf", str(self.dest), "model.gguf",
                        self.logs.append, self.events.append, ev,
                        expected_size=self.SIZE)
            except BaseException as e:
                result_box["e"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        self.assertTrue(started.wait(timeout=3))
        time.sleep(0.3)
        t0 = time.time()
        ev.set()
        t.join(timeout=3)
        elapsed = time.time() - t0
        self.assertFalse(t.is_alive(), "chunked download did not return after cancel")
        self.assertLess(elapsed, 1.5, "cancel response too slow during urlopen")
        self.assertIsInstance(result_box.get("e"), md.DownloadCancelled)
        blocker.set()

    def test_small_file_uses_single_stream(self):
        def _open(req, timeout=None):
            return _resp(200, self.data[:512], cl=512)

        with mock.patch("urllib.request.urlopen", side_effect=_open), \
             mock.patch("gui_app.models_downloader._fetch_sizes",
                        return_value={"model.gguf": 512}):
            r = md.download_files(
                "hf", "unsloth/Qwen-GGUF", ["model.gguf"], str(self.dest.parent),
                log_fn=self.logs.append)
        self.assertTrue(r["ok"])
        self.assertEqual((self.dest.parent / "unsloth" / "Qwen-GGUF" / "model.gguf")
                         .read_bytes(), self.data[:512])
        self.assertFalse(any("分片下载" in s for s in self.logs))

    def test_oversized_part_reset_and_redownloaded(self):
        Path(str(self.dest) + ".part0").write_bytes(b"X" * 999)
        server = self._server()
        self._chunked(server)
        self.assertEqual(self.dest.read_bytes(), self.data)
        self.assertIn((0, 511), server.calls)   # 异常段重置后从段首重下


class TestDownloadFiles(unittest.TestCase):
    """download_files：两级目录组织 / 失败继续（保留断点）/ 取消中断 / 非法路径拒绝。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.root = Path(self._td.name) / "models"
        self.logs = []

    def _run(self, urlopen_side_effect, files=("model.gguf", "mmproj-x.gguf"),
             cancel=None, cleanup=False):
        sizes = {f: 10 for f in files}
        with mock.patch("urllib.request.urlopen",
                        side_effect=urlopen_side_effect), \
             mock.patch("gui_app.models_downloader.time.sleep"), \
             mock.patch("gui_app.models_downloader._fetch_sizes", return_value=sizes):
            return md.download_files(
                "hf", "unsloth/Qwen-GGUF", list(files), str(self.root),
                log_fn=self.logs.append, cancel_event=cancel,
                cleanup_on_cancel=cleanup)

    def test_two_level_layout_and_progress_order(self):
        events = []
        with mock.patch("urllib.request.urlopen", side_effect=[
                _resp(200, b"a" * 10, cl=10),
                _resp(200, b"b" * 10, cl=10)]), \
             mock.patch("gui_app.models_downloader.time.sleep"), \
             mock.patch("gui_app.models_downloader._fetch_sizes",
                        return_value={"model.gguf": 10, "mmproj-x.gguf": 10}):
            r = md.download_files(
                "hf", "unsloth/Qwen-GGUF", ["model.gguf", "mmproj-x.gguf"],
                str(self.root), log_fn=self.logs.append, progress_cb=events.append)
        self.assertTrue(r["ok"])
        self.assertEqual(r["downloaded"], 2)
        self.assertEqual(r["failed"], [])
        self.assertFalse(r["cancelled"])
        self.assertEqual((self.root / "unsloth" / "Qwen-GGUF" / "model.gguf")
                         .read_bytes(), b"a" * 10)
        self.assertEqual((self.root / "unsloth" / "Qwen-GGUF" / "mmproj-x.gguf")
                         .read_bytes(), b"b" * 10)
        self.assertTrue(r["dir"].endswith(os.path.join("unsloth", "Qwen-GGUF")))
        self.assertIn(str(self.root), r["dir"])
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "file_start")
        self.assertEqual(types[-1], "file_done")
        starts = [e for e in events if e["type"] == "file_start"]
        self.assertEqual([e["file"] for e in starts],
                         ["model.gguf", "mmproj-x.gguf"])

    def test_failure_continues_next(self):
        boom = OSError("reset")
        r = self._run([boom, boom, boom, _resp(200, b"b" * 10, cl=10)])
        self.assertFalse(r["ok"])
        self.assertEqual(r["downloaded"], 1)
        self.assertEqual(len(r["failed"]), 1)
        self.assertEqual(r["failed"][0]["file"], "model.gguf")
        self.assertIn("失败: model.gguf", "\n".join(self.logs))
        self.assertTrue((self.root / "unsloth" / "Qwen-GGUF" / "mmproj-x.gguf")
                        .exists())

    def test_cancel_stops_remaining(self):
        ev = threading.Event()
        ev.set()
        r = self._run([_resp(200, b"a" * 10, cl=10)], cancel=ev)
        self.assertTrue(r["cancelled"])
        self.assertEqual(r["downloaded"], 0)
        self.assertFalse((self.root / "unsloth" / "Qwen-GGUF" / "model.gguf").exists())
        self.assertTrue(any("已取消" in s for s in self.logs))

    def test_cancel_with_cleanup_deletes_repo_dir(self):
        ev = threading.Event()
        ev.set()
        r = self._run([_resp(200, b"a" * 10, cl=10)], cancel=ev, cleanup=True)
        self.assertTrue(r["cancelled"])
        self.assertFalse((self.root / "unsloth" / "Qwen-GGUF").exists())
        self.assertTrue(any("已清理" in s for s in self.logs), self.logs)

    def test_cancel_without_cleanup_removes_partial(self):
        ev = threading.Event()
        ev.set()
        r = self._run([_resp(200, b"a" * 10, cl=10)], cancel=ev)
        self.assertTrue(r["cancelled"])
        self.assertTrue((self.root / "unsloth" / "Qwen-GGUF").exists())
        self.assertFalse((self.root / "unsloth" / "Qwen-GGUF" / "model.gguf").exists())
        self.assertTrue(any("已取消" in s for s in self.logs), self.logs)

    def test_invalid_path_rejected_without_download(self):
        r = self._run([_resp(200, b"a" * 10, cl=10)],
                      files=("../escape.gguf", "ok.gguf"))
        self.assertFalse(r["ok"])
        self.assertEqual(r["failed"][0]["file"], "../escape.gguf")
        self.assertEqual(r["failed"][0]["error"], "路径不合法")
        self.assertTrue((self.root / "unsloth" / "Qwen-GGUF" / "ok.gguf").exists())


class TestDownloadUrls(unittest.TestCase):
    """download_urls：直连下载 / label 显示 / 取消 / 磁盘预检。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.root = Path(self._td.name)
        self.logs = []

    def _dl(self, urlopen_side_effect, items, cancel=None):
        with mock.patch("urllib.request.urlopen",
                        side_effect=urlopen_side_effect), \
             mock.patch("urllib.request.Request", wraps=urllib.request.Request), \
             mock.patch("gui_app.models_downloader.time.sleep"):
            return md.download_urls(items, str(self.root),
                                    log_fn=self.logs.append, cancel_event=cancel)

    def test_download_with_label(self):
        r = self._dl([_resp(200, b"x" * 10, cl=10)],
                     [{"url": "http://x/pkg.zip", "filename": "pkg.zip",
                       "size": 10, "label": "预编译"}])
        self.assertTrue(r["ok"])
        self.assertEqual(r["downloaded"], 1)
        self.assertEqual((self.root / "pkg.zip").read_bytes(), b"x" * 10)
        self.assertTrue(any("开始下载: 预编译" in s for s in self.logs))
        self.assertTrue((self.root / "pkg.zip").exists())

    def test_resume_206(self):
        dest = self.root / "pkg.zip"
        dest.write_bytes(b"A" * 5)
        r = self._dl([_resp(206, b"y" * 5, cl=5)],
                     [{"url": "http://x/pkg.zip", "filename": "pkg.zip", "size": 10}])
        self.assertTrue(r["ok"])
        self.assertEqual(dest.read_bytes(), b"A" * 5 + b"y" * 5)

    def test_cancelled_flag(self):
        ev = threading.Event()
        ev.set()
        r = self._dl([_resp(200, b"x" * 10, cl=10)],
                     [{"url": "http://x/pkg.zip", "filename": "pkg.zip", "size": 10}],
                     cancel=ev)
        self.assertTrue(r["cancelled"])
        self.assertFalse((self.root / "pkg.zip").exists())

    def test_disk_shortage_rejected(self):
        with mock.patch("gui_app.models_downloader.shutil.disk_usage") as du, \
             mock.patch("gui_app.models_downloader.os.makedirs"):
            du.return_value.free = 5
            r = md.download_urls([{"url": "http://x/a.bin", "filename": "a.bin",
                                   "size": 10 ** 9}], str(self.root))
        self.assertFalse(r["ok"])
        self.assertIn("磁盘空间不足", r["error"])

    def test_invalid_filename_rejected(self):
        r = self._dl([_resp(200, b"x" * 10, cl=10)],
                     [{"url": "http://x/a", "filename": "../a", "size": 10}])
        self.assertFalse(r["ok"])
        self.assertEqual(r["failed"][0]["error"], "文件名不合法")


class TestConcurrencyGuard(unittest.TestCase):
    """begin/end_download 占坑：重复发起被拒绝。"""

    def test_begin_twice_rejected(self):
        self.assertTrue(md.begin_download())
        try:
            self.assertFalse(md.begin_download())  # 占坑中 → 拒绝
        finally:
            md.end_download()
        self.assertTrue(md.begin_download())  # 释放后可再次开始
        md.end_download()


class TestModelDownloadMixin(unittest.TestCase):
    """Mixin：返回结构 + js_pusher 进度推送 + 并发保护。"""

    def setUp(self):
        from gui_app.api_mixins.models import ModelDownloadMixin
        self.api = ModelDownloadMixin()

    def test_search_models_ok(self):
        with mock.patch("gui_app.models_downloader.api_search_models",
                        return_value=([{"id": "unsloth/Qwen-GGUF", "downloads": 3}], "cur1")) as m:
            r = self.api.model_search("hf", "gguf", sort="downloads")
        self.assertTrue(r["ok"])
        self.assertEqual(r["source"], "hf")
        self.assertEqual(r["results"][0]["id"], "unsloth/Qwen-GGUF")
        self.assertEqual(r["next_cursor"], "cur1")
        self.assertTrue(r["has_more"])
        m.assert_called_once_with("hf", "gguf", sort="downloads", limit=20, cursor="")

    def test_search_models_last_page(self):
        with mock.patch("gui_app.models_downloader.api_search_models",
                        return_value=([], None)):
            r = self.api.model_search("hf", "gguf")
        self.assertTrue(r["ok"])
        self.assertEqual(r["next_cursor"], "")
        self.assertFalse(r["has_more"])

    def test_open_repo(self):
        with mock.patch("webbrowser.open") as mo:
            r = self.api.model_open_repo("hf", "unsloth/Qwen-GGUF")
        self.assertTrue(r["ok"])
        self.assertEqual(r["url"], md.HF_ENDPOINT + "/unsloth/Qwen-GGUF")
        mo.assert_called_once_with(r["url"])

    def test_open_repo_rejects_bad_id(self):
        with mock.patch("webbrowser.open") as mo:
            r = self.api.model_open_repo("hf", "../etc/passwd")
        self.assertFalse(r["ok"])
        mo.assert_not_called()

    def test_search_models_error(self):
        with mock.patch("gui_app.models_downloader.api_search_models",
                        side_effect=OSError("network unreachable")):
            r = self.api.model_search("hf", "gguf")
        self.assertFalse(r["ok"])
        self.assertIn("搜索失败", r["error"])

    def test_repo_files_ok(self):
        files = [{"type": "file", "path": "model.gguf", "size": 10}]
        with mock.patch("gui_app.models_downloader.api_list_files", return_value=files) as m, \
             mock.patch("gui_app.llama_cpp.get_models_dir",
                        return_value=Path("M:/models")):
            r = self.api.model_repo_files("hf", "unsloth/Qwen-GGUF")
        self.assertTrue(r["ok"])
        self.assertEqual([f["name"] for f in r["gguf"]], ["model.gguf"])
        self.assertEqual(r["mmproj"], [])
        self.assertEqual(r["models_dir"], str(Path("M:/models")))
        m.assert_called_once_with("hf", "unsloth/Qwen-GGUF")

    def test_download_models_pushes_progress(self):
        events = []

        def _fake_download(source, repo_id, files, dest_root, log_fn=None,
                           progress_cb=None, cancel_event=None, revision="",
                           cleanup_on_cancel=False):
            log_fn("开始下载")
            progress_cb({"type": "file_start", "file": "model.gguf", "idx": 1, "count": 1})
            progress_cb({"type": "progress", "file": "model.gguf",
                         "done": 50, "total": 100, "pct": 0.5})
            progress_cb({"type": "file_done", "file": "model.gguf", "size": 100})
            return {"ok": True, "downloaded": 1, "failed": [], "cancelled": False,
                    "dir": "M:/models/u/q"}

        with mock.patch("gui_app.models_downloader.begin_download", return_value=True) as begin, \
             mock.patch("gui_app.models_downloader.download_files", side_effect=_fake_download), \
             mock.patch("gui_app.models_downloader.end_download") as end, \
             mock.patch("gui_app.llama_cpp.get_models_dir",
                        return_value=Path("M:/models")), \
             mock.patch("gui_app.api_mixins.models.js_pusher") as pusher:
            r = self.api.model_download("hf", "u/q", ["model.gguf"])

        self.assertTrue(r["ok"])
        self.assertEqual(r["downloaded"], 1)
        begin.assert_called_once_with()
        end.assert_called_once_with()
        # 进度事件推送到前端
        pushed = [c.args[0] for c in pusher.push.call_args_list]
        self.assertIn("hfDownloadProgress", pushed)
        self.assertIn("appendLog", pushed)
        ev = [c.args[1] for c in pusher.push.call_args_list
              if c.args[0] == "hfDownloadProgress"]
        self.assertEqual(ev[0]["type"], "file_start")
        # 结束事件（后台下载收尾用）：结果与返回值一致，且只推一次
        done = [c.args[1] for c in pusher.push.call_args_list
                if c.args[0] == "hfDownloadDone"]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["downloaded"], 1)
        self.assertFalse(done[0]["cancelled"])

    def test_download_rejected_when_busy(self):
        with mock.patch("gui_app.models_downloader.begin_download", return_value=False), \
             mock.patch("gui_app.models_downloader.download_files") as dl:
            r = self.api.model_download("hf", "u/q", ["model.gguf"])
        self.assertFalse(r["ok"])
        self.assertIn("进行中", r["error"])
        dl.assert_not_called()

    def test_download_empty_files_rejected(self):
        with mock.patch("gui_app.models_downloader.begin_download") as begin:
            r = self.api.model_download("hf", "u/q", [])
        self.assertFalse(r["ok"])
        begin.assert_not_called()

    def test_cancel(self):
        with mock.patch("gui_app.models_downloader.set_cancel") as sc:
            r = self.api.model_cancel_download()
        self.assertTrue(r["ok"])
        sc.assert_called_once_with()

    def test_download_exception_still_pushes_done(self):
        # 异常路径也必须推 hfDownloadDone，否则后台下载时前端无法收尾（按钮/圆环卡住）
        with mock.patch("gui_app.models_downloader.begin_download", return_value=True), \
             mock.patch("gui_app.models_downloader.download_files",
                        side_effect=RuntimeError("boom")), \
             mock.patch("gui_app.models_downloader.end_download"), \
             mock.patch("gui_app.llama_cpp.get_models_dir",
                        return_value=Path("M:/models")), \
             mock.patch("gui_app.api_mixins.models.js_pusher") as pusher:
            r = self.api.model_download("hf", "u/q", ["model.gguf"])
        self.assertFalse(r["ok"])
        self.assertIn("boom", r["error"])
        done = [c.args[1] for c in pusher.push.call_args_list
                if c.args[0] == "hfDownloadDone"]
        self.assertEqual(len(done), 1)
        self.assertIn("boom", done[0]["error"])


if __name__ == "__main__":
    unittest.main()
