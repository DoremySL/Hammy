import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app import llama_cpp
from gui_app import models_downloader as md


class TestDownloadUrlsParallelStall(unittest.TestCase):
    """download_urls_parallel（llama.cpp 安装入口）：取消立即解除停滞的网络读。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.dest = Path(self._td.name) / "pkg.zip"
        self.logs = []

    def test_cancel_unblocks_stalled_read(self):
        ev = threading.Event()
        release = threading.Event()
        result = {}

        class StalledResp:
            """read() 阻塞直到 close()：模拟网络停滞的连接。"""
            status = 200
            headers = {"Content-Length": "100"}

            def read(self, n=-1):
                release.wait(10)  # 连接被 watcher 关闭前一直阻塞
                raise OSError("connection closed")

            def close(self):
                release.set()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _run():
            try:
                r = md.download_urls_parallel(
                    [{"url": "http://x/pkg.zip", "filename": self.dest.name,
                      "size": 100, "label": "预编译"}],
                    str(self.dest.parent), log_fn=self.logs.append,
                    cancel_event=ev)
                result["r"] = r
            except BaseException as e:
                result["exc"] = e

        with mock.patch("urllib.request.urlopen", return_value=StalledResp()), \
             mock.patch("gui_app.models_downloader.time.sleep"):
            t = threading.Thread(target=_run)
            t.start()
            time.sleep(0.3)  # 等监视线程就绪、read 进入阻塞
            t0 = time.time()
            ev.set()         # 用户点击停止：watcher 立即关闭响应
            t.join(5)
        elapsed = time.time() - t0
        self.assertFalse(t.is_alive())  # 秒级停止，未等满超时周期
        self.assertLess(elapsed, 2.0)
        self.assertNotIn("exc", result)
        self.assertTrue(result["r"]["cancelled"])
        self.assertFalse(self.dest.exists())


class TestInstallPartialKeep(unittest.TestCase):
    """install：下载失败/取消时清理半成品压缩包，下次安装从头下载。"""

    def _fake_build(self):
        return {"type": "cuda", "ver": "13", "key": "cuda-13",
                "precompiled_url": "http://x/pre.zip",
                "precompiled_size": 100,
                "cudart_url": "http://x/cudart.zip",
                "cudart_size": 100}

    def _patch_env(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        llama_dir = Path(td.name) / "llama.cpp"
        p1 = mock.patch.object(llama_cpp, "LLAMA_DIR", llama_dir)
        p2 = mock.patch("gui_app.llama_cpp.get_latest_release_assets",
                        return_value={"ok": True, "builds": [self._fake_build()]})
        p3 = mock.patch("gui_app.llama_cpp._parse_build_sel",
                        return_value=self._fake_build())
        p1.start(); p2.start(); p3.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop); self.addCleanup(p3.stop)
        return llama_dir

    @staticmethod
    def _fake_dl_result(items, dest_root, status="fail", cancel_event=None):
        for it in items:
            dest = Path(dest_root) / it["filename"]
            if status == "zip":
                with zipfile.ZipFile(dest, "w") as z:
                    z.writestr("llama-server.exe", "MZ")
            elif status == "readme_zip":
                with zipfile.ZipFile(dest, "w") as z:
                    z.writestr("readme.txt", "hi")
            else:
                dest.write_bytes(b"partial")
        if status == "cancel":
            if cancel_event is not None:
                cancel_event.set()
            return {"ok": False, "cancelled": True, "downloaded": 0,
                    "failed": [], "dir": dest_root}
        if status == "fail":
            return {"ok": False, "cancelled": False, "downloaded": 0,
                    "failed": [{"file": "预编译", "error": "connection reset"}],
                    "dir": dest_root}
        return {"ok": True, "cancelled": False, "downloaded": len(items),
                "failed": [], "dir": dest_root}

    def test_failed_download_cleans_partial(self):
        llama_dir = self._patch_env()
        pre_zip = llama_dir / "_pre_cuda-13.zip"

        def _boom(items, dest_root, log_fn=None, progress_cb=None,
                  cancel_event=None, cleanup_on_cancel=False):
            return self._fake_dl_result(items, dest_root, status="fail")

        with mock.patch("gui_app.models_downloader.download_urls_parallel", side_effect=_boom):
            r = llama_cpp.install("cuda-13")
        self.assertFalse(r["ok"])  # 下载失败 → 安装失败
        self.assertIn("下载失败", r["error"])
        self.assertFalse(pre_zip.exists())  # 半成品已清理，下次安装从头下载

    def test_failed_download_logs_manual_hint(self):
        llama_dir = self._patch_env()
        logs = []

        def _boom(items, dest_root, log_fn=None, progress_cb=None,
                  cancel_event=None, cleanup_on_cancel=False):
            log_fn("engine log")
            return self._fake_dl_result(items, dest_root, status="fail")

        with mock.patch("gui_app.models_downloader.download_urls_parallel", side_effect=_boom):
            r = llama_cpp.install("cuda-13", log_fn=logs.append)
        self.assertFalse(r["ok"])
        # 网络失败时日志给出「稍后再试 / 手动下载解压」引导与 release 页地址
        self.assertTrue(any("请稍后再试" in s for s in logs), logs)
        self.assertTrue(any("手动下载解压" in s for s in logs), logs)
        self.assertTrue(any("github.com/ggml-org/llama.cpp/releases" in s for s in logs), logs)
        self.assertTrue(any("llama.cpp" in s and "文件夹" in s for s in logs), logs)

    def test_cancel_mid_download_cleans_zips(self):
        """下载中点停止：返回 cancelled，半成品压缩包已清理，stop_event 传入下载。"""
        llama_dir = self._patch_env()
        pre_zip = llama_dir / "_pre_cuda-13.zip"
        ev = threading.Event()
        seen = {}

        def _cancel_dl(items, dest_root, log_fn=None, progress_cb=None,
                       cancel_event=None, cleanup_on_cancel=False):
            seen["cancel_event"] = cancel_event
            return self._fake_dl_result(items, dest_root, status="cancel",
                                        cancel_event=cancel_event)

        with mock.patch("gui_app.models_downloader.download_urls_parallel", side_effect=_cancel_dl):
            r = llama_cpp.install("cuda-13", stop_event=ev)
        self.assertFalse(r["ok"])
        self.assertTrue(r["cancelled"])
        self.assertIs(seen["cancel_event"], ev)
        self.assertFalse(pre_zip.exists())  # 半成品已清理，下次安装从头下载

    def test_cancel_before_start_no_partial(self):
        """安装开始前已取消：直接返回 cancelled，不产生任何压缩包。"""
        llama_dir = self._patch_env()
        ev = threading.Event()
        ev.set()
        r = llama_cpp.install("cuda-13", stop_event=ev)
        self.assertFalse(r["ok"])
        self.assertTrue(r["cancelled"])
        self.assertFalse((llama_dir / "_pre_cuda-13.zip").exists())

    def test_release_fetch_failure_logs_manual_hint(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        llama_dir = Path(td.name) / "llama.cpp"
        logs = []

        with mock.patch.object(llama_cpp, "LLAMA_DIR", llama_dir), \
             mock.patch("gui_app.llama_cpp.get_latest_release_assets",
                        side_effect=OSError("network unreachable")):
            r = llama_cpp.install("cuda-13", log_fn=logs.append)
        self.assertFalse(r["ok"])
        self.assertIn("获取 release 失败", r["error"])
        self.assertTrue(any("请稍后再试" in s for s in logs), logs)
        self.assertTrue(any("手动下载解压" in s for s in logs), logs)

    def test_unpack_failure_cleans_zips(self):
        llama_dir = self._patch_env()
        pre_zip = llama_dir / "_pre_cuda-13.zip"

        def _write_zip(items, dest_root, log_fn=None, progress_cb=None,
                       cancel_event=None, cleanup_on_cancel=False):
            return self._fake_dl_result(items, dest_root, status="readme_zip")

        with mock.patch("gui_app.models_downloader.download_urls_parallel", side_effect=_write_zip):
            r = llama_cpp.install("cuda-13")
        self.assertFalse(r["ok"])  # 解压后无 llama-server.exe → 布局异常
        self.assertFalse(pre_zip.exists())  # 下载已完成 → 清理 zip

    def _capture_llama_config(self):
        """patch update_llama_config：捕获 mutator 应用后的配置。"""
        writes = []

        def _fake_update(mutator):
            cfg = {"enabled": False, "integrate": False}
            mutator(cfg)
            writes.append(dict(cfg))

        return mock.patch("gui_app.config_store.update_llama_config",
                          side_effect=_fake_update), writes

    def test_install_success_auto_enables(self):
        """自动下载安装成功后自动开启总开关与本地推理集成。"""
        llama_dir = self._patch_env()
        p, writes = self._capture_llama_config()

        def _write_zip(items, dest_root, log_fn=None, progress_cb=None,
                       cancel_event=None, cleanup_on_cancel=False):
            return self._fake_dl_result(items, dest_root, status="zip")

        with p, mock.patch("gui_app.models_downloader.download_urls_parallel",
                           side_effect=_write_zip):
            r = llama_cpp.install("cuda-13")
        self.assertTrue(r["ok"])
        self.assertEqual(writes[-1], {"enabled": True, "integrate": True})
        self.assertFalse((llama_dir / "_pre_cuda-13.zip").exists())  # 成功后清包

    def test_manual_install_auto_enables(self):
        """手动安装模式同样自动开启总开关与本地推理集成。"""
        self._patch_env()
        p, writes = self._capture_llama_config()
        with p:
            r = llama_cpp.install("manual")
        self.assertTrue(r["ok"])
        self.assertTrue(r["manual"])
        self.assertEqual(writes[-1], {"enabled": True, "integrate": True})

    def test_remove_success_auto_disables(self):
        """卸载成功后自动关闭总开关与本地推理集成。"""
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        llama_dir = Path(td.name) / "llama.cpp"
        llama_dir.mkdir(parents=True)
        (llama_dir / "llama-server.exe").write_bytes(b"MZ")
        p, writes = self._capture_llama_config()

        with mock.patch.object(llama_cpp, "LLAMA_DIR", llama_dir), \
             mock.patch("gui_app.llama_cpp.stop"), p:
            r = llama_cpp.remove()
        self.assertTrue(r["ok"])
        self.assertEqual(writes[-1], {"enabled": False, "integrate": False})
        self.assertFalse(llama_dir.exists())

    def test_remove_failure_keeps_switches(self):
        """删除失败时如实返回，不自动关闭开关（保持原状态供重试）。"""
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        llama_dir = Path(td.name) / "llama.cpp"
        llama_dir.mkdir(parents=True)
        (llama_dir / "llama-server.exe").write_bytes(b"MZ")

        with mock.patch.object(llama_cpp, "LLAMA_DIR", llama_dir), \
             mock.patch("gui_app.llama_cpp.stop"), \
             mock.patch("gui_app.llama_cpp.shutil.rmtree",
                        side_effect=OSError("拒绝访问")), \
             mock.patch("gui_app.config_store.update_llama_config") as upd:
            r = llama_cpp.remove()
        self.assertFalse(r["ok"])
        upd.assert_not_called()


def _fake_release(tag="b123"):
    return {"tag_name": tag,
            "assets": [{"name": "llama-b123-bin-win-cpu-x64.zip",
                        "browser_download_url": "http://x/cpu.zip", "size": 1}]}


class TestReleaseCache(unittest.TestCase):
    """get_latest_release_assets 的缓存 TTL 与并发等待语义。"""

    def setUp(self):
        llama_cpp._cached_release = None
        llama_cpp._cached_release_at = 0.0
        llama_cpp._release_fetching = False
        self.addCleanup(setattr, llama_cpp, "_cached_release", None)
        self.addCleanup(setattr, llama_cpp, "_cached_release_at", 0.0)
        self.addCleanup(setattr, llama_cpp, "_release_fetching", False)

    def _gpu(self):
        return mock.patch.object(llama_cpp, "detect_gpu",
                                 return_value={"cuda_max": "", "gpu_name": "x"})

    def test_waiter_rejects_stale_cache_and_refetches(self):
        # 抓取在途时等待者不得返回过期缓存；抓取失败后等待者自行重试（也失败则报错）
        llama_cpp._cached_release = {"ok": True, "stale": True}
        llama_cpp._cached_release_at = time.time() - 10 * llama_cpp._CACHE_TTL
        calls = []

        def fetch(*a, **k):
            calls.append(1)
            time.sleep(0.4)
            raise RuntimeError("network down")

        with mock.patch.object(llama_cpp, "_fetch_json", side_effect=fetch), self._gpu():
            def _run_a():
                try:
                    llama_cpp.get_latest_release_assets()
                except RuntimeError:
                    pass
            ta = threading.Thread(target=_run_a)
            ta.start()
            time.sleep(0.1)  # A 已进入抓取
            with self.assertRaises(RuntimeError):
                llama_cpp.get_latest_release_assets()  # B：等待者
            ta.join()
        self.assertEqual(len(calls), 2)  # A 失败后 B 自己重抓了一次

    def test_waiter_gets_fresh_cache_after_success(self):
        # 首个线程抓取成功后，等待者直接复用同一结果（不重复抓取）
        result = {}
        calls = []

        def fetch(*a, **k):
            calls.append(1)
            time.sleep(0.3)
            return [_fake_release()]

        with mock.patch.object(llama_cpp, "_fetch_json", side_effect=fetch), self._gpu():
            ta = threading.Thread(target=lambda: result.update(
                r=llama_cpp.get_latest_release_assets()))
            ta.start()
            time.sleep(0.1)
            rb = llama_cpp.get_latest_release_assets()
            ta.join()
        self.assertIs(result["r"], rb)
        self.assertEqual(len(calls), 1)

    def test_cache_timestamp_set_after_fetch(self):
        # TTL 从抓取完成时刻起算：抓取耗时超过 TTL 后缓存仍有效
        calls = []

        def slow_fetch(*a, **k):
            calls.append(1)
            time.sleep(0.35)
            return [_fake_release()]

        with mock.patch.object(llama_cpp, "_CACHE_TTL", 0.2), \
             mock.patch.object(llama_cpp, "_fetch_json", side_effect=slow_fetch), self._gpu():
            r1 = llama_cpp.get_latest_release_assets()
            r2 = llama_cpp.get_latest_release_assets()
        self.assertIs(r2, r1)  # 若时间戳写于抓取前，age 0.35s > TTL 0.2s 会 miss 重抓
        self.assertEqual(len(calls), 1)


class TestParallelDownloadBehavior(unittest.TestCase):
    """download_urls_parallel：两包并发、共用总进度、取消仅清理未完成文件。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.root = Path(self._td.name)

    def _items(self):
        return [{"url": "u1", "filename": "a.zip", "size": 100, "label": "预编译"},
                {"url": "u2", "filename": "b.zip", "size": 200, "label": "cudart"}]

    def test_combined_progress_and_success(self):
        events, logs = [], []

        def fake_df(url, dest, filename, log_fn, progress_cb=None, cancel_event=None,
                    expected_size=0, timeout=30):
            for done in (expected_size // 2, expected_size):
                if progress_cb:
                    progress_cb({"type": "progress", "file": filename,
                                 "done": done, "total": expected_size,
                                 "pct": done / expected_size})
            Path(dest).write_bytes(b"x" * expected_size)
            if progress_cb:
                progress_cb({"type": "file_done", "file": filename, "size": expected_size})

        with mock.patch.object(md, "_download_file", side_effect=fake_df):
            r = md.download_urls_parallel(self._items(), str(self.root),
                                          log_fn=logs.append, progress_cb=events.append)
        self.assertTrue(r["ok"])
        self.assertEqual(r["downloaded"], 2)
        progress = [e for e in events if e["type"] == "progress"]
        self.assertTrue(progress)
        self.assertTrue(all(p["total"] == 300 for p in progress))
        dones = [p["done"] for p in progress]
        self.assertEqual(dones, sorted(dones))
        self.assertEqual(progress[-1]["done"], 300)
        self.assertAlmostEqual(progress[-1]["pct"], 1.0)
        self.assertTrue((self.root / "a.zip").exists())
        self.assertTrue((self.root / "b.zip").exists())

    def test_cancel_keeps_completed_cleans_partial(self):
        ev = threading.Event()
        a_done = threading.Event()

        def fake_df(url, dest, filename, log_fn, progress_cb=None, cancel_event=None,
                    expected_size=0, timeout=30):
            if filename == "预编译":
                Path(dest).write_bytes(b"x" * expected_size)
                a_done.set()
                return
            a_done.wait(5)
            Path(dest).write_bytes(b"partial")
            ev.set()
            raise md.DownloadCancelled()

        with mock.patch.object(md, "_download_file", side_effect=fake_df):
            r = md.download_urls_parallel(self._items(), str(self.root),
                                          cancel_event=ev)
        self.assertFalse(r["ok"])
        self.assertTrue(r["cancelled"])
        self.assertFalse((self.root / "b.zip").exists())
        self.assertTrue((self.root / "a.zip").exists())

    def test_stopped_before_start_no_files(self):
        ev = threading.Event()
        ev.set()
        r = md.download_urls_parallel(self._items(), str(self.root), cancel_event=ev)
        self.assertTrue(r["cancelled"])
        self.assertFalse((self.root / "a.zip").exists())
        self.assertFalse((self.root / "b.zip").exists())

    def test_staggered_connection_start(self):
        """并发起步按序号错峰：仅第 2 个条目延迟 1 秒再连。"""
        sleeps, calls = [], []

        def fake_df(url, dest, filename, log_fn, progress_cb=None, cancel_event=None,
                    expected_size=0, timeout=30):
            calls.append(filename)
            Path(dest).write_bytes(b"x")

        def fake_sleep(sec):
            sleeps.append(sec)

        with mock.patch.object(md, "_download_file", side_effect=fake_df), \
             mock.patch("gui_app.models_downloader.time.sleep", side_effect=fake_sleep):
            r = md.download_urls_parallel(self._items(), str(self.root))
        self.assertTrue(r["ok"])
        self.assertEqual(sleeps, [1])
        self.assertEqual(calls[0], "预编译")

    def test_big_file_uses_single_stream(self):
        """llama 安装不走分片多连接：超大文件也只用单流下载。"""
        calls = []

        def fake_chunked(*a, **k):
            calls.append("chunked")
            return None

        def fake_df(url, dest, filename, log_fn, progress_cb=None, cancel_event=None,
                    expected_size=0, timeout=30):
            calls.append("single")
            Path(dest).write_bytes(b"x" * expected_size)

        with mock.patch.object(md, "_download_file_chunked", side_effect=fake_chunked), \
             mock.patch.object(md, "_download_file", side_effect=fake_df):
            r = md.download_urls_parallel(
                [{"url": "u", "filename": "big.zip",
                  "size": 300 * 1024 * 1024, "label": "预编译"}],
                str(self.root))
        self.assertTrue(r["ok"])
        self.assertEqual(calls, ["single"])


if __name__ == "__main__":
    unittest.main()
