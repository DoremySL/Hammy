import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app import llama_cpp


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


class TestDownloadResume(unittest.TestCase):
    """_download：断点续传（Range / 206 / 200 / 416）+ 失败重试。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.dest = Path(self._td.name) / "pkg.zip"
        self.logs = []

    def _run(self, urlopen_side_effect, prefill=0):
        if prefill:
            self.dest.write_bytes(b"A" * prefill)
        with mock.patch("urllib.request.urlopen",
                        side_effect=urlopen_side_effect) as m, \
             mock.patch("urllib.request.Request",
                        wraps=urllib.request.Request) as req, \
             mock.patch("gui_app.llama_cpp.time.sleep"):
            llama_cpp._download("http://x/pkg.zip", self.dest,
                                self.logs.append, "包")
        return m, req

    def test_fresh_download(self):
        m, req = self._run([_resp(200)])
        self.assertEqual(self.dest.read_bytes(), b"x" * 100)
        self.assertEqual(req.call_args.args[0], "http://x/pkg.zip")
        self.assertNotIn("Range", req.call_args.kwargs["headers"])
        self.assertTrue(any("开始下载" in s for s in self.logs))
        self.assertTrue(any("[完成]" in s for s in self.logs))

    def test_resume_from_206(self):
        m, req = self._run([_resp(206, b"y" * 50, cl=50)], prefill=100)
        self.assertEqual(self.dest.read_bytes(), b"A" * 100 + b"y" * 50)
        self.assertEqual(req.call_args.kwargs["headers"]["Range"], "bytes=100-")
        self.assertTrue(any("继续下载" in s for s in self.logs))
        self.assertTrue(any("（100%" in s for s in self.logs))  # 150/150 到 100%

    def test_server_ignores_range_restarts(self):
        m, req = self._run([_resp(200, b"z" * 30, cl=30)], prefill=100)
        self.assertEqual(self.dest.read_bytes(), b"z" * 30)  # 从头覆盖旧内容
        self.assertEqual(req.call_args.kwargs["headers"]["Range"], "bytes=100-")
        self.assertTrue(any("开始下载" in s for s in self.logs))

    def test_416_means_complete(self):
        e = urllib.error.HTTPError("http://x/pkg.zip", 416,
                                   "Range Not Satisfiable", {}, None)
        m, req = self._run([e], prefill=100)
        self.assertEqual(self.dest.read_bytes(), b"A" * 100)  # 原样保留
        self.assertTrue(any("已下载完整" in s for s in self.logs))

    def test_retry_then_success(self):
        boom = OSError("connection reset")
        m, req = self._run([boom, _resp(206, b"y" * 50, cl=50)], prefill=100)
        self.assertEqual(self.dest.read_bytes(), b"A" * 100 + b"y" * 50)
        self.assertTrue(any("重试" in s for s in self.logs))
        self.assertEqual(m.call_count, 2)

    def test_give_up_after_attempts(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=OSError("reset")), \
             mock.patch("gui_app.llama_cpp.time.sleep"):
            with self.assertRaises(OSError):
                llama_cpp._download("http://x/pkg.zip", self.dest,
                                    self.logs.append, "包")
        retries = [s for s in self.logs if "重试" in s]
        self.assertEqual(len(retries), 2)  # 3 次尝试 = 2 次重试日志


class TestDownloadCancelImmediate(unittest.TestCase):
    """取消监视线程：网络停滞（阻塞读取）时置位取消也能秒级中断。"""

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
                llama_cpp._download("http://x/pkg.zip", self.dest,
                                    self.logs.append, "包", cancel_event=ev)
                result["ok"] = True
            except BaseException as e:
                result["exc"] = e

        with mock.patch("urllib.request.urlopen", return_value=StalledResp()), \
             mock.patch("gui_app.llama_cpp.time.sleep"):
            t = threading.Thread(target=_run)
            t.start()
            time.sleep(0.3)  # 等监视线程就绪、read 进入阻塞
            ev.set()         # 用户点击停止：watcher 立即关闭响应
            t.join(5)
        self.assertFalse(t.is_alive())  # 秒级停止，未等满超时周期
        self.assertIsInstance(result.get("exc"), llama_cpp.InstallCancelled)


class TestInstallPartialKeep(unittest.TestCase):
    """install：下载失败/取消时清理半成品压缩包，下次安装从头下载。"""

    def _fake_build(self):
        return {"type": "cuda", "ver": "13", "key": "cuda-13",
                "precompiled_url": "http://x/pre.zip",
                "cudart_url": "http://x/cudart.zip"}

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

    def test_failed_download_cleans_partial(self):
        llama_dir = self._patch_env()
        pre_zip = llama_dir / "_pre_cuda-13.zip"

        def _boom(url, dest, log_fn, label, expected_size=0, threads=0, cancel_event=None):
            dest.write_bytes(b"partial")
            raise OSError("connection reset")

        with mock.patch("gui_app.llama_cpp._download", side_effect=_boom):
            r = llama_cpp.install("cuda-13")
        self.assertFalse(r["ok"])  # 下载失败 → 安装失败
        self.assertFalse(pre_zip.exists())  # 半成品已清理，下次安装从头下载

    def test_cancel_mid_download_cleans_zips(self):
        """下载中点停止：返回 cancelled，半成品压缩包已清理，下次安装从头下载。"""
        llama_dir = self._patch_env()
        pre_zip = llama_dir / "_pre_cuda-13.zip"
        ev = threading.Event()
        seen = {}

        def _cancel_dl(url, dest, log_fn, label, expected_size=0, threads=0, cancel_event=None):
            seen["cancel_event"] = cancel_event
            dest.write_bytes(b"partial")
            if cancel_event is not None:
                cancel_event.set()
            raise llama_cpp.InstallCancelled()

        with mock.patch("gui_app.llama_cpp._download", side_effect=_cancel_dl):
            r = llama_cpp.install("cuda-13", stop_event=ev)
        self.assertFalse(r["ok"])
        self.assertTrue(r["cancelled"])
        self.assertIs(seen["cancel_event"], ev)  # stop_event 一路传入下载
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

    def test_failed_download_logs_manual_hint(self):
        llama_dir = self._patch_env()
        logs = []

        def _boom(url, dest, log_fn, label, expected_size=0, threads=0, cancel_event=None):
            dest.write_bytes(b"partial")
            raise OSError("connection reset")

        with mock.patch("gui_app.llama_cpp._download", side_effect=_boom):
            r = llama_cpp.install("cuda-13", log_fn=logs.append)
        self.assertFalse(r["ok"])
        # 网络失败时日志给出「稍后再试 / 手动下载解压」引导与 release 页地址
        self.assertTrue(any("请稍后再试" in s for s in logs), logs)
        self.assertTrue(any("手动下载解压" in s for s in logs), logs)
        self.assertTrue(any("github.com/ggml-org/llama.cpp/releases" in s for s in logs), logs)
        self.assertTrue(any("llama.cpp" in s and "文件夹" in s for s in logs), logs)

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

        def _write_zip(url, dest, log_fn, label, expected_size=0, threads=0, cancel_event=None):
            with zipfile.ZipFile(dest, "w") as z:
                z.writestr("readme.txt", "hi")

        with mock.patch("gui_app.llama_cpp._download", side_effect=_write_zip):
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

        def _write_zip(url, dest, log_fn, label, expected_size=0, threads=0, cancel_event=None):
            with zipfile.ZipFile(dest, "w") as z:
                z.writestr("llama-server.exe", "MZ")

        with p, mock.patch("gui_app.llama_cpp._download", side_effect=_write_zip):
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


if __name__ == "__main__":
    unittest.main()
