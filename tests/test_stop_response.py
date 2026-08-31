"""手动停止响应速度优化的回归测试。

覆盖：
- 改动①：analyze_frames 在 AI 请求进行中收到 stop_event，立即返回 cancel
- 改动②：AnalyzeStream.close 在 stop_event 置位时走快路径，unregister/unlink 仍执行
- 改动④：llama_cpp.stop 把 grace 透传给 _terminate_process
"""
import sys
import threading
import time
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from batch_rename.ai import analyze_frames
from batch_rename.types import Frame


class TestAnalyzeFramesCancel(unittest.TestCase):
    """改动①：AI 调用进行中置位 stop_event，应立即返回 cancel。"""

    def test_cancel_during_request(self):
        blocker = threading.Event()
        started = threading.Event()

        def blocking_create(**kwargs):
            started.set()
            blocker.wait(timeout=5)

        client = MagicMock()
        client.chat.completions.create.side_effect = blocking_create

        config = MagicMock()
        config.ai_timeout = 60
        config.max_tokens = 100
        config.temperature = 0.3
        config.top_p = 1.0
        config.enforce_json_mode = False
        config.retry_times = 0
        config.system_prompt = ""
        config.frame_time_tags = False
        config.prompt = "p"

        frames = [Frame(ts=0.0, b64="")]
        stop_event = threading.Event()
        result_box = {}

        def run():
            result_box["r"] = analyze_frames(
                client, "m", frames, config, stop_event, "v.mp4", 10.0
            )

        t = threading.Thread(target=run, daemon=True)
        t.start()
        self.assertTrue(started.wait(timeout=3))
        time.sleep(0.3)
        stop_event.set()
        t.join(timeout=3)
        self.assertFalse(t.is_alive())
        r = result_box["r"]
        self.assertEqual(r.error_kind, "cancel")
        self.assertEqual(r.err_msg, "已取消")
        self.assertEqual(r.retries, 0)
        blocker.set()


class TestAnalyzeStreamCloseFastPath(unittest.TestCase):
    """改动②：stop_event 置位时 close 走快路径，unregister/unlink 仍执行。"""

    def _mk_stream(self, stop_set, broken):
        from gui_app.pixai_tagger import AnalyzeStream
        stream = AnalyzeStream.__new__(AnalyzeStream)
        se = threading.Event()
        if stop_set:
            se.set()
        stream.stop_event = se
        stream.broken = broken
        stream.p = MagicMock()
        stream.p.returncode = 0
        stream._deadline = time.time() + 100
        stream._out_thread = MagicMock()
        stream.per_video = []
        stream._error = None
        stream._err_tail = []
        stream._params_path = "/tmp/nonexistent_params.json"
        return stream

    def test_close_fast_path_on_stop(self):
        stream = self._mk_stream(stop_set=True, broken=False)
        with mock.patch("gui_app.pixai_tagger.unregister_subprocess") as mock_unreg, \
             mock.patch("gui_app.pixai_tagger.os.unlink") as mock_unlink:
            r = stream.close()
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"], "已停止")
        stream.p.terminate.assert_called_once()
        mock_unreg.assert_called_once_with(stream.p)
        mock_unlink.assert_called_once_with(stream._params_path)

    def test_close_normal_path_when_no_stop(self):
        stream = self._mk_stream(stop_set=False, broken=False)
        with mock.patch("gui_app.pixai_tagger.unregister_subprocess"), \
             mock.patch("gui_app.pixai_tagger.os.unlink"):
            r = stream.close()
        self.assertTrue(r["ok"])
        self.assertIsNone(r["error"])
        stream.p.stdin.close.assert_called_once()


class TestStopGrace(unittest.TestCase):
    """改动④：stop(grace=...) 透传给 _terminate_process(wait=...)。"""

    def setUp(self):
        from gui_app import llama_cpp
        llama_cpp._paused_launch = None
        llama_cpp._llama_proc = None
        llama_cpp._llama_state.update({
            "running": False, "pid": None, "port": None, "model": None,
            "launch_params": None, "starting": False, "launch_failed": None,
        })

    def _patch_stop_env(self, proc):
        return (
            mock.patch("gui_app.llama_cpp._llama_proc", proc),
            mock.patch("gui_app.llama_cpp.unregister_subprocess"),
            mock.patch("gui_app.llama_cpp._llama_state",
                       {"running": True, "starting": False, "launch_failed": None}),
            mock.patch("gui_app.llama_cpp._terminate_process"),
        )

    def test_stop_passes_grace_to_terminate(self):
        from gui_app import llama_cpp
        proc = MagicMock()
        proc.poll.return_value = None
        p1, p2, p3, p4 = self._patch_stop_env(proc)
        with p1, p2, p3, p4 as mock_term:
            r = llama_cpp.stop(grace=2.0)
        mock_term.assert_called_once_with(proc, wait=2.0)
        self.assertTrue(r["ok"])

    def test_stop_default_grace_is_5(self):
        from gui_app import llama_cpp
        proc = MagicMock()
        proc.poll.return_value = None
        p1, p2, p3, p4 = self._patch_stop_env(proc)
        with p1, p2, p3, p4 as mock_term:
            llama_cpp.stop()
        mock_term.assert_called_once_with(proc, wait=5.0)


if __name__ == "__main__":
    unittest.main()
