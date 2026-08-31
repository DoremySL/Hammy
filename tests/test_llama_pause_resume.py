"""llama-server 显存让出/恢复（pause_for_task / resume_after_task）与启动状态测试。

覆盖：pause/resume 的暂存-恢复语义、GPU 任务（标签获取/转录）前后挂钩、
launch 的 show_logs 终端接管、启动 4 态（starting/launch_failed）流转、
_reader_thread 无管道模式的退出等待。
"""
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from gui_app import llama_cpp
from gui_app.config_store import _default_config, _pixai_defaults


def _reset_runtime_state():
    """恢复 llama_cpp 模块级运行时状态（测试可能污染）。"""
    llama_cpp._paused_launch = None
    llama_cpp._llama_proc = None
    llama_cpp._llama_state.update({
        "running": False, "pid": None, "port": None, "model": None,
        "launch_params": None,
        "starting": False, "launch_failed": None,
    })


class TestPauseForTask(unittest.TestCase):
    """pause_for_task：其他高显存任务开始前让出 llama-server。"""

    def setUp(self):
        _reset_runtime_state()

    def test_running_saves_params_and_stops(self):
        proc = mock.MagicMock()
        proc.poll.return_value = None  # 运行中
        state = {"model": "C:/m.gguf", "launch_params": {"ctx": 8192, "port": 8080}}
        with mock.patch("gui_app.llama_cpp._llama_proc", proc), \
             mock.patch("gui_app.llama_cpp._llama_state", state), \
             mock.patch("gui_app.llama_cpp.stop",
                        return_value={"ok": True}) as stop_mock:
            r = llama_cpp.pause_for_task()
        self.assertTrue(r["ok"])
        self.assertTrue(r["was_running"])
        stop_mock.assert_called_once()
        # 启动参数已暂存，供任务完成后恢复
        self.assertEqual(llama_cpp._paused_launch, {
            "model": "C:/m.gguf", "params": {"ctx": 8192, "port": 8080}})

    def test_not_running_noop(self):
        proc = mock.MagicMock()
        proc.poll.return_value = 0  # 进程已退出
        with mock.patch("gui_app.llama_cpp._llama_proc", proc), \
             mock.patch("gui_app.llama_cpp._llama_state", {}), \
             mock.patch("gui_app.llama_cpp.stop") as stop_mock:
            r = llama_cpp.pause_for_task()
        self.assertTrue(r["ok"])
        self.assertFalse(r["was_running"])
        stop_mock.assert_not_called()
        self.assertIsNone(llama_cpp._paused_launch)

    def test_no_proc_noop(self):
        with mock.patch("gui_app.llama_cpp._llama_proc", None), \
             mock.patch("gui_app.llama_cpp.stop") as stop_mock:
            r = llama_cpp.pause_for_task()
        self.assertFalse(r["was_running"])
        stop_mock.assert_not_called()

    def test_stop_failure_clears_saved(self):
        proc = mock.MagicMock()
        proc.poll.return_value = None
        with mock.patch("gui_app.llama_cpp._llama_proc", proc), \
             mock.patch("gui_app.llama_cpp._llama_state",
                        {"model": "C:/m.gguf", "launch_params": {}}), \
             mock.patch("gui_app.llama_cpp.stop",
                        return_value={"ok": False, "error": "x"}) as stop_mock:
            r = llama_cpp.pause_for_task()
        self.assertFalse(r["ok"])
        self.assertTrue(r["was_running"])
        stop_mock.assert_called_once()
        # 停止失败：不留下待恢复状态，避免任务结束后误重启
        self.assertIsNone(llama_cpp._paused_launch)


class TestResumeAfterTask(unittest.TestCase):
    """resume_after_task：高显存任务完成后恢复 llama-server。"""

    def setUp(self):
        _reset_runtime_state()

    def test_no_saved_noop(self):
        with mock.patch("gui_app.llama_cpp.launch") as launch_mock:
            r = llama_cpp.resume_after_task()
        self.assertTrue(r["ok"])
        self.assertFalse(r["restarted"])
        launch_mock.assert_not_called()

    def test_saved_relaunches_with_saved_params(self):
        llama_cpp._paused_launch = {
            "model": "C:/m.gguf", "params": {"port": 9999, "ctx": 8192}}
        with mock.patch("gui_app.llama_cpp.launch",
                        return_value={"ok": True, "model": "C:/m.gguf"}) as launch_mock:
            r = llama_cpp.resume_after_task()
        self.assertTrue(r["ok"])
        self.assertTrue(r["restarted"])
        launch_mock.assert_called_once_with(
            "C:/m.gguf", {"port": 9999, "ctx": 8192}, log_fn=mock.ANY)
        self.assertIsNone(llama_cpp._paused_launch)  # 暂存被消费

    def test_saved_but_already_running_skips(self):
        # 任务期间用户手动启动了服务 → 不重复启动
        llama_cpp._paused_launch = {"model": "C:/m.gguf", "params": {}}
        proc = mock.MagicMock()
        proc.poll.return_value = None
        with mock.patch("gui_app.llama_cpp._llama_proc", proc), \
             mock.patch("gui_app.llama_cpp.launch") as launch_mock:
            r = llama_cpp.resume_after_task()
        self.assertTrue(r["ok"])
        self.assertFalse(r["restarted"])
        launch_mock.assert_not_called()

    def test_relaunch_failure_reported(self):
        llama_cpp._paused_launch = {"model": "C:/m.gguf", "params": {}}
        with mock.patch("gui_app.llama_cpp.launch",
                        return_value={"ok": False, "error": "启动失败"}) as launch_mock:
            r = llama_cpp.resume_after_task()
        self.assertFalse(r["ok"])
        self.assertNotIn("restarted", r)
        launch_mock.assert_called_once()


class TestLaunchSavesParams(unittest.TestCase):
    """launch 成功启动后保存完整参数，供 pause_for_task 让出后恢复。"""

    def setUp(self):
        _reset_runtime_state()

    def test_launch_params_recorded(self):
        proc = mock.MagicMock()
        proc.pid = 1234
        proc.poll.return_value = None
        with mock.patch("pathlib.Path.is_file", return_value=True), \
             mock.patch("gui_app.llama_cpp.scan_models"), \
             mock.patch("gui_app.llama_cpp.scan_mmprojs", return_value=[]), \
             mock.patch("gui_app.llama_cpp._pick_mmproj", return_value=""), \
             mock.patch("gui_app.llama_cpp._build_args", return_value=[]), \
             mock.patch("gui_app.llama_cpp.subprocess.Popen", return_value=proc), \
             mock.patch("gui_app.llama_cpp.register_subprocess"), \
             mock.patch("gui_app.llama_cpp._reader_thread"), \
             mock.patch("gui_app.llama_cpp._wait_for_health", return_value=True):
            r = llama_cpp.launch("C:/m.gguf", {"ctx": 8192, "port": 9090})
        self.assertTrue(r["ok"])
        saved = llama_cpp._llama_state["launch_params"]
        self.assertEqual(saved["ctx"], 8192)
        self.assertEqual(saved["port"], 9090)
        self.assertEqual(llama_cpp._llama_state["model"],
                         str(Path("C:/m.gguf")))  # launch 记录解析后的实际路径


class _FakeStream:
    """假分析流：捕获 send、永不返回结果、close 报告成功。

    测试环境防真实启动推理子进程/加载模型（mock start_analyze_stream 的返回）。
    当前两个用例抽帧均无帧（无 send），走「全部抽帧失败」路径即可。
    """

    def __init__(self):
        self.sent: list = []
        self.closed = False
        self.broken = False  # send-ahead 主循环会检查的引擎状态位

    def send(self, item: dict) -> bool:
        self.sent.append(item)
        return True

    def next_result(self):
        return None

    def try_next_result(self):
        return None  # 机会性排空：无就绪结果

    def close(self):
        self.closed = True
        return {"ok": True, "per_video": [], "error": None}


class TestTaskHooks(unittest.TestCase):
    """detect_ip_tags / detect_speech：任务开始让出、结束恢复本地推理服务。"""

    def _api(self):
        from gui_app.api_mixins.experimental import ExperimentalMixin
        return ExperimentalMixin()

    def _pixai_patches(self, extract):
        """detect_ip_tags 的公共 mock：模块就绪、配置默认、不写真实 workspace、
        假分析流（start_analyze_stream 在函数体内导入，patch 源模块）。"""
        return [
            mock.patch("gui_app.pixai_tagger.get_status",
                       return_value={"ready": True}),
            mock.patch("gui_app.api_mixins.experimental.load_pixai_config",
                       side_effect=_pixai_defaults),
            mock.patch("gui_app.api_mixins.experimental.update_json"),  # 标签落盘不入真实工作区
            mock.patch("batch_rename.dependencies.ffmpeg_tools.ffmpeg",
                       "/fake/ffmpeg.exe"),
            mock.patch("gui_app.pixai_tagger.cls_model_available",
                       return_value=False),
            mock.patch("gui_app.pixai_frames.extract_frames_for_tagger",
                       **extract),
            mock.patch("gui_app.pixai_tagger.start_analyze_stream",
                       return_value=_FakeStream()),
        ]

    def test_detect_ip_tags_pauses_and_resumes(self):
        api = self._api()
        with mock.patch("gui_app.llama_cpp.pause_for_task") as pause_mock, \
             mock.patch("gui_app.llama_cpp.resume_after_task") as resume_mock, \
             ExitStack() as stack:
            for p in self._pixai_patches({"return_value": []}):
                stack.enter_context(p)
            r = api.detect_ip_tags([["vid1", "C:/v.mp4"]])
        self.assertTrue(r["ok"])
        pause_mock.assert_called_once()   # 开始前让出显存
        resume_mock.assert_called_once()  # 结束（含失败路径）后恢复

    def test_detect_ip_tags_degrades_on_extract_error(self):
        # 抽帧异常按视频降级（该视频记 error，不整体失败）：
        # 异常路径同样要恢复本地推理服务
        api = self._api()
        with mock.patch("gui_app.llama_cpp.pause_for_task") as pause_mock, \
             mock.patch("gui_app.llama_cpp.resume_after_task") as resume_mock, \
             ExitStack() as stack:
            for p in self._pixai_patches({"side_effect": RuntimeError("boom")}):
                stack.enter_context(p)
            r = api.detect_ip_tags([["vid1", "C:/v.mp4"]])
        self.assertTrue(r["ok"])                          # 单视频异常不影响整体
        self.assertEqual(r["results"]["vid1"]["error"], "抽帧失败")
        pause_mock.assert_called_once()   # 开始前让出显存
        resume_mock.assert_called_once()  # 结束（含失败路径）后恢复

    def test_detect_speech_pauses_and_resumes(self):
        api = self._api()
        with mock.patch("gui_app.faster_whisper.get_status",
                        return_value={"ready": True}), \
             mock.patch("gui_app.api_mixins.experimental.load_config",
                        return_value=_default_config()), \
             mock.patch("gui_app.faster_whisper.run_transcription_batch",
                        return_value={"ok": False, "per_video": [],
                                      "error": "x"}), \
             mock.patch("gui_app.llama_cpp.pause_for_task") as pause_mock, \
             mock.patch("gui_app.llama_cpp.resume_after_task") as resume_mock:
            r = api.detect_speech([["v1", "C:/v.mp4"]])
        self.assertFalse(r["ok"])  # 转录失败路径
        pause_mock.assert_called_once()
        resume_mock.assert_called_once()


class TestLaunchTerminal(unittest.TestCase):
    """「程序内显示 llama.cpp 日志」：show_logs 决定输出接管还是继承程序终端。"""

    def setUp(self):
        _reset_runtime_state()

    def _launch(self, params):
        proc = mock.MagicMock()
        proc.pid = 1234
        proc.poll.return_value = None
        with mock.patch("pathlib.Path.is_file", return_value=True), \
             mock.patch("gui_app.llama_cpp.scan_models"), \
             mock.patch("gui_app.llama_cpp.scan_mmprojs", return_value=[]), \
             mock.patch("gui_app.llama_cpp._pick_mmproj", return_value=""), \
             mock.patch("gui_app.llama_cpp._build_args", return_value=[]), \
             mock.patch("gui_app.llama_cpp.subprocess.Popen",
                        return_value=proc) as popen_mock, \
             mock.patch("gui_app.llama_cpp.register_subprocess"), \
             mock.patch("gui_app.llama_cpp._reader_thread") as reader_mock, \
             mock.patch("gui_app.llama_cpp.threading.Thread") as thread_cls, \
             mock.patch("gui_app.llama_cpp._wait_for_health",
                        return_value=True):
            r = llama_cpp.launch("C:/m.gguf", params)
        self.assertTrue(r["ok"])
        return proc, popen_mock, reader_mock, thread_cls

    def test_default_inherits_program_terminal(self):
        # 默认 show_logs=False：不接管输出，llama-server 直接写到程序终端
        proc, popen_mock, reader_mock, thread_cls = self._launch({})
        kw = popen_mock.call_args.kwargs
        self.assertIsNone(kw["stdout"])
        self.assertIsNone(kw["stderr"])
        # 不能带 CREATE_NO_WINDOW：该 flag 使 console 子进程标准句柄不被设置，
        # llama-server 日志会静默丢失（MSDN 文档行为）
        self.assertEqual(kw["creationflags"], 0)
        # 监听线程只等退出做状态清理，不读管道（read_pipe=False）
        thread_cls.assert_called_once()
        self.assertEqual(thread_cls.call_args.kwargs["target"], reader_mock)
        self.assertEqual(thread_cls.call_args.kwargs["args"],
                         (proc, mock.ANY, False, False))

    def test_show_logs_pipes_to_log_bar(self):
        # show_logs=True：接管管道逐行转发到日志栏，保留防弹窗 flag
        proc, popen_mock, reader_mock, thread_cls = \
            self._launch({"show_logs": True})
        kw = popen_mock.call_args.kwargs
        self.assertIs(kw["stdout"], llama_cpp.subprocess.PIPE)
        self.assertIs(kw["stderr"], llama_cpp.subprocess.STDOUT)
        self.assertIsNotNone(kw["creationflags"])  # SUBPROCESS_KWARGS 的 flag
        thread_cls.assert_called_once()
        self.assertEqual(thread_cls.call_args.kwargs["args"],
                         (proc, mock.ANY, True, True))

    def test_launch_success_clears_starting(self):
        # 健康检查通过后：starting 收起、无失败残留 → 胶囊「运行中」
        self._launch({})
        self.assertTrue(llama_cpp._llama_state["running"])
        self.assertFalse(llama_cpp._llama_state["starting"])
        self.assertIsNone(llama_cpp._llama_state["launch_failed"])


class TestLaunchState(unittest.TestCase):
    """启动状态 4 态的后端支撑：starting / launch_failed 状态流转。"""

    def setUp(self):
        _reset_runtime_state()

    def _mk_proc(self, poll_value):
        proc = mock.MagicMock()
        proc.pid = 1234
        proc.poll.return_value = poll_value
        return proc

    def _launch_patches(self):
        """launch 到 Popen 为止的公共 mock（health 结果由调用方另行 patch）。"""
        return [
            mock.patch("pathlib.Path.is_file", return_value=True),
            mock.patch("gui_app.llama_cpp.scan_models"),
            mock.patch("gui_app.llama_cpp.scan_mmprojs", return_value=[]),
            mock.patch("gui_app.llama_cpp._pick_mmproj", return_value=""),
            mock.patch("gui_app.llama_cpp._build_args", return_value=[]),
            mock.patch("gui_app.llama_cpp.register_subprocess"),
            mock.patch("gui_app.llama_cpp._reader_thread"),
            mock.patch("gui_app.llama_cpp.threading.Thread"),
        ]

    def test_popen_failure_records_launch_failed(self):
        # Popen 抛异常（如 exe 无法拉起）：胶囊「启动失败」+ 失败原因
        with mock.patch("pathlib.Path.is_file", return_value=True), \
             mock.patch("gui_app.llama_cpp.scan_models"), \
             mock.patch("gui_app.llama_cpp.scan_mmprojs", return_value=[]), \
             mock.patch("gui_app.llama_cpp._pick_mmproj", return_value=""), \
             mock.patch("gui_app.llama_cpp._build_args", return_value=[]), \
             mock.patch("gui_app.llama_cpp.subprocess.Popen",
                        side_effect=OSError("boom")):
            r = llama_cpp.launch("C:/m.gguf", {}, log_fn=lambda m: None)
        self.assertFalse(r["ok"])
        st = llama_cpp._llama_state
        self.assertFalse(st["starting"])
        self.assertIn("boom", st["launch_failed"])

    def test_health_timeout_stops_and_records_failed(self):
        # 健康检查超时但进程仍存活：回收进程 + 胶囊「启动失败」
        proc = self._mk_proc(None)
        with ExitStack() as stack:
            for p in self._launch_patches():
                stack.enter_context(p)
            stack.enter_context(mock.patch("gui_app.llama_cpp.subprocess.Popen",
                                           return_value=proc))
            stack.enter_context(mock.patch("gui_app.llama_cpp.unregister_subprocess"))
            stack.enter_context(mock.patch("gui_app.llama_cpp._wait_for_health",
                                           return_value=False))
            r = llama_cpp.launch("C:/m.gguf", {}, log_fn=lambda m: None)
        self.assertFalse(r["ok"])
        proc.terminate.assert_called_once()  # 未就绪进程被回收
        st = llama_cpp._llama_state
        self.assertFalse(st["running"])
        self.assertFalse(st["starting"])
        self.assertIn("秒", st["launch_failed"])

    def test_proc_exited_during_start_records_failed(self):
        # 启动过程中进程异常退出（模型加载失败/显存不足）：胶囊「启动失败」
        proc = self._mk_proc(0)
        with ExitStack() as stack:
            for p in self._launch_patches():
                stack.enter_context(p)
            stack.enter_context(mock.patch("gui_app.llama_cpp.subprocess.Popen",
                                           return_value=proc))
            stack.enter_context(mock.patch("gui_app.llama_cpp._wait_for_health",
                                           return_value=False))
            r = llama_cpp.launch("C:/m.gguf", {}, log_fn=lambda m: None)
        self.assertFalse(r["ok"])
        st = llama_cpp._llama_state
        self.assertFalse(st["running"])
        self.assertFalse(st["starting"])
        self.assertIn("进程已退出", st["launch_failed"])

    def test_stop_clears_starting_and_launch_failed(self):
        # 运行中停止：胶囊回到「未启动」（无失败残留）
        proc = self._mk_proc(None)
        with mock.patch("gui_app.llama_cpp._llama_proc", proc), \
             mock.patch("gui_app.llama_cpp.unregister_subprocess"), \
             mock.patch("gui_app.llama_cpp._llama_state",
                        {"running": True, "starting": True,
                         "launch_failed": "旧失败"}):
            r = llama_cpp.stop()
        self.assertTrue(r["ok"])
        st = llama_cpp._llama_state
        self.assertFalse(st["running"])
        self.assertFalse(st["starting"])
        self.assertIsNone(st["launch_failed"])

    def test_stop_no_proc_clears_starting_and_launch_failed(self):
        # 无进程（含上次启动失败后）：停止入口同样清掉失败残留
        with mock.patch("gui_app.llama_cpp._llama_proc", None), \
             mock.patch("gui_app.llama_cpp._llama_state",
                        {"starting": True, "launch_failed": "旧失败"}):
            r = llama_cpp.stop()
        self.assertTrue(r["ok"])
        st = llama_cpp._llama_state
        self.assertFalse(st["starting"])
        self.assertIsNone(st["launch_failed"])

    def test_get_status_exposes_starting_and_launch_failed(self):
        # 胶囊 4 态所需字段如实上报；进程不存在时 running 收敛为 False
        with mock.patch("gui_app.llama_cpp._llama_proc", None), \
             mock.patch("gui_app.llama_cpp._llama_state",
                        {"running": True, "starting": True,
                         "launch_failed": "加载超时", "pid": 1, "port": 8080,
                         "model": "C:/m.gguf",
                         "launch_params": {}}), \
             mock.patch("gui_app.llama_cpp.scan_models", return_value=[]):
            st = llama_cpp.get_status()
        self.assertTrue(st["starting"])
        self.assertEqual(st["launch_failed"], "加载超时")
        self.assertFalse(st["running"])


class TestReaderThreadNoPipe(unittest.TestCase):
    """_reader_thread 无管道模式：等进程真正退出后才清状态（防 running 误清）。"""

    def setUp(self):
        _reset_runtime_state()

    def _run(self, read_pipe):
        proc = mock.MagicMock()
        proc.stdout = None
        state = {"running": True}
        with mock.patch("gui_app.llama_cpp._llama_proc", proc), \
             mock.patch("gui_app.llama_cpp._llama_state", state), \
             mock.patch("gui_app.llama_cpp.unregister_subprocess"):
            llama_cpp._reader_thread(proc, lambda m: None,
                                     show_logs=False, read_pipe=read_pipe)
        return proc, state

    def test_no_pipe_waits_without_timeout(self):
        # 无管道：wait() 不带超时（进程退出前阻塞，不吞超时清状态）
        proc, state = self._run(read_pipe=False)
        _, kwargs = proc.wait.call_args
        self.assertNotIn("timeout", kwargs)
        # wait 返回（进程已退出）后才清 running
        self.assertFalse(state["running"])

    def test_pipe_waits_with_short_timeout(self):
        # 管道模式保持原有快速收尾语义（EOF 后进程基本已退出）
        proc, state = self._run(read_pipe=True)
        _, kwargs = proc.wait.call_args
        self.assertEqual(kwargs.get("timeout"), 5)
        self.assertFalse(state["running"])

    def test_reader_clears_starting_on_exit(self):
        # 进程在「正在启动」窗口内退出：收起启动中标记
        # （失败原因由 launch 的健康检查分支补记 launch_failed）
        proc = mock.MagicMock()
        proc.stdout = None
        state = {"running": True, "starting": True}
        with mock.patch("gui_app.llama_cpp._llama_proc", proc), \
             mock.patch("gui_app.llama_cpp._llama_state", state), \
             mock.patch("gui_app.llama_cpp.unregister_subprocess"):
            llama_cpp._reader_thread(proc, lambda m: None,
                                     show_logs=False, read_pipe=False)
        self.assertFalse(state["running"])
        self.assertFalse(state["starting"])


if __name__ == "__main__":
    unittest.main()
