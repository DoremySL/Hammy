import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from gui_app import llama_integration as li
from gui_app import llama_cpp
from gui_app.config_store import _default_config

_ACTIVE = {"enabled": True, "integrate": True}


class TestIsActive(unittest.TestCase):
    """集成生效条件：llama 模块启用 + 集成开关开启 + llama.cpp 已安装。"""

    def _is_active(self, llama_cfg, is_file=True):
        with mock.patch("gui_app.config_store.load_llama_config",
                        return_value=llama_cfg), \
             mock.patch("gui_app.llama_cpp.LLAMA_DIR", Path("C:/fake/llama")), \
             mock.patch("gui_app.llama_cpp.EXE_NAME", "llama-server.exe"), \
             mock.patch("pathlib.Path.is_file", return_value=is_file):
            return li.is_active()

    def test_default_off(self):
        # 未启用模块 → 不激活
        self.assertFalse(self._is_active({}))
        self.assertFalse(self._is_active({"integrate": True}))  # 开了集成但总开关未开
        self.assertFalse(self._is_active({"enabled": True}))   # 开了总开关但集成未开
        with mock.patch("gui_app.config_store.load_llama_config", return_value={}):
            self.assertIsNone(li.ai_override())

    def test_active_when_installed(self):
        self.assertTrue(self._is_active(_ACTIVE))

    def test_inactive_when_exe_missing(self):
        # 可插拔：开关全开但 llama-server.exe 不存在（未安装/被删）→ 自动失效
        self.assertFalse(self._is_active(_ACTIVE, is_file=False))


class TestAiOverride(unittest.TestCase):
    def setUp(self):
        self.llama_dir = mock.patch("gui_app.llama_cpp.LLAMA_DIR", Path("C:/fake/llama"))
        self.exe = mock.patch("gui_app.llama_cpp.EXE_NAME", "llama-server.exe")
        self.is_file = mock.patch("pathlib.Path.is_file", return_value=True)
        for p in (self.llama_dir, self.exe, self.is_file):
            p.start()
            self.addCleanup(p.stop)
        # load_llama_config 在 ai_override 内两次延迟导入调用，返回已启动的 mock
        # 实例，测试内可改 return_value
        patcher = mock.patch("gui_app.config_store.load_llama_config",
                             return_value=dict(_ACTIVE))
        self.llama_cfg = patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_endpoint(self):
        ov = li.ai_override()
        self.assertEqual(ov, {
            "model": "model",
            "api_key": "not-needed",
            "base_url": "http://127.0.0.1:8080/v1",
            "ai_workers": 1,
        })

    def test_workers_follow_parallel(self):
        self.llama_cfg.return_value = {**_ACTIVE, "parallel": 4}
        ov = li.ai_override()
        self.assertEqual(ov["ai_workers"], 4)

    def test_custom_host_port(self):
        # 0.0.0.0 是监听地址，本机调用须归一为回环（_local_host）；端口仍取配置值
        self.llama_cfg.return_value = {**_ACTIVE, "host": "0.0.0.0", "port": 11434}
        ov = li.ai_override()
        self.assertEqual(ov["base_url"], "http://127.0.0.1:11434/v1")
        # 非监听地址的自定义 host 原样保留
        self.llama_cfg.return_value = {**_ACTIVE, "host": "192.168.1.5", "port": 8081}
        ov = li.ai_override()
        self.assertEqual(ov["base_url"], "http://192.168.1.5:8081/v1")


class TestBuildEngineConfigOverride(unittest.TestCase):
    """主链路：build_engine_config 在集成激活时覆盖 model/base_url/api_key。"""

    def test_override_applied(self):
        from gui_app import runner
        c = _default_config()
        c["ai"] = {"model": "remote-model", "base_url": "https://remote/v1", "api_key": "sk-x"}
        with mock.patch("gui_app.llama_cpp.LLAMA_DIR", Path("C:/fake/llama")), \
             mock.patch("gui_app.llama_cpp.EXE_NAME", "llama-server.exe"), \
             mock.patch("pathlib.Path.is_file", return_value=True), \
             mock.patch("gui_app.config_store.load_llama_config",
                        return_value={**_ACTIVE, "port": 9999}), \
             mock.patch("gui_app.runner.prompts.get_active",
                        return_value={"preset": {}, "prompt": "p", "system_prompt": "s"}):
            eng = runner.build_engine_config(c)
        self.assertEqual(eng.model, "model")
        self.assertEqual(eng.api_key, "not-needed")
        self.assertEqual(eng.base_url, "http://127.0.0.1:9999/v1")
        # 用户参数在传入 dict 中保持不变
        self.assertEqual(c["ai"]["model"], "remote-model")

    def test_no_override_when_inactive(self):
        from gui_app import runner
        c = _default_config()
        c["ai"] = {"model": "remote-model", "base_url": "https://remote/v1", "api_key": "sk-x"}
        with mock.patch("gui_app.config_store.load_llama_config", return_value={}), \
             mock.patch("gui_app.runner.prompts.get_active",
                        return_value={"preset": {}, "prompt": "p", "system_prompt": "s"}):
            eng = runner.build_engine_config(c)
        self.assertEqual(eng.model, "remote-model")
        self.assertEqual(eng.base_url, "https://remote/v1")


class TestLaunchParallelSnapshot(unittest.TestCase):
    """launch 成功后把实际生效的 -np 写入全局 llama_config.json（集成接管依据）。"""

    def setUp(self):
        import tempfile

        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        llama_dir = Path(self.td.name) / "llama.cpp"
        llama_dir.mkdir()
        (llama_dir / "llama-server.exe").write_bytes(b"MZ")
        self.model_file = Path(self.td.name) / "m.gguf"
        self.model_file.write_bytes(b"gguf")
        self.writes = []

        def _capture(mutator):
            cfg = {}
            mutator(cfg)
            self.writes.append(dict(cfg))

        def _fake_proc(*a, **k):
            p = mock.MagicMock()
            p.poll.return_value = None
            p.pid = 1234
            return p

        self._health_ok = True
        patches = [
            mock.patch.object(llama_cpp, "LLAMA_DIR", llama_dir),
            mock.patch.object(llama_cpp, "_port_in_use", return_value=False),
            mock.patch.object(llama_cpp, "_wait_for_health",
                              return_value=True),
            mock.patch.object(llama_cpp, "_model_layers", return_value=0),
            mock.patch.object(llama_cpp, "_reader_thread", lambda *a, **k: None),
            mock.patch.object(llama_cpp, "register_subprocess", lambda *a: None),
            mock.patch.object(llama_cpp, "unregister_subprocess", lambda *a: None),
            mock.patch.object(llama_cpp, "_notify_state_change", lambda: None),
            mock.patch.object(llama_cpp, "_terminate_process", lambda *a: None),
            mock.patch("gui_app.llama_cpp.subprocess.Popen", _fake_proc),
            mock.patch("gui_app.config_store.update_llama_config",
                       side_effect=_capture),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        llama_cpp._llama_proc = None
        self.addCleanup(setattr, llama_cpp, "_llama_proc", None)

    def test_success_writes_parallel_snapshot(self):
        r = llama_cpp.launch(str(self.model_file), {"parallel": 4})
        self.assertTrue(r["ok"])
        self.assertEqual(self.writes, [{"parallel": 4}])

    def test_unhealthy_no_snapshot(self):
        self._health_ok = False
        with mock.patch.object(llama_cpp, "_wait_for_health", return_value=False):
            r = llama_cpp.launch(str(self.model_file), {"parallel": 4})
        self.assertFalse(r["ok"])
        self.assertEqual(self.writes, [])

    def test_snapshot_overwrites_previous(self):
        r1 = llama_cpp.launch(str(self.model_file), {"parallel": 4})
        llama_cpp._llama_proc = None  # 模拟 stop 后重启
        r2 = llama_cpp.launch(str(self.model_file), {"parallel": 8})
        self.assertTrue(r1["ok"] and r2["ok"])
        self.assertEqual(self.writes[-1], {"parallel": 8})


if __name__ == "__main__":
    unittest.main()
