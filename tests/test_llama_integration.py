import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from gui_app import llama_integration as li
from gui_app.config_store import _default_config


def _cfg(**over):
    """构造一份基础配置 dict，over 可覆盖 experimental 段。"""
    c = _default_config()
    c["experimental"].update(over)
    return c


class TestIsActive(unittest.TestCase):
    """集成生效条件：llama 模块启用 + 集成开关开启 + llama.cpp 已安装。"""

    def test_default_off(self):
        # 默认配置：未启用模块、未开集成 → 不激活
        self.assertFalse(li.is_active(_cfg()))
        self.assertIsNone(li.ai_override(_cfg()))

    def test_integrate_on_but_module_disabled(self):
        c = _cfg()
        c["experimental"]["llama_integrate"] = True  # 开了集成但总开关未开
        self.assertFalse(li.is_active(c))

    def test_module_on_but_integrate_off(self):
        c = _cfg(llama_enabled=True)
        self.assertFalse(li.is_active(c))

    def test_active_when_installed(self):
        c = _cfg(llama_enabled=True)
        c["experimental"]["llama_integrate"] = True
        with mock.patch("gui_app.llama_cpp.LLAMA_DIR", Path("C:/fake/llama")), \
             mock.patch("gui_app.llama_cpp.EXE_NAME", "llama-server.exe"), \
             mock.patch("pathlib.Path.is_file", return_value=True):
            self.assertTrue(li.is_active(c))

    def test_inactive_when_exe_missing(self):
        # 可插拔：开关全开但 llama-server.exe 不存在（未安装/被删）→ 自动失效
        c = _cfg(llama_enabled=True)
        c["experimental"]["llama_integrate"] = True
        with mock.patch("gui_app.llama_cpp.LLAMA_DIR", Path("C:/fake/llama")), \
             mock.patch("gui_app.llama_cpp.EXE_NAME", "llama-server.exe"), \
             mock.patch("pathlib.Path.is_file", return_value=False):
            self.assertFalse(li.is_active(c))
            self.assertIsNone(li.ai_override(c))


class TestAiOverride(unittest.TestCase):
    def setUp(self):
        self.c = _cfg(llama_enabled=True)
        self.c["experimental"]["llama_integrate"] = True
        self.llama_dir = mock.patch("gui_app.llama_cpp.LLAMA_DIR", Path("C:/fake/llama"))
        self.exe = mock.patch("gui_app.llama_cpp.EXE_NAME", "llama-server.exe")
        self.is_file = mock.patch("pathlib.Path.is_file", return_value=True)
        # host/port 从独立文件 llama_config.json 读（ai_override 内延迟导入）
        self.llama_cfg = mock.patch("gui_app.config_store.load_llama_config",
                                    return_value={})
        self.llama_dir.start()
        self.exe.start()
        self.is_file.start()
        self.llama_cfg.start()

    def tearDown(self):
        self.llama_dir.stop()
        self.exe.stop()
        self.is_file.stop()
        self.llama_cfg.stop()

    def test_default_endpoint(self):
        ov = li.ai_override(self.c)
        self.assertEqual(ov, {
            "model": "model",
            "api_key": "not-needed",
            "base_url": "http://127.0.0.1:8080/v1",
        })

    def test_custom_host_port(self):
        self.llama_cfg.stop()
        self.llama_cfg = mock.patch("gui_app.config_store.load_llama_config",
                                    return_value={"host": "0.0.0.0", "port": 11434})
        self.llama_cfg.start()
        ov = li.ai_override(self.c)
        self.assertEqual(ov["base_url"], "http://0.0.0.0:11434/v1")

    def test_override_does_not_mutate_input(self):
        before = dict(self.c)
        li.ai_override(self.c)
        self.assertEqual(self.c, before)  # 磁盘配置/传入 dict 完全不动

    def test_user_ai_params_untouched(self):
        # 用户填写的远程参数仍在 cfg["ai"] 中，覆盖只发生在返回值里
        self.c["ai"] = {"model": "my-remote-model", "base_url": "https://remote.example/v1",
                        "api_key": "sk-secret"}
        ov = li.ai_override(self.c)
        self.assertEqual(ov["model"], "model")
        self.assertEqual(ov["api_key"], "not-needed")
        self.assertEqual(self.c["ai"]["model"], "my-remote-model")


class TestBuildEngineConfigOverride(unittest.TestCase):
    """主链路：build_engine_config 在集成激活时覆盖 model/base_url/api_key。"""

    def test_override_applied(self):
        from gui_app import runner
        c = _cfg(llama_enabled=True)
        c["experimental"]["llama_integrate"] = True
        c["ai"] = {"model": "remote-model", "base_url": "https://remote/v1", "api_key": "sk-x"}
        with mock.patch("gui_app.llama_cpp.LLAMA_DIR", Path("C:/fake/llama")), \
             mock.patch("gui_app.llama_cpp.EXE_NAME", "llama-server.exe"), \
             mock.patch("pathlib.Path.is_file", return_value=True), \
             mock.patch("gui_app.config_store.load_llama_config",
                        return_value={"port": 9999}), \
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
        c = _cfg()
        c["ai"] = {"model": "remote-model", "base_url": "https://remote/v1", "api_key": "sk-x"}
        with mock.patch("gui_app.runner.prompts.get_active",
                        return_value={"preset": {}, "prompt": "p", "system_prompt": "s"}):
            eng = runner.build_engine_config(c)
        self.assertEqual(eng.model, "remote-model")
        self.assertEqual(eng.base_url, "https://remote/v1")


if __name__ == "__main__":
    unittest.main()
