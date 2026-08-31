"""llama per-model 配置记忆 / 目录变更清理 / 模块文件存储测试。

覆盖：model_config_updates、apply_model_configs、ensure_model_configs、
set_llama_config 落盘链路、auto_run/ensure_llama_running 取参链、
launch_llama 记录 last_model、remove_llama 清理、purge_stale_after_dir_change。
"""
import sys
import json
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from gui_app import llama_cpp
from gui_app.config_store import _default_config


def _cfg(**over):
    """构造一份基础配置 dict，over 可覆盖 experimental 段。"""
    c = _default_config()
    c["experimental"].update(over)
    return c


def _model(path="C:/models/llama.gguf", **over):
    """构造 scan_models() 返回的模型条目。"""
    m = {"path": path, "name": "llama.gguf", "size_mb": 100.0, "mmprojs": [],
         "moe": False, "expert_count": 0, "ctx": 8192, "layers": 80}
    m.update(over)
    return m


def _redirect_workspace_files(ws_dir):
    """把 config.json / 模块文件（llama/config.json 等）重定向到临时目录。

    绑定语义：config_store 模块级导入的常量 patch 其命名空间；
    llama_cpp 函数内延迟导入的常量 patch 源模块 workspace_paths。
    """
    from gui_app import config_store, workspace_paths
    config_file = ws_dir / "config.json"
    llama_dir = ws_dir / "llama"
    llama_dir.mkdir(parents=True, exist_ok=True)
    llama_file = llama_dir / "config.json"
    mcfgs_file = llama_dir / "model_configs.json"
    patchers = [
        mock.patch.object(config_store, "CONFIG_FILE", config_file),
        mock.patch.object(config_store, "LLAMA_CONFIG_FILE", llama_file),
        mock.patch.object(config_store, "LLAMA_MODEL_CONFIGS_FILE", mcfgs_file),
        mock.patch.object(workspace_paths, "LLAMA_SCAN_CACHE_FILE", llama_dir / "scan_cache.json"),
    ]
    return patchers, config_file, llama_file, mcfgs_file


class _FileStoreTestCase(unittest.TestCase):
    """文件级测试基类：真实读写，仅重定向存储路径。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        ws_dir = Path(self._tmp.name)
        patchers, self.config_file, self.llama_file, self.mcfgs_file = \
            _redirect_workspace_files(ws_dir)
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)


class TestModelConfigUpdates(unittest.TestCase):
    """model_config_updates：拆分 per-model 更新（纯函数）。"""

    def test_no_model_no_update(self):
        self.assertEqual(llama_cpp.model_config_updates({"host": "x"}), ("", {}))

    def test_full_params_split(self):
        params = {
            "model": "C:/m.gguf", "host": "0.0.0.0", "port": 11434,
            "threads": 4, "threads_batch": 4, "ngl": 999, "ctx": 8192,
            "batch": 1024, "ubatch": 512, "parallel": 1, "npredict": -1,
            "timeout": 1200, "kv_quant": "q8_0", "mmproj": "C:/mmproj.gguf",
            "mmproj_auto": False, "no_mmproj_offload": False, "reasoning_mode": "off",
            "kv_preset": "kv_unified", "load_mode": "none",
            "extra_args": "", "n_cpu_moe": 0,
        }
        model, updates = llama_cpp.model_config_updates(params)
        self.assertEqual(model, "C:/m.gguf")
        self.assertNotIn("model", updates)
        # 整套参数（含 host/port）都保留
        self.assertEqual(updates["host"], "0.0.0.0")
        self.assertEqual(updates["port"], 11434)
        self.assertEqual(updates["npredict"], -1)
        # 新开关键（缓存预设 / 加载模式 / mmproj CPU 计算 / 自动检测）整套保留
        self.assertEqual(updates["kv_preset"], "kv_unified")
        self.assertEqual(updates["load_mode"], "none")
        self.assertIs(updates["no_mmproj_offload"], False)
        self.assertIs(updates["mmproj_auto"], False)
        # n_cpu_moe 为 0 → None 标记删除（不记忆）
        self.assertIsNone(updates["n_cpu_moe"])

    def test_n_cpu_moe_positive_saved(self):
        _, u = llama_cpp.model_config_updates({"model": "m", "n_cpu_moe": 16})
        self.assertEqual(u["n_cpu_moe"], 16)

    def test_n_cpu_moe_empty_and_bad(self):
        _, u = llama_cpp.model_config_updates({"model": "m", "n_cpu_moe": ""})
        self.assertIsNone(u["n_cpu_moe"])
        _, u2 = llama_cpp.model_config_updates({"model": "m", "n_cpu_moe": "abc"})
        self.assertIsNone(u2["n_cpu_moe"])


class TestApplyModelConfigs(unittest.TestCase):
    """apply_model_configs：把提交参数合并进 per-model 条目（写独立文件）。"""

    def setUp(self):
        self.mcfgs = {}

    def _install_store(self):
        """mock config_store.update_llama_model_configs（函数内延迟导入，patch 来源模块）。"""
        p = mock.patch("gui_app.config_store.update_llama_model_configs",
                       side_effect=lambda m: m(self.mcfgs))
        p.start()
        self.addCleanup(p.stop)

    def test_writes_whole_set(self):
        self._install_store()
        params = {"model": "C:/m.gguf", "host": "0.0.0.0", "port": 9999,
                  "ctx": 8192, "n_cpu_moe": 0}
        self.assertTrue(llama_cpp.apply_model_configs(params))
        entry = self.mcfgs["C:/m.gguf"]
        self.assertEqual(entry["host"], "0.0.0.0")
        self.assertEqual(entry["port"], 9999)
        self.assertNotIn("n_cpu_moe", entry)

    def test_moe_positive_saved(self):
        self._install_store()
        llama_cpp.apply_model_configs({"model": "m", "n_cpu_moe": 32})
        self.assertEqual(self.mcfgs["m"]["n_cpu_moe"], 32)

    def test_n_cpu_moe_removed_from_existing(self):
        self._install_store()
        llama_cpp.apply_model_configs({"model": "m", "n_cpu_moe": 32})
        llama_cpp.apply_model_configs({"model": "m", "n_cpu_moe": 0})
        self.assertNotIn("n_cpu_moe", self.mcfgs["m"])

    def test_no_model_no_write(self):
        self._install_store()
        self.assertFalse(llama_cpp.apply_model_configs({"host": "x"}))
        self.assertEqual(self.mcfgs, {})

    def test_global_default_untouched(self):
        # per-model 落盘走独立文件，不经过 config.json → 全局兜底参数不受影响
        self._install_store()
        with mock.patch("gui_app.config_store.update_config") as uc:
            llama_cpp.apply_model_configs({"model": "m", "host": "0.0.0.0"})
            uc.assert_not_called()
        self.assertEqual(self.mcfgs["m"]["host"], "0.0.0.0")


class TestEnsureModelConfigs(unittest.TestCase):
    """ensure_model_configs：扫描到新模型时初始化 per-model 条目（幂等，写独立文件）。"""

    def setUp(self):
        # 用户改过的全局参数（独立文件 llama/config.json）作为新模型的默认值
        self.llama_cfg = {"ngl": 100, "host": "0.0.0.0"}
        self.mcfgs = {}

    def _install_fake_store(self):
        """mock config_store 的读写函数（ensure_model_configs 内延迟导入）。

        load_llama_model_configs 返回与 update 修改同一 dict 对象，
        模拟真实「读-改-写」语义。
        """
        p1 = mock.patch("gui_app.config_store.load_llama_config",
                        side_effect=lambda: self.llama_cfg)
        p2 = mock.patch("gui_app.config_store.load_llama_model_configs",
                        side_effect=lambda: self.mcfgs)
        self.update_mock = mock.Mock(side_effect=lambda m: m(self.mcfgs))
        p3 = mock.patch("gui_app.config_store.update_llama_model_configs",
                        new=self.update_mock)
        for p in (p1, p2, p3):
            p.start()
            self.addCleanup(p.stop)

    def test_initializes_missing_models(self):
        self._install_fake_store()
        mcfgs = llama_cpp.ensure_model_configs(
            [_model("C:/a.gguf"), _model("C:/b.gguf")])
        self.assertEqual(set(mcfgs), {"C:/a.gguf", "C:/b.gguf"})
        for p in ("C:/a.gguf", "C:/b.gguf"):
            e = mcfgs[p]
            self.assertEqual(e["ctx"], 8192)     # 上下文窗口固定 8192
            self.assertEqual(e["ngl"], 100)      # 其余继承当前全局参数
            self.assertEqual(e["host"], "0.0.0.0")
            self.assertNotIn("n_cpu_moe", e)     # 非 MoE 不写 --n-cpu-moe
            self.assertNotIn("layers", e)        # 层数上限由扫描结果提供，不落盘
            # 全局独占键不进入 per-model 条目
            for k in ("model", "auto_run", "show_logs",
                      "models_dir", "last_model"):
                self.assertNotIn(k, e)
        self.update_mock.assert_called_once()

    def test_existing_entries_keep_params(self):
        # 已有条目保留原参数，仅初始化缺失的模型（不补写已存在条目）
        self.mcfgs["C:/a.gguf"] = {"host": "h", "ctx": 4096}
        self._install_fake_store()
        mcfgs = llama_cpp.ensure_model_configs(
            [_model("C:/a.gguf"), _model("C:/b.gguf")])
        e = mcfgs["C:/a.gguf"]
        self.assertEqual(e["host"], "h")         # 已有参数不被覆盖
        self.assertEqual(e["ctx"], 4096)
        self.assertNotIn("layers", e)            # 已存在条目不做任何补写
        self.assertIn("C:/b.gguf", mcfgs)

    def test_existing_entry_with_layers_no_write(self):
        # 已有条目含 layers → 无新模型也无补写 → 零写盘（幂等）
        self.mcfgs["C:/a.gguf"] = {"layers": 80}
        self._install_fake_store()
        llama_cpp.ensure_model_configs([_model("C:/a.gguf")])
        self.update_mock.assert_not_called()

    def test_layers_unreadable_not_written(self):
        # 扫描读不到 block_count（0）→ 不写 layers，且不重复触发写盘
        self._install_fake_store()
        llama_cpp.ensure_model_configs([_model("C:/a.gguf", layers=0)])
        e = self.mcfgs["C:/a.gguf"]
        self.assertNotIn("layers", e)
        llama_cpp.ensure_model_configs([_model("C:/a.gguf", layers=0)])
        self.update_mock.assert_called_once()    # 第二次无变化 → 零写盘

    def test_no_models_no_write(self):
        self._install_fake_store()
        self.assertEqual(llama_cpp.ensure_model_configs([]), {})
        self.assertEqual(self.mcfgs, {})
        self.update_mock.assert_not_called()  # 无新模型 → 零写盘


class TestSetLlamaConfigPersists(unittest.TestCase):
    """set_llama_config：「应用」/「启动服务」落盘链路（带/不带 model）。"""

    def setUp(self):
        self.llama_cfg = {}
        self.mcfgs = {}

    def _api_with_store(self, cfg):
        from gui_app.api_mixins.experimental import ExperimentalMixin
        api = ExperimentalMixin()
        p1 = mock.patch("gui_app.api_mixins.experimental.load_config",
                        return_value=cfg)
        p3 = mock.patch("gui_app.api_mixins.experimental.load_llama_config",
                        side_effect=lambda: self.llama_cfg)
        p4 = mock.patch("gui_app.api_mixins.experimental.update_llama_config",
                        side_effect=lambda m: m(self.llama_cfg))
        p5 = mock.patch("gui_app.config_store.update_llama_model_configs",
                        side_effect=lambda m: m(self.mcfgs))
        p1.start(); p3.start(); p4.start(); p5.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p3.stop)
        self.addCleanup(p4.stop)
        self.addCleanup(p5.stop)
        return api

    def test_with_model_writes_per_model(self):
        cfg = _cfg()
        api = self._api_with_store(cfg)
        r = api.set_llama_config({
            "model": "C:/m.gguf", "host": "0.0.0.0", "port": 9999,
            "ctx": 8192, "n_cpu_moe": 0,
            "reasoning_mode": "off", "kv_preset": "kv_unified", "load_mode": "none",
        })
        self.assertTrue(r["ok"])
        self.assertEqual(self.llama_cfg["model"], "C:/m.gguf")  # model 偏好仍写全局
        self.assertNotIn("host", self.llama_cfg)  # 运行时参数只进 per-model，不写全局
        self.assertNotIn("model_configs", self.llama_cfg)    # 全局配置不再携带 per-model
        entry = self.mcfgs["C:/m.gguf"]
        self.assertEqual(entry["port"], 9999)       # per-model 整套落盘（独立文件）
        self.assertEqual(entry["kv_preset"], "kv_unified")    # 新开关键同样整套落盘
        self.assertEqual(entry["load_mode"], "none")
        self.assertNotIn("n_cpu_moe", entry)
        # config.json 的 experimental 段无 llama 段（参数不进 config.json）
        self.assertNotIn("llama", cfg["experimental"])

    def test_without_model_runtime_not_saved(self):
        cfg = _cfg()
        api = self._api_with_store(cfg)
        api.set_llama_config({"host": "0.0.0.0"})  # 无 model → 运行时参数不落盘
        self.assertEqual(self.mcfgs, {})
        self.assertNotIn("host", self.llama_cfg)

    def test_integrate_goes_to_llama_file(self):
        # 「本地推理集成」开关写入 llama 模块文件（llama/config.json）的 integrate 键
        cfg = _cfg()
        api = self._api_with_store(cfg)
        r = api.set_llama_config({"integrate": True, "host": "0.0.0.0"})
        self.assertTrue(r["ok"])
        self.assertIs(self.llama_cfg["integrate"], True)  # 开关写独立文件
        # config.json 的 experimental 段不写任何 llama 键（可插拔：模块键不进 config.json）
        self.assertNotIn("llama", cfg["experimental"])
        self.assertNotIn("llama_integrate", cfg["experimental"])
        self.assertIs(r["config"]["integrate"], True)  # 返回结构含 integrate 键
        self.assertEqual(r["config"]["host"], "127.0.0.1")  # 运行时参数不写全局 → 回默认


class TestAutoRunMergesModelConfig(unittest.TestCase):
    """auto_run_llama：程序启动自启用「上次成功运行的模型」+ 该模型 per-model 配置。"""

    def _make_api(self, cfg, llama_cfg, is_file, mcfgs):
        from gui_app.api_mixins.experimental import ExperimentalMixin
        api = ExperimentalMixin()
        p1 = mock.patch("gui_app.api_mixins.experimental.load_config",
                        return_value=cfg)
        p2 = mock.patch("gui_app.api_mixins.experimental.load_llama_config",
                        side_effect=lambda: llama_cfg)
        p3 = mock.patch("gui_app.api_mixins.experimental.update_llama_config",
                        side_effect=lambda m: m(llama_cfg))
        p4 = mock.patch("gui_app.llama_cpp.get_status",
                        return_value={"ready": True, "running": False})
        p5 = mock.patch("pathlib.Path.is_file", new=is_file)
        p6 = mock.patch("gui_app.llama_cpp.launch")
        launch_mock = p6.start()
        launch_mock.return_value = {"ok": True, "model": "C:/m.gguf"}
        p7 = mock.patch("gui_app.api_mixins.experimental.load_llama_model_configs",
                        side_effect=lambda: mcfgs)
        p1.start(); p2.start(); p3.start(); p4.start(); p5.start(); p7.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)
        self.addCleanup(p3.stop); self.addCleanup(p4.stop)
        self.addCleanup(p5.stop); self.addCleanup(p6.stop)
        self.addCleanup(p7.stop)
        return api, launch_mock

    def test_uses_last_run_model_config(self):
        cfg = _cfg(llama_enabled=True)
        llama = {"auto_run": True,
                 "last_model": "C:/m.gguf",       # 上次成功运行的模型
                 "model": "C:/other.gguf"}        # 设置页当前选中（应被忽略）
        mcfgs = {"C:/m.gguf": {"host": "0.0.0.0", "port": 9999, "ctx": 8192}}
        api, launch_mock = self._make_api(cfg, llama,
                                          is_file=lambda self: True, mcfgs=mcfgs)
        r = api.auto_run_llama()
        self.assertTrue(r["ok"])
        args = launch_mock.call_args[0]
        self.assertEqual(args[0], "C:/m.gguf")   # 默认启动记录的模型
        params = args[1]
        self.assertEqual(params["port"], 9999)   # per-model 参数生效
        self.assertEqual(params["ctx"], 8192)
        self.assertNotIn("model_configs", params)
        self.assertNotIn("integrate", params)    # integrate 属全局键，不进 per-model 参数

    def test_missing_last_model_falls_back_to_selected(self):
        # 记录的模型已不存在 → 回退设置页选中的模型
        cfg = _cfg(llama_enabled=True)
        llama = {"auto_run": True,
                 "last_model": "C:/gone.gguf",
                 "model": "C:/m.gguf"}
        mcfgs = {"C:/m.gguf": {"port": 7777}}
        api, launch_mock = self._make_api(
            cfg, llama, is_file=lambda self: self.name != "gone.gguf", mcfgs=mcfgs)
        api.auto_run_llama()
        args = launch_mock.call_args[0]
        self.assertEqual(args[0], "C:/m.gguf")
        self.assertEqual(args[1]["port"], 7777)  # 回退模型的 per-model 参数生效

    def test_no_records_falls_back_to_auto_pick(self):
        # 无记录且未选中模型 → 传空串交给 launch 自动选择
        cfg = _cfg(llama_enabled=True)
        llama = {"auto_run": True}
        api, launch_mock = self._make_api(cfg, llama,
                                          is_file=lambda self: False, mcfgs={})
        r = api.auto_run_llama()
        self.assertTrue(r["ok"])
        self.assertEqual(launch_mock.call_args[0][0], "")


class TestEnsureLlamaRunning(unittest.TestCase):
    """ensure_llama_running：「开始处理」前置保障——服务未运行自动拉起。"""

    def _make_api(self, cfg, llama_cfg, status, is_file, mcfgs):
        from gui_app.api_mixins.experimental import ExperimentalMixin
        api = ExperimentalMixin()
        p1 = mock.patch("gui_app.api_mixins.experimental.load_config",
                        return_value=cfg)
        p2 = mock.patch("gui_app.api_mixins.experimental.load_llama_config",
                        side_effect=lambda: llama_cfg)
        p3 = mock.patch("gui_app.api_mixins.experimental.update_llama_config",
                        side_effect=lambda m: m(llama_cfg))
        p4 = mock.patch("gui_app.llama_cpp.get_status", return_value=status)
        p5 = mock.patch("pathlib.Path.is_file", new=is_file)
        p6 = mock.patch("gui_app.llama_cpp.launch")
        launch_mock = p6.start()
        launch_mock.return_value = {"ok": True, "model": "C:/m.gguf"}
        p7 = mock.patch("gui_app.api_mixins.experimental.load_llama_model_configs",
                        side_effect=lambda: mcfgs)
        p1.start(); p2.start(); p3.start(); p4.start(); p5.start(); p7.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)
        self.addCleanup(p3.stop); self.addCleanup(p4.stop)
        self.addCleanup(p5.stop); self.addCleanup(p6.stop)
        self.addCleanup(p7.stop)
        return api, launch_mock

    def test_running_returns_ready(self):
        # 已运行 → 直接就绪，不再发起启动
        api, launch_mock = self._make_api(
            _cfg(llama_enabled=True), {}, {"ready": True, "running": True},
            is_file=lambda self: True, mcfgs={})
        r = api.ensure_llama_running()
        self.assertEqual(r, {"ok": True, "ready": True})
        launch_mock.assert_not_called()

    def test_starting_returns_starting(self):
        # 正在启动（如本地推理页启动中）→ 返回 starting，前端轮询等待
        api, launch_mock = self._make_api(
            _cfg(llama_enabled=True), {}, {"ready": True, "running": False, "starting": True},
            is_file=lambda self: True, mcfgs={})
        r = api.ensure_llama_running()
        self.assertEqual(r, {"ok": True, "ready": False, "starting": True})
        launch_mock.assert_not_called()

    def test_not_installed_errors(self):
        api, launch_mock = self._make_api(
            _cfg(llama_enabled=True), {}, {"ready": False, "running": False},
            is_file=lambda self: True, mcfgs={})
        r = api.ensure_llama_running()
        self.assertFalse(r["ok"])
        self.assertIn("未安装", r["error"])
        launch_mock.assert_not_called()

    def test_master_switch_off_blocks_launch(self):
        # 总开关关闭 → 直接报错，不拉起服务（防御门槛与 auto_run_llama 一致）
        api, launch_mock = self._make_api(
            _cfg(), {}, {"ready": True, "running": False},
            is_file=lambda self: True, mcfgs={})
        r = api.ensure_llama_running()
        self.assertFalse(r["ok"])
        self.assertIn("总开关", r["error"])
        launch_mock.assert_not_called()

    def test_launches_with_last_run_model_config(self):
        # 未运行 → 按「上次成功运行的模型 + per-model 配置」自动拉起
        cfg = _cfg(llama_enabled=True)
        llama = {"last_model": "C:/m.gguf",
                 "model": "C:/other.gguf"}       # 应被忽略
        mcfgs = {"C:/m.gguf": {"port": 9999, "ctx": 8192}}
        api, launch_mock = self._make_api(
            cfg, llama, {"ready": True, "running": False},
            is_file=lambda self: True, mcfgs=mcfgs)
        r = api.ensure_llama_running()
        self.assertTrue(r["ok"])
        args = launch_mock.call_args[0]
        self.assertEqual(args[0], "C:/m.gguf")
        params = args[1]
        self.assertEqual(params["port"], 9999)   # per-model 参数生效
        self.assertEqual(params["ctx"], 8192)
        self.assertNotIn("model_configs", params)

    def test_missing_last_model_falls_back_to_selected(self):
        cfg = _cfg(llama_enabled=True)
        llama = {"last_model": "C:/gone.gguf",
                 "model": "C:/m.gguf"}
        api, launch_mock = self._make_api(
            cfg, llama, {"ready": True, "running": False},
            is_file=lambda self: self.name != "gone.gguf", mcfgs={})
        api.ensure_llama_running()
        self.assertEqual(launch_mock.call_args[0][0], "C:/m.gguf")

    def test_no_records_falls_back_to_auto_pick(self):
        api, launch_mock = self._make_api(
            _cfg(llama_enabled=True), {}, {"ready": True, "running": False},
            is_file=lambda self: False, mcfgs={})
        r = api.ensure_llama_running()
        self.assertTrue(r["ok"])
        self.assertEqual(launch_mock.call_args[0][0], "")


class TestLaunchRecordsLastModel(unittest.TestCase):
    """launch_llama：启动成功后记录 last_model（下拉框默认选中 + auto_run 默认启动）。"""

    def _api_with_store(self, llama_cfg):
        from gui_app.api_mixins.experimental import ExperimentalMixin
        api = ExperimentalMixin()
        p1 = mock.patch("gui_app.api_mixins.experimental.load_llama_config",
                        side_effect=lambda: llama_cfg)
        p2 = mock.patch("gui_app.api_mixins.experimental.update_llama_config",
                        side_effect=lambda m: m(llama_cfg))
        p1.start(); p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        return api

    def test_success_records_resolved_model(self):
        llama_cfg = {}
        api = self._api_with_store(llama_cfg)
        with mock.patch("gui_app.llama_cpp.launch",
                        return_value={"ok": True, "model": "C:/m.gguf"}) as launch_mock:
            r = api.launch_llama("", {})
        self.assertTrue(r["ok"])
        # 记录 launch 解析后的实际路径（自动选择的情况也准确）
        self.assertEqual(llama_cfg["last_model"], "C:/m.gguf")
        self.assertEqual(launch_mock.call_args[0][0], "")

    def test_success_no_model_key_falls_back_to_arg(self):
        llama_cfg = {}
        api = self._api_with_store(llama_cfg)
        with mock.patch("gui_app.llama_cpp.launch", return_value={"ok": True}):
            api.launch_llama("C:/m.gguf", {})
        self.assertEqual(llama_cfg["last_model"], "C:/m.gguf")

    def test_failure_does_not_record(self):
        llama_cfg = {}
        api = self._api_with_store(llama_cfg)
        with mock.patch("gui_app.llama_cpp.launch",
                        return_value={"ok": False, "error": "x"}):
            api.launch_llama("C:/m.gguf", {})
        self.assertNotIn("last_model", llama_cfg)


class TestRemoveClearsModuleData(unittest.TestCase):
    """remove_llama：卸载成功 → 删除模块数据文件夹（配置/last_model 随之清除）。"""

    def test_success_removes_module_dir(self):
        from gui_app.api_mixins.experimental import ExperimentalMixin
        api = ExperimentalMixin()
        with mock.patch("gui_app.llama_cpp.remove", return_value={"ok": True}), \
             mock.patch("gui_app.api_mixins.experimental.shutil.rmtree") as rm:
            r = api.remove_llama()
        self.assertTrue(r["ok"])
        rm.assert_called_once()  # _workspace/llama/ 整个删除（含 last_model）

    def test_failure_keeps_module_dir(self):
        from gui_app.api_mixins.experimental import ExperimentalMixin
        api = ExperimentalMixin()
        with mock.patch("gui_app.llama_cpp.remove",
                        return_value={"ok": False, "error": "x"}), \
             mock.patch("gui_app.api_mixins.experimental.shutil.rmtree") as rm:
            api.remove_llama()
        rm.assert_not_called()  # 卸载失败不动模块数据


class TestDirHelpers(unittest.TestCase):
    """目录归一/包含判断（purge 的边界正确性基础）。"""

    def test_path_in_dir_basic(self):
        self.assertTrue(llama_cpp._path_in_dir("C:/new/b.gguf", "C:/new"))
        self.assertTrue(llama_cpp._path_in_dir("C:/new/sub/b.gguf", "C:/new"))
        self.assertFalse(llama_cpp._path_in_dir("C:/old/a.gguf", "C:/new"))

    def test_prefix_no_false_positive(self):
        # C:/models2 不是 C:/models 的子目录：前缀匹配必须带分隔符边界
        self.assertFalse(llama_cpp._path_in_dir("C:/models2/y.gguf", "C:/models"))
        self.assertTrue(llama_cpp._path_in_dir("C:/models/x.gguf", "C:/models"))

    def test_slash_backslash_mixed_paths(self):
        self.assertTrue(llama_cpp._path_in_dir("C:/new\\sub\\m.gguf", "C:/new"))
        self.assertTrue(llama_cpp._path_in_dir("C:\\new/sub\\m.gguf", "C:/new"))

    def test_empty_dir_matches_nothing(self):
        self.assertFalse(llama_cpp._path_in_dir("C:/x/m.gguf", ""))

    def test_abs_dir_empty_defaults_to_models(self):
        self.assertEqual(llama_cpp._abs_dir(""), str(llama_cpp.MODELS_DIR))

    def test_abs_dir_relative_resolved_against_app_root(self):
        from gui_app.env import APP_ROOT
        self.assertEqual(llama_cpp._abs_dir("models/x"),
                         str(APP_ROOT / "models" / "x"))

    def test_abs_dir_absolute_unchanged(self):
        # 绝对路径保留（Windows 上 Path 归一分隔符）
        self.assertEqual(llama_cpp._abs_dir("D:/m"), str(Path("D:/m")))


class TestPurgeAfterDirChange(_FileStoreTestCase):
    """purge_stale_after_dir_change：目录变更后清独立文件失效条目 + last_model。"""

    def _seed(self, mcfgs=None, last_model="C:/old/a.gguf", models_dir="C:/old"):
        if mcfgs is not None:
            self.mcfgs_file.write_text(json.dumps(mcfgs, ensure_ascii=False),
                                       encoding="utf-8")
        self.llama_file.write_text(json.dumps(
            {"models_dir": models_dir, "last_model": last_model},
            ensure_ascii=False), encoding="utf-8")
        self.config_file.write_text(json.dumps(_cfg(), ensure_ascii=False),
                                    encoding="utf-8")

    def _read_mcfgs(self):
        return json.loads(self.mcfgs_file.read_text(encoding="utf-8"))

    def _read_llama(self):
        return json.loads(self.llama_file.read_text(encoding="utf-8"))

    def test_purges_stale_entries_and_last_model(self):
        self._seed({
            "C:/old/a.gguf": {"port": 1111},
            "C:/new/b.gguf": {"port": 2222},
        })
        r = llama_cpp.purge_stale_after_dir_change("C:/old", "C:/new")
        self.assertTrue(r)
        self.assertEqual(self._read_mcfgs(), {"C:/new/b.gguf": {"port": 2222}})
        self.assertNotIn("last_model", self._read_llama())  # 指向旧目录 → 清空

    def test_keeps_valid_last_model(self):
        self._seed({
            "C:/old/a.gguf": {"port": 1},
            "C:/new/b.gguf": {"port": 2},
        }, last_model="C:/new/b.gguf")
        r = llama_cpp.purge_stale_after_dir_change("C:/old", "C:/new")
        self.assertTrue(r)
        self.assertEqual(self._read_mcfgs(), {"C:/new/b.gguf": {"port": 2}})
        self.assertEqual(self._read_llama()["last_model"], "C:/new/b.gguf")  # 新目录下保留

    def test_same_dir_noop(self):
        self._seed({"C:/old/a.gguf": {"port": 1}}, last_model="C:/old/a.gguf")
        r = llama_cpp.purge_stale_after_dir_change("C:/old", "C:/old")
        self.assertFalse(r)
        self.assertEqual(self._read_mcfgs(), {"C:/old/a.gguf": {"port": 1}})
        self.assertEqual(self._read_llama()["last_model"], "C:/old/a.gguf")

    def test_no_stale_returns_false(self):
        self._seed({"C:/new/b.gguf": {"port": 2}}, last_model="C:/new/b.gguf",
                   models_dir="C:/old")
        r = llama_cpp.purge_stale_after_dir_change("C:/old", "C:/new")
        self.assertFalse(r)
        self.assertEqual(self._read_mcfgs(), {"C:/new/b.gguf": {"port": 2}})

    def test_empty_new_dir_defaults_to_models(self):
        self._seed({"C:/old/a.gguf": {"port": 1}}, last_model="C:/old/a.gguf")
        with mock.patch.object(llama_cpp, "MODELS_DIR", Path("D:/models")):
            r = llama_cpp.purge_stale_after_dir_change("C:/old", "")
        self.assertTrue(r)
        self.assertEqual(self._read_mcfgs(), {})
        self.assertNotIn("last_model", self._read_llama())


class TestSetLlamaConfigDirChange(unittest.TestCase):
    """set_llama_config：models_dir 变化时自动清理失效配置（集成链路）。"""

    def setUp(self):
        self.llama_cfg = {}
        self.mcfgs = {}

    def _api_with_store(self, cfg):
        from gui_app.api_mixins.experimental import ExperimentalMixin
        api = ExperimentalMixin()
        p1 = mock.patch("gui_app.api_mixins.experimental.load_config",
                        return_value=cfg)
        p3 = mock.patch("gui_app.api_mixins.experimental.load_llama_config",
                        side_effect=lambda: self.llama_cfg)
        p4 = mock.patch("gui_app.api_mixins.experimental.update_llama_config",
                        side_effect=lambda m: m(self.llama_cfg))
        # purge_stale_after_dir_change / apply_model_configs 内延迟导入 config_store
        p5 = mock.patch("gui_app.config_store.load_llama_config",
                        side_effect=lambda: self.llama_cfg)
        p6 = mock.patch("gui_app.config_store.update_llama_config",
                        side_effect=lambda m: m(self.llama_cfg))
        p7 = mock.patch("gui_app.config_store.load_llama_model_configs",
                        side_effect=lambda: self.mcfgs)
        p8 = mock.patch("gui_app.config_store.update_llama_model_configs",
                        side_effect=lambda m: m(self.mcfgs))
        p1.start(); p3.start(); p4.start(); p5.start(); p6.start(); p7.start(); p8.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p3.stop); self.addCleanup(p4.stop)
        self.addCleanup(p5.stop); self.addCleanup(p6.stop)
        self.addCleanup(p7.stop); self.addCleanup(p8.stop)
        return api

    def test_dir_change_purges_stale(self):
        cfg = _cfg()
        self.llama_cfg = {"models_dir": "C:/old", "last_model": "C:/old/a.gguf"}
        self.mcfgs = {
            "C:/old/a.gguf": {"port": 1111},
            "C:/new/b.gguf": {"port": 2222},
        }
        api = self._api_with_store(cfg)
        r = api.set_llama_config({"models_dir": "C:/new"})
        self.assertTrue(r["ok"])
        self.assertEqual(self.llama_cfg["models_dir"], "C:/new")
        self.assertEqual(list(self.mcfgs), ["C:/new/b.gguf"])
        self.assertNotIn("last_model", self.llama_cfg)  # 旧目录记录被清空

    def test_same_dir_keeps_everything(self):
        cfg = _cfg()
        self.llama_cfg = {"models_dir": "C:/new", "last_model": "C:/old/a.gguf"}
        self.mcfgs = {"C:/old/a.gguf": {"port": 1}}
        api = self._api_with_store(cfg)
        api.set_llama_config({"models_dir": "C:/new"})
        self.assertEqual(self.llama_cfg["last_model"], "C:/old/a.gguf")
        self.assertEqual(list(self.mcfgs), ["C:/old/a.gguf"])

    def test_not_submitting_dir_keeps_everything(self):
        cfg = _cfg()
        self.llama_cfg = {"models_dir": "C:/new", "last_model": "C:/old/a.gguf"}
        self.mcfgs = {"C:/old/a.gguf": {"port": 1}}
        api = self._api_with_store(cfg)
        api.set_llama_config({"host": "0.0.0.0"})  # 与目录无关的键
        self.assertEqual(self.llama_cfg["last_model"], "C:/old/a.gguf")
        self.assertEqual(list(self.mcfgs), ["C:/old/a.gguf"])

    def test_reset_to_default_purges_custom(self):
        cfg = _cfg()
        self.llama_cfg = {"models_dir": "C:/old", "last_model": "C:/old/a.gguf"}
        self.mcfgs = {"C:/old/a.gguf": {"port": 1}}
        api = self._api_with_store(cfg)
        with mock.patch.object(llama_cpp, "MODELS_DIR", Path("D:/models")):
            r = api.set_llama_config({"models_dir": ""})  # 恢复默认目录
        self.assertTrue(r["ok"])
        self.assertEqual(self.llama_cfg["models_dir"], "")
        self.assertEqual(self.mcfgs, {})
        self.assertNotIn("last_model", self.llama_cfg)


class TestModelConfigsFileStore(_FileStoreTestCase):
    """load/update_llama_model_configs：独立文件原子读写 + 容错。"""

    def test_empty_when_missing(self):
        from gui_app.config_store import load_llama_model_configs
        self.assertEqual(load_llama_model_configs(), {})

    def test_roundtrip(self):
        from gui_app.config_store import (load_llama_model_configs,
                                          update_llama_model_configs)

        def _seed(m):
            m["C:/m.gguf"] = {"port": 9999}
            return m

        def _bump(m):
            m["C:/m.gguf"]["port"] = 10000
            return m

        update_llama_model_configs(_seed)
        update_llama_model_configs(_bump)
        self.assertEqual(load_llama_model_configs(),
                         {"C:/m.gguf": {"port": 10000}})

    def test_corrupt_file_tolerated(self):
        from gui_app.config_store import load_llama_model_configs
        self.mcfgs_file.write_text("{broken", encoding="utf-8")
        self.assertEqual(load_llama_model_configs(), {})

    def test_non_dict_tolerated(self):
        from gui_app.config_store import load_llama_model_configs
        self.mcfgs_file.write_text(json.dumps([1, 2]), encoding="utf-8")
        self.assertEqual(load_llama_model_configs(), {})


class TestLlamaConfigFileStore(_FileStoreTestCase):
    """load/update_llama_config：llama 全局配置独立文件原子读写 + 容错。"""

    def test_empty_when_missing(self):
        from gui_app.config_store import load_llama_config
        self.assertEqual(load_llama_config(), {})

    def test_roundtrip(self):
        from gui_app.config_store import load_llama_config, update_llama_config

        def _seed(c):
            c["host"] = "0.0.0.0"
            c["port"] = 11434
            c["last_model"] = "C:/m.gguf"
            return c

        def _bump(c):
            c["port"] = 10000
            return c

        update_llama_config(_seed)
        update_llama_config(_bump)
        self.assertEqual(load_llama_config(),
                         {"host": "0.0.0.0", "port": 10000,
                          "last_model": "C:/m.gguf"})

    def test_corrupt_file_tolerated(self):
        from gui_app.config_store import load_llama_config
        self.llama_file.write_text("{broken", encoding="utf-8")
        self.assertEqual(load_llama_config(), {})

    def test_non_dict_tolerated(self):
        from gui_app.config_store import load_llama_config
        self.llama_file.write_text(json.dumps([1, 2]), encoding="utf-8")
        self.assertEqual(load_llama_config(), {})


class TestGetLlamaStatusConfig(unittest.TestCase):
    """get_llama_status：config = DEFAULTS + 模块文件；enabled/integrate 从模块文件读。"""

    def test_assembles_config_from_independent_file(self):
        from gui_app.api_mixins.experimental import ExperimentalMixin
        api = ExperimentalMixin()
        llama_cfg = {"host": "0.0.0.0", "port": 9999,
                     "models_dir": "C:/models", "model": "C:/m.gguf",
                     "last_model": "C:/m.gguf", "auto_run": True,
                     "show_logs": True, "integrate": True, "enabled": True}
        with mock.patch("gui_app.llama_cpp.get_status",
                        return_value={"ready": True, "running": False,
                                      "models": []}), \
             mock.patch("gui_app.llama_cpp.ensure_model_configs",
                        return_value={}), \
             mock.patch("gui_app.api_mixins.experimental.load_llama_config",
                        side_effect=lambda: llama_cfg), \
             mock.patch("gui_app.api_mixins.experimental.load_llama_model_configs",
                        return_value={}):
            st = api.get_llama_status()
        c = st["config"]
        # 独立文件值覆盖 DEFAULTS
        self.assertEqual(c["host"], "0.0.0.0")
        self.assertEqual(c["port"], 9999)
        # 未覆盖键由 DEFAULTS 兜底（前端读取的完整结构）
        self.assertEqual(c["ngl"], 999)
        self.assertEqual(c["load_mode"], "none")
        self.assertEqual(c["models_dir"], "C:/models")
        # 偏好键齐全
        for k in ("models_dir", "model", "last_model", "auto_run", "show_logs"):
            self.assertIn(k, c)
        # integrate / enabled 从 llama 模块文件读（config.json 不再持有开关）
        self.assertIs(c["integrate"], True)
        self.assertIs(st["enabled"], True)

    def test_switches_default_false_when_missing(self):
        from gui_app.api_mixins.experimental import ExperimentalMixin
        api = ExperimentalMixin()
        with mock.patch("gui_app.llama_cpp.get_status",
                        return_value={"ready": True, "running": False,
                                      "models": []}), \
             mock.patch("gui_app.llama_cpp.ensure_model_configs",
                        return_value={}), \
             mock.patch("gui_app.api_mixins.experimental.load_llama_config",
                        side_effect=lambda: {}), \
             mock.patch("gui_app.api_mixins.experimental.load_llama_model_configs",
                        return_value={}):
            st = api.get_llama_status()
        # 模块文件缺失（未安装）→ 开关默认 False，程序可用
        self.assertIs(st["config"]["integrate"], False)
        self.assertIs(st["enabled"], False)


if __name__ == "__main__":
    unittest.main()
