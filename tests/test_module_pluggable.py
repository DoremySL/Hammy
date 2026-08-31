"""可插拔模块（llama/whisper/pixai）测试。
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from gui_app.config_store import _default_config, _pixai_defaults, _whisper_defaults


def _cfg(**over):
    """构造一份基础配置 dict，over 覆盖 experimental 段。"""
    c = _default_config()
    c["experimental"].update(over)
    return c


def _redirect_workspace_files(ws_dir):
    """把 config.json / 三个模块文件夹全部重定向到临时目录。

    绑定语义：模块级导入的常量 patch 消费方命名空间（config_store /
    experimental），函数内延迟导入的常量 patch 源模块 workspace_paths
    （pixai_tagger / faster_whisper / runner / llama_cpp）。
    """
    from gui_app import config_store, workspace_paths
    from gui_app.api_mixins import experimental
    config_file = ws_dir / "config.json"
    pixai_file = ws_dir / "pixai" / "config.json"
    pixai_tags = ws_dir / "pixai" / "tags.json"
    whisper_file = ws_dir / "whisper" / "config.json"
    whisper_tx = ws_dir / "whisper" / "transcripts.json"
    whisper_srt = ws_dir / "whisper" / "srt"
    llama_file = ws_dir / "llama" / "config.json"
    mcfgs_file = ws_dir / "llama" / "model_configs.json"
    scan_cache = ws_dir / "llama" / "scan_cache.json"
    patchers = [
        # config_store（模块级导入）
        mock.patch.object(config_store, "CONFIG_FILE", config_file),
        mock.patch.object(config_store, "PIXAI_CONFIG_FILE", pixai_file),
        mock.patch.object(config_store, "WHISPER_CONFIG_FILE", whisper_file),
        mock.patch.object(config_store, "LLAMA_CONFIG_FILE", llama_file),
        mock.patch.object(config_store, "LLAMA_MODEL_CONFIGS_FILE", mcfgs_file),
        # experimental（模块级导入）
        mock.patch.object(experimental, "PIXAI_TAGS_FILE", pixai_tags),
        mock.patch.object(experimental, "WHISPER_TRANSCRIPTS_FILE", whisper_tx),
        mock.patch.object(experimental, "WHISPER_SRT_DIR", whisper_srt),
        mock.patch.object(experimental, "LLAMA_CONFIG_FILE", llama_file),
        # 源模块 workspace_paths（pixai_tagger / faster_whisper / runner 函数内导入）
        mock.patch.object(workspace_paths, "PIXAI_CONFIG_FILE", pixai_file),
        mock.patch.object(workspace_paths, "PIXAI_TAGS_FILE", pixai_tags),
        mock.patch.object(workspace_paths, "WHISPER_CONFIG_FILE", whisper_file),
        mock.patch.object(workspace_paths, "WHISPER_TRANSCRIPTS_FILE", whisper_tx),
        mock.patch.object(workspace_paths, "WHISPER_SRT_DIR", whisper_srt),
        mock.patch.object(workspace_paths, "LLAMA_SCAN_CACHE_FILE", scan_cache),
    ]
    paths = {
        "config": config_file, "pixai": pixai_file, "pixai_tags": pixai_tags,
        "whisper": whisper_file, "whisper_tx": whisper_tx,
        "whisper_srt": whisper_srt, "llama": llama_file, "mcfgs": mcfgs_file,
        "scan_cache": scan_cache,
    }
    return patchers, paths


class _WorkspaceTestCase(unittest.TestCase):
    """文件级测试基类：真实读写，仅重定向存储路径。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        ws_dir = Path(self._tmp.name)
        patchers, self.paths = _redirect_workspace_files(ws_dir)
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)

    def _read(self, key):
        return json.loads(self.paths[key].read_text(encoding="utf-8"))

    def _write(self, key, data):
        p = self.paths[key]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _write_config(self, cfg):
        self.paths["config"].write_text(
            json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    def _disk_cfg(self):
        return json.loads(self.paths["config"].read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════════
# load_config 组装 experimental 段（前端结构不变，组装不落盘）
# ═══════════════════════════════════════════════════════════════

def _PIXAI_KEYMAP_DEFAULTS():
    from gui_app import config_store
    return {config_store._PIXAI_KEY_MAP[k]: v
            for k, v in _pixai_defaults().items()}


def _WHISPER_KEYMAP_DEFAULTS():
    from gui_app import config_store
    return {config_store._WHISPER_KEY_MAP[k]: v
            for k, v in _whisper_defaults().items()}


class TestAssemble(_WorkspaceTestCase):
    """_assemble_experimental：模块文件夹 → experimental 段。"""

    def test_assemble_defaults(self):
        from gui_app import config_store
        self._write_config(_cfg())
        exp = config_store.load_config()["experimental"]
        # pixai（含 classify）+ whisper + llama 2 开关，全部默认值
        self.assertEqual(exp, {
            **_PIXAI_KEYMAP_DEFAULTS(), **_WHISPER_KEYMAP_DEFAULTS(),
            "llama_enabled": False, "llama_integrate": False,
        })
        # 「识别二次元作品」默认开启
        self.assertIs(exp["pixai_classify"], True)

    def test_assemble_reflects_module_files(self):
        from gui_app import config_store
        self._write("pixai", {"enabled": True, "frames": 9, "classify": False})
        self._write("whisper", {"enabled": True, "batch": True})
        self._write("llama", {"enabled": True, "integrate": True})
        self._write_config(_cfg())
        exp = config_store.load_config()["experimental"]
        self.assertIs(exp["pixai_tagger_enabled"], True)
        self.assertEqual(exp["pixai_frames"], 9)
        self.assertIs(exp["pixai_classify"], False)
        self.assertIs(exp["whisper_enabled"], True)
        self.assertIs(exp["whisper_batch"], True)
        self.assertIs(exp["llama_enabled"], True)
        self.assertIs(exp["llama_integrate"], True)

    def test_assemble_not_persisted(self):
        # 组装只发生在返回值（内存）：磁盘 config.json 不写任何模块键
        from gui_app import config_store
        self._write("pixai", {"enabled": True, "frames": 9})
        self._write("whisper", {"enabled": True, "batch": True})
        self._write("llama", {"enabled": True, "integrate": True})
        self._write_config(_cfg())
        config_store.load_config()
        disk = self._disk_cfg()
        self.assertEqual(disk.get("experimental", {}), {})
        self.assertEqual(
            [k for k in disk if "pixai" in k or "whisper" in k or "llama" in k],
            [])

    def test_module_dir_deleted_still_usable(self):
        # 模拟卸载：模块文件夹整个删除后，load_config 用默认值组装，程序可用
        from gui_app import config_store
        self._write("pixai", {"enabled": True, "frames": 9})
        self._write("whisper", {"enabled": True})
        self._write("llama", {"enabled": True, "integrate": True})
        self._write_config(_cfg())
        shutil.rmtree(self.paths["pixai"].parent)   # 卸载 pixai
        shutil.rmtree(self.paths["whisper"].parent)  # 卸载 whisper
        shutil.rmtree(self.paths["llama"].parent)    # 卸载 llama
        exp = config_store.load_config()["experimental"]
        self.assertIs(exp["pixai_tagger_enabled"], False)
        self.assertIs(exp["whisper_enabled"], False)
        self.assertIs(exp["llama_enabled"], False)
        self.assertIs(exp["llama_integrate"], False)
        # 磁盘 config.json 的 experimental 段仍为空（无模块键残留）
        self.assertEqual(self._disk_cfg().get("experimental", {}), {})


# ═══════════════════════════════════════════════════════════════
# save_config 拆键路由
# ═══════════════════════════════════════════════════════════════

class TestSaveConfigSplit(_WorkspaceTestCase):
    """save_config：experimental 提交的 pixai_*/whisper_* 路由到模块文件。"""

    def test_pixai_keys_route_to_module_file(self):
        from gui_app import config_store
        self._write_config(_cfg())
        r = config_store.save_config({
            "experimental": {"pixai_tagger_enabled": True, "pixai_frames": 6},
        })
        self.assertTrue(r["ok"])
        # 模块文件：默认值 + 提交值
        self.assertEqual(self._read("pixai")["enabled"], True)
        self.assertEqual(self._read("pixai")["frames"], 6)
        # config.json：experimental 段为空（模块键不落盘）
        self.assertEqual(self._disk_cfg().get("experimental", {}), {})

    def test_pixai_classify_routes_to_module_file(self):
        # 「识别二次元作品」开关同样路由到 pixai/config.json 的 classify 键
        from gui_app import config_store
        self._write_config(_cfg())
        config_store.save_config({"experimental": {"pixai_classify": False}})
        self.assertIs(self._read("pixai")["classify"], False)
        self.assertEqual(self._disk_cfg().get("experimental", {}), {})

    def test_whisper_keys_route_to_module_file(self):
        from gui_app import config_store
        self._write_config(_cfg())
        config_store.save_config({
            "experimental": {"whisper_enabled": True, "whisper_language": "en"},
        })
        self.assertEqual(self._read("whisper")["enabled"], True)
        self.assertEqual(self._read("whisper")["language"], "en")
        self.assertEqual(self._disk_cfg().get("experimental", {}), {})

    def test_mixed_exp_routes(self):
        from gui_app import config_store
        self._write_config(_cfg())
        config_store.save_config({
            "experimental": {"pixai_threshold": 0.8, "whisper_batch": True},
        })
        self.assertEqual(self._read("pixai")["threshold"], 0.8)
        self.assertEqual(self._read("whisper")["batch"], True)
        self.assertEqual(self._disk_cfg().get("experimental", {}), {})

    def test_rest_keys_stay_in_config(self):
        from gui_app import config_store
        self._write_config(_cfg())
        config_store.save_config({
            "experimental": {"custom_key": 1, "pixai_frames": 7},
        })
        self.assertEqual(self._read("pixai")["frames"], 7)
        self.assertEqual(self._disk_cfg()["experimental"],
                         {"custom_key": 1})  # 非模块键仍在 config.json

    def test_llama_switches_dropped_not_persisted(self):
        # llama 开关不经 save_config（权威在 llama/config.json，只走
        # set_llama_enabled/set_llama_config）：提交即丢弃，config.json 不残留
        from gui_app import config_store
        self._write_config(_cfg())
        config_store.save_config({
            "experimental": {"llama_enabled": True, "llama_integrate": True},
        })
        disk = self._disk_cfg()
        self.assertNotIn("llama_enabled", disk.get("experimental", {}))
        self.assertNotIn("llama_integrate", disk.get("experimental", {}))
        self.assertFalse(self.paths["llama"].exists())  # 不写模块文件
        # 组装仍以模块文件为准（默认 False）
        exp = config_store.load_config()["experimental"]
        self.assertIs(exp["llama_enabled"], False)

    def test_no_exp_noop(self):
        from gui_app import config_store
        self._write_config(_cfg())
        config_store.save_config({"theme": "dark"})
        self.assertEqual(self._disk_cfg()["theme"], "dark")
        self.assertFalse(self.paths["pixai"].exists())


# ═══════════════════════════════════════════════════════════════
# set_*_enabled：写模块文件；模块未安装时 no-op（防止卸载后复活）
# ═══════════════════════════════════════════════════════════════

class TestSetEnabled(unittest.TestCase):
    """set_*_enabled 写入模块文件；安装目录不存在时 no-op 不落盘。"""

    def _api(self):
        from gui_app.api_mixins.experimental import ExperimentalMixin
        return ExperimentalMixin()

    def _tmp_dir(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    # ── pixai ──

    def test_pixai_enabled_writes_module_config(self):
        api = self._api()
        with mock.patch("gui_app.pixai_tagger.PIXAI_TAGGER_DIR",
                        self._tmp_dir()), \
             mock.patch("gui_app.api_mixins.experimental.update_pixai_config") as up:
            r = api.set_pixai_tagger_enabled(True)
        self.assertEqual(r, {"ok": True, "enabled": True})
        up.assert_called_once()
        cfg = {"enabled": False, "frames": 5}
        out = up.call_args[0][0](cfg)
        self.assertIs(out, cfg)          # 就地修改
        self.assertIs(cfg["enabled"], True)

    def test_pixai_enabled_noop_when_not_installed(self):
        # 卸载成功后前端补调 set(false)：此时安装目录与模块文件夹已删，
        # 若落盘会重新创建模块文件夹 → 必须 no-op
        api = self._api()
        with mock.patch("gui_app.pixai_tagger.PIXAI_TAGGER_DIR",
                        Path("C:/definitely/not/exists")), \
             mock.patch("gui_app.api_mixins.experimental.update_pixai_config") as up:
            r = api.set_pixai_tagger_enabled(False)
        self.assertEqual(r, {"ok": True, "enabled": False})
        up.assert_not_called()

    def test_get_pixai_status_reads_module_config(self):
        api = self._api()
        with mock.patch("gui_app.pixai_tagger.get_status",
                        return_value={"ready": True}), \
             mock.patch("gui_app.api_mixins.experimental.load_pixai_config",
                        return_value={"enabled": True, "frames": 5}):
            st = api.get_pixai_tagger_status()
        self.assertIs(st["enabled"], True)

    # ── whisper ──

    def test_whisper_enabled_writes_module_config(self):
        api = self._api()
        with mock.patch("gui_app.faster_whisper.WHISPER_DIR",
                        self._tmp_dir()), \
             mock.patch("gui_app.api_mixins.experimental.update_whisper_config") as uw:
            r = api.set_whisper_enabled(True)
        self.assertEqual(r, {"ok": True, "enabled": True})
        uw.assert_called_once()
        cfg = {"enabled": False, "vad": True}
        out = uw.call_args[0][0](cfg)
        self.assertIs(out, cfg)
        self.assertIs(cfg["enabled"], True)

    def test_whisper_enabled_noop_when_not_installed(self):
        api = self._api()
        with mock.patch("gui_app.faster_whisper.WHISPER_DIR",
                        Path("C:/definitely/not/exists")), \
             mock.patch("gui_app.api_mixins.experimental.update_whisper_config") as uw:
            r = api.set_whisper_enabled(False)
        self.assertEqual(r, {"ok": True, "enabled": False})
        uw.assert_not_called()

    def test_get_whisper_status_reads_module_config(self):
        api = self._api()
        with mock.patch("gui_app.faster_whisper.get_status",
                        return_value={"ready": True}), \
             mock.patch("gui_app.api_mixins.experimental.load_whisper_config",
                        return_value={"enabled": True, "vad": True}):
            st = api.get_whisper_status()
        self.assertIs(st["enabled"], True)

    # ── llama ──

    def test_llama_enabled_writes_module_config(self):
        api = self._api()
        with mock.patch("gui_app.llama_cpp.LLAMA_DIR", self._tmp_dir()), \
             mock.patch("gui_app.api_mixins.experimental.update_llama_config") as ul:
            r = api.set_llama_enabled(True)
        self.assertEqual(r, {"ok": True, "enabled": True})
        ul.assert_called_once()
        cfg = {"enabled": False, "host": "0.0.0.0"}
        out = ul.call_args[0][0](cfg)
        self.assertIs(out, cfg)
        self.assertIs(cfg["enabled"], True)

    def test_llama_enabled_noop_when_not_installed(self):
        api = self._api()
        with mock.patch("gui_app.llama_cpp.LLAMA_DIR",
                        Path("C:/definitely/not/exists")), \
             mock.patch("gui_app.api_mixins.experimental.update_llama_config") as ul:
            r = api.set_llama_enabled(False)
        self.assertEqual(r, {"ok": True, "enabled": False})
        ul.assert_not_called()


# ═══════════════════════════════════════════════════════════════
# 卸载：安装目录 + 模块文件夹一并删除
# ═══════════════════════════════════════════════════════════════

class TestRemove(unittest.TestCase):
    """remove_*：删除安装目录 + 模块配置文件夹（可插拔）。"""

    def test_remove_pixai_deletes_install_and_module_dir(self):
        from gui_app.pixai_tagger import remove_pixai_tagger
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        inst = base / "pixai-tagger"
        inst.mkdir()
        mod = base / "pixai"
        mod.mkdir()
        with mock.patch("gui_app.pixai_tagger.PIXAI_TAGGER_DIR", inst), \
             mock.patch("gui_app.workspace_paths.PIXAI_CONFIG_FILE",
                        mod / "config.json"):
            r = remove_pixai_tagger()
        self.assertTrue(r["ok"])
        self.assertFalse(inst.exists())   # 安装目录已删
        self.assertFalse(mod.exists())    # 模块文件夹已删

    def test_remove_pixai_not_installed_still_cleans_module_dir(self):
        from gui_app.pixai_tagger import remove_pixai_tagger
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        mod = base / "pixai"
        mod.mkdir()
        with mock.patch("gui_app.pixai_tagger.PIXAI_TAGGER_DIR",
                        base / "pixai-tagger"), \
             mock.patch("gui_app.workspace_paths.PIXAI_CONFIG_FILE",
                        mod / "config.json"):
            r = remove_pixai_tagger()
        self.assertTrue(r["ok"])
        self.assertFalse(mod.exists())    # 残留模块文件夹仍被清掉

    def test_remove_pixai_failure_keeps_module_dir(self):
        from gui_app.pixai_tagger import remove_pixai_tagger
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        inst = base / "pixai-tagger"
        inst.mkdir()
        mod = base / "pixai"
        mod.mkdir()
        with mock.patch("gui_app.pixai_tagger.PIXAI_TAGGER_DIR", inst), \
             mock.patch("gui_app.pixai_tagger.shutil.rmtree",
                        side_effect=OSError("拒绝访问")), \
             mock.patch("gui_app.workspace_paths.PIXAI_CONFIG_FILE",
                        mod / "config.json"):
            r = remove_pixai_tagger()
        self.assertFalse(r["ok"])
        self.assertTrue(inst.exists())
        self.assertTrue(mod.exists())  # 删除失败 → 模块文件夹不动

    def test_remove_whisper_deletes_install_and_module_dir(self):
        from gui_app.faster_whisper import remove_faster_whisper
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        inst = base / "faster-whisper"
        inst.mkdir()
        mod = base / "whisper"
        mod.mkdir()
        with mock.patch("gui_app.faster_whisper.WHISPER_DIR", inst), \
             mock.patch("gui_app.workspace_paths.WHISPER_CONFIG_FILE",
                        mod / "config.json"):
            r = remove_faster_whisper()
        self.assertTrue(r["ok"])
        self.assertFalse(inst.exists())
        self.assertFalse(mod.exists())

    def test_remove_whisper_not_installed_still_cleans_module_dir(self):
        from gui_app.faster_whisper import remove_faster_whisper
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        mod = base / "whisper"
        mod.mkdir()
        with mock.patch("gui_app.faster_whisper.WHISPER_DIR",
                        base / "faster-whisper"), \
             mock.patch("gui_app.workspace_paths.WHISPER_CONFIG_FILE",
                        mod / "config.json"):
            r = remove_faster_whisper()
        self.assertTrue(r["ok"])
        self.assertFalse(mod.exists())

    def test_remove_whisper_failure_keeps_module_dir(self):
        from gui_app.faster_whisper import remove_faster_whisper
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        inst = base / "faster-whisper"
        inst.mkdir()
        mod = base / "whisper"
        mod.mkdir()
        with mock.patch("gui_app.faster_whisper.WHISPER_DIR", inst), \
             mock.patch("gui_app.faster_whisper.shutil.rmtree",
                        side_effect=OSError("拒绝访问")), \
             mock.patch("gui_app.workspace_paths.WHISPER_CONFIG_FILE",
                        mod / "config.json"):
            r = remove_faster_whisper()
        self.assertFalse(r["ok"])
        self.assertTrue(inst.exists())
        self.assertTrue(mod.exists())  # 删除失败 → 模块文件夹不动


# ═══════════════════════════════════════════════════════════════
# 模块配置存储（pixai / whisper config.json）
# ═══════════════════════════════════════════════════════════════

class TestModuleConfigStore(_WorkspaceTestCase):
    """load/update_pixai_config、load/update_whisper_config：roundtrip + 容错。"""

    def test_pixai_defaults_when_missing(self):
        from gui_app.config_store import load_pixai_config
        self.assertEqual(load_pixai_config(), _pixai_defaults())

    def test_pixai_roundtrip(self):
        from gui_app.config_store import load_pixai_config, update_pixai_config

        def _seed(c):
            c["enabled"] = True
            c["frames"] = 8
            return c

        def _bump(c):
            c["threshold"] = 0.75
            c["classify"] = False
            return c

        update_pixai_config(_seed)
        update_pixai_config(_bump)
        cfg = load_pixai_config()
        self.assertIs(cfg["enabled"], True)
        self.assertEqual(cfg["frames"], 8)
        self.assertEqual(cfg["threshold"], 0.75)
        self.assertIs(cfg["classify"], False)
        self.assertEqual(cfg["short_side"], 448)  # 未覆盖键保留默认

    def test_pixai_corrupt_tolerated(self):
        from gui_app.config_store import load_pixai_config, update_pixai_config
        self.paths["pixai"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["pixai"].write_text("{broken", encoding="utf-8")
        self.assertEqual(load_pixai_config(), _pixai_defaults())
        update_pixai_config(lambda c: c.update(enabled=True) or c)
        self.assertIs(load_pixai_config()["enabled"], True)  # 损坏后仍可写

    def test_pixai_non_dict_tolerated(self):
        from gui_app.config_store import load_pixai_config
        self.paths["pixai"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["pixai"].write_text(json.dumps([1, 2]), encoding="utf-8")
        self.assertEqual(load_pixai_config(), _pixai_defaults())

    def test_whisper_defaults_when_missing(self):
        from gui_app.config_store import load_whisper_config
        self.assertEqual(load_whisper_config(), _whisper_defaults())

    def test_whisper_roundtrip(self):
        from gui_app.config_store import load_whisper_config, update_whisper_config

        def _seed(c):
            c["enabled"] = True
            c["language"] = "zh"
            return c

        def _bump(c):
            c["max_chars"] = 1200
            return c

        update_whisper_config(_seed)
        update_whisper_config(_bump)
        cfg = load_whisper_config()
        self.assertIs(cfg["enabled"], True)
        self.assertEqual(cfg["language"], "zh")
        self.assertEqual(cfg["max_chars"], 1200)
        self.assertIs(cfg["batch"], False)   # 未覆盖键保留默认

    def test_whisper_corrupt_tolerated(self):
        from gui_app.config_store import load_whisper_config
        self.paths["whisper"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["whisper"].write_text("{broken", encoding="utf-8")
        self.assertEqual(load_whisper_config(), _whisper_defaults())

    def test_whisper_non_dict_tolerated(self):
        from gui_app.config_store import load_whisper_config
        self.paths["whisper"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["whisper"].write_text(json.dumps("x"), encoding="utf-8")
        self.assertEqual(load_whisper_config(), _whisper_defaults())


if __name__ == "__main__":
    unittest.main()
