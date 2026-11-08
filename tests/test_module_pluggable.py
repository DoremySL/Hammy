"""可插拔模块（llama/语音转录/tagger）测试。
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from gui_app.config_store import _default_config, _tagger_defaults, _transcribe_defaults


def _cfg(**over):
    """构造一份基础配置 dict，over 覆盖 experimental 段。"""
    c = _default_config()
    c["experimental"].update(over)
    return c


def _redirect_workspace_files(ws_dir):
    """把 config.json / 三个模块文件夹全部重定向到临时目录。

    绑定语义：模块级导入的常量 patch 消费方命名空间（config_store /
    experimental），函数内延迟导入的常量 patch 源模块 workspace_paths
    （tagger / transcribe_cpp / runner / llama_cpp）。
    """
    from gui_app import config_store, workspace_paths
    from gui_app.api_mixins import experimental
    config_file = ws_dir / "config.json"
    tagger_file = ws_dir / "tagger" / "config.json"
    tagger_tags = ws_dir / "tagger" / "tags.json"
    transcribe_file = ws_dir / "transcribe" / "config.json"
    transcribe_tx = ws_dir / "transcribe" / "transcripts.json"
    transcribe_srt = ws_dir / "transcribe" / "srt"
    llama_file = ws_dir / "llama" / "config.json"
    mcfgs_file = ws_dir / "llama" / "model_configs.json"
    scan_cache = ws_dir / "llama" / "scan_cache.json"
    patchers = [
        # config_store（模块级导入）
        mock.patch.object(config_store, "CONFIG_FILE", config_file),
        mock.patch.object(config_store, "TAGGER_CONFIG_FILE", tagger_file),
        mock.patch.object(config_store, "TRANSCRIBE_CONFIG_FILE", transcribe_file),
        mock.patch.object(config_store, "LLAMA_CONFIG_FILE", llama_file),
        mock.patch.object(config_store, "LLAMA_MODEL_CONFIGS_FILE", mcfgs_file),
        # experimental（模块级导入）
        mock.patch.object(experimental, "TAGGER_TAGS_FILE", tagger_tags),
        mock.patch.object(experimental, "TRANSCRIBE_TRANSCRIPTS_FILE", transcribe_tx),
        mock.patch.object(experimental, "TRANSCRIBE_SRT_DIR", transcribe_srt),
        mock.patch.object(experimental, "LLAMA_CONFIG_FILE", llama_file),
        # 源模块 workspace_paths（tagger / transcribe_cpp / runner 函数内导入）
        mock.patch.object(workspace_paths, "TAGGER_CONFIG_FILE", tagger_file),
        mock.patch.object(workspace_paths, "TAGGER_TAGS_FILE", tagger_tags),
        mock.patch.object(workspace_paths, "TRANSCRIBE_CONFIG_FILE", transcribe_file),
        mock.patch.object(workspace_paths, "TRANSCRIBE_TRANSCRIPTS_FILE", transcribe_tx),
        mock.patch.object(workspace_paths, "TRANSCRIBE_SRT_DIR", transcribe_srt),
        mock.patch.object(workspace_paths, "LLAMA_SCAN_CACHE_FILE", scan_cache),
    ]
    paths = {
        "config": config_file, "tagger": tagger_file, "tagger_tags": tagger_tags,
        "transcribe": transcribe_file, "transcribe_tx": transcribe_tx,
        "transcribe_srt": transcribe_srt, "llama": llama_file, "mcfgs": mcfgs_file,
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

def _TAGGER_KEYMAP_DEFAULTS():
    from gui_app import config_store
    # 只取映射过的键：未映射键（precision 内部状态）不进 experimental 段
    return {config_store._TAGGER_KEY_MAP[k]: v
            for k, v in _tagger_defaults().items() if k in config_store._TAGGER_KEY_MAP}


def _TRANSCRIBE_KEYMAP_DEFAULTS():
    from gui_app import config_store
    return {config_store._TRANSCRIBE_KEY_MAP[k]: v
            for k, v in _transcribe_defaults().items()}


class TestAssemble(_WorkspaceTestCase):
    """_assemble_experimental：模块文件夹 → experimental 段。"""

    def test_assemble_defaults(self):
        from gui_app import config_store
        self._write_config(_cfg())
        exp = config_store.load_config()["experimental"]
        # tagger + 语音转录 + llama 2 开关，全部默认值
        self.assertEqual(exp, {
            **_TAGGER_KEYMAP_DEFAULTS(), **_TRANSCRIBE_KEYMAP_DEFAULTS(),
            "llama_enabled": False, "llama_integrate": False,
        })
        # 内部状态键（precision）不进设置段
        self.assertNotIn("precision", exp)

    def test_assemble_reflects_module_files(self):
        from gui_app import config_store
        self._write("tagger", {"enabled": True, "frames": 9})
        self._write("transcribe", {"enabled": True, "language": "en"})
        self._write("llama", {"enabled": True, "integrate": True})
        self._write_config(_cfg())
        exp = config_store.load_config()["experimental"]
        self.assertIs(exp["tagger_enabled"], True)
        self.assertEqual(exp["tagger_frames"], 9)
        self.assertIs(exp["transcribe_enabled"], True)
        self.assertEqual(exp["transcribe_language"], "en")
        self.assertIs(exp["llama_enabled"], True)
        self.assertIs(exp["llama_integrate"], True)

    def test_assemble_not_persisted(self):
        # 组装只发生在返回值（内存）：磁盘 config.json 不写任何模块键
        from gui_app import config_store
        self._write("tagger", {"enabled": True, "frames": 9})
        self._write("transcribe", {"enabled": True, "language": "en"})
        self._write("llama", {"enabled": True, "integrate": True})
        self._write_config(_cfg())
        config_store.load_config()
        disk = self._disk_cfg()
        self.assertEqual(disk.get("experimental", {}), {})
        self.assertEqual(
            [k for k in disk if "tagger" in k or "transcribe" in k or "llama" in k],
            [])

    def test_module_dir_deleted_still_usable(self):
        # 模拟卸载：模块文件夹整个删除后，load_config 用默认值组装，程序可用
        from gui_app import config_store
        self._write("tagger", {"enabled": True, "frames": 9})
        self._write("transcribe", {"enabled": True})
        self._write("llama", {"enabled": True, "integrate": True})
        self._write_config(_cfg())
        shutil.rmtree(self.paths["tagger"].parent)   # 卸载 tagger
        shutil.rmtree(self.paths["transcribe"].parent)  # 卸载语音转录
        shutil.rmtree(self.paths["llama"].parent)    # 卸载 llama
        exp = config_store.load_config()["experimental"]
        self.assertIs(exp["tagger_enabled"], False)
        self.assertIs(exp["transcribe_enabled"], False)
        self.assertIs(exp["llama_enabled"], False)
        self.assertIs(exp["llama_integrate"], False)
        # 磁盘 config.json 的 experimental 段仍为空（无模块键残留）
        self.assertEqual(self._disk_cfg().get("experimental", {}), {})


# ═══════════════════════════════════════════════════════════════
# save_config 拆键路由
# ═══════════════════════════════════════════════════════════════

class TestSaveConfigSplit(_WorkspaceTestCase):
    """save_config：experimental 提交的 tagger_*/transcribe_* 路由到模块文件。"""

    def test_tagger_keys_route_to_module_file(self):
        from gui_app import config_store
        self._write_config(_cfg())
        r = config_store.save_config({
            "experimental": {"tagger_enabled": True, "tagger_frames": 6},
        })
        self.assertTrue(r["ok"])
        # 模块文件：默认值 + 提交值
        self.assertEqual(self._read("tagger")["enabled"], True)
        self.assertEqual(self._read("tagger")["frames"], 6)
        # config.json：experimental 段为空（模块键不落盘）
        self.assertEqual(self._disk_cfg().get("experimental", {}), {})

    def test_transcribe_keys_route_to_module_file(self):
        from gui_app import config_store
        self._write_config(_cfg())
        config_store.save_config({
            "experimental": {"transcribe_enabled": True, "transcribe_language": "en"},
        })
        self.assertEqual(self._read("transcribe")["enabled"], True)
        self.assertEqual(self._read("transcribe")["language"], "en")
        self.assertEqual(self._disk_cfg().get("experimental", {}), {})

    def test_mixed_exp_routes(self):
        from gui_app import config_store
        self._write_config(_cfg())
        config_store.save_config({
            "experimental": {"tagger_threshold": 0.8, "transcribe_language": "ja"},
        })
        self.assertEqual(self._read("tagger")["threshold"], 0.8)
        self.assertEqual(self._read("transcribe")["language"], "ja")
        self.assertEqual(self._disk_cfg().get("experimental", {}), {})

    def test_rest_keys_stay_in_config(self):
        from gui_app import config_store
        self._write_config(_cfg())
        config_store.save_config({
            "experimental": {"custom_key": 1, "tagger_frames": 7},
        })
        self.assertEqual(self._read("tagger")["frames"], 7)
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
        self.assertFalse(self.paths["tagger"].exists())


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

    # ── tagger ──

    def test_tagger_enabled_writes_module_config(self):
        api = self._api()
        with mock.patch("gui_app.tagger.TAGGER_DIR",
                        self._tmp_dir()), \
             mock.patch("gui_app.api_mixins.experimental.update_tagger_config") as up:
            r = api.set_tagger_enabled(True)
        self.assertEqual(r, {"ok": True, "enabled": True})
        up.assert_called_once()
        cfg = {"enabled": False, "frames": 5}
        self.assertIsNone(up.call_args[0][0](cfg))  # 原地修改，返回 None
        self.assertIs(cfg["enabled"], True)

    def test_tagger_enabled_noop_when_not_installed(self):
        # 卸载成功后前端补调 set(false)：此时安装目录与模块文件夹已删，
        # 若落盘会重新创建模块文件夹 → 必须 no-op
        api = self._api()
        with mock.patch("gui_app.tagger.TAGGER_DIR",
                        Path("C:/definitely/not/exists")), \
             mock.patch("gui_app.api_mixins.experimental.update_tagger_config") as up:
            r = api.set_tagger_enabled(False)
        self.assertEqual(r, {"ok": True, "enabled": False})
        up.assert_not_called()

    def test_get_tagger_status_reads_module_config(self):
        api = self._api()
        with mock.patch("gui_app.tagger.get_status",
                        return_value={"ready": True}), \
             mock.patch("gui_app.api_mixins.experimental.load_tagger_config",
                        return_value={"enabled": True, "frames": 5}):
            st = api.get_tagger_status()
        self.assertIs(st["enabled"], True)

    # ── 语音转录 ──

    def test_transcribe_enabled_writes_module_config(self):
        api = self._api()
        with mock.patch("gui_app.transcribe_cpp.TRANSCRIBE_DIR",
                        self._tmp_dir()), \
             mock.patch("gui_app.api_mixins.experimental.update_transcribe_config") as uw:
            r = api.set_transcribe_enabled(True)
        self.assertEqual(r, {"ok": True, "enabled": True})
        uw.assert_called_once()
        cfg = {"enabled": False, "language": ""}
        self.assertIsNone(uw.call_args[0][0](cfg))  # 原地修改，返回 None
        self.assertIs(cfg["enabled"], True)

    def test_transcribe_enabled_noop_when_not_installed(self):
        api = self._api()
        with mock.patch("gui_app.transcribe_cpp.TRANSCRIBE_DIR",
                        Path("C:/definitely/not/exists")), \
             mock.patch("gui_app.api_mixins.experimental.update_transcribe_config") as uw:
            r = api.set_transcribe_enabled(False)
        self.assertEqual(r, {"ok": True, "enabled": False})
        uw.assert_not_called()

    def test_get_transcribe_status_reads_module_config(self):
        api = self._api()
        with mock.patch("gui_app.transcribe_cpp.get_status",
                        return_value={"ready": True}), \
             mock.patch("gui_app.api_mixins.experimental.load_transcribe_config",
                        return_value={"enabled": True, "language": ""}):
            st = api.get_transcribe_status()
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
        self.assertIsNone(ul.call_args[0][0](cfg))  # 原地修改，返回 None
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

    def test_remove_tagger_deletes_install_and_module_dir(self):
        from gui_app.tagger import remove_tagger
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        inst = base / "tagger-tagger"
        inst.mkdir()
        mod = base / "tagger"
        mod.mkdir()
        with mock.patch("gui_app.tagger.TAGGER_DIR", inst), \
             mock.patch("gui_app.workspace_paths.TAGGER_CONFIG_FILE",
                        mod / "config.json"):
            r = remove_tagger()
        self.assertTrue(r["ok"])
        self.assertFalse(inst.exists())   # 安装目录已删
        self.assertFalse(mod.exists())    # 模块文件夹已删

    def test_remove_tagger_not_installed_still_cleans_module_dir(self):
        from gui_app.tagger import remove_tagger
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        mod = base / "tagger"
        mod.mkdir()
        with mock.patch("gui_app.tagger.TAGGER_DIR",
                        base / "tagger-tagger"), \
             mock.patch("gui_app.workspace_paths.TAGGER_CONFIG_FILE",
                        mod / "config.json"):
            r = remove_tagger()
        self.assertTrue(r["ok"])
        self.assertFalse(mod.exists())    # 残留模块文件夹仍被清掉

    def test_remove_tagger_failure_keeps_module_dir(self):
        from gui_app.tagger import remove_tagger
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        inst = base / "tagger-tagger"
        inst.mkdir()
        mod = base / "tagger"
        mod.mkdir()
        with mock.patch("gui_app.tagger.TAGGER_DIR", inst), \
             mock.patch("gui_app.tagger.shutil.rmtree",
                        side_effect=OSError("拒绝访问")), \
             mock.patch("gui_app.workspace_paths.TAGGER_CONFIG_FILE",
                        mod / "config.json"):
            r = remove_tagger()
        self.assertFalse(r["ok"])
        self.assertTrue(inst.exists())
        self.assertTrue(mod.exists())  # 删除失败 → 模块文件夹不动

    def test_remove_transcribe_deletes_install_and_module_dir(self):
        from gui_app.transcribe_cpp import remove_transcribe
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        inst = base / "transcribe"
        inst.mkdir()
        mod = base / "transcribe-ws"
        mod.mkdir()
        with mock.patch("gui_app.transcribe_cpp.TRANSCRIBE_DIR", inst), \
             mock.patch("gui_app.workspace_paths.TRANSCRIBE_CONFIG_FILE",
                        mod / "config.json"):
            r = remove_transcribe()
        self.assertTrue(r["ok"])
        self.assertFalse(inst.exists())
        self.assertFalse(mod.exists())

    def test_remove_transcribe_not_installed_still_cleans_module_dir(self):
        from gui_app.transcribe_cpp import remove_transcribe
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        mod = base / "transcribe-ws"
        mod.mkdir()
        with mock.patch("gui_app.transcribe_cpp.TRANSCRIBE_DIR",
                        base / "transcribe"), \
             mock.patch("gui_app.workspace_paths.TRANSCRIBE_CONFIG_FILE",
                        mod / "config.json"):
            r = remove_transcribe()
        self.assertTrue(r["ok"])
        self.assertFalse(mod.exists())

    def test_remove_transcribe_failure_keeps_module_dir(self):
        from gui_app.transcribe_cpp import remove_transcribe
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        inst = base / "transcribe"
        inst.mkdir()
        mod = base / "transcribe-ws"
        mod.mkdir()
        with mock.patch("gui_app.transcribe_cpp.TRANSCRIBE_DIR", inst), \
             mock.patch("gui_app.transcribe_cpp.shutil.rmtree",
                        side_effect=OSError("拒绝访问")), \
             mock.patch("gui_app.workspace_paths.TRANSCRIBE_CONFIG_FILE",
                        mod / "config.json"):
            r = remove_transcribe()
        self.assertFalse(r["ok"])
        self.assertTrue(inst.exists())
        self.assertTrue(mod.exists())  # 删除失败 → 模块文件夹不动

    def test_remove_ragvec_resets_module_config(self):
        """卸载 ragvec：安装目录删除 + rag_vec_enabled/device 复位（不留痕迹）。"""
        from gui_app import config_store
        from gui_app.ort_embedding import remove_ort_embedding
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        inst = base / "ort-embedding"
        inst.mkdir()
        cfg_file = base / "config.json"
        cfg_file.write_text(json.dumps(
            _cfg(rag_vec_enabled=True, rag_vec_device="gpu"),
            ensure_ascii=False), encoding="utf-8")
        with mock.patch("gui_app.ort_embedding.ORT_DIR", inst), \
             mock.patch.object(config_store, "CONFIG_FILE", cfg_file):
            r = remove_ort_embedding()
        self.assertTrue(r["ok"])
        self.assertFalse(inst.exists())
        disk = json.loads(cfg_file.read_text(encoding="utf-8"))
        self.assertFalse(disk["experimental"].get("rag_vec_enabled", False))
        self.assertEqual(disk["experimental"].get("rag_vec_device", ""), "")


# ═══════════════════════════════════════════════════════════════
# 模块配置存储（tagger / 语音转录 config.json）
# ═══════════════════════════════════════════════════════════════

class TestModuleConfigStore(_WorkspaceTestCase):
    """load/update_tagger_config、load/update_transcribe_config：roundtrip + 容错。"""

    def test_tagger_defaults_when_missing(self):
        from gui_app.config_store import load_tagger_config
        self.assertEqual(load_tagger_config(), _tagger_defaults())

    def test_tagger_roundtrip(self):
        from gui_app.config_store import load_tagger_config, update_tagger_config

        def _seed(c):
            c["enabled"] = True
            c["frames"] = 8
            return c

        def _bump(c):
            c["threshold"] = 0.75
            return c

        update_tagger_config(_seed)
        update_tagger_config(_bump)
        cfg = load_tagger_config()
        self.assertIs(cfg["enabled"], True)
        self.assertEqual(cfg["frames"], 8)
        self.assertEqual(cfg["threshold"], 0.75)
        self.assertNotIn("short_side", cfg)  # v1.0 起废弃键不再注入默认值

    def test_tagger_corrupt_tolerated(self):
        from gui_app.config_store import load_tagger_config, update_tagger_config
        self.paths["tagger"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["tagger"].write_text("{broken", encoding="utf-8")
        self.assertEqual(load_tagger_config(), _tagger_defaults())
        update_tagger_config(lambda c: c.update(enabled=True) or c)
        self.assertIs(load_tagger_config()["enabled"], True)  # 损坏后仍可写

    def test_tagger_non_dict_tolerated(self):
        from gui_app.config_store import load_tagger_config
        self.paths["tagger"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["tagger"].write_text(json.dumps([1, 2]), encoding="utf-8")
        self.assertEqual(load_tagger_config(), _tagger_defaults())

    def test_transcribe_defaults_when_missing(self):
        from gui_app.config_store import load_transcribe_config
        self.assertEqual(load_transcribe_config(), _transcribe_defaults())

    def test_transcribe_roundtrip(self):
        from gui_app.config_store import load_transcribe_config, update_transcribe_config

        def _seed(c):
            c["enabled"] = True
            c["language"] = "zh"
            return c

        def _bump(c):
            c["max_chars"] = 1200
            return c

        update_transcribe_config(_seed)
        update_transcribe_config(_bump)
        cfg = load_transcribe_config()
        self.assertIs(cfg["enabled"], True)
        self.assertEqual(cfg["language"], "zh")
        self.assertEqual(cfg["max_chars"], 1200)
        self.assertIs(cfg["inject_timestamps"], False)   # 未覆盖键保留默认

    def test_transcribe_corrupt_tolerated(self):
        from gui_app.config_store import load_transcribe_config
        self.paths["transcribe"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["transcribe"].write_text("{broken", encoding="utf-8")
        self.assertEqual(load_transcribe_config(), _transcribe_defaults())

    def test_transcribe_non_dict_tolerated(self):
        from gui_app.config_store import load_transcribe_config
        self.paths["transcribe"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["transcribe"].write_text(json.dumps("x"), encoding="utf-8")
        self.assertEqual(load_transcribe_config(), _transcribe_defaults())


if __name__ == "__main__":
    unittest.main()
