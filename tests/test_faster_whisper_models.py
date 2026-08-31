"""faster-whisper 模型管理单元测试：元数据、目录定位、完整性检测、status。

只测纯函数与目录逻辑（不触发安装/下载/转录子进程）：
- WHISPER_MODELS 元数据完整（三个模型、仓库 ID、推荐标记）
- _model_subdir 按 hf 布局解析（models/<author>/<repo>/），未知模型返回 None
- is_model_installed 以 model.bin 判定完整性
- get_current_model 配置持久化 + 异常回退默认
- get_status 扩展字段（models/installed/current_model）与 ready 判定
- _download_model 未知模型参数校验
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app import faster_whisper as fw


class TestModelMeta(unittest.TestCase):
    """模型元数据表与目录定位。"""

    def test_three_models_with_full_meta(self):
        self.assertEqual(set(fw.WHISPER_MODELS.keys()), {"v3-turbo", "large-v3", "ja-1.5B"})
        for key, meta in fw.WHISPER_MODELS.items():
            self.assertTrue(meta["title"])
            self.assertIn("/", meta["repo"], f"{key} 仓库 ID 应为 author/repo")
            self.assertTrue(meta["desc"])
            self.assertTrue(meta["size_label"])

    def test_default_model_is_recommended_v3_turbo(self):
        self.assertEqual(fw.DEFAULT_MODEL, "v3-turbo")
        self.assertTrue(fw.WHISPER_MODELS["v3-turbo"]["recommended"])
        # 恰好一个推荐模型（安装弹窗默认选中唯一）
        rec = [k for k, m in fw.WHISPER_MODELS.items() if m["recommended"]]
        self.assertEqual(rec, ["v3-turbo"])

    def test_ja_only_models(self):
        self.assertIn("ja-1.5B", fw.JA_ONLY_MODELS)

    def test_model_subdir_hf_layout(self):
        sub = fw._model_subdir("large-v3")
        self.assertEqual(sub, fw.MODEL_DIR / "Systran" / "faster-whisper-large-v3")
        self.assertIn("models", sub.parts)
        # 未知模型 -> None
        self.assertIsNone(fw._model_subdir("unknown-model"))


class TestModelInstallCheck(unittest.TestCase):
    """模型完整性检测（model.bin 判定）与 status 组装。"""

    def _make_model_dir(self, base: Path) -> Path:
        """构造一个含 model.bin 的模型目录并返回。"""
        sub = base / "deepdml" / "faster-whisper-large-v3-turbo-ct2"
        sub.mkdir(parents=True)
        (sub / "model.bin").write_bytes(b"x")
        (sub / "config.json").write_text("{}", encoding="utf-8")
        return sub

    def test_is_model_installed_true_with_bin(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(fw, "MODEL_DIR", Path(td)):
                self._make_model_dir(Path(td))
                self.assertTrue(fw.is_model_installed("v3-turbo"))

    def test_is_model_installed_false_when_missing_dir(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(fw, "MODEL_DIR", Path(td)):
                self.assertFalse(fw.is_model_installed("v3-turbo"))

    def test_is_model_installed_false_when_no_bin(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(fw, "MODEL_DIR", Path(td)):
                sub = Path(td) / "deepdml" / "faster-whisper-large-v3-turbo-ct2"
                sub.mkdir(parents=True)
                (sub / "config.json").write_text("{}", encoding="utf-8")
                self.assertFalse(fw.is_model_installed("v3-turbo"))

    def test_get_installed_models_scan(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(fw, "MODEL_DIR", Path(td)):
                self._make_model_dir(Path(td))
                inst = fw.get_installed_models()
                self.assertEqual(inst["v3-turbo"], True)
                self.assertEqual(inst["large-v3"], False)
                self.assertEqual(inst["ja-1.5B"], False)

    def test_get_status_fields_and_ready(self):
        with tempfile.TemporaryDirectory() as td:
            venv_py = Path(td) / "venv" / "Scripts" / "python.exe"
            with mock.patch.object(fw, "MODEL_DIR", Path(td)), \
                 mock.patch.object(fw, "VENV_DIR", Path(td) / "venv"), \
                 mock.patch.object(fw, "venv_python_path", return_value=venv_py), \
                 mock.patch("gui_app.config_store.load_whisper_config",
                            return_value={"model": "v3-turbo"}):
                # venv 缺失 + 模型缺失 -> 未就绪
                st = fw.get_status()
                self.assertFalse(st["ready"])
                self.assertEqual(st["current_model"], "v3-turbo")
                self.assertEqual(len(st["models"]), 3)
                self.assertIn("key", st["models"][0])
                self.assertFalse(st["installed"]["v3-turbo"])
                # 模型下载后但 venv 缺失 -> 仍未就绪（venv_ok 是必要条件）
                self._make_model_dir(Path(td))
                st = fw.get_status()
                self.assertTrue(st["model_exists"])
                self.assertFalse(st["ready"])
                # venv + 模型齐备 -> 就绪
                venv_py.parent.mkdir(parents=True)
                venv_py.write_bytes(b"x")
                st = fw.get_status()
                self.assertTrue(st["ready"])


class TestCurrentModel(unittest.TestCase):
    """当前模型读取：配置持久化 + 异常回退。"""

    def test_current_from_config(self):
        with mock.patch("gui_app.config_store.load_whisper_config",
                        return_value={"model": "ja-1.5B"}):
            self.assertEqual(fw.get_current_model(), "ja-1.5B")

    def test_current_fallback_default_when_unknown(self):
        with mock.patch("gui_app.config_store.load_whisper_config",
                        return_value={"model": "bad-key"}):
            self.assertEqual(fw.get_current_model(), fw.DEFAULT_MODEL)

    def test_current_fallback_default_when_missing(self):
        with mock.patch("gui_app.config_store.load_whisper_config",
                        return_value={}):
            self.assertEqual(fw.get_current_model(), fw.DEFAULT_MODEL)

    def test_current_fallback_on_exception(self):
        def _boom(*a, **k):
            raise RuntimeError("config 损坏")
        with mock.patch("gui_app.config_store.load_whisper_config", _boom):
            self.assertEqual(fw.get_current_model(), fw.DEFAULT_MODEL)


class TestDownloadValidation(unittest.TestCase):
    """下载函数参数校验（不真正下载）。"""

    def test_download_unknown_model_rejected(self):
        r = fw._download_model("not-a-model")
        self.assertFalse(r["ok"])
        self.assertIn("未知模型", r["error"])

    def test_download_conflict_returns_error(self):
        # begin_download 已被占坑 -> 直接返回冲突错误，不触发网络
        with mock.patch("gui_app.hf_downloader.begin_download", return_value=False) as bd:
            r = fw._download_model("v3-turbo")
            self.assertFalse(r["ok"])
            self.assertIn("进行中", r["error"])
            bd.assert_called_once_with()


class TestInstallSyncsModel(unittest.TestCase):
    """安装成功后把所选模型写为当前模型（避免 ready 指向未下载的默认模型）。"""

    def _run_install(self, model="large-v3"):
        """mock 掉全部子进程步骤，仅验证步骤 4 后的配置写入。"""
        import tempfile

        def _fake_install(model_key, log_fn=None, progress_cb=None,
                          cancel_event=None, cleanup_on_cancel=False):
            return {"ok": True, "downloaded": 1, "failed": [], "cancelled": False}

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            with mock.patch.object(fw, "WHISPER_DIR", base), \
                 mock.patch.object(fw, "VENV_DIR", base / "venv"), \
                 mock.patch.object(fw, "MODEL_DIR", base / "models"), \
                 mock.patch.object(fw, "_ensure_uv", return_value="uv"), \
                 mock.patch.object(fw, "_run_subprocess_streaming",
                                   return_value=(0, ["ok"])), \
                 mock.patch.object(fw, "venv_python_path",
                                   return_value=base / "venv" / "python"), \
                 mock.patch.object(fw, "detect_gpu",
                                   return_value={"has_nvidia": False}), \
                 mock.patch.object(fw, "_download_model", side_effect=_fake_install), \
                 mock.patch("gui_app.config_store.update_whisper_config") as uwc:
                r = fw.install_dependencies(pypi_mirror="nju", model=model)
        return r, uwc

    def test_install_writes_selected_model(self):
        r, uwc = self._run_install(model="large-v3")
        self.assertTrue(r["ok"])
        # update_whisper_config 的 mutator 应写入所选模型
        mutator = uwc.call_args.args[0]
        cfg = mutator({})
        self.assertEqual(cfg, {"model": "large-v3"})

    def test_install_default_model_idempotent(self):
        r, uwc = self._run_install(model="v3-turbo")
        self.assertTrue(r["ok"])
        mutator = uwc.call_args.args[0]
        cfg = mutator({"model": "v3-turbo", "enabled": True})
        self.assertEqual(cfg["model"], "v3-turbo")
        self.assertEqual(cfg["enabled"], True)  # 其他键不被破坏

    def test_install_download_failure_does_not_write_model(self):
        """下载失败时不写当前模型（安装中断，配置保持原样）。"""

        def _fail(model_key, log_fn=None, progress_cb=None,
                  cancel_event=None, cleanup_on_cancel=False):
            return {"ok": False, "error": "网络错误"}

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            with mock.patch.object(fw, "WHISPER_DIR", base), \
                 mock.patch.object(fw, "VENV_DIR", base / "venv"), \
                 mock.patch.object(fw, "_ensure_uv", return_value="uv"), \
                 mock.patch.object(fw, "_run_subprocess_streaming",
                                   return_value=(0, ["ok"])), \
                 mock.patch.object(fw, "venv_python_path",
                                   return_value=base / "venv" / "python"), \
                 mock.patch.object(fw, "detect_gpu",
                                   return_value={"has_nvidia": False}), \
                 mock.patch.object(fw, "_download_model", side_effect=_fail), \
                 mock.patch("gui_app.config_store.update_whisper_config") as uwc:
                r = fw.install_dependencies(pypi_mirror="nju", model="large-v3")
        self.assertFalse(r["ok"])
        uwc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
