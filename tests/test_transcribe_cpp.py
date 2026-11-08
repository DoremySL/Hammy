"""transcribe.cpp 语音转录单元测试：模型元数据、目录定位、完整性检测、status、安装。

只测纯函数与目录逻辑（不触发安装/下载/转录子进程）：
- TRANSCRIBE_MODELS 元数据完整（两个模型、GGUF 文件名、仓库 ID、家族标记）
- 模型路径按 author/repo 布局解析，未知模型返回空
- is_model_installed 以 GGUF 文件判定完整性
- get_current_model 配置持久化 + 异常回退默认
- get_status 扩展字段（models/installed/current_model）与 ready 判定
- _download_model 未知模型参数校验
- install_dependencies 的 NVIDIA/CPU 分支与配置写入
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_app import transcribe_cpp as tc


class TestModelMeta(unittest.TestCase):
    """模型元数据表。"""

    def test_three_models_with_full_meta(self):
        self.assertEqual(set(tc.TRANSCRIBE_MODELS.keys()),
                         {"whisper-turbo", "moss", "sensevoice"})
        for key, meta in tc.TRANSCRIBE_MODELS.items():
            self.assertTrue(meta["title"])
            self.assertIn(meta["family"], ("whisper", "moss", "sensevoice"))
            self.assertTrue(meta["gguf"].endswith(".gguf"))
            self.assertIn("/", meta["hf_repo"], f"{key} 仓库 ID 应为 author/repo")
            self.assertTrue(meta["desc"])
            self.assertTrue(meta["size_label"])
        # 字幕模型族 vs 注入候选：sensevoice 不作字幕模型（family 守卫）
        self.assertNotIn(tc.TRANSCRIBE_MODELS["sensevoice"]["family"], ("whisper", "moss"))
        self.assertIn(tc.TRANSCRIBE_MODELS["whisper-turbo"]["family"], ("whisper", "moss"))

    def test_install_bundles(self):
        self.assertEqual(tc.DEFAULT_BUNDLE, "whisper+sv")
        self.assertEqual(tc.INSTALL_BUNDLES["whisper+sv"], ["whisper-turbo", "sensevoice"])
        self.assertEqual(tc.INSTALL_BUNDLES["moss+sv"], ["moss", "sensevoice"])

    def test_default_model_is_recommended_whisper_turbo(self):
        self.assertEqual(tc.DEFAULT_MODEL, "whisper-turbo")
        self.assertTrue(tc.TRANSCRIBE_MODELS["whisper-turbo"]["recommended"])
        # 恰好一个推荐模型
        rec = [k for k, m in tc.TRANSCRIBE_MODELS.items() if m["recommended"]]
        self.assertEqual(rec, ["whisper-turbo"])

    def test_msfallback_empty_means_hf(self):
        # ms_repo 留空 = 走 hf-mirror
        for meta in tc.TRANSCRIBE_MODELS.values():
            self.assertIn("ms_repo", meta)
            self.assertIn("hf_repo", meta)

    def test_gguf_names_match_upstream(self):
        self.assertEqual(tc.TRANSCRIBE_MODELS["whisper-turbo"]["gguf"],
                         "whisper-large-v3-turbo-Q8_0.gguf")
        self.assertEqual(tc.TRANSCRIBE_MODELS["moss"]["gguf"],
                         "MOSS-Transcribe-Diarize-Q8_0.gguf")
        self.assertEqual(tc.TRANSCRIBE_MODELS["sensevoice"]["gguf"],
                         "SenseVoiceSmall-Q8_0.gguf")


class TestModelInstallCheck(unittest.TestCase):
    """模型完整性检测与 status 组装。"""

    def _make_model_file(self, base: Path, repo: str) -> Path:
        author, _, name = repo.partition("/")
        sub = base / author / name
        sub.mkdir(parents=True)
        gguf = sub / tc.TRANSCRIBE_MODELS["whisper-turbo"]["gguf"]
        gguf.write_bytes(b"x")
        return gguf

    def test_is_model_installed_true_with_gguf(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(tc, "MODEL_DIR", Path(td)):
                self._make_model_file(Path(td), tc.TRANSCRIBE_MODELS["whisper-turbo"]["hf_repo"])
                self.assertTrue(tc.is_model_installed("whisper-turbo"))

    def test_is_model_installed_accepts_ms_layout(self):
        # ms_repo 填入后，魔搭布局下的文件同样算已安装
        meta = dict(tc.TRANSCRIBE_MODELS["moss"])
        meta["ms_repo"] = "DoremySL/hammy-asr-gguf"
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(tc, "MODEL_DIR", Path(td)), \
                 mock.patch.object(tc, "TRANSCRIBE_MODELS", {"moss": meta}):
                author, _, name = meta["ms_repo"].partition("/")
                sub = Path(td) / author / name
                sub.mkdir(parents=True)
                (sub / meta["gguf"]).write_bytes(b"x")
                self.assertTrue(tc.is_model_installed("moss"))

    def test_is_model_installed_false_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(tc, "MODEL_DIR", Path(td)):
                self.assertFalse(tc.is_model_installed("moss"))

    def test_model_candidates_unknown_model(self):
        self.assertEqual(tc._model_candidates("unknown-model"), [])

    def test_get_installed_models_scan(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(tc, "MODEL_DIR", Path(td)):
                self._make_model_file(Path(td), tc.TRANSCRIBE_MODELS["whisper-turbo"]["hf_repo"])
                inst = tc.get_installed_models()
                self.assertEqual(inst["whisper-turbo"], True)
                self.assertEqual(inst["moss"], False)

    def test_get_status_fields_and_ready(self):
        with tempfile.TemporaryDirectory() as td:
            venv_py = Path(td) / "venv" / "Scripts" / "python.exe"
            with mock.patch.object(tc, "MODEL_DIR", Path(td)), \
                 mock.patch.object(tc, "VENV_DIR", Path(td) / "venv"), \
                 mock.patch.object(tc, "venv_python_path", return_value=venv_py), \
                 mock.patch("gui_app.config_store.load_transcribe_config",
                            return_value={"model": "whisper-turbo"}):
                # venv 缺失 + 模型缺失 -> 未就绪
                st = tc.get_status()
                self.assertFalse(st["ready"])
                self.assertEqual(st["current_model"], "whisper-turbo")
                # models=字幕模型（2），inject_models=全部候选（3）
                self.assertEqual({m["key"] for m in st["models"]},
                                 {"whisper-turbo", "moss"})
                self.assertEqual({m["key"] for m in st["inject_models"]},
                                 {"whisper-turbo", "moss", "sensevoice"})
                self.assertEqual(st["inject_model"], "sensevoice")
                self.assertEqual(set(st["installed"].keys()),
                                 {"whisper-turbo", "moss", "sensevoice"})
                self.assertIn("key", st["models"][0])
                self.assertFalse(st["installed"]["whisper-turbo"])
                # 模型下载后但 venv 缺失 -> 仍未就绪（venv_ok 是必要条件）
                self._make_model_file(Path(td), tc.TRANSCRIBE_MODELS["whisper-turbo"]["hf_repo"])
                st = tc.get_status()
                self.assertTrue(st["model_exists"])
                self.assertFalse(st["ready"])
                # venv + 模型齐备 -> 就绪
                venv_py.parent.mkdir(parents=True)
                venv_py.write_bytes(b"x")
                st = tc.get_status()
                self.assertTrue(st["ready"])


class TestCurrentModel(unittest.TestCase):
    """当前模型读取：配置持久化 + 异常回退。"""

    def test_current_from_config(self):
        with mock.patch("gui_app.config_store.load_transcribe_config",
                        return_value={"model": "moss"}):
            self.assertEqual(tc.get_current_model(), "moss")

    def test_current_fallback_default_when_unknown(self):
        with mock.patch("gui_app.config_store.load_transcribe_config",
                        return_value={"model": "bad-key"}):
            self.assertEqual(tc.get_current_model(), tc.DEFAULT_MODEL)

    def test_current_fallback_default_when_missing(self):
        with mock.patch("gui_app.config_store.load_transcribe_config",
                        return_value={}):
            self.assertEqual(tc.get_current_model(), tc.DEFAULT_MODEL)

    def test_current_fallback_on_exception(self):
        def _boom(*a, **k):
            raise RuntimeError("config 损坏")
        with mock.patch("gui_app.config_store.load_transcribe_config", _boom):
            self.assertEqual(tc.get_current_model(), tc.DEFAULT_MODEL)


class TestDownloadValidation(unittest.TestCase):
    """下载函数参数校验。"""

    def test_download_unknown_model_rejected(self):
        r = tc._download_model("not-a-model")
        self.assertFalse(r["ok"])
        self.assertIn("未知模型", r["error"])

    def test_download_conflict_returns_error(self):
        # begin_download 已被占坑 -> 直接返回冲突错误，不触发网络
        with mock.patch("gui_app.models_downloader.begin_download", return_value=False) as bd:
            r = tc._download_model("whisper-turbo")
            self.assertFalse(r["ok"])
            self.assertIn("进行中", r["error"])
            bd.assert_called_once_with()


class TestInstallSyncsModel(unittest.TestCase):
    """安装成功后把所选模型写为当前模型。"""

    def _run_install(self, bundle="moss+sv", has_nvidia=False):
        """mock 掉全部子进程步骤，仅验证步骤 4 后的配置写入。"""
        def _fake_download(model_key, log_fn=None, progress_cb=None,
                           cancel_event=None, cleanup_on_cancel=False):
            return {"ok": True, "downloaded": 1, "failed": [], "cancelled": False}

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            patches = [
                mock.patch.object(tc, "TRANSCRIBE_DIR", base),
                mock.patch.object(tc, "VENV_DIR", base / "venv"),
                mock.patch.object(tc, "MODEL_DIR", base / "models"),
                mock.patch.object(tc, "_ensure_uv", return_value="uv"),
                mock.patch.object(tc, "_run_subprocess_streaming",
                                  return_value=(0, ["ok"])),
                mock.patch.object(tc, "venv_python_path",
                                  return_value=base / "venv" / "python"),
                mock.patch.object(tc, "detect_gpu",
                                  return_value={"has_nvidia": has_nvidia}),
                mock.patch.object(tc, "_download_model", side_effect=_fake_download),
            ]
            if has_nvidia:
                wheel = base / "binding.whl"
                wheel.write_bytes(b"x")
                patches.append(mock.patch.object(tc, "_download_engine_wheels",
                                                 return_value=[wheel]))
            with mock.patch.object(tc, "WHEELS_DIR", base / "wheels"):
                for p in patches:
                    p.start()
                uwc = mock.patch("gui_app.config_store.update_transcribe_config").start()
                try:
                    r = tc.install_dependencies(pypi_mirror="nju", bundle=bundle)
                finally:
                    mock.patch.stopall()
        return r, uwc

    def test_install_cpu_writes_selected_model(self):
        r, uwc = self._run_install(bundle="moss+sv", has_nvidia=False)
        self.assertTrue(r["ok"])
        # mutator 应写入组合主模型
        mutator = uwc.call_args.args[0]
        cfg = {}
        self.assertIsNone(mutator(cfg))
        self.assertEqual(cfg, {"model": "moss"})

    def test_install_nvidia_uses_engine_wheels(self):
        r, uwc = self._run_install(bundle="whisper+sv", has_nvidia=True)
        self.assertTrue(r["ok"])
        mutator = uwc.call_args.args[0]
        cfg = {"model": "whisper-turbo", "enabled": True}
        mutator(cfg)
        self.assertEqual(cfg["model"], "whisper-turbo")
        self.assertEqual(cfg["enabled"], True)  # 其他键不被破坏

    def test_install_download_failure_does_not_write_model(self):
        """下载失败时不写当前模型。"""
        def _fail(model_key, log_fn=None, progress_cb=None,
                  cancel_event=None, cleanup_on_cancel=False):
            return {"ok": False, "error": "网络错误"}

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            with mock.patch.object(tc, "TRANSCRIBE_DIR", base), \
                 mock.patch.object(tc, "VENV_DIR", base / "venv"), \
                 mock.patch.object(tc, "MODEL_DIR", base / "models"), \
                 mock.patch.object(tc, "_ensure_uv", return_value="uv"), \
                 mock.patch.object(tc, "_run_subprocess_streaming",
                                   return_value=(0, ["ok"])), \
                 mock.patch.object(tc, "venv_python_path",
                                   return_value=base / "venv" / "python"), \
                 mock.patch.object(tc, "detect_gpu",
                                   return_value={"has_nvidia": False}), \
                 mock.patch.object(tc, "_download_model", side_effect=_fail), \
                 mock.patch("gui_app.config_store.update_transcribe_config") as uwc:
                r = tc.install_dependencies(pypi_mirror="nju", bundle="moss+sv")
        self.assertFalse(r["ok"])
        uwc.assert_not_called()

    def test_install_skips_download_when_model_exists(self):
        """组合内模型都在盘上时步骤 4 跳过下载。"""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for mk in tc.INSTALL_BUNDLES["whisper+sv"]:
                meta = tc.TRANSCRIBE_MODELS[mk]
                author, _, name = meta["hf_repo"].partition("/")
                sub = base / "models" / author / name
                sub.mkdir(parents=True)
                (sub / meta["gguf"]).write_bytes(b"x")
            with mock.patch.object(tc, "TRANSCRIBE_DIR", base), \
                 mock.patch.object(tc, "VENV_DIR", base / "venv"), \
                 mock.patch.object(tc, "MODEL_DIR", base / "models"), \
                 mock.patch.object(tc, "_ensure_uv", return_value="uv"), \
                 mock.patch.object(tc, "_run_subprocess_streaming",
                                   return_value=(0, ["ok"])), \
                 mock.patch.object(tc, "venv_python_path",
                                   return_value=base / "venv" / "python"), \
                 mock.patch.object(tc, "detect_gpu",
                                   return_value={"has_nvidia": False}), \
                 mock.patch.object(tc, "_download_model") as dl, \
                 mock.patch("gui_app.config_store.update_transcribe_config"):
                r = tc.install_dependencies(pypi_mirror="nju", bundle="whisper+sv")
        self.assertTrue(r["ok"])
        dl.assert_not_called()

    def test_run_batch_requires_ffmpeg(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            # 伪造已安装模型 + venv，使 ready=True，命中 ffmpeg 前置校验
            meta = tc.TRANSCRIBE_MODELS["moss"]
            author, _, name = meta["hf_repo"].partition("/")
            sub = base / "models" / author / name
            sub.mkdir(parents=True)
            (sub / meta["gguf"]).write_bytes(b"x")
            vpy = base / "venv" / "Scripts" / "python.exe"
            vpy.parent.mkdir(parents=True)
            vpy.write_bytes(b"x")
            with mock.patch.object(tc, "TRANSCRIBE_DIR", base), \
                 mock.patch.object(tc, "VENV_DIR", base / "venv"), \
                 mock.patch.object(tc, "venv_python_path", return_value=vpy), \
                 mock.patch("gui_app.config_store.load_transcribe_config",
                            return_value={"model": "moss"}), \
                 mock.patch.object(tc, "detect_gpu",
                                   return_value={"has_nvidia": False}), \
                 mock.patch.object(tc, "FFMPEG_EXE", ""), \
                 mock.patch.object(tc, "MODEL_DIR", base / "models"):
                r = tc.run_transcription_batch(["C:/v.mp4"])
        self.assertFalse(r["ok"])
        self.assertIn("ffmpeg", r["error"])


    def test_run_batch_rejects_sensevoice(self):
        """SenseVoice 只作注入，不允许作为右键字幕模型。"""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            meta = tc.TRANSCRIBE_MODELS["sensevoice"]
            author, _, name = meta["hf_repo"].partition("/")
            sub = base / "models" / author / name
            sub.mkdir(parents=True)
            (sub / meta["gguf"]).write_bytes(b"x")
            vpy = base / "venv" / "Scripts" / "python.exe"
            vpy.parent.mkdir(parents=True)
            vpy.write_bytes(b"x")
            with mock.patch.object(tc, "TRANSCRIBE_DIR", base),                  mock.patch.object(tc, "VENV_DIR", base / "venv"),                  mock.patch.object(tc, "venv_python_path", return_value=vpy),                  mock.patch("gui_app.config_store.load_transcribe_config",
                            return_value={"model": "sensevoice"}),                  mock.patch.object(tc, "MODEL_DIR", base / "models"):
                r = tc.run_transcription_batch(["C:/v.mp4"])
        self.assertFalse(r["ok"])
        self.assertIn("不支持字幕转录", r["error"])

    def test_inject_ready(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            vpy = base / "venv" / "Scripts" / "python.exe"
            with mock.patch.object(tc, "VENV_DIR", base / "venv"),                  mock.patch.object(tc, "venv_python_path", return_value=vpy),                  mock.patch.object(tc, "MODEL_DIR", base / "models"),                  mock.patch("gui_app.config_store.load_transcribe_config",
                            return_value={"inject_model": "sensevoice"}):
                # venv 缺失 -> 未就绪
                self.assertFalse(tc.inject_ready())
                # venv 就绪但模型缺失 -> 未就绪
                vpy.parent.mkdir(parents=True)
                vpy.write_bytes(b"x")
                self.assertFalse(tc.inject_ready())
                # 注入模型就绪
                meta = tc.TRANSCRIBE_MODELS["sensevoice"]
                author, _, name = meta["hf_repo"].partition("/")
                sub = base / "models" / author / name
                sub.mkdir(parents=True)
                (sub / meta["gguf"]).write_bytes(b"x")
                self.assertTrue(tc.inject_ready())

    def test_inject_ready_with_configured_model(self):
        """注入模型可配置为 whisper-turbo / MOSS。"""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            vpy = base / "venv" / "Scripts" / "python.exe"
            vpy.parent.mkdir(parents=True)
            vpy.write_bytes(b"x")
            with mock.patch.object(tc, "VENV_DIR", base / "venv"),                  mock.patch.object(tc, "venv_python_path", return_value=vpy),                  mock.patch.object(tc, "MODEL_DIR", base / "models"),                  mock.patch("gui_app.config_store.load_transcribe_config",
                            return_value={"inject_model": "whisper-turbo"}):
                # 注入模型未下载 -> 未就绪
                self.assertFalse(tc.inject_ready())
                meta = tc.TRANSCRIBE_MODELS["whisper-turbo"]
                author, _, name = meta["hf_repo"].partition("/")
                sub = base / "models" / author / name
                sub.mkdir(parents=True)
                (sub / meta["gguf"]).write_bytes(b"x")
                self.assertTrue(tc.inject_ready())
                # 未知 inject_model 回退默认 sensevoice
                with mock.patch("gui_app.config_store.load_transcribe_config",
                                return_value={"inject_model": "bad-key"}):
                    self.assertEqual(tc.get_inject_model(), "sensevoice")

    def test_transcribe_defaults_integration_on(self):
        from gui_app.config_store import _transcribe_defaults
        self.assertIs(_transcribe_defaults()["integration"], True)


if __name__ == "__main__":
    unittest.main()
