"""transcribe.cpp 语音转写单元测试：模型元数据、目录定位、完整性检测、status、安装。

只测纯函数与目录逻辑（不触发安装/下载/转写子进程）：
- TRANSCRIBE_MODELS 元数据完整（三个模型、GGUF 文件名、仓库 ID、家族标记）
- 模型路径按 author/repo 布局解析，未知模型返回空
- is_model_installed 以 GGUF 文件判定完整性
- get_current_model / get_inject_model 配置持久化 + 未知 key 回退默认
- get_status 扩展字段（models/installed/current_model）与 ready 判定
- _download_model 未知模型参数校验
- install_dependencies 的模型配置写入
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
        # family 决定引擎输出模式（whisper/moss 出 SRT，sensevoice 出纯文本）
        self.assertNotIn(tc.TRANSCRIBE_MODELS["sensevoice"]["family"], ("whisper", "moss"))
        self.assertIn(tc.TRANSCRIBE_MODELS["whisper-turbo"]["family"], ("whisper", "moss"))

    def test_default_model_is_recommended_whisper_turbo(self):
        self.assertEqual(tc.DEFAULT_MODEL, "whisper-turbo")
        self.assertTrue(tc.TRANSCRIBE_MODELS["whisper-turbo"]["recommended"])
        # 恰好一个推荐模型
        rec = [k for k, m in tc.TRANSCRIBE_MODELS.items() if m["recommended"]]
        self.assertEqual(rec, ["whisper-turbo"])

    def test_models_have_hf_source_only(self):
        # 全部内置模型仅有 hf 源
        for meta in tc.TRANSCRIBE_MODELS.values():
            self.assertIn("hf_repo", meta)
            self.assertNotIn("ms_repo", meta)

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

    def test_is_model_installed_false_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(tc, "MODEL_DIR", Path(td)):
                self.assertFalse(tc.is_model_installed("moss"))

    def test_model_path_unknown_model(self):
        self.assertIsNone(tc._model_path("unknown-model"))

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
                            return_value={"model": "whisper-turbo",
                                          "inject_model": "sensevoice"}):
                # venv 缺失 + 模型缺失 -> 未就绪
                st = tc.get_status()
                self.assertFalse(st["ready"])
                self.assertEqual(st["current_model"], "whisper-turbo")
                # models 为全部候选（3）
                self.assertEqual({m["key"] for m in st["models"]},
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
    """安装成功后把所选模型写为当前模型与注入模型。"""

    def _run_install(self, model="moss"):
        """mock 掉全部子进程步骤，仅验证步骤 4 后的配置写入。"""
        def _fake_download(model_key, log_fn=None, progress_cb=None,
                           cancel_event=None, cleanup_on_cancel=False):
            return {"ok": True, "downloaded": 1, "failed": [], "cancelled": False}

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
                 mock.patch.object(tc, "_download_model", side_effect=_fake_download), \
                 mock.patch.object(tc, "_download_vad", return_value=True), \
                 mock.patch("gui_app.config_store.update_transcribe_config") as uwc:
                r = tc.install_dependencies(pypi_mirror="nju", model=model)
        return r, uwc

    def test_install_writes_selected_model(self):
        r, uwc = self._run_install(model="moss")
        self.assertTrue(r["ok"])
        # mutator 应同时写入当前模型与辅助重命名模型
        mutator = uwc.call_args.args[0]
        cfg = {}
        self.assertIsNone(mutator(cfg))
        self.assertEqual(cfg, {"model": "moss", "inject_model": "moss"})

    def test_install_keeps_other_keys(self):
        r, uwc = self._run_install(model="whisper-turbo")
        self.assertTrue(r["ok"])
        mutator = uwc.call_args.args[0]
        cfg = {"enabled": True}
        mutator(cfg)
        self.assertEqual(cfg["model"], "whisper-turbo")
        self.assertEqual(cfg["inject_model"], "whisper-turbo")
        self.assertEqual(cfg["enabled"], True)  # 其他键不被破坏

    def test_install_unknown_model_rejected(self):
        """未知模型 key 直接报错，不回退默认。"""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            with mock.patch.object(tc, "TRANSCRIBE_DIR", base):
                r = tc.install_dependencies(model="bad-key")
        self.assertFalse(r["ok"])
        self.assertIn("未知模型", r["error"])

    def test_install_downloads_selected_model_once(self):
        calls = []
        def _record(model_key, log_fn=None, progress_cb=None,
                    cancel_event=None, cleanup_on_cancel=False):
            calls.append(model_key)
            return {"ok": True, "downloaded": 1, "failed": [], "cancelled": False}

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
                 mock.patch.object(tc, "_download_model", side_effect=_record), \
                 mock.patch.object(tc, "_download_vad", return_value=True), \
                 mock.patch("gui_app.config_store.update_transcribe_config"):
                r = tc.install_dependencies(pypi_mirror="nju", model="whisper-turbo")
        self.assertTrue(r["ok"])
        self.assertEqual(calls, ["whisper-turbo"])

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
                 mock.patch.object(tc, "_download_model", side_effect=_fail), \
                 mock.patch.object(tc, "_download_vad", return_value=True), \
                 mock.patch("gui_app.config_store.update_transcribe_config") as uwc:
                r = tc.install_dependencies(pypi_mirror="nju", model="moss")
        self.assertFalse(r["ok"])
        uwc.assert_not_called()

    def test_install_skips_download_when_model_exists(self):
        """所选模型已在盘上时步骤 4 跳过下载。"""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            meta = tc.TRANSCRIBE_MODELS["whisper-turbo"]
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
                 mock.patch.object(tc, "_download_vad", return_value=True), \
                 mock.patch.object(tc, "_download_model") as dl, \
                 mock.patch("gui_app.config_store.update_transcribe_config"):
                r = tc.install_dependencies(pypi_mirror="nju", model="whisper-turbo")
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
                 mock.patch.object(tc, "FFMPEG_EXE", ""), \
                 mock.patch.object(tc, "MODEL_DIR", base / "models"):
                r = tc.run_transcription_batch(["C:/v.mp4"])
        self.assertFalse(r["ok"])
        self.assertIn("ffmpeg", r["error"])


    def test_run_batch_allows_sensevoice(self):
        """SenseVoice 可作为右键转写模型（无 family 拦截，命中 ffmpeg 前置校验）。"""
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
                            return_value={"model": "sensevoice"}),                  mock.patch.object(tc, "MODEL_DIR", base / "models"),                  mock.patch.object(tc, "FFMPEG_EXE", ""):
                r = tc.run_transcription_batch(["C:/v.mp4"])
        self.assertFalse(r["ok"])
        self.assertIn("ffmpeg", r["error"])
        self.assertNotIn("不支持", r["error"])

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
        self.assertEqual(_transcribe_defaults()["max_chars"], 2000)


class TestWorkDirCleanup(unittest.TestCase):
    """批量脚本工作目录由父进程持有：运行中存在、结束后清理（含子进程被杀残留）。"""

    def _run(self, fake):
        with mock.patch.object(tc, "_run_venv_script", fake):
            return tc._run_batch_script({"video_paths": ["a.mp4"]}, [], None,
                                        None, None, "转写")

    def test_workdir_exists_during_and_removed_after(self):
        import os
        box = {}

        def fake(venv_dir, script, params, **kw):
            wd = params["work_dir"]
            box["wd"] = wd
            os.makedirs(os.path.join(wd, "v1"), exist_ok=True)
            with open(os.path.join(wd, "v1", "a.wav"), "wb") as f:
                f.write(b"x")
            box["exists_during"] = os.path.isdir(wd)
            return 0, "", ""

        error, _ = self._run(fake)
        self.assertTrue(box["exists_during"])  # 运行中存在
        self.assertEqual(error, "转写无输出")
        self.assertFalse(os.path.isdir(box["wd"]))  # 结束后清理（含残留 wav）

    def test_workdir_removed_on_script_error(self):
        import os
        box = {}

        def fake(venv_dir, script, params, **kw):
            box["wd"] = params["work_dir"]
            return 1, "", "boom"

        error, _ = self._run(fake)
        self.assertIn("失败", error)
        self.assertFalse(os.path.isdir(box["wd"]))


class TestTranscriptionParams(unittest.TestCase):
    """字幕转写参数传导：vad_model 与注入模式一致。"""

    def test_params_carry_vad_model(self):
        captured = {}

        def fake_batch(params, per_video, on_video_done, on_log, stop_event, err_label):
            captured["params"] = params
            return None, []

        with tempfile.TemporaryDirectory() as td:
            vpy = Path(td) / "Scripts" / "python.exe"
            vpy.parent.mkdir(parents=True)
            vpy.write_bytes(b"x")
            with mock.patch.object(tc, "get_current_model", lambda: "sensevoice"), \
                 mock.patch.object(tc, "VENV_DIR", Path(td)), \
                 mock.patch.object(tc, "_model_path", lambda k: Path("m.gguf")), \
                 mock.patch.object(tc, "FFMPEG_EXE", "ffmpeg"), \
                 mock.patch.object(tc, "_run_batch_script", fake_batch):
                r = tc.run_transcription_batch(["a.mp4"])
        self.assertTrue(r["ok"])
        self.assertIn("vad_model", captured["params"])
        self.assertEqual(captured["params"]["mode"], "srt")


if __name__ == "__main__":
    unittest.main()
