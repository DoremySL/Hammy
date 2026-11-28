"""在后台线程中运行 BatchPipeline：装配 Config、转发引擎日志、逐文件写 history 并推送前端。"""
from __future__ import annotations

import base64
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import discovery, prompts
from .constants import PROBE_POOL_WORKERS
from .rename_aux import format_transcribe_block
from .workspace_paths import NFO_DIR, stable_id
from .workspace_store import (
    append_history_entry,
    begin_batch,
    flush_batch,
    get_history_by_id,
    read_json,
    update_adhoc_path,
)
from .workspace_service import find_cached_nfo
from batch_rename.config import Config
from batch_rename.utils import safe_get
from batch_rename.dependencies import DependencyError, ensure_dependencies, bindings
from batch_rename.env import SUBPROCESS_KWARGS, logger as engine_logger
from batch_rename.pipeline import BatchPipeline, ready_gate
from batch_rename.term_recall import TermRecall


def _asr_missing_list(paths: List[str], srt_dir: Path,
                      stop_event: Optional[threading.Event] = None) -> List[str]:
    """缺 SRT 的视频并行探针（与探针池同限流），无音轨不进转写；停止时提前退出。"""
    no_srt = [vp for vp in paths
              if not (srt_dir / f"{stable_id(vp)}.srt").exists()]
    probes: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(PROBE_POOL_WORKERS) as ex:
        futures = {ex.submit(discovery.probe_video, vp): vp for vp in no_srt}
        for fut in as_completed(futures):
            if stop_event is not None and stop_event.is_set():
                for f in futures:
                    f.cancel()
                return []
            probes[futures[fut]] = fut.result()
    # as_completed 完成序不定，按输入顺序输出
    return [vp for vp in no_srt if probes.get(vp, {}).get("has_audio") is not False]


_THUMB_TIME_RE = re.compile(r"\d{1,3}:\d{2}:\d{2}")


def _thumb_time_seconds(thumb_time: str) -> Optional[int]:
    """history 的 thumb_time（HH:MM:SS）→ int 秒（与提示词时间标签同为四舍五入口径）。"""
    raw = (thumb_time or "").strip()
    if not isinstance(raw, str) or not _THUMB_TIME_RE.fullmatch(raw):
        return None
    h, m, s = (int(x) for x in raw.split(":"))
    return h * 3600 + m * 60 + s


def _llama_vram_warning(on_log) -> None:
    """集成辅助与本地推理共存前的显存提示（警告引导，不自动改变配置）。"""
    try:
        import subprocess
        from .llama_integration import is_active
        from .llama_cpp import get_status
        if not (is_active() and get_status().get("running")):
            return
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, **SUBPROCESS_KWARGS)
        frees = [int(x) for x in (r.stdout or "").split() if x.strip().isdigit()]
        if frees and max(frees) < 2048:
            on_log("warn", "[显存提示] 空闲显存不足 2GB，集成辅助与本地推理共存"
                           "可能显存不足；如遇失败建议关闭集成，改用右键手动分析/转写。")
    except Exception:
        pass  # 检测失败不阻断流程


# ── 日志转发 handler ──


class _UIHandler(logging.Handler):
    """把 batch_rename 日志行转发到 GUI 回调。"""

    # GUI 日志为精简视图：这些前缀不推界面（日志流中按级别可查）
    _FILTER_PREFIXES = ("[改名]", "[NFO] 已生成",
                        "[标签辅助] 集成完成", "[语音辅助] 已生成")

    def __init__(self, callback: Callable[[str, str], None], verbose: bool = False):
        super().__init__()
        self._cb = callback
        self._verbose = verbose  # debug 模式：跳过精简过滤，全量推送
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        try:
            msg = self.format(record)
            if not msg.strip():
                return
            # 非 debug：GUI 日志只保留精简视图，过滤引擎明细与横线分隔行
            if not self._verbose:
                if msg.startswith(self._FILTER_PREFIXES):
                    return
                stripped = msg.strip()
                if stripped and set(stripped) <= set("━─═"):
                    return
            lv = (
                "err" if record.levelno >= logging.ERROR
                else "warn" if record.levelno >= logging.WARNING
                else "debug" if record.levelno <= logging.DEBUG or msg.startswith("[debug]")
                else ""
            )
            self._cb(lv, msg.strip())
        except Exception:
            pass


# ── 从 config.json + 预设构建 batch_rename.Config ──


def build_engine_config(cfg_dict: Dict[str, Any],
                        llama_override: Optional[Dict[str, Any]] = None) -> Config:
    """从 GUI 的 config.json 构建 batch_rename.Config 对象。

    llama_override：本地推理集成的显式覆盖值（与启动计划同源）；缺省回退 ai_override()。"""
    cfg = Config()

    def _section(name: str) -> Dict[str, Any]:
        v = cfg_dict.get(name)
        return v if isinstance(v, dict) else {}

    ai = _section("ai")
    cfg.model = ai.get("model", cfg.model)
    cfg.base_url = ai.get("base_url", cfg.base_url)
    cfg.api_key = ai.get("api_key", cfg.api_key)

    # 本地推理集成：激活时覆盖连接参数，远程 API 配置保持不动
    from .llama_integration import ai_override
    ov = llama_override if llama_override is not None else ai_override()
    if ov:
        cfg.model = ov["model"]
        cfg.base_url = ov["base_url"]
        cfg.api_key = ov["api_key"]
        cfg.llama_slots = max(1, safe_get(ov, "ai_workers", 1))

    cfg.max_tokens = safe_get(ai, "max_tokens", cfg.max_tokens)
    cfg.temperature = safe_get(ai, "temperature", cfg.temperature, cast=float)
    cfg.top_p = safe_get(ai, "top_p", cfg.top_p, cast=float)
    cfg.retry_times = safe_get(ai, "retry_times", cfg.retry_times)
    cfg.ai_timeout = safe_get(ai, "ai_timeout", cfg.ai_timeout)
    cfg.ai_workers = safe_get(ai, "ai_workers", cfg.ai_workers)
    if ov:
        cfg.ai_workers = max(1, safe_get(ov, "ai_workers", cfg.ai_workers))
    cfg.enforce_json_mode = bool(ai.get("enforce_json_mode", cfg.enforce_json_mode))
    cfg.rag_top_k = safe_get(ai, "rag_top_k", cfg.rag_top_k)
    exp = cfg_dict.get("experimental") if isinstance(cfg_dict.get("experimental"), dict) else {}
    cfg.rag_vec_enabled = bool(exp.get("rag_vec_enabled", False))
    cfg.rag_vec_device = str(exp.get("rag_vec_device", "") or "")
    cfg.rag_vec_threshold = safe_get(exp, "rag_vec_threshold", cfg.rag_vec_threshold, cast=float)
    cfg.rag_vec_top_n = safe_get(exp, "rag_vec_top_n", cfg.rag_vec_top_n)

    video = _section("video")
    cfg.sampling_points = safe_get(video, "sampling_points", cfg.sampling_points)
    cfg.frames_per_point = safe_get(video, "frames_per_point", cfg.frames_per_point)
    cfg.frame_max_side = safe_get(video, "frame_max_side", cfg.frame_max_side)
    cfg.frame_time_tags = safe_get(video, "frame_time_tags", cfg.frame_time_tags)
    cfg.frame_extract_mode = str(video.get("frame_extract_mode", cfg.frame_extract_mode) or "auto")

    naming = _section("naming")
    cfg.naming_mode = str(naming.get("naming_mode", cfg.naming_mode) or "date")

    active = prompts.get_active(with_thumb_time=(cfg.frame_time_tags == 2))
    cfg.prompt = active["prompt"]
    cfg.system_prompt = active["system_prompt"]

    # debug 模式（config.json 顶层手动开关）
    cfg.debug_log = bool(cfg_dict.get("debug_log", False))

    # NFO 输出：默认写缓存目录，开启自动导出后直接写视频目录
    if not cfg_dict.get("nfo_auto_export", False):
        cfg.nfo_target_dir = str(NFO_DIR)
    else:
        cfg.nfo_target_dir = None

    cfg.validate()
    return cfg


_SRT_TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),\d{3} --> ")


def srt_to_text(srt_content: str, inject_timestamps: bool = False) -> str:
    """SRT 字幕清洗为纯文本（供注入 AI 提示词）。"""
    raw = srt_content.splitlines()
    lines: List[str] = []
    for i, line in enumerate(raw):
        s = line.strip()
        if not s:
            continue
        if "-->" in s:
            if inject_timestamps:
                m = _SRT_TS_RE.match(s)
                if m:
                    h, mn, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    total_sec = h * 3600 + mn * 60 + sec
                    mm, ss = divmod(total_sec, 60)
                    lines.append(f"[{mm:02d}:{ss:02d}]")
            continue
        # 序号行：纯数字且下一行紧接时间轴（对话文本中的纯数字行保留）
        if s.isdigit() and i + 1 < len(raw) and "-->" in raw[i + 1].strip():
            continue
        lines.append(s)
    return " ".join(lines)


# ── PipelineRunner ──


class PipelineRunner:
    """在后台线程中运行 BatchPipeline，推送实时日志/进度/每文件状态。"""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop: threading.Event = threading.Event()
        self._start_lock: threading.Lock = threading.Lock()
        self._handler: Optional[_UIHandler] = None
        self._client = None
        self._bindings = None  # batch_rename.dependencies.AIClientBindings
        # history 落盘完成信号，供 app.py 退出清理轮询
        self._history_flushed: threading.Event = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── 扩展功能上下文注入 ──

    def _build_tagger_meta_map(self, paths: List[str],
                               tcfg: Dict[str, Any]) -> Dict[str, tuple]:
        """构建 tagger 注入上下文：vp → (A 组优先条目, C 段查询词)。"""
        if not tcfg.get("enabled", False):
            return {}
        from .workspace_paths import TAGGER_TAGS_FILE
        tags_store = read_json(TAGGER_TAGS_FILE, {})
        if not tags_store:
            return {}
        from .tagger import load_term_library_map, tagger_injection
        lib = load_term_library_map()
        gen_n = max(0, safe_get(tcfg, "general_count", 15))

        result: Dict[str, tuple] = {}
        for vp in paths:
            entry = tags_store.get(stable_id(vp))
            if not entry:
                continue
            hits, q = tagger_injection(entry, lib, gen_n)
            if hits or q:
                result[vp] = (hits, q)
        return result

    def _build_transcribe_meta_map(self, paths: List[str], wcfg: Dict[str, Any]) -> Dict[str, str]:
        """构建语音转写上下文映射（已有 SRT；缺失视频由 ASR worker 逐视频交付）。"""
        if not wcfg.get("enabled", False):
            return {}
        from .workspace_paths import TRANSCRIBE_SRT_DIR, TRANSCRIBE_TRANSCRIPTS_FILE
        if not TRANSCRIBE_SRT_DIR.exists():
            return {}
        max_chars = max(100, safe_get(wcfg, "max_chars", 2000))
        inject_ts = wcfg.get("inject_timestamps", False)
        store = read_json(TRANSCRIBE_TRANSCRIPTS_FILE, {})
        result: Dict[str, str] = {}
        for vp in paths:
            vid = stable_id(vp)
            srt_file = TRANSCRIBE_SRT_DIR / f"{vid}.srt"
            if not srt_file.exists():
                continue
            try:
                srt_content = srt_file.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not srt_content:
                continue
            text = srt_to_text(srt_content, inject_ts)
            lang = store.get(vid, {}).get("language", "")
            block = format_transcribe_block(text, lang, max_chars)
            if block:
                result[vp] = block
        return result

    def run(
        self,
        paths: List[str],
        on_log: Optional[Callable[[str, str], None]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_file_done: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_done: Optional[Callable[[Dict[str, Any]], None]] = None,
        llama_ensure: Optional[Callable[[threading.Event], bool]] = None,
        llama_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """启动处理。llama_ensure：AI 阶段前置就绪门；llama_override：显式集成覆盖值。"""
        if self.is_running:
            return {"ok": False, "error": "已有自动重命名任务在运行"}

        from .config_store import load_config
        cfg_dict = load_config()
        llama_gate = ready_gate(llama_ensure, self._stop) if llama_ensure else None
        engine_cfg = build_engine_config(cfg_dict, llama_override)

        # 进度统计（线程安全）
        progress_lock = threading.Lock()
        progress = {"done": 0, "total": len(paths)}

        def _emit_file_done(orig: str, new_path: Optional[str], status: str,
                             info: Dict[str, Any], title: str, plot: str,
                             tags: List[str],
                             thumb_lookup: Optional[Dict[int, Any]] = None) -> None:
            """引擎 on_file_done 回调 → 写 history + 推送前端。"""
            vid = stable_id(orig)
            entry = {
                "id": vid,
                "original_path": orig,
                "new_path": new_path or orig,
                "original_name": Path(orig).name,
                "new_name": Path(new_path or orig).name,
                "status": status,
                "title": title,
                "plot": plot,
                "tags": list(tags) if tags else [],
                "info": {
                    "duration": info.get("duration", 0),
                    "size": info.get("size", 0),
                    "resolution": _format_resolution(info),
                    "codec": (info.get("video") or {}).get("codec", ""),
                },
                "processed_at": time.time(),
            }
            thumb_time = (info or {}).get("thumb_time") or ""
            if thumb_time:
                entry["thumb_time"] = thumb_time
            # 写 history（status=ok/skipped 才算"done"）
            if status in ("ok", "skipped"):
                append_history_entry(entry)
                if engine_cfg.frame_time_tags == 2:
                    # 先于前端推送重生成缩略图：优先 thumb_time 匹配的本次缓存帧
                    discovery.invalidate_thumbnail(vid)
                    refreshed = False
                    if thumb_lookup:
                        sec = _thumb_time_seconds(thumb_time)
                        f = thumb_lookup.get(sec) if sec is not None else None
                        if f is not None:
                            try:
                                refreshed = discovery.generate_thumbnail_from_jpeg(
                                    vid, base64.b64decode(f.b64)) is not None
                            except Exception:
                                refreshed = False
                    if not refreshed:
                        try:
                            discovery.generate_thumbnail(new_path or orig, vid)
                        except Exception:
                            pass
                    entry["thumb_refresh"] = True
            # stop 后逐文件落盘，防 daemon 线程退出丢记录
            if self._stop.is_set():
                try:
                    flush_batch(keep_mode=True)
                except Exception as e:
                    sys.stderr.write(f"[runner] stop 检查点 history 落盘失败: {e}\n")
            # 重命名成功同步 manifest 的 adhoc 路径
            if status == "ok" and new_path and new_path != orig:
                try:
                    update_adhoc_path(orig, new_path)
                except Exception as e:
                    sys.stderr.write(f"[runner] 更新 adhoc 路径失败: {e}\n")

            with progress_lock:
                progress["done"] += 1
                cur, tot = progress["done"], progress["total"]
            if on_progress:
                try:
                    on_progress(cur, tot)
                except Exception:
                    pass
            if on_file_done:
                try:
                    on_file_done(entry)
                except Exception:
                    pass

        def _runner():
            self._handler = _UIHandler(on_log or (lambda lv, msg: None),
                                       verbose=engine_cfg.debug_log)
            engine_logger.addHandler(self._handler)
            old_level = engine_logger.level
            engine_logger.setLevel(logging.DEBUG if engine_cfg.debug_log else logging.INFO)

            try:
                try:
                    ensure_dependencies()
                except DependencyError as e:
                    if on_log:
                        on_log("err", f"依赖检查失败: {e}")
                    if on_done:
                        on_done({"ok": False, "error": "依赖缺失", "ok_count": 0,
                                  "error_count": 0, "skipped_count": 0})
                    return

                # 复用全局 bindings
                self._bindings = bindings

                try:
                    self._client = self._bindings.OpenAI(
                        api_key=engine_cfg.api_key,
                        base_url=engine_cfg.base_url,
                    )
                except Exception as e:
                    if on_log:
                        on_log("err", f"创建 AI 客户端失败: {e}")
                    if on_done:
                        on_done({"ok": False, "error": str(e),
                                  "ok_count": 0, "error_count": 0, "skipped_count": 0})
                    return

                begin_batch()  # history 变更累积到内存，结束时一次性落盘
                # 扩展功能上下文
                from .config_store import load_tagger_config, load_transcribe_config
                tcfg = load_tagger_config()
                wcfg = load_transcribe_config()
                _log_cb = on_log or (lambda lv, msg: None)

                from .workspace_paths import TRANSCRIBE_SRT_DIR, TAGGER_TAGS_FILE
                from .api_mixins.experimental import _gpu_task_lock
                from .rename_aux import AsrInjectWorker, TaggerInjectWorker
                from . import tagger as tagger_mod, transcribe_cpp

                # 注入转写集成：开关 + 就绪 + 存在缺失 SRT 的视频
                asr_active = (bool(wcfg.get("enabled", False))
                              and bool(wcfg.get("integration", False))
                              and transcribe_cpp.inject_ready())
                asr_missing: List[str] = []
                if asr_active:
                    asr_missing = _asr_missing_list(paths, TRANSCRIBE_SRT_DIR, self._stop)
                asr_active = bool(asr_missing)

                # Tagger 集成：开关 + 引擎就绪，未就绪降级不阻断
                tagger_active = (bool(tcfg.get("enabled", False))
                                 and bool(tcfg.get("integration", False)))
                if tagger_active:
                    st = tagger_mod.get_status()
                    files_err = tagger_mod.ensure_model_files()
                    if not st.get("ready") or files_err:
                        _log_cb("warn", f"[标签辅助] 集成不可用（{files_err or '引擎未就绪'}），"
                                        f"本次跳过标签推理，重命名不受影响")
                        tagger_active = False

                aux_workers: List[Any] = []
                aux_stages: set = set()
                prefill: Dict[str, Dict[str, Any]] = {}
                if asr_active:
                    max_chars = max(100, safe_get(wcfg, "max_chars", 2000))
                    aux_workers.append(AsrInjectWorker(asr_missing, max_chars,
                                                       _log_cb, self._stop))
                    aux_stages.add("asr")
                if tagger_active:
                    threshold = tcfg.get("threshold")
                    try:
                        threshold = min(1, max(0.1, float(threshold)))
                    except (TypeError, ValueError):
                        threshold = 0.75
                    aux_workers.append(TaggerInjectWorker(
                        self._stop, _log_cb, threshold=threshold,
                        general_count=max(0, safe_get(tcfg, "general_count", 15)),
                        save_fn=tagger_mod.save_tag_results))
                    aux_stages.add("tagger")

                # 已有结果的复用上下文（prefill），其余视频由 aux worker 交付
                for vp, pair in self._build_tagger_meta_map(paths, tcfg).items():
                    prefill.setdefault(vp, {})["tagger_hits"] = pair[0]
                    prefill.setdefault(vp, {})["literal_meta"] = pair[1]
                for vp, block in self._build_transcribe_meta_map(paths, wcfg).items():
                    prefill.setdefault(vp, {})["extra_meta"] = block
                if tagger_active:
                    tags_store = read_json(TAGGER_TAGS_FILE, {})
                    tagged_skip = {vp for vp in paths if tags_store.get(stable_id(vp))}
                else:
                    tagged_skip = set()

                # 集成辅助开启时持 GPU 锁整批，隔离右键手动入口
                hold_gpu_lock = bool(asr_missing) or (
                    tagger_active and len(tagged_skip) < len(paths))
                if hold_gpu_lock:
                    acquired = False
                    while not acquired:
                        if self._stop.is_set():
                            break
                        acquired = _gpu_task_lock.acquire(blocking=False)
                        if not acquired:
                            time.sleep(0.2)
                    if not acquired:
                        if on_done:
                            on_done({"ok": False, "error": "已停止", "ok_count": 0,
                                      "error_count": 0, "skipped_count": 0})
                        return
                    _llama_vram_warning(_log_cb)
                try:
                    pt = prompts.load_term_library()
                    # 停用组过滤：黑名单命中组的条目不进入检索（"" 为默认分组）
                    pt_disabled = set(pt["disabled_groups"])
                    pt_items = [it for it in pt["items"] if it["group"] not in pt_disabled]
                    pt_skipped = len(pt["items"]) - len(pt_items)
                    dense = None
                    pt_mode = pt["mode"]
                    if engine_cfg.rag_vec_enabled and pt_mode == "enhanced":
                        from .ort_embedding import OrtEmbedBackend, get_status as ort_status
                        if ort_status().get("ready"):
                            dense = OrtEmbedBackend(
                                device=engine_cfg.rag_vec_device,
                                log_fn=lambda m: engine_logger.info(f"[RAG] {m}"))
                        else:
                            engine_logger.info("[RAG] 向量检索已启用但 ort-embedding 未就绪，"
                                               "增强档本次降级为开启（仅首轮字面召回）")
                    term_recall = TermRecall(items=pt_items,
                                           mode=pt_mode,
                                           top_k=engine_cfg.rag_top_k,
                                           dense=dense,
                                           vec_threshold=engine_cfg.rag_vec_threshold,
                                           vec_top_n=engine_cfg.rag_vec_top_n)
                    warm_thread: Optional[threading.Thread] = None
                    if dense is not None:
                        engine_logger.info(f"[RAG] 正在后台加载嵌入模型: {dense.model_title}"
                                           f"（device={dense.device}）…")

                        def _warm() -> None:
                            if self._stop.is_set():
                                return
                            try:
                                term_recall.warmup_dense(self._stop)
                                if term_recall.dense is None:
                                    engine_logger.info("[RAG] 向量索引构建失败，增强档降级为单轮")
                            except Exception as e:
                                engine_logger.info(f"[RAG] 向量索引预热失败，增强档降级为单轮: {e}")

                        # 预热后台化
                        warm_thread = threading.Thread(target=_warm, name="rag-warmup", daemon=True)
                        warm_thread.start()
                    if term_recall.mode != "off":
                        if term_recall.mode == "enhanced" and not term_recall.has_dense:
                            mode_label = "增强（向量不可用，本次降级为开启）"
                        else:
                            mode_label = {"on": "开启（单轮·字面召回注入）",
                                          "enhanced": "增强（首轮字面注入+第二轮向量召回）"}[term_recall.mode]
                        skip_note = f"（停用组跳过 {pt_skipped} 条）" if pt_skipped else ""
                        engine_logger.info(
                            f"[RAG] 词条检索模式: {mode_label} | 词条库 {term_recall.total_items} 条{skip_note}")
                    elif pt_mode != "off":
                        engine_logger.info("[RAG] 词条检索已开启但启用组内无标签，本次不注入")
                    pipeline = BatchPipeline(
                        paths, engine_cfg, self._client, self._stop,
                        on_file_done=_emit_file_done,
                        # NFO 缓存模式以 stable_id 命名，防同 stem 覆盖
                        nfo_namer=lambda vp: f"{stable_id(vp)}.nfo",
                        term_recall=term_recall,
                        aux_stages=aux_stages,
                        aux_workers=aux_workers,
                        prefill=prefill,
                        llama_gate=llama_gate,
                    )
                    # 预完成复用已有结果的辅助阶段（与 prefill 同一口径）
                    missing_set = set(asr_missing)
                    for ctx in pipeline.contexts:
                        if "asr" in aux_stages and ctx.vp not in missing_set:
                            ctx.complete_stage("asr")
                        if "tagger" in aux_stages and ctx.vp in tagged_skip:
                            ctx.complete_stage("tagger")
                    pipeline.run()
                    if warm_thread is not None:
                        # 预热线程收尾
                        warm_thread.join(timeout=2.0)
                finally:
                    if hold_gpu_lock:
                        _gpu_task_lock.release()

                # 先落盘 history 再通知前端（on_done 会重读）
                flush_batch()

                stats = getattr(pipeline, "stats", None)
                if on_done:
                    if stats is not None:
                        on_done({
                            "ok": stats.fatal_error is None,
                            "error": stats.fatal_error or "",
                            "ok_count": stats.ok,
                            "error_count": stats.error,
                            "skipped_count": stats.skipped,
                            "cancelled_count": stats.cancelled,
                            "total": stats.total,
                            "processed": progress["done"],
                        })
                    else:
                        # stats 为 None 表示无可处理视频
                        on_done({
                            "ok": True,
                            "ok_count": 0,
                            "error_count": 0,
                            "skipped_count": 0,
                            "total": 0,
                            "processed": 0,
                        })

            except Exception as e:
                if on_log:
                    on_log("err", f"自动重命名异常: {e}")
                if on_done:
                    on_done({"ok": False, "error": str(e),
                              "ok_count": 0, "error_count": 0, "skipped_count": 0})
            finally:
                # 异常/停止路径同样落盘
                try:
                    flush_batch()
                    # 落盘完成信号，app.py 退出清理据此放行
                    self._history_flushed.set()
                except Exception:
                    pass
                if self._handler is not None:
                    engine_logger.removeHandler(self._handler)
                    self._handler = None
                engine_logger.setLevel(old_level)
                if self._client is not None:
                    try:
                        self._client.close()
                    except Exception:
                        pass
                    self._client = None

        with self._start_lock:
            if self.is_running:
                return {"ok": False, "error": "已有自动重命名任务在运行"}
            self._stop.clear()
            self._history_flushed.clear()
            self._thread = threading.Thread(target=_runner, name="pipeline-runner", daemon=True)
            self._thread.start()
        return {"ok": True}

    def stop(self) -> None:
        self._stop.set()


def _format_resolution(info: Dict[str, Any]) -> str:
    v = info.get("video") or {}
    if v.get("width") and v.get("height"):
        return f"{v['width']}x{v['height']}"
    return ""


# ── NFO 导出：缓存 → 视频目录 ──


def export_nfo(vid: str) -> Dict[str, Any]:
    """缓存 NFO 复制到视频目录 <新名>.nfo，返回 {ok, message, target_path}。"""
    import shutil
    from batch_rename.utils import path_exists, to_long_path
    entry = get_history_by_id(vid)
    if not entry:
        return {"ok": False, "message": "未找到已处理记录"}

    new_path = entry.get("new_path") or entry.get("original_path") or ""
    if not new_path or not path_exists(new_path):
        return {"ok": False, "message": f"视频文件不存在: {new_path}"}

    cache = find_cached_nfo(vid)
    if cache is None:
        return {"ok": False, "message": "缓存中无 NFO（可能未开启 NFO 模式或处理时未生成）"}

    target = Path(new_path).with_suffix(".nfo")
    try:
        shutil.copy2(to_long_path(str(cache)), to_long_path(str(target)))
    except OSError as e:
        return {"ok": False, "message": f"复制失败: {e}"}
    return {"ok": True, "message": f"已导出到: {target.name}", "target_path": str(target)}


def export_nfo_batch(vids: List[str]) -> Dict[str, Any]:
    """批量导出 NFO。"""
    ok = 0
    failed = 0
    errors = []
    for vid in vids:
        r = export_nfo(vid)
        if r.get("ok"):
            ok += 1
        else:
            failed += 1
            errors.append(r.get("message", "未知错误"))
    return {"ok": failed == 0, "ok_count": ok, "failed_count": failed, "errors": errors[:10]}
