"""在后台线程中运行 BatchPipeline：装配 Config、转发引擎日志、逐文件写 history 并推送前端。"""
from __future__ import annotations

import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import discovery, prompts
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
from batch_rename.dependencies import DependencyError, ensure_dependencies, bindings
from batch_rename.env import logger as engine_logger
from batch_rename.pipeline import BatchPipeline


def _safe_int(d: dict, key: str, default: int) -> int:
    try:
        return int(d.get(key, default))
    except (TypeError, ValueError):
        return default


def _safe_float(d: dict, key: str, default: float) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


# ── 日志转发 handler ──


class _UIHandler(logging.Handler):
    """把 batch_rename 日志行转发到 GUI 回调。"""

    def __init__(self, callback: Callable[[str, str], None]):
        super().__init__()
        self._cb = callback
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        try:
            msg = self.format(record)
            if not msg.strip():
                return
            # GUI 日志只保留 [i/N] 进度行，过滤引擎的 [改名]/[NFO 已生成] 细节
            if msg.startswith("[改名]") or msg.startswith("[NFO] 已生成"):
                return
            # 过滤纯横线分隔行（引擎的 ━/─ 分隔符）
            stripped = msg.strip()
            if stripped and set(stripped) <= set("━─═"):
                return
            lv = (
                "err" if record.levelno >= logging.ERROR
                else "warn" if record.levelno >= logging.WARNING
                else ""
            )
            self._cb(lv, msg.strip())
        except Exception:
            pass


# ── 从 config.json + 预设构建 batch_rename.Config ──


def build_engine_config(cfg_dict: Dict[str, Any]) -> Config:
    """从 GUI 的 config.json 构建 batch_rename.Config 对象。"""
    cfg = Config()

    def _section(name: str) -> Dict[str, Any]:
        v = cfg_dict.get(name)
        return v if isinstance(v, dict) else {}

    ai = _section("ai")
    cfg.model = ai.get("model", cfg.model)
    cfg.base_url = ai.get("base_url", cfg.base_url)
    cfg.api_key = ai.get("api_key", cfg.api_key)

    # 本地推理集成（可插拔）：开启后覆盖为本地 llama-server 地址与默认
    # 模型/密钥，磁盘上用户填写的远程 API 参数保持不动；未激活时返回 None
    from .llama_integration import ai_override
    ov = ai_override(cfg_dict)
    if ov:
        cfg.model = ov["model"]
        cfg.base_url = ov["base_url"]
        cfg.api_key = ov["api_key"]

    cfg.max_tokens = _safe_int(ai, "max_tokens", cfg.max_tokens)
    cfg.temperature = _safe_float(ai, "temperature", cfg.temperature)
    cfg.top_p = _safe_float(ai, "top_p", cfg.top_p)
    cfg.retry_times = _safe_int(ai, "retry_times", cfg.retry_times)
    cfg.ai_timeout = _safe_int(ai, "ai_timeout", cfg.ai_timeout)
    cfg.ai_workers = _safe_int(ai, "ai_workers", cfg.ai_workers)
    cfg.enforce_json_mode = bool(ai.get("enforce_json_mode", cfg.enforce_json_mode))

    video = _section("video")
    cfg.sampling_points = _safe_int(video, "sampling_points", cfg.sampling_points)
    cfg.frames_per_point = _safe_int(video, "frames_per_point", cfg.frames_per_point)
    cfg.frame_max_side = _safe_int(video, "frame_max_side", cfg.frame_max_side)
    cfg.frame_time_tags = _safe_int(video, "frame_time_tags", cfg.frame_time_tags)

    naming = _section("naming")
    cfg.include_date = bool(naming.get("include_date", cfg.include_date))
    cfg.include_original = bool(naming.get("include_original", cfg.include_original))

    active = prompts.get_active(with_thumb_time=(cfg.frame_time_tags == 2))
    cfg.prompt = active["prompt"]
    cfg.system_prompt = active["system_prompt"]

    # NFO 输出模式：默认（nfo_auto_export 关闭）缓存到 _workspace/nfo/，导出时再复制到视频目录；
    # 开启「自动输出 NFO 至视频目录」后直接写视频所在目录（nfo_target_dir=None）
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
        # 退出期 history 落盘完成信号：finally 的 flush_batch 成功后置位，
        # 供 app.py 退出清理轮询——比「硬等线程结束」更能区分「已落盘」与「线程未退出」
        self._history_flushed: threading.Event = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── 扩展功能上下文注入 ──

    def _build_pixai_meta_map(self, paths: List[str], exp_cfg: Dict[str, Any]) -> Dict[str, str]:
        """构建 pixai 标签上下文映射（video_path → 提示词文本）。

        仅当功能开启且标签存在时才注入。exp_cfg 为 experimental 段配置。
        """
        if not exp_cfg.get("pixai_tagger_enabled", False):
            return {}
        from .workspace_paths import PIXAI_TAGS_FILE
        tags_store = read_json(PIXAI_TAGS_FILE, {})
        if not tags_store:
            return {}
        result: Dict[str, str] = {}
        for vp in paths:
            vid = stable_id(vp)
            entry = tags_store.get(vid)
            if not entry:
                continue
            char_tags = entry.get("character_tags", [])
            ip_tags = entry.get("ip_tags", [])
            if not char_tags and not ip_tags:
                continue
            parts = ["【IP辅助参考】",
                     "以下为自动识别结果，可能存在误判（如相似角色混淆），仅供参考。请结合视频画面自行判断，若与画面明显矛盾则忽略："]
            if char_tags:
                chars = ", ".join(f"{t['name']} ({int(t['score']*100)}%)" for t in char_tags)
                parts.append(f"- 疑似角色: {chars}")
            if ip_tags:
                ips = ", ".join(t["name"] for t in ip_tags)
                parts.append(f"- 疑似IP/作品: {ips}")
            parts.append("生成 tags 与plot时，上述角色/IP 名称请统一改用常用中文名")
            result[vp] = "\n".join(parts)
        return result

    def _build_whisper_meta_map(self, paths: List[str], exp_cfg: Dict[str, Any]) -> Dict[str, str]:
        """构建 whisper 转录上下文映射（video_path → 提示词文本）。

        仅当功能开启且 SRT 文件存在时才注入。exp_cfg 为 experimental 段配置。
        """
        if not exp_cfg.get("whisper_enabled", False):
            return {}
        from .workspace_paths import WHISPER_SRT_DIR, WHISPER_TRANSCRIPTS_FILE
        if not WHISPER_SRT_DIR.exists():
            return {}
        max_chars = max(100, _safe_int(exp_cfg, "whisper_max_chars", 800))
        inject_ts = exp_cfg.get("whisper_inject_timestamps", False)
        store = read_json(WHISPER_TRANSCRIPTS_FILE, {})
        result: Dict[str, str] = {}
        for vp in paths:
            vid = stable_id(vp)
            srt_file = WHISPER_SRT_DIR / f"{vid}.srt"
            if not srt_file.exists():
                continue
            try:
                srt_content = srt_file.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not srt_content:
                continue
            text = srt_to_text(srt_content, inject_ts)
            if len(text) > max_chars:
                text = text[:max_chars] + "…（已截断）"
            lang = store.get(vid, {}).get("language", "")
            parts = ["【语音辅助参考】",
                     "以下为自动语音转录结果，仅供参考。请结合视频画面自行判断："]
            if lang:
                parts.append(f"- 语言: {lang}")
            parts.append(f"- 转录文本: {text}")
            parts.append("过长的转录文本可能被截断，注意判断")
            result[vp] = "\n".join(parts)
        return result

    def run(
        self,
        paths: List[str],
        on_log: Optional[Callable[[str, str], None]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_file_done: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_done: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """启动处理。"""
        if self.is_running:
            return {"ok": False, "error": "已有处理任务在运行"}

        from .config_store import load_config
        cfg_dict = load_config()
        engine_cfg = build_engine_config(cfg_dict)

        # 进度统计（线程安全）
        progress_lock = threading.Lock()
        progress = {"done": 0, "total": len(paths)}

        def _emit_file_done(orig: str, new_path: Optional[str], status: str,
                             info: Dict[str, Any], title: str, plot: str,
                             tags: List[str]) -> None:
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
                    # 先于前端推送用 thumb_time 重生成缩略图（append 后 history 内存缓存立即可读）
                    discovery.invalidate_thumbnail(vid)
                    try:
                        discovery.generate_thumbnail(new_path or orig, vid)
                    except Exception:
                        pass
                    entry["thumb_refresh"] = True
            # 停止检查点：stop 后每完成一个文件即落盘一次，避免 daemon 线程退出时整批记录丢失。
            if self._stop.is_set():
                try:
                    flush_batch(keep_mode=True)
                except Exception as e:
                    sys.stderr.write(f"[runner] stop 检查点 history 落盘失败: {e}\n")
            # 重命名成功后同步更新 manifest 中的 adhoc 路径（旧→新），避免重新扫描时找不到文件
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
            self._handler = _UIHandler(on_log or (lambda lv, msg: None))
            engine_logger.addHandler(self._handler)
            old_level = engine_logger.level
            engine_logger.setLevel(logging.DEBUG)

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

                # 2. 复用全局 bindings（ensure_dependencies 已完成加载）
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
                # 扩展功能上下文（功能开启且数据存在时注入提示词；均关闭时为空 dict）
                exp_cfg = cfg_dict.get("experimental", {})
                extra_meta_map = self._build_pixai_meta_map(paths, exp_cfg)
                for vp, txt in self._build_whisper_meta_map(paths, exp_cfg).items():
                    if vp in extra_meta_map:
                        extra_meta_map[vp] += "\n\n" + txt
                    else:
                        extra_meta_map[vp] = txt
                pipeline = BatchPipeline(
                    paths, engine_cfg, self._client, self._stop,
                    on_file_done=_emit_file_done,
                    # NFO 缓存模式（nfo_target_dir 非空）下直接以 stable_id 命名写入，
                    # 避免不同目录下同 stem 视频映射到同一文件互相覆盖
                    nfo_namer=lambda vp: f"{stable_id(vp)}.nfo",
                    extra_meta_map=extra_meta_map,
                )
                pipeline.run()

                # 4. 先把累积的 history 落盘，再通知前端完成
                #    （on_done 里会 scan_all 重读 history，必须保证读到的是已持久化的数据）
                flush_batch()

                stats = getattr(pipeline, "stats", None)
                if on_done:
                    if stats is not None:
                        on_done({
                            "ok": True,
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
                    on_log("err", f"处理异常: {e}")
                if on_done:
                    on_done({"ok": False, "error": str(e),
                              "ok_count": 0, "error_count": 0, "skipped_count": 0})
            finally:
                # 把批量累积的 history 变更一次性落盘（异常/停止路径同样保证写入）
                try:
                    flush_batch()
                    # 落盘完成信号：app.py 退出清理据此判断可以放行，
                    # 而不是靠硬性线程超时猜测
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
                return {"ok": False, "error": "已有处理任务在运行"}
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
    """把缓存的 NFO 复制到视频同目录，命名为 <新名>.nfo。

    返回 {ok, message, target_path}。
    """
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
