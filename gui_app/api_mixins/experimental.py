"""ExperimentalMixin — 扩展功能 API（llama.cpp / faster-whisper / pixai-tagger）。"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..config_store import (load_config, load_llama_config, load_llama_model_configs,
                            load_pixai_config, load_whisper_config,
                            update_llama_config, update_pixai_config, update_whisper_config)
from ..installer import DEFAULT_PYPI_MIRROR, DEFAULT_TRT_SITE
from ..js_push import js_pusher
from ..workspace_paths import (LLAMA_CONFIG_FILE, PIXAI_TAGS_FILE,
                               WHISPER_SRT_DIR, WHISPER_TRANSCRIPTS_FILE)
from ..workspace_store import read_json, update_json, write_json
from batch_rename.utils import safe_float, safe_int

# GPU 密集任务互斥（转录/标签获取/翻译同刻只允许一个）
_gpu_task_lock = threading.Lock()
# 安装/卸载/清缓存互斥
_install_lock = threading.Lock()
# GPU 任务取消事件（开始前 clear，stop_gpu_task 置位）
_gpu_stop_event = threading.Event()
# 安装取消事件（开始前 clear，stop_install 置位）
_install_stop_event = threading.Event()


def _push_log(msg: str, level: str = "info") -> None:
    js_pusher.push("appendLog", msg, level)


def _push_llama_log(msg: str) -> None:
    js_pusher.push("appendLlamaLog", msg)


def _load_pixai_tags() -> Dict[str, Any]:
    """读取 pixai 标签存储（缺失/损坏按空 dict 处理）。"""
    return read_json(PIXAI_TAGS_FILE, {})


def _record_last_model(path: str) -> None:
    """记录本次成功运行的模型：本地推理下拉框默认选中 + auto_run 默认启动。"""
    path = str(path or "").strip()
    if not path:
        return

    def _mutate(c):
        c["last_model"] = path
        return c

    update_llama_config(_mutate)


class ExperimentalMixin:
    """扩展功能相关 API。"""

    # ── GPU 任务停止 ──

    def stop_gpu_task(self) -> Dict[str, Any]:
        """停止当前运行中的 GPU 任务（转录/标签获取/翻译）。"""
        if not _gpu_task_lock.locked():
            return {"ok": False, "error": "当前没有运行中的任务"}
        if _gpu_stop_event.is_set():
            return {"ok": True, "message": "已在停止中"}
        _gpu_stop_event.set()
        _push_log("正在停止当前任务（进行中的步骤完成后生效）…")
        return {"ok": True}

    def stop_install(self) -> Dict[str, Any]:
        """停止当前正在进行的安装（llama.cpp / faster-whisper / pixai-tagger）。"""
        if not _install_lock.locked():
            return {"ok": False, "error": "当前没有正在进行的安装"}
        if _install_stop_event.is_set():
            return {"ok": True, "message": "已在停止中"}
        _install_stop_event.set()
        try:
            from .. import models_downloader
            models_downloader.set_cancel()
        except Exception:
            pass
        _push_log("正在停止安装…")
        return {"ok": True}

    # ── UV 包管理工具 ──

    def get_uv_status(self) -> Dict[str, Any]:
        """获取 UV 包管理工具状态（是否安装、是否位于 UV-Tool 目录）。"""
        from ..installer import get_uv_status as _status
        return _status()

    def clean_uv_cache(self) -> Dict[str, Any]:
        """清理 UV 包缓存，释放磁盘空间。"""
        from ..installer import clean_uv_cache as _clean
        if not _install_lock.acquire(blocking=False):
            return {"ok": False, "error": "已有模块正在安装，请等待完成后再清理缓存"}
        try:
            return _clean(log_fn=_push_log)
        finally:
            _install_lock.release()

    def uninstall_uv(self) -> Dict[str, Any]:
        """卸载 UV（删除 UV-Tool 含缓存）。已装模块 venv 仍可运行。"""
        from ..installer import uninstall_uv as _uninstall
        if not _install_lock.acquire(blocking=False):
            return {"ok": False, "error": "已有模块正在安装，请等待完成后再卸载 UV"}
        try:
            return _uninstall(log_fn=_push_log)
        finally:
            _install_lock.release()

    def _auto_clean_uv_cache(self) -> None:
        """安装成功后自动清空 UV 包缓存（与清理按钮同一逻辑，保留 UV 本体）。"""
        from ..installer import clean_uv_cache
        clean_uv_cache(log_fn=_push_log)

    # ── 扩展功能页聚合状态 ──

    def get_experimental_status(self) -> Dict[str, Any]:
        """扩展功能页一次拉齐全部状态与配置。"""
        return {
            "cfg": self.get_config(),
            "uv": self.get_uv_status(),
            "llama": self.get_llama_status(),
            "pixai": self.get_pixai_tagger_status(),
            "whisper": self.get_whisper_status(),
            "ragvec": self.get_rag_vec_status(),
        }

    def get_rag_vec_status(self) -> Dict[str, Any]:
        """向量检索模块安装状态与启用状态。"""
        from ..ort_embedding import get_status
        exp = self.get_config().get("experimental") or {}
        status = get_status()
        status["enabled"] = bool(exp.get("rag_vec_enabled", False))
        try:
            threshold = float(exp.get("rag_vec_threshold", 0.45))
            top_n = int(exp.get("rag_vec_top_n", 20))
        except (TypeError, ValueError):
            threshold, top_n = 0.45, 20
        status["cfg"] = {
            "device": str(exp.get("rag_vec_device", "") or ""),
            "threshold": threshold,
            "top_n": top_n,
        }
        return status

    def get_rag_vec_mirrors(self) -> Dict[str, Any]:
        """向量检索安装选项：运行设备（按 GPU 检测推荐）+ PyPI 镜像。"""
        from ..installer import detect_gpu, get_mirror_groups
        info = get_mirror_groups(["pypi"])
        gpu = detect_gpu()
        info["gpu"] = gpu
        has_gpu = bool(gpu.get("has_nvidia"))
        # onnxruntime-gpu 需 CUDA 13 驱动（≥580）
        rec = "gpu" if has_gpu and gpu.get("recommended") == "cu132" else "cpu"
        if has_gpu and rec == "cpu":
            gpu_desc = (f"检测到驱动支持 CUDA {gpu.get('cuda_max') or '?'}，"
                        "GPU 版需 ≥580：请升级驱动，或选择 CPU 版")
        elif has_gpu:
            gpu_desc = "N 卡加速，约 1.27GB"
        else:
            gpu_desc = "未检测到 NVIDIA 显卡"
        info["devices"] = [
            {"id": "gpu", "name": "GPU（fp32 模型）", "desc": gpu_desc,
             "recommended": rec == "gpu"},
            {"id": "cpu", "name": "CPU（int8 模型）",
             "desc": "约 345MB",
             "recommended": rec == "cpu"},
        ]
        return info

    def install_rag_vec(self, device: str = "gpu",
                        site: str = DEFAULT_PYPI_MIRROR) -> Dict[str, Any]:
        """安装向量检索依赖（uv + venv + onnxruntime + 所选设备的模型文件）。"""
        from ..ort_embedding import install_dependencies
        if not _install_lock.acquire(blocking=False):
            return {"ok": False, "error": "已有模块正在安装，请等待完成后再试"}
        _install_stop_event.clear()
        try:
            r = install_dependencies(device=device, site=site,
                                     log_fn=_push_log,
                                     stop_event=_install_stop_event)
            if r.get("ok"):
                self._auto_clean_uv_cache()
            return r
        finally:
            _install_stop_event.clear()
            _install_lock.release()

    def remove_rag_vec(self) -> Dict[str, Any]:
        """删除 ort-embedding 模块（venv + 模型 + 索引缓存）。"""
        from ..ort_embedding import remove_ort_embedding as _remove
        if not _install_lock.acquire(blocking=False):
            return {"ok": False, "error": "已有模块正在安装，请等待完成后再卸载"}
        try:
            return _remove()
        finally:
            _install_lock.release()

    def stop_rag_vec(self) -> Dict[str, Any]:
        """停止嵌入后台进程，释放显存/内存；下次向量检索自动重启。"""
        from ..ort_embedding import stop_worker, get_status
        stop_worker()
        return {"ok": True, "message": "已停止嵌入后台进程", "worker_running": get_status().get("worker_running")}

    # ── PixAI Tagger 标签获取 ──

    def get_pixai_tagger_status(self) -> Dict[str, Any]:
        """获取 pixai-tagger 安装状态与启用状态。"""
        from ..pixai_tagger import get_status
        enabled = load_pixai_config().get("enabled", False)
        status = get_status()
        status["enabled"] = enabled
        return status

    def set_pixai_tagger_enabled(self, enabled: bool) -> Dict[str, Any]:
        """启用/禁用 pixai-tagger 功能。"""
        from ..pixai_tagger import PIXAI_TAGGER_DIR
        if not PIXAI_TAGGER_DIR.is_dir():
            return {"ok": True, "enabled": enabled}
        update_pixai_config(lambda c: c.update(enabled=enabled) or c)
        return {"ok": True, "enabled": enabled}

    def get_pixai_mirrors(self) -> Dict[str, Any]:
        """获取 pixai-tagger 安装选项（尺寸/站点/GPU 检测）。"""
        from ..pixai_tagger import get_mirrors_info
        return get_mirrors_info()

    def install_pixai_tagger(self, input_size: int = 1008,
                             site: str = DEFAULT_TRT_SITE) -> Dict[str, Any]:
        """安装 pixai-tagger 依赖（uv + venv + tensorrt + 模型下载 + TRT 引擎构建）。"""
        from ..pixai_tagger import install_dependencies

        if not _install_lock.acquire(blocking=False):
            return {"ok": False, "error": "已有模块正在安装，请等待完成后再试"}
        _install_stop_event.clear()
        try:
            r = install_dependencies(input_size=input_size,
                                     site=site, log_fn=_push_log,
                                     stop_event=_install_stop_event)
            if r.get("ok"):
                self._auto_clean_uv_cache()
            return r
        finally:
            _install_stop_event.clear()
            _install_lock.release()

    def remove_pixai_tagger(self) -> Dict[str, Any]:
        """删除 pixai-tagger 文件夹与模块数据。"""
        from ..pixai_tagger import remove_pixai_tagger as _remove
        if not _install_lock.acquire(blocking=False):
            return {"ok": False, "error": "已有模块正在安装，请等待完成后再卸载"}
        try:
            return _remove()
        finally:
            _install_lock.release()

    def detect_ip_tags(self, items: List) -> Dict[str, Any]:
        """对选中视频执行 PixAI 标签获取（items: [[video_id, video_path], ...]）。"""
        from ..pixai_tagger import (get_status, start_analyze_stream,
                                    extract_frames_for_tagger, ensure_model_files,
                                    resolve_precision, normalize_size,
                                    ANIME_CLS_THRESHOLD, TAG_THRESHOLD)
        from batch_rename.dependencies import ffmpeg_tools
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

        status = get_status()
        if not status["ready"]:
            return {"ok": False, "results": {}, "error": "pixai-tagger 未安装或未就绪，请先在扩展功能页安装依赖"}

        pcfg = load_pixai_config()
        frames_n = max(1, safe_int(pcfg.get("frames"), 15))
        threshold = min(1, max(0.1, safe_float(pcfg.get("threshold"), TAG_THRESHOLD)))
        skip_real = bool(pcfg.get("classify", False))
        size = normalize_size(pcfg.get("input_size"))
        precision = resolve_precision("auto")

        if not ffmpeg_tools.ffmpeg:
            try:
                ffmpeg_tools.locate()
            except Exception as e:
                return {"ok": False, "results": {}, "error": f"ffmpeg 不可用: {e}"}

        if not _gpu_task_lock.acquire(blocking=False):
            return {"ok": False, "results": {}, "error": "已有 GPU 任务在运行，请等待完成后再试"}
        _gpu_stop_event.clear()
        if _gpu_stop_event.is_set():
            _gpu_task_lock.release()
            return {"ok": False, "results": {}, "error": "已停止"}

        vid_list = [item[0] for item in items]
        video_paths = [item[1] for item in items]
        total = len(video_paths)
        results: Dict[str, Any] = {}

        try:
            _push_log(f"正在分析 {total} 个视频的IP信息…")
            t0 = time.time()

            # ONNX 预检：缺失则补下载
            files_err = ensure_model_files(size, precision,
                                           log_fn=_push_log, stop_event=_gpu_stop_event)
            if files_err:
                return {"ok": False, "results": {}, "ok_count": 0, "total": total,
                        "real_count": 0, "error": files_err}

            # 让出显存
            from ..llama_cpp import pause_for_task
            pause_for_task(log_fn=_push_log)

            # 启动推理子进程（与抽帧并行）
            stream = start_analyze_stream(
                total,
                skip_real=skip_real,
                anime_threshold=ANIME_CLS_THRESHOLD,
                tag_threshold=threshold,
                ip_threshold=threshold,
                input_size=size,
                stop_event=_gpu_stop_event,
                on_log=_push_log,
                on_video_result=None,
            )
            if stream is None:
                return self._pixai_engine_unavailable(
                    vid_list, video_paths, results, frames_n, total)

            # 抽帧流水线（线程池 + 有界背压 + send-ahead）
            frame_workers = min(6, max(1, total))
            max_pending = frame_workers * 4
            pipe_depth = 2  # 已 send 未收结果上限

            done = 0
            sent: Dict[int, Tuple[str, str]] = {}

            def _record_result(vr: Dict[str, Any]) -> None:
                """收下一条推理结果。"""
                nonlocal done
                done += 1
                vi = int(vr.get("index", -1))
                vid, name = sent.pop(vi, (None, f"视频{vi + 1}"))
                if vid is None:
                    return
                if vr.get("error"):
                    results[vid] = {"character_tags": [], "ip_tags": [],
                                    "error": str(vr["error"])}
                    _push_log(f"  [{done}/{total}] {name} — 分析失败: {vr['error']}", "err")
                else:
                    entry = {"character_tags": vr.get("character_tags", []),
                             "ip_tags": vr.get("ip_tags", []), "error": None}
                    # 附带分类信息（前端角标用）
                    if "anime_score" in vr:
                        entry["anime_score"] = vr["anime_score"]
                        entry["is_anime"] = vr.get("is_anime")  # None = 不确定
                    results[vid] = entry
                    if entry.get("is_anime") is False and skip_real:
                        _push_log(f"  [{done}/{total}] {name} — 非二次元作品"
                                  f"（{entry['anime_score']:.0%}），跳过标签获取")
                    else:
                        chars = ", ".join(t["name"] for t in entry["character_tags"]) or "无"
                        ips = ", ".join(t["name"] for t in entry["ip_tags"]) or "无"
                        _push_log(f"  [{done}/{total}] {name} — 角色: {chars} | IP: {ips}")
                js_pusher.push("setProgress", done, total)

            def _drain_ready() -> None:
                """机会性排空：收走已就绪结果。"""
                while True:
                    vr = stream.try_next_result()
                    if vr is None:
                        return  # 暂无就绪 / 引擎已终止
                    _record_result(vr)

            def _deliver_frame(f, i) -> bool:
                """处理一个完成的抽帧任务：有帧送推理，无帧记失败；停止/管道满/引擎终止返回 False。"""
                nonlocal done
                if _gpu_stop_event.is_set():
                    return False
                vid, vpath = vid_list[i], video_paths[i]
                name = Path(vpath).name
                # 先机会性排空
                _drain_ready()
                if stream.broken:
                    return False
                try:
                    frames = f.result()
                except Exception as e:
                    frames = None
                    _push_log(f"  [{done + 1}/{total}] {name} — 抽帧异常: {str(e)[:100]}", "err")
                if not frames:
                    done += 1
                    results[vid] = {"character_tags": [], "ip_tags": [], "error": "抽帧失败"}
                    _push_log(f"  [{done}/{total}] {name} — 抽帧失败", "err")
                    js_pusher.push("setProgress", done, total)
                    return True
                # 管道满 → 阻塞收取一条腾位
                while len(sent) >= pipe_depth:
                    vr = stream.next_result()
                    if vr is None:
                        return False
                    _record_result(vr)
                if not stream.send({"index": i, "name": name, "frames": frames}):
                    done += 1
                    results[vid] = {"character_tags": [], "ip_tags": [],
                                    "error": "推理中断"}
                    _push_log(f"  [{done}/{total}] {name} — 推理中断（引擎已退出）", "err")
                    js_pusher.push("setProgress", done, total)
                    return False
                sent[i] = (vid, name)
                del frames
                return True

            ex = ThreadPoolExecutor(max_workers=frame_workers,
                                    thread_name_prefix="pixai-frame")
            try:
                futures: Dict[Any, int] = {}
                pipeline_ok = True
                for i, vpath in enumerate(video_paths):
                    if _gpu_stop_event.is_set():
                        break
                    # 有界背压：先处理最快完成的一个
                    while len(futures) >= max_pending:
                        for f in wait(futures, return_when=FIRST_COMPLETED)[0]:
                            if not _deliver_frame(f, futures.pop(f)):
                                pipeline_ok = False
                        if not pipeline_ok:
                            break
                    if not pipeline_ok:
                        break
                    futures[ex.submit(extract_frames_for_tagger, vpath,
                                      _gpu_stop_event, frames_n, size)] = i
                # 排空剩余抽帧任务
                while futures and pipeline_ok:
                    for f in wait(futures, return_when=FIRST_COMPLETED)[0]:
                        if not _deliver_frame(f, futures.pop(f)):
                            pipeline_ok = False
                # 收尾排空：收取已 send 的结果
                while sent and pipeline_ok:
                    vr = stream.next_result()
                    if vr is None:
                        pipeline_ok = False
                        break
                    _record_result(vr)
            finally:
                ex.shutdown(wait=False, cancel_futures=True)

            # 收尾：关闭引擎并补记未出结果的视频
            close_result = stream.close()
            stopped = _gpu_stop_event.is_set()
            if stopped:
                _push_log("  分析已停止", "warn")
            elif not close_result["ok"]:
                err_txt = close_result.get("error") or "推理中断"
                for i, vid in enumerate(vid_list):
                    if vid not in results:
                        results[vid] = {"character_tags": [], "ip_tags": [],
                                        "error": err_txt}
                _push_log(f"  推理中断: {str(err_txt)[:200]}", "err")

            # 保存结果（取消/失败时保留已完成部分）
            self._save_pixai_tags(results)

            ok_count = sum(1 for r in results.values() if not r.get("error"))
            real_count = sum(1 for r in results.values() if r.get("is_anime") is False)
            if stopped:
                _push_log(f"IP分析已停止（完成 {ok_count}/{total}，结果已保存）", "warn")
                return {"ok": False, "results": results, "ok_count": ok_count,
                        "total": total, "real_count": real_count, "error": "已停止"}
            if not close_result["ok"]:
                err_txt = close_result.get("error") or "推理失败"
                _push_log(f"IP分析中止：{str(err_txt)[:200]}", "err")
                return {"ok": False, "results": results, "ok_count": ok_count,
                        "total": total, "real_count": real_count, "error": err_txt}
            _push_log(f"IP分析完成：成功 {ok_count}/{total}"
                      + (f"，已跳过 {real_count} 个非二次元视频"
                         if skip_real and real_count else "")
                      + f"（耗时 {time.time() - t0:.1f}s）")
            return {"ok": True, "results": results, "ok_count": ok_count,
                    "total": total, "real_count": real_count, "error": None}
        except Exception as e:
            # 异常路径收尾：关闭推理子进程
            if "stream" in locals():
                try:
                    stream.close()
                except Exception:
                    pass
            ok_count = sum(1 for r in results.values() if not r.get("error"))
            return {"ok": False, "results": results, "ok_count": ok_count,
                    "total": total, "real_count": 0, "error": str(e)}
        finally:
            # 恢复本地推理服务
            from ..llama_cpp import resume_after_task
            resume_after_task(log_fn=_push_log, server_log_fn=_push_llama_log)
            _gpu_task_lock.release()

    def _pixai_engine_unavailable(self, vid_list: List, video_paths: List,
                                  results: Dict[str, Any], frames_n: int,
                                  total: int) -> Dict[str, Any]:
        """引擎启动失败降级：仍尝试逐个抽帧并记录结果。"""
        from ..pixai_tagger import extract_frames_for_tagger
        _push_log("分析引擎启动失败，正在尝试抽帧…", "err")
        any_frames = False
        for i, vpath in enumerate(video_paths):
            if _gpu_stop_event.is_set():
                break
            vid, name = vid_list[i], Path(vpath).name
            try:
                frames = extract_frames_for_tagger(
                    vpath, _gpu_stop_event, frames_n)
            except Exception:
                frames = None
            if frames:
                any_frames = True
                results[vid] = {"character_tags": [], "ip_tags": [],
                                "error": "分析引擎启动失败"}
                _push_log(f"  [{i + 1}/{total}] {name} — 分析引擎启动失败", "err")
            else:
                results[vid] = {"character_tags": [], "ip_tags": [], "error": "抽帧失败"}
                _push_log(f"  [{i + 1}/{total}] {name} — 抽帧失败", "err")
            js_pusher.push("setProgress", i + 1, total)
        self._save_pixai_tags(results)
        if not any_frames:
            _push_log("IP分析完成（无可分析的视频，均已抽帧失败）", "warn")
            return {"ok": True, "results": results, "ok_count": 0,
                    "total": total, "real_count": 0, "error": None}
        _push_log("IP分析失败：分析引擎启动失败", "err")
        return {"ok": False, "results": results, "ok_count": 0,
                "total": total, "real_count": 0, "error": "分析引擎启动失败"}

    def get_pixai_tags(self, video_id: str) -> Dict[str, Any]:
        """获取指定视频的已保存 pixai 标签（附带 zh 中文翻译）。"""
        entry = _load_pixai_tags().get(video_id)
        if not entry:
            return {"ok": False, "character_tags": [], "ip_tags": []}
        data = {"ok": True, **entry}
        if load_pixai_config().get("output_zh", True):
            from ..pixai_tagger import load_tag_zh
            zh = load_tag_zh()
            if zh:
                for group in ("character_tags", "ip_tags"):
                    for t in data.get(group) or []:
                        z = zh.get(str(t.get("name") or ""))
                        if z:
                            t["zh"] = z
        return data

    def clear_pixai_tags(self) -> Dict[str, Any]:
        """清除所有已保存的 pixai 标签。"""
        write_json(PIXAI_TAGS_FILE, {})
        return {"ok": True, "message": "已清除所有IP标签数据"}

    def get_pixai_tagged_ids(self) -> List[str]:
        """返回已有 IP 标签数据的视频 ID 列表（供前端 IP 角标显示）。"""
        return [vid for vid, data in _load_pixai_tags().items()
                if data.get("character_tags") or data.get("ip_tags")]

    def get_pixai_real_ids(self) -> List[str]:
        """返回被预筛为非二次元的视频 ID 列表（供前端 REAL 角标显示）。"""
        return [vid for vid, data in _load_pixai_tags().items()
                if data.get("is_anime") is False]

    def get_pixai_anime_ids(self) -> List[str]:
        """返回被预筛为二次元的视频 ID 列表（供前端 ANIME 角标显示）。"""
        return [vid for vid, data in _load_pixai_tags().items()
                if data.get("is_anime") is True]

    def get_pixai_uncertain_ids(self) -> List[str]:
        """返回被预筛为不确定的视频 ID 列表（供前端 UNC 角标显示）。"""
        return [vid for vid, data in _load_pixai_tags().items()
                if "is_anime" in data and data["is_anime"] is None]

    def _save_pixai_tags(self, results: Dict[str, Any]) -> None:
        """将标签获取结果合并保存到 pixai/tags.json（失败条目不落盘）。"""
        def _mutate(current):
            if not isinstance(current, dict):
                current = {}
            for vid, data in results.items():
                if data.get("error"):
                    continue
                entry = {
                    "character_tags": data.get("character_tags", []),
                    "ip_tags": data.get("ip_tags", []),
                }
                # 保存分类信息（仅预筛过的视频携带）
                if "anime_score" in data:
                    entry["anime_score"] = data["anime_score"]
                if "is_anime" in data:
                    entry["is_anime"] = data["is_anime"]
                current[vid] = entry
            return current

        update_json(PIXAI_TAGS_FILE, _mutate, default_factory=dict)

    # ── Faster-Whisper 语音转录 ──

    def get_whisper_status(self) -> Dict[str, Any]:
        """获取 faster-whisper 安装状态与启用状态。"""
        from ..faster_whisper import get_status
        enabled = load_whisper_config().get("enabled", False)
        status = get_status()
        status["enabled"] = enabled
        return status

    def set_whisper_enabled(self, enabled: bool) -> Dict[str, Any]:
        """启用/禁用 faster-whisper 功能。"""
        from ..faster_whisper import WHISPER_DIR
        if not WHISPER_DIR.is_dir():
            return {"ok": True, "enabled": enabled}
        update_whisper_config(lambda c: c.update(enabled=enabled) or c)
        return {"ok": True, "enabled": enabled}

    def get_whisper_mirrors(self) -> Dict[str, Any]:
        """获取 faster-whisper 安装可选镜像（仅通用 PyPI）。"""
        from ..installer import detect_gpu, get_mirror_groups
        info = get_mirror_groups(["pypi"])
        info["gpu"] = detect_gpu()
        return info

    def install_whisper(self, pypi_mirror: str = DEFAULT_PYPI_MIRROR,
                        model: str = "v3-turbo") -> Dict[str, Any]:
        """安装 faster-whisper 依赖（uv + venv + packages + 所选模型）。"""
        from ..faster_whisper import install_dependencies

        if not _install_lock.acquire(blocking=False):
            return {"ok": False, "error": "已有模块正在安装，请等待完成后再试"}
        _install_stop_event.clear()
        try:
            r = install_dependencies(pypi_mirror=pypi_mirror, model=model,
                                     log_fn=_push_log,
                                     stop_event=_install_stop_event)
            if r.get("ok"):
                self._auto_clean_uv_cache()
            return r
        finally:
            _install_stop_event.clear()
            _install_lock.release()

    def download_whisper_model(self, model_key: str = "") -> Dict[str, Any]:
        """下载未安装的 whisper 模型（hf-mirror.com，取消时清理未完成文件）。"""
        from ..faster_whisper import _download_model

        if not model_key:
            return {"ok": False, "error": "未指定模型"}
        if not _install_lock.acquire(blocking=False):
            return {"ok": False, "busy": True, "error": "已有模块正在安装，请等待完成后再试"}

        def _progress(ev: Dict[str, Any]):
            js_pusher.push("whisperModelProgress", ev)

        try:
            result = _download_model(model_key, log_fn=_push_log,
                                     progress_cb=_progress)
        except Exception as e:
            result = {"ok": False, "error": f"下载过程异常: {e}"}
        finally:
            js_pusher.push("whisperModelDone", result)
            _install_lock.release()
        return result

    def remove_whisper(self) -> Dict[str, Any]:
        """删除 faster-whisper 文件夹与模块数据。"""
        from ..faster_whisper import remove_faster_whisper as _remove
        if not _install_lock.acquire(blocking=False):
            return {"ok": False, "error": "已有模块正在安装，请等待完成后再卸载"}
        try:
            return _remove()
        finally:
            _install_lock.release()

    def detect_speech(self, items: List) -> Dict[str, Any]:
        """对选中视频执行语音转录（items: [[video_id, video_path], ...]）。"""
        from ..faster_whisper import get_status, run_transcription_batch

        status = get_status()
        if not status["ready"]:
            return {"ok": False, "results": {}, "error": "faster-whisper 未安装或未就绪，请先在扩展功能页安装依赖"}

        cfg_exp = load_config().get("experimental", {})
        use_vad = cfg_exp.get("whisper_vad", True)
        language = cfg_exp.get("whisper_language", "")
        use_batch = cfg_exp.get("whisper_batch", False)
        # 转录并发（0 = 自动）
        workers = max(0, safe_int(cfg_exp.get("whisper_workers"), 0))
        beam_size = max(1, safe_int(cfg_exp.get("whisper_beam_size"), 5))
        cr_raw = cfg_exp.get("whisper_compression_ratio", 2.4)
        compression_ratio = None if cr_raw is None else safe_float(cr_raw, 2.4)

        if not _gpu_task_lock.acquire(blocking=False):
            return {"ok": False, "results": {}, "error": "已有 GPU 任务在运行，请等待完成后再试"}
        _gpu_stop_event.clear()
        if _gpu_stop_event.is_set():
            _gpu_task_lock.release()
            return {"ok": False, "results": {}, "error": "已停止"}

        vid_list = [item[0] for item in items]
        video_paths = [item[1] for item in items]
        total = len(video_paths)
        results: Dict[str, Any] = {}

        try:
            _push_log(f"正在转录 {total} 个视频的语音…")
            js_pusher.push("setProgress", 0, total)

            # 让出显存
            from ..llama_cpp import pause_for_task
            pause_for_task(log_fn=_push_log)

            t0 = time.time()

            # 进度按完成数计
            done = 0

            def _on_done(idx: int, entry: Dict[str, Any]):
                nonlocal done
                done += 1
                name = Path(video_paths[idx]).name if idx < len(video_paths) else f"视频{idx+1}"
                if entry.get("ok"):
                    seg_count = entry.get("srt", "").count("-->")
                    _push_log(f"  [{done}/{total}] {name} — 完成 {seg_count} 段，"
                              f"语言={entry.get('language') or '未知'}")
                else:
                    err_txt = str(entry.get("error", ""))
                    _push_log(f"  [{done}/{total}] {name} — 转录失败"
                              + (f": {err_txt[:200]}" if err_txt else ""), "err")
                    if err_txt and any(k in err_txt.lower()
                                       for k in ("out of memory", "cuda", "oom", "alloc")):
                        _push_log("    （显存不足，可尝试减少并发或关闭批处理模式）", "warn")
                js_pusher.push("setProgress", done, total)

            batch_result = run_transcription_batch(
                video_paths, vad=use_vad, language=language, batch=use_batch,
                workers=workers,
                beam_size=beam_size, compression_ratio=compression_ratio,
                on_video_done=_on_done, on_log=_push_log,
                stop_event=_gpu_stop_event,
            )
            elapsed = time.time() - t0

            per_video = batch_result.get("per_video") or []
            has_done = any((tr or {}).get("ok") for tr in per_video)
            if not batch_result["ok"]:
                err_txt = str(batch_result.get("error") or "")
                if not has_done:
                    _push_log(f"  转录失败: {err_txt[:200]}", "err")
                    return {"ok": False, "results": {}, "error": batch_result.get("error")}
                _push_log(f"  转录中止（{err_txt}），保留已完成的 {sum(1 for tr in per_video if (tr or {}).get('ok'))} 个结果", "warn")

            for idx, vid in enumerate(vid_list):
                tr = per_video[idx] if idx < len(per_video) else {"ok": False, "srt": "", "error": "无结果"}
                if not tr.get("ok"):
                    results[vid] = {"srt": "", "error": tr.get("error", "转录失败")}
                else:
                    results[vid] = {"srt": tr["srt"], "language": tr.get("language", ""), "error": None}

            if batch_result["ok"]:
                _push_log(f"  转录完成（{total} 个视频，总耗时 {elapsed:.1f}s）")

            # 保存 SRT 文件 + 索引
            self._save_whisper_srt(results)
            ok_count = sum(1 for r in results.values() if not r.get("error"))
            if not batch_result["ok"]:
                # 「已取消」归一为「已停止」
                err_txt = "已停止" if str(batch_result.get("error") or "") == "已取消" \
                    else batch_result.get("error")
                return {"ok": False, "results": results, "ok_count": ok_count,
                        "total": total, "error": err_txt}
            return {"ok": True, "results": results, "ok_count": ok_count, "total": total, "error": None}
        except Exception as e:
            ok_count = sum(1 for r in results.values() if not r.get("error"))
            return {"ok": False, "results": results, "ok_count": ok_count,
                    "total": total, "error": str(e)}
        finally:
            # 恢复本地推理服务
            from ..llama_cpp import resume_after_task
            resume_after_task(log_fn=_push_log, server_log_fn=_push_llama_log)
            _gpu_task_lock.release()

    def get_whisper_transcript(self, video_id: str) -> Dict[str, Any]:
        """获取指定视频的转录内容（读 SRT 文件）。"""
        srt_file = WHISPER_SRT_DIR / f"{video_id}.srt"
        if srt_file.exists():
            try:
                srt_text = srt_file.read_text(encoding="utf-8")
            except OSError:
                return {"ok": False, "text": ""}
            store = read_json(WHISPER_TRANSCRIPTS_FILE, {})
            lang = store.get(video_id, {}).get("language", "")
            return {"ok": True, "text": srt_text, "language": lang}
        return {"ok": False, "text": ""}

    def clear_whisper_transcripts(self) -> Dict[str, Any]:
        """清除所有转录数据（SRT 文件 + 索引）。"""
        write_json(WHISPER_TRANSCRIPTS_FILE, {})
        if WHISPER_SRT_DIR.exists():
            shutil.rmtree(WHISPER_SRT_DIR, ignore_errors=True)
        return {"ok": True, "message": "已清除所有语音转录数据"}

    def get_whisper_transcribed_ids(self) -> List[str]:
        """返回已有转录的视频 ID 列表（轻量，供前端角标显示）。"""
        store = read_json(WHISPER_TRANSCRIPTS_FILE, {})
        return list(store.keys())

    def _save_whisper_srt(self, results: Dict[str, Any]) -> None:
        """保存 SRT 文件 + 轻量索引 JSON（索引只存语言）。"""
        WHISPER_SRT_DIR.mkdir(parents=True, exist_ok=True)
        langs = {}
        for vid, data in results.items():
            if data.get("error"):
                continue
            try:
                (WHISPER_SRT_DIR / f"{vid}.srt").write_text(data.get("srt", ""), encoding="utf-8")
            except OSError:
                continue
            langs[vid] = {"language": data.get("language", "")}

        def _mutate(current):
            if not isinstance(current, dict):
                current = {}
            current.update(langs)
            return current

        update_json(WHISPER_TRANSCRIPTS_FILE, _mutate, default_factory=dict)

    def export_srt(self, items: List) -> Dict[str, Any]:
        """导出 SRT 字幕到视频同目录（items: [[video_id, video_path], ...]）。"""
        exported = 0
        errors = []
        for item in items:
            vid, vpath = item[0], item[1]
            src = WHISPER_SRT_DIR / f"{vid}.srt"
            if not src.exists():
                errors.append(f"{Path(vpath).name}: 无转录数据")
                continue
            dest = Path(vpath).with_suffix('.srt')
            try:
                shutil.copy2(src, dest)
                exported += 1
            except Exception as e:
                errors.append(f"{Path(vpath).name}: {e}")
        return {"ok": exported > 0, "exported": exported, "errors": errors}

    def export_srt_translated(self, items: List) -> Dict[str, Any]:
        """导出 SRT 字幕并翻译为中文（.zh.srt，items: [[video_id, video_path], ...]）。"""
        from ..srt_translate import translate_srt_file, TranslationCancelled

        if not _gpu_task_lock.acquire(blocking=False):
            return {"ok": False, "exported": 0, "errors": ["已有 GPU 任务在运行，请等待完成后再试"]}
        _gpu_stop_event.clear()
        if _gpu_stop_event.is_set():
            _gpu_task_lock.release()
            return {"ok": False, "exported": 0, "errors": ["已停止"]}

        try:
            cfg = load_config()
            ai_cfg = cfg.get("ai", {})

            # 本地推理集成：翻译改走本地 llama-server
            from ..llama_integration import ai_override
            ov = ai_override(cfg)
            if ov:
                ai_cfg = {**ai_cfg, **ov}

            api_key = ai_cfg.get("api_key", "")
            base_url = ai_cfg.get("base_url", "")
            model = ai_cfg.get("model", "")
            workers = max(1, safe_int(ai_cfg.get("ai_workers"), 4))
            if not base_url:
                return {"ok": False, "exported": 0, "errors": ["未配置 AI 服务地址（请先在 AI 配置页填写）"]}

            tasks = []  # [(idx, vid, vpath, src, dest)]
            errors = []
            for idx, item in enumerate(items):
                vid, vpath = item[0], item[1]
                src = WHISPER_SRT_DIR / f"{vid}.srt"
                name = Path(vpath).name
                if not src.exists():
                    errors.append(f"{name}: 无转录数据")
                    _push_log(f"  [{idx+1}/{len(items)}] {name} — 无转录数据", "err")
                    continue
                dest = Path(vpath).with_suffix('.zh.srt')
                tasks.append((idx, vid, vpath, str(src), str(dest)))

            if not tasks:
                return {"ok": False, "exported": 0, "errors": errors}

            total = len(items)
            _push_log(f"正在翻译 {len(tasks)} 个字幕文件（单文件 {workers} 并发）…")

            exported = 0
            cancelled = False
            for idx, vid, vpath, src, dest in tasks:
                if _gpu_stop_event.is_set():
                    _push_log(f"  翻译已停止，剩余 {len(tasks) - idx} 个文件未处理", "warn")
                    cancelled = True
                    break
                name = Path(vpath).name
                try:
                    translate_srt_file(src, dest, api_key, base_url, model,
                                       workers=workers, log_fn=_push_log,
                                       stop_event=_gpu_stop_event)
                    exported += 1
                    _push_log(f"  [{idx+1}/{total}] {name} — 已导出 {Path(dest).name}")
                except TranslationCancelled:
                    _push_log(f"  翻译已停止，已完成 {exported}/{total}", "warn")
                    cancelled = True
                    break
                except Exception as e:
                    errors.append(f"{name}: {e}")
                    _push_log(f"  [{idx+1}/{total}] {name} — 翻译失败: {str(e)[:100]}", "err")

            _push_log(f"  翻译{'中止' if cancelled else '完成'}：成功 {exported}/{total}")
            return {"ok": exported > 0, "exported": exported, "errors": errors,
                    **({"cancelled": True} if cancelled else {})}
        finally:
            _gpu_task_lock.release()

    # ── llama.cpp 本地推理 ──

    def get_llama_status(self) -> Dict[str, Any]:
        """获取 llama.cpp 安装/运行状态与模型列表。"""
        from ..llama_cpp import get_status, DEFAULTS, ensure_model_configs
        st = get_status()
        ensure_model_configs(st.get("models") or [])
        llama_cfg = load_llama_config()
        st["config"] = {
            **DEFAULTS,
            **llama_cfg,
            "integrate": bool(llama_cfg.get("integrate", False)),
        }
        st["model_configs"] = load_llama_model_configs()
        st["defaults"] = DEFAULTS
        st["enabled"] = bool(llama_cfg.get("enabled", False))
        return st

    def get_llama_releases(self, force: bool = False) -> Dict[str, Any]:
        """获取最新 release 的构建列表（含推荐版本）。force 忽略缓存。"""
        from ..llama_cpp import get_latest_release_assets
        try:
            return get_latest_release_assets(force=force)
        except Exception as e:
            return {"ok": False, "error": f"获取 llama.cpp 发布信息失败: {e}"}

    def install_llama(self, build_sel: str = "", proxy: str = "") -> Dict[str, Any]:
        """安装 llama.cpp。build_sel: manual（仅建目录）/ cuda-<ver>（自动下载）。"""
        from ..llama_cpp import install as _install

        if not build_sel:
            return {"ok": False, "error": "未指定构建"}
        if not _install_lock.acquire(blocking=False):
            return {"ok": False, "error": "已有模块正在安装，请等待完成后再试"}
        _install_stop_event.clear()
        try:
            return _install(build_sel, log_fn=_push_log, proxy=proxy,
                            stop_event=_install_stop_event)
        except Exception as e:
            return {"ok": False, "error": f"安装过程异常: {e}"}
        finally:
            _install_stop_event.clear()
            _install_lock.release()

    def remove_llama(self) -> Dict[str, Any]:
        """卸载 llama.cpp（先停止服务再删目录）。"""
        from ..llama_cpp import remove as _remove

        if not _install_lock.acquire(blocking=False):
            return {"ok": False, "error": "已有模块正在安装，请等待完成后再卸载"}
        try:
            r = _remove(log_fn=_push_log)
            if r.get("ok"):
                shutil.rmtree(LLAMA_CONFIG_FILE.parent, ignore_errors=True)
            return r
        finally:
            _install_lock.release()

    def scan_llama_models(self) -> Dict[str, Any]:
        """扫描模型文件夹下的 .gguf 模型；新模型同时初始化 per-model 配置。"""
        from ..llama_cpp import scan_models, get_models_dir, ensure_model_configs
        models = scan_models(force=True)
        ensure_model_configs(models)
        return {"ok": True, "models": models, "models_dir": str(get_models_dir())}

    def launch_llama(self, model_path: str = "", params: Dict[str, Any] = None) -> Dict[str, Any]:
        """启动 llama-server。model_path 为空时自动选取模型文件夹中唯一模型。"""
        from ..llama_cpp import launch as _launch

        params = dict(params or {})

        r = _launch(model_path or "", params, log_fn=_push_log,
                    server_log_fn=_push_llama_log)
        if r.get("ok"):
            # 记录本次成功运行的模型
            _record_last_model(r.get("model") or model_path or "")
        return r

    def _llama_autostart_target(self) -> str:
        """自动启动的模型选择：上次成功运行的模型 → 设置页选中的模型 → 空（交给 launch 自动选）。"""
        llama_cfg = load_llama_config() or {}
        for key in ("last_model", "model"):
            target = str(llama_cfg.get(key) or "")
            if target and Path(target).is_file():
                return target
        return ""

    def _llama_autostart_params(self, target: str) -> Dict[str, Any]:
        """自动启动参数：全局运行参数 + 目标模型 per-model 配置叠加（剔除非运行键）。"""
        from ..llama_cpp import _GLOBAL_ONLY_KEYS
        llama = load_llama_config() or {}
        params = {k: v for k, v in llama.items() if k not in _GLOBAL_ONLY_KEYS}
        mcfg = load_llama_model_configs().get(target, {})
        params.update({k: v for k, v in mcfg.items() if v is not None})
        return params

    def auto_run_llama(self) -> Dict[str, Any]:
        """程序启动时按开关自动启动上次使用的模型。"""
        from ..llama_cpp import launch as _launch, get_status

        cfg = load_config().get("experimental", {})
        if not cfg.get("llama_enabled", False):
            return {"ok": False, "skipped": "module_disabled"}
        llama_cfg = load_llama_config() or {}
        if not llama_cfg.get("auto_run", False):
            return {"ok": False, "skipped": "auto_run_off"}
        st = get_status()
        if not st.get("ready"):
            return {"ok": False, "skipped": "not_installed"}
        if st.get("running"):
            return {"ok": False, "skipped": "already_running"}

        target = self._llama_autostart_target()
        _push_log("检测到自动运行已开启，正在启动本地推理服务…")
        r = _launch(target, self._llama_autostart_params(target), log_fn=_push_log,
                    server_log_fn=_push_llama_log)
        if r.get("ok"):
            _push_log("自动运行启动成功。")
            _record_last_model(r.get("model") or target or "")
        else:
            _push_log(f"自动运行启动失败：{r.get('error', '未知错误')}")
        return r

    def ensure_llama_running(self) -> Dict[str, Any]:
        """「开始自动重命名」前置保障（本地推理集成模式）：服务未运行时自动拉起。"""
        from ..llama_cpp import launch as _launch, get_status

        cfg = load_config().get("experimental", {})
        if not cfg.get("llama_enabled", False):
            return {"ok": False, "error": "llama.cpp 总开关已关闭，无法自动启动服务"}

        st = get_status()
        if st.get("running"):
            return {"ok": True, "ready": True}
        if st.get("starting"):
            return {"ok": True, "ready": False, "starting": True}
        if not st.get("ready"):
            return {"ok": False, "error": "llama.cpp 未安装，请先在扩展功能页安装"}

        target = self._llama_autostart_target()
        _push_log("本地推理服务未运行，「开始自动重命名」触发自动启动…")
        r = _launch(target, self._llama_autostart_params(target), log_fn=_push_log,
                    server_log_fn=_push_llama_log)
        if r.get("ok"):
            _push_log("本地推理服务已自动启动。")
            _record_last_model(r.get("model") or target or "")
        else:
            _push_log(f"本地推理服务自动启动失败：{r.get('error', '未知错误')}")
        return r

    def stop_llama(self) -> Dict[str, Any]:
        """停止运行中的 llama-server。"""
        from ..llama_cpp import stop as _stop
        return _stop(log_fn=_push_log, grace=2.0)

    def open_llama_webui(self) -> Dict[str, Any]:
        """用默认浏览器打开 llama-server 自带 webui 聊天界面（需服务运行中）。"""
        import webbrowser
        from ..llama_cpp import get_status, _local_host

        st = get_status()
        if not st.get("running"):
            return {"ok": False, "error": "本地推理服务未运行，请先启动服务"}
        llama_cfg = load_llama_config() or {}
        host = _local_host(llama_cfg.get("host") or "127.0.0.1")
        port = st.get("port") or llama_cfg.get("port") or 8080
        url = f"http://{host}:{port}/"
        try:
            webbrowser.open(url)
            return {"ok": True, "url": url}
        except Exception as e:
            return {"ok": False, "error": f"打开浏览器失败: {e}"}

    def set_llama_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """保存 llama.cpp 配置到独立文件（增量合并：仅更新传入的键）。"""
        from ..llama_cpp import (apply_model_configs,
                                 purge_stale_after_dir_change, DEFAULTS,
                                 _GLOBAL_ONLY_KEYS)

        cfg = dict(cfg or {})
        integrate = cfg.pop("integrate", None)

        old_dir = new_dir = None

        def _mutate(c):
            nonlocal old_dir, new_dir
            old_dir = str(c.get("models_dir") or "")
            for k, v in cfg.items():
                if v is not None and k in _GLOBAL_ONLY_KEYS:
                    c[k] = v
            new_dir = str(c.get("models_dir") or "")
            return c

        update_llama_config(_mutate)
        if integrate is not None:
            update_llama_config(lambda c: c.update(integrate=bool(integrate)) or c)
        # 模型目录变更：清理旧目录的 per-model 配置与 last_model
        if old_dir is not None and old_dir != new_dir:
            purge_stale_after_dir_change(old_dir, new_dir)
        apply_model_configs(cfg or {})
        final = load_llama_config()
        cfg_full = {
            **DEFAULTS,
            **final,
            "integrate": bool(final.get("integrate", False)),
        }
        return {"ok": True, "config": cfg_full}

    def set_llama_enabled(self, enabled: bool) -> Dict[str, Any]:
        """启用/禁用 llama.cpp 功能（控制设置页「本地推理」页签是否显示）。"""
        from ..llama_cpp import LLAMA_DIR, stop as _stop
        if not LLAMA_DIR.is_dir():
            return {"ok": True, "enabled": enabled}
        if not enabled:
            r = _stop(log_fn=_push_log, grace=2.0)
            if not r.get("ok"):
                return {"ok": False, "error": f"停止本地推理服务失败: {r.get('error', '未知错误')}"}
        update_llama_config(lambda c: c.update(enabled=enabled) or c)
        return {"ok": True, "enabled": enabled}
