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
from ..js_push import js_pusher
from ..workspace_paths import (LLAMA_CONFIG_FILE, PIXAI_TAGS_FILE,
                               WHISPER_SRT_DIR, WHISPER_TRANSCRIPTS_FILE)
from ..workspace_store import read_json, update_json, write_json
from batch_rename.utils import safe_float, safe_int

# 后端互斥：pywebview 每次调用新开线程，前端标志拦不住并发 RPC
# GPU 密集任务（转录/标签获取/翻译）互斥：同刻只允许一个
_gpu_task_lock = threading.Lock()
# 安装基础设施互斥：安装/卸载/清缓存同刻只允许一个
# （卸载与安装并发会互相破坏：装 llama 时删目录、装包时删 UV/缓存）
_install_lock = threading.Lock()
# GPU 任务取消事件：开始前 clear，stop_gpu_task 置位；
# 抽帧/子进程推理/翻译分批在检查点响应（单步完成后才生效）
_gpu_stop_event = threading.Event()
# 安装取消事件：开始前 clear，stop_install 置位；
# 子进程由 run_subprocess_streaming 终止，下载监视线程即时断连，秒级生效
_install_stop_event = threading.Event()


def _push_log(msg: str, level: str = "info") -> None:
    js_pusher.push("appendLog", msg, level)


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

    # ── GPU 任务停止（转录 / 标签获取 / 翻译共用） ──

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
        # 联动模型下载取消：模型管理弹窗的下载监听 downloader 全局取消事件，
        # 让主界面「停止」按钮对下载任务同样生效；无下载任务时置位无副作用
        try:
            from .. import models_downloader
            models_downloader.set_cancel()
        except Exception:
            pass
        _push_log("正在停止安装…")
        return {"ok": True}

    # ── UV 包管理工具（whisper / pixai 共享的安装基础设施） ──

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

    # ── 扩展功能页聚合状态（单次桥接调用，替代前端 5 个并行 RPC） ──

    def get_experimental_status(self) -> Dict[str, Any]:
        """扩展功能页一次拉齐全部状态与配置。"""
        return {
            "cfg": self.get_config(),
            "uv": self.get_uv_status(),
            "llama": self.get_llama_status(),
            "pixai": self.get_pixai_tagger_status(),
            "whisper": self.get_whisper_status(),
        }

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
        """获取 pixai-tagger 安装可选镜像（PyTorch CUDA + 通用 PyPI + GPU 检测）。"""
        from ..pixai_tagger import get_mirrors_info
        return get_mirrors_info()

    def install_pixai_tagger(self, pytorch_mirror: str = "nju-cu128",
                             pypi_mirror: str = "nju") -> Dict[str, Any]:
        """安装 pixai-tagger 依赖（uv + venv + torch/timm + 两个模型）。"""
        from ..pixai_tagger import install_dependencies

        if not _install_lock.acquire(blocking=False):
            return {"ok": False, "error": "已有模块正在安装，请等待完成后再试"}
        _install_stop_event.clear()
        try:
            return install_dependencies(pytorch_mirror=pytorch_mirror,
                                        pypi_mirror=pypi_mirror, log_fn=_push_log,
                                        stop_event=_install_stop_event)
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
        """对选中视频执行 PixAI 标签获取。
        items: [[video_id, video_path], ...]，id 为前端视频对象的稳定 ID
        """
        from ..pixai_tagger import get_status, start_analyze_stream, ANIME_CLS_THRESHOLD
        from ..pixai_frames import extract_frames_for_tagger
        from batch_rename.dependencies import ffmpeg_tools
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

        status = get_status()
        if not status["ready"]:
            return {"ok": False, "results": {}, "error": "pixai-tagger 未安装或未就绪，请先在扩展功能页安装依赖"}

        pcfg = load_pixai_config()
        frames_n = max(1, safe_int(pcfg.get("frames"), 15))
        short_side = max(64, safe_int(pcfg.get("short_side"), 448))
        crop_square = bool(pcfg.get("crop_square", False))
        crop_portrait = bool(pcfg.get("crop_portrait", False))
        threshold = min(0.99, max(0.5, safe_float(pcfg.get("threshold"), 0.9)))
        skip_real = bool(pcfg.get("classify", False))

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

            # 让出显存：先停止本地推理服务，任务完成后恢复（两者都占用大量显存）
            from ..llama_cpp import pause_for_task
            pause_for_task(log_fn=_push_log)

            # 第一步（与抽帧并行）：立即启动推理子进程，加载双模型
            stream = start_analyze_stream(
                total,
                skip_real=skip_real,
                anime_threshold=ANIME_CLS_THRESHOLD,
                tag_threshold=threshold,
                stop_event=_gpu_stop_event,
                on_log=_push_log,
                on_video_result=None,  # 结果统一在主循环取（next_result），避免跨线程进度竞争
            )
            if stream is None:
                return self._pixai_engine_unavailable(
                    vid_list, video_paths, results, frames_n,
                    short_side, crop_square, crop_portrait, total)

            # 第二步（流水线）：抽帧线程池 + 有界背压 + send-ahead 管道
            # （抽帧为 ffmpeg 等待型 IO，CPU 线程池即可；并发 6 路喂饱管道）
            frame_workers = min(6, max(1, total))
            max_pending = frame_workers * 4
            pipe_depth = 2  # 已 send 未收结果的上限（子进程预取队列深度也是 2）

            done = 0
            sent: Dict[int, Tuple[str, str]] = {}

            def _record_result(vr: Dict[str, Any]) -> None:
                """收下一条推理结果：记录 + 日志 + 进度（done 单调递增）。"""
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
                    # 预筛模型可用时附带分类信息（供前端 ANIME/REAL/UNC 角标）
                    if "anime_score" in vr:
                        entry["anime_score"] = vr["anime_score"]
                        entry["is_anime"] = vr.get("is_anime")  # None = 不确定
                    results[vid] = entry
                    if entry.get("is_anime") is False and skip_real:
                        # 开关开启时非二次元视频确实未做标签获取（子进程直接跳过）
                        _push_log(f"  [{done}/{total}] {name} — 非二次元作品"
                                  f"（{entry['anime_score']:.0%}），跳过标签获取")
                    else:
                        chars = ", ".join(t["name"] for t in entry["character_tags"]) or "无"
                        ips = ", ".join(t["name"] for t in entry["ip_tags"]) or "无"
                        _push_log(f"  [{done}/{total}] {name} — 角色: {chars} | IP: {ips}")
                js_pusher.push("setProgress", done, total)

            def _drain_ready() -> None:
                """机会性排空：收走所有已就绪结果（无就绪时即刻返回，不阻塞抽帧）。"""
                while True:
                    vr = stream.try_next_result()
                    if vr is None:
                        return  # 暂无就绪 / 引擎已终止（broken 已置位）
                    _record_result(vr)

            def _deliver_frame(f, i) -> bool:
                """处理一个完成的抽帧任务：无帧 → 失败记步；有帧 → send-ahead
                （送出即返回不等结果）；管道满或引擎终止时返回 False。"""
                nonlocal done
                vid, vpath = vid_list[i], video_paths[i]
                name = Path(vpath).name
                # 先机会性排空：及时收已完成结果（进度实时 + 为管道腾位）
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
                # 管道满 → 阻塞收取一条腾位（响应顺序 = 请求顺序）；
                # 停止/超时/引擎退出返回 None，剩余已送视频由 close 收尾补记
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
                del frames  # 已送出：父进程侧帧引用即刻释放（子进程处理完也释放）
                return True

            ex = ThreadPoolExecutor(max_workers=frame_workers,
                                    thread_name_prefix="pixai-frame")
            try:
                futures: Dict[Any, int] = {}
                pipeline_ok = True
                for i, vpath in enumerate(video_paths):
                    if _gpu_stop_event.is_set():
                        break
                    # 有界背压：在途达到上限时先处理最快完成的一个（帧即送出/释放）
                    while len(futures) >= max_pending:
                        for f in wait(futures, return_when=FIRST_COMPLETED)[0]:
                            if not _deliver_frame(f, futures.pop(f)):
                                pipeline_ok = False
                        if not pipeline_ok:
                            break
                    if not pipeline_ok:
                        break
                    futures[ex.submit(extract_frames_for_tagger, vpath,
                                      _gpu_stop_event,
                                      frames_n, short_side, crop_square,
                                      crop_portrait)] = i
                # 排空剩余抽帧任务（流水线中断后不再送推理，在途任务直接丢弃）
                while futures and pipeline_ok:
                    for f in wait(futures, return_when=FIRST_COMPLETED)[0]:
                        if not _deliver_frame(f, futures.pop(f)):
                            pipeline_ok = False
                # 收尾排空：已 send 的结果全部收取（引擎中断则剩余由 close 补记）
                while sent and pipeline_ok:
                    vr = stream.next_result()
                    if vr is None:
                        pipeline_ok = False
                        break
                    _record_result(vr)
            finally:
                ex.shutdown(wait=False, cancel_futures=True)

            # 收尾：关闭 stdin 触发引擎正常退出；异常时补记未出结果的视频
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

            # 保存结果到 pixai/tags.json（取消/失败时已完成的部分同样保留）
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
                      # 开关开时 REAL 视频未获取标签，汇总其数量；
                      # 关闭时不单列（前端 toast 已提示非二次元数量）
                      + (f"，已跳过 {real_count} 个非二次元视频"
                         if skip_real and real_count else "")
                      + f"（耗时 {time.time() - t0:.1f}s）")
            return {"ok": True, "results": results, "ok_count": ok_count,
                    "total": total, "real_count": real_count, "error": None}
        except Exception as e:
            # 异常路径也要收尾：关闭推理子进程（卸载模型/释放显存/删 params 临时文件）
            if "stream" in locals():
                try:
                    stream.close()
                except Exception:
                    pass
            ok_count = sum(1 for r in results.values() if not r.get("error"))
            return {"ok": False, "results": results, "ok_count": ok_count,
                    "total": total, "real_count": 0, "error": str(e)}
        finally:
            # 恢复本地推理服务（按原参数重新加载）
            from ..llama_cpp import resume_after_task
            resume_after_task(log_fn=_push_log)
            _gpu_task_lock.release()

    def _pixai_engine_unavailable(self, vid_list: List, video_paths: List,
                                  results: Dict[str, Any], frames_n: int,
                                  short_side: int, crop_square: bool,
                                  crop_portrait: bool,
                                  total: int) -> Dict[str, Any]:
        """引擎启动失败降级：仍尝试抽帧——全部抽帧失败时结果仅抽帧失败（ok）；
        有抽帧成功视频时整体失败（该视频无法分析）。"""
        from ..pixai_frames import extract_frames_for_tagger
        _push_log("分析引擎启动失败，正在尝试抽帧…", "err")
        any_frames = False
        for i, vpath in enumerate(video_paths):
            if _gpu_stop_event.is_set():
                break
            vid, name = vid_list[i], Path(vpath).name
            try:
                frames = extract_frames_for_tagger(
                    vpath, _gpu_stop_event, frames_n, short_side,
                    crop_square, crop_portrait)
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
        """获取指定视频的已保存 pixai 标签。"""
        entry = _load_pixai_tags().get(video_id)
        if entry:
            return {"ok": True, **entry}
        return {"ok": False, "character_tags": [], "ip_tags": []}

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

    def install_whisper(self, pypi_mirror: str = "nju",
                        model: str = "v3-turbo") -> Dict[str, Any]:
        """安装 faster-whisper 依赖（uv + venv + packages + 所选模型）。"""
        from ..faster_whisper import install_dependencies

        if not _install_lock.acquire(blocking=False):
            return {"ok": False, "error": "已有模块正在安装，请等待完成后再试"}
        _install_stop_event.clear()
        try:
            return install_dependencies(pypi_mirror=pypi_mirror, model=model,
                                        log_fn=_push_log,
                                        stop_event=_install_stop_event)
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
        """对选中视频执行语音转录（模型只加载一次，逐视频流式回推进度）。
        items: [[video_id, video_path], ...]，id 为前端视频对象的稳定 ID
        """
        from ..faster_whisper import get_status, run_transcription_batch

        status = get_status()
        if not status["ready"]:
            return {"ok": False, "results": {}, "error": "faster-whisper 未安装或未就绪，请先在扩展功能页安装依赖"}

        cfg_exp = load_config().get("experimental", {})
        use_vad = cfg_exp.get("whisper_vad", True)
        language = cfg_exp.get("whisper_language", "")
        use_batch = cfg_exp.get("whisper_batch", False)
        # 视频间转录并发（0 = 自动：GPU 4 路 / CPU 串行；显式值由脚本按视频数收敛）
        workers = max(0, safe_int(cfg_exp.get("whisper_workers"), 0))

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

            # 让出显存：先停止本地推理服务，任务完成后恢复（两者都占用大量显存）
            from ..llama_cpp import pause_for_task
            pause_for_task(log_fn=_push_log)

            t0 = time.time()

            # 进度按完成数计（并发下完成顺序不定，用视频序号做前缀会显得乱）；
            # 回调来自单一读行线程串行触发，计数器无需加锁
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

            # 保存 SRT 文件 + 轻量索引（取消/失败时已完成的部分同样保留）
            self._save_whisper_srt(results)
            ok_count = sum(1 for r in results.values() if not r.get("error"))
            if not batch_result["ok"]:
                # 「已取消」归一为「已停止」，与 IP 分析一致，前端据此展示部分完成
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
            # 恢复本地推理服务（按原参数重新加载）
            from ..llama_cpp import resume_after_task
            resume_after_task(log_fn=_push_log)
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
        """导出 SRT 字幕到视频同目录。
        items: [[video_id, video_path], ...]，id 为原始 stable_id（重命名前）
        """
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
        """导出 SRT 字幕并翻译为中文（.zh.srt）。
        items: [[video_id, video_path], ...]，id 为原始 stable_id（重命名前）
        """
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

            # 本地推理集成：开启后翻译改走本地 llama-server（磁盘配置不变）
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
        # 「显示运行日志」是扩展功能页卡片的偏好，不在启动参数面板里——启动时自动并入
        llama_cfg = load_llama_config() or {}
        params.setdefault("show_logs", llama_cfg.get("show_logs", False))

        r = _launch(model_path or "", params, log_fn=_push_log)
        if r.get("ok"):
            # 记录本次成功运行的模型（用 launch 解析后的实际路径，含自动选择的情况）
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
        # show_logs 存在全局但属启动参数（与 launch_llama 的 setdefault 语义一致）
        params["show_logs"] = bool(llama.get("show_logs", False))
        mcfg = load_llama_model_configs().get(target, {})
        params.update({k: v for k, v in mcfg.items() if v is not None})
        return params

    def auto_run_llama(self) -> Dict[str, Any]:
        """程序启动时调用：若启用模块且开启 auto_run，且已安装，则自动启动
        「上次成功运行的模型」（无记录时回退设置页选中的模型）。"""
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
        r = _launch(target, self._llama_autostart_params(target), log_fn=_push_log)
        if r.get("ok"):
            _push_log("自动运行启动成功。")
            _record_last_model(r.get("model") or target or "")
        else:
            _push_log(f"自动运行启动失败：{r.get('error', '未知错误')}")
        return r

    def ensure_llama_running(self) -> Dict[str, Any]:
        """「开始处理」前置保障（本地推理集成模式）：服务未运行时自动拉起。"""
        from ..llama_cpp import launch as _launch, get_status

        # 防御性兜底：总开关关闭时禁止自动拉起（与 auto_run_llama 判定一致）
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
        _push_log("本地推理服务未运行，「开始处理」触发自动启动…")
        r = _launch(target, self._llama_autostart_params(target), log_fn=_push_log)
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
        from ..llama_cpp import get_status

        st = get_status()
        if not st.get("running"):
            return {"ok": False, "error": "本地推理服务未运行，请先启动服务"}
        llama_cfg = load_llama_config() or {}
        host = llama_cfg.get("host") or "127.0.0.1"
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"  # 监听所有网卡时，浏览器同样走本机回环
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
