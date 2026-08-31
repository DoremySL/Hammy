"""ModelDownloadMixin — HF-Mirror 模型搜索/下载 API（本地推理页「下载模型」）。"""
from __future__ import annotations

from typing import Any, Dict, List

from ..js_push import js_pusher


class ModelDownloadMixin:
    """HF-Mirror 模型搜索与下载相关 API。"""

    def hf_search_models(self, keyword: str = "", sort: str = "trendingScore",
                         limit: int = 20, cursor: str = "") -> Dict[str, Any]:
        """搜索模型仓库（hf-mirror.com）。cursor: 上一页的 next_cursor，空串为第一页。"""
        from .. import hf_downloader
        try:
            results, next_cursor = hf_downloader.api_search_models(
                keyword, sort=sort, limit=limit, cursor=cursor)
            return {"ok": True, "results": results,
                    "next_cursor": next_cursor or "",
                    "has_more": bool(next_cursor) and bool(results)}
        except Exception as e:
            return {"ok": False, "error": f"搜索失败: {e}"}

    def hf_open_repo(self, repo_id: str) -> Dict[str, Any]:
        """用默认浏览器打开仓库页面（查看模型介绍）。"""
        import urllib.parse
        import webbrowser
        from .. import hf_downloader
        if not hf_downloader._safe_relpath(repo_id):
            return {"ok": False, "error": "仓库名不合法"}
        url = f"{hf_downloader.HF_ENDPOINT}/{urllib.parse.quote(repo_id)}"
        try:
            webbrowser.open(url)
            return {"ok": True, "url": url}
        except Exception as e:
            return {"ok": False, "error": f"打开浏览器失败: {e}"}

    def hf_repo_files(self, repo_id: str) -> Dict[str, Any]:
        """获取仓库文件列表并按 gguf / mmproj 分类。"""
        from .. import hf_downloader
        from ..llama_cpp import get_models_dir
        try:
            files = hf_downloader.api_list_files(repo_id)
            classified = hf_downloader.classify_files(files)
            return {"ok": True, "repo_id": repo_id,
                    "gguf": classified["gguf"], "mmproj": classified["mmproj"],
                    "models_dir": str(get_models_dir())}
        except Exception as e:
            return {"ok": False, "error": f"获取文件列表失败: {e}"}

    def hf_download_models(self, repo_id: str, files: List[str]) -> Dict[str, Any]:
        """下载勾选的模型文件到模型文件夹（作者/仓库名两级目录）。"""
        from .. import hf_downloader
        from ..llama_cpp import get_models_dir
        from .experimental import _install_lock

        if not files:
            return {"ok": False, "error": "未选择任何文件"}
        if not _install_lock.acquire(blocking=False):
            return {"ok": False, "busy": True, "error": "已有模块正在安装，请等待完成后再试"}
        try:
            if not hf_downloader.begin_download():
                return {"ok": False, "busy": True, "error": "已有下载任务进行中，请稍后再试"}

            def _log(msg: str):
                js_pusher.push("appendLog", msg, "info")

            def _progress(ev: Dict[str, Any]):
                js_pusher.push("hfDownloadProgress", ev)

            result: Dict[str, Any] = {}
            try:
                result = hf_downloader.download_files(
                    repo_id, files, str(get_models_dir()),
                    log_fn=_log, progress_cb=_progress,
                    cancel_event=hf_downloader.get_cancel_event())
            except Exception as e:
                result = {"ok": False, "error": f"下载过程异常: {e}"}
            finally:
                js_pusher.push("hfDownloadDone", result)
                hf_downloader.end_download()
            return result
        finally:
            _install_lock.release()

    def hf_cancel_download(self) -> Dict[str, Any]:
        """请求取消当前下载（立即返回，下载线程在下一个分块处中断）。"""
        from .. import hf_downloader
        hf_downloader.set_cancel()
        return {"ok": True}
