"""ProcessingMixin — 启动/停止批处理管线 + AI 连接检测。"""
from __future__ import annotations

import threading
from typing import Any, Dict, List

from .. import config_store, discovery, runner
from ..js_push import js_pusher


# ── AI 连接检测（模块级辅助函数） ──


def _check_ai_connection_impl() -> Dict[str, Any]:
    """实际检测函数：尝试 list models + 轻量推理。"""
    try:
        from batch_rename.dependencies import AIClientBindings, DependencyError
        cfg = config_store.load_config()
        ai = cfg.get("ai", {})

        # 本地推理集成（可插拔）：开启后连接检测指向本地 llama-server，
        # 磁盘上用户填写的远程 API 参数保持不动
        from ..llama_integration import ai_override
        ov = ai_override(cfg)
        if ov:
            ai = {**ai, **ov}

        b = AIClientBindings()
        try:
            b.load()
        except DependencyError:
            return {"ok": False, "message": "openai 库未安装"}

        client = b.OpenAI(
            api_key=ai.get("api_key", "not-needed"),
            base_url=ai.get("base_url", "http://localhost:8080/v1"),
        )
        try:
            client.models.list(timeout=8)
            return {"ok": True, "message": "AI 服务已连接", "kind": "ok"}
        except Exception:
            # 列模型失败，再试轻量推理
            try:
                client.chat.completions.create(
                    model=ai.get("model", "model"),
                    messages=[{"role": "user", "content": "1"}],
                    max_tokens=1, timeout=15,
                )
                return {"ok": True, "message": "AI 服务已连接（轻量推理通过）", "kind": "ok"}
            except Exception as e2:
                return _classify_conn_err(e2, ai.get("base_url", ""))
        finally:
            try:
                client.close()
            except Exception:
                pass
    except Exception as e:
        return {"ok": False, "message": f"检测异常: {e}", "kind": "other"}


def _classify_conn_err(e: Exception, base_url: str) -> Dict[str, Any]:
    """分类连接错误，给出友好提示。"""
    msg = str(e)
    cls_name = type(e).__name__
    if cls_name == "APIConnectionError":
        return {"ok": False, "message": f"无法连接服务器: {base_url}", "kind": "conn"}
    if cls_name == "APITimeoutError":
        return {"ok": False, "message": "连接超时", "kind": "timeout"}
    if cls_name == "APIStatusError":
        sc = getattr(e, "status_code", 0)
        if sc in (401, 403):
            return {"ok": False, "message": f"认证失败 ({sc})，请检查 api_key", "kind": "auth"}
        if sc == 404:
            return {"ok": False, "message": f"端点不存在 ({sc})，请检查 base_url", "kind": "url"}
        if sc and 500 <= sc < 600:
            return {"ok": False, "message": f"服务器错误 ({sc})", "kind": "server"}
        return {"ok": False, "message": f"API 错误 ({sc})", "kind": "api"}
    return {"ok": False, "message": msg[:120], "kind": "other"}


# ── Mixin ──


class ProcessingMixin:
    """批处理管线控制 + AI 连接检测。"""

    # 防重入锁：pywebview 桥接每次调用新开线程，
    # 「检查是否有任务 + 创建 runner」必须整体在临界区内，否则双击可起两条管线
    _process_lock = threading.Lock()
    # 连接检测串行化：单次检测最长 ~23s，避免重复点击在桥接线程上堆积
    _check_conn_lock = threading.Lock()

    # ── 处理 ──

    def process(self, paths: List[str]) -> Dict[str, Any]:
        """启动批量处理。"""
        with self._process_lock:
            if self._runner and self._runner.is_running:
                return {"ok": False, "error": "已有处理任务在运行"}
            self._runner = runner.PipelineRunner()

            def _on_log(level: str, line: str):
                js_pusher.push("appendLog", line, level)
                # 提取 [N/M] 进度（引擎日志格式）
                m = self._progress_re.match(line)
                if m:
                    js_pusher.push("setProgress", int(m.group(1)), int(m.group(2)))

            def _on_progress(cur: int, tot: int):
                js_pusher.push("setProgress", cur, tot)

            def _on_file_done(entry: Dict[str, Any]):
                js_pusher.push("onFileDone", entry)

            def _on_done(summary: Dict[str, Any]):
                # 完成后重新扫描，推送更新后的列表
                try:
                    result = discovery.scan_all()
                    js_pusher.push("onProcessDone", summary, result)
                except Exception as e:
                    js_pusher.push("onProcessDone", summary, {"error": str(e)})

            return self._runner.run(
                paths=paths,
                on_log=_on_log,
                on_progress=_on_progress,
                on_file_done=_on_file_done,
                on_done=_on_done,
            )

    def stop(self) -> Dict[str, Any]:
        if self._runner:
            self._runner.stop()
            return {"ok": True}
        return {"ok": False, "error": "无运行中的任务"}

    # ── AI 连接检测 ──

    def check_connection(self) -> Dict[str, Any]:
        with self._check_conn_lock:
            r = _check_ai_connection_impl()
        # 附带当前生效的服务地址（本地推理集成开启时为本地服务），供前端 hover 显示
        try:
            cfg = config_store.load_config()
            ai = cfg.get("ai", {})
            from ..llama_integration import ai_override
            ov = ai_override(cfg)
            r["base_url"] = (ov or ai).get("base_url", "")
        except Exception:
            r["base_url"] = ""
        return r
