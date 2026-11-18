"""ort_embed_worker.py — ONNX Runtime 嵌入 worker（在 ort-embedding venv 内运行）。

模型：EmbeddingGemma-300m ONNX（图内含 mean-pooling + 归一化，输出 [B,768] 句向量）。
fp32 图（model.onnx）输入 int64；int8_qat 图（model_int8_qat.onnx）输入 int32 且需要
position_ids——按 sess.get_inputs() 的实际签名组装 feeds，两种变体共用本 worker。

协议（stdin/stdout，JSON 行）：
  ← {"event": "ready", "dim", "device", "file", "load_s"}
  → {"op": "build", "hash", "texts"}   → {"event": "progress"*, "built"}
  → {"op": "query", "parts", "top_k"}  → {"event": "matches"}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

MODEL_FILES = {"gpu": "model.onnx", "cpu": "model_int8_qat.onnx"}
MAX_SEQ_LEN = 2048


def _emit(obj) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _pick_model_file(model_dir: Path, device: str) -> Path:
    """按设备优先选模型文件；缺文件时用另一变体兜底（fp32/int8 均可跑任意 EP）。"""
    order = [MODEL_FILES.get(device, "model.onnx"), "model.onnx", "model_int8_qat.onnx"]
    for name in order:
        p = model_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"{model_dir} 下没有模型文件")


def _load(model_path: Path, device: str):
    import contextlib
    import onnxruntime as ort
    if device == "gpu":
        # Windows：把 pip 安装的 CUDA/cuDNN DLL 目录暴露给加载器。缺 DLL 时 ORT
        # 会 print 警告，重定向到 stderr——stdout 是 JSON 协议流，不能被污染
        with contextlib.redirect_stdout(sys.stderr):
            try:
                ort.preload_dlls()
            except Exception:
                pass
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device == "gpu" else ["CPUExecutionProvider"])
    path = _pick_model_file(model_path, device)
    sess = ort.InferenceSession(str(path), so, providers=providers)
    return sess, path


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as f:
        params = json.load(f)
    model_path = Path(str(params["model_path"]))
    device = "gpu" if str(params.get("device", "")).lower() == "gpu" else "cpu"
    index_path = str(params.get("index_path", "") or "")
    batch = 32

    t0 = time.time()
    try:
        sess, model_file = _load(model_path, device)
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        tok.enable_truncation(max_length=MAX_SEQ_LEN)
    except Exception as e:
        _emit({"event": "error", "message": f"模型加载失败: {e}"})
        return

    # 图签名 → 输入组装（fp32: int64；int8_qat: int32 + position_ids）
    dtypes = {i.name: (np.int32 if "int32" in i.type else np.int64)
              for i in sess.get_inputs()}
    # 输出选择：社区 fp32 图带多个输出（last_hidden_state 在前），
    # 句向量统一按名取；无名时取最后一个（ST 导出把最终输出放末位）
    out_names = [o.name for o in sess.get_outputs()]
    out_idx = (out_names.index("sentence_embedding")
               if "sentence_embedding" in out_names else len(out_names) - 1)
    out_name = out_names[out_idx]

    def _encode(texts):
        """文本列表 → [B, dim] 归一化嵌入矩阵（Gemma 编码自带 BOS，pad id = 0）。"""
        texts = list(texts)
        encs = tok.encode_batch(texts)
        B = len(encs)
        S = max((len(e.ids) for e in encs), default=1)
        ids = np.zeros((B, S), dtype=np.int64)
        mask = np.zeros((B, S), dtype=np.int64)
        for i, e in enumerate(encs):
            L = len(e.ids)
            ids[i, :L] = e.ids
            mask[i, :L] = 1
        feeds = {}
        for name, dt in dtypes.items():
            if name == "position_ids":
                feeds[name] = np.tile(np.arange(S, dtype=dt), (B, 1))
            elif name == "attention_mask":
                feeds[name] = mask.astype(dt)
            elif name == "input_ids":
                feeds[name] = ids.astype(dt)
            else:  # 未知输入（如 token_type_ids）：全零占位
                feeds[name] = np.zeros((B, S), dtype=dt)
        out = sess.run([out_name], feeds)[0].astype(np.float32)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        np.maximum(n, 1e-12, out=n)
        return out / n

    try:
        dim = int(_encode(["预热"]).shape[1])  # 预热编码：验证管线 + 取输出维度
    except Exception as e:
        _emit({"event": "error", "message": f"预热编码失败: {e}"})
        return
    actual = "cuda" if "CUDAExecutionProvider" in sess.get_providers() else "cpu"
    _emit({"event": "ready", "dim": dim, "device": actual, "file": model_file.name,
           "load_s": round(time.time() - t0, 1)})

    # 缓存键带模型文件名：换装变体（fp32 ↔ int8）后旧索引不会误命中
    model_key = f"{model_path.as_posix()}|{model_file.name}"

    state = {"hash": None, "mat": None, "count": 0}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            _emit({"event": "error", "message": "无法解析的请求行"})
            continue
        op = req.get("op")

        if op == "build":
            h = str(req.get("hash", "") or "")
            texts = list(req.get("texts", []) or [])
            mat, cached, encode_s = None, False, None
            if index_path and h and Path(index_path).is_file():
                try:
                    z = np.load(index_path, allow_pickle=False)
                    if str(z["hash"]) == h and str(z["model"]) == model_key:
                        v = z["vectors"]
                        if v.shape[0] == len(texts):
                            mat = v.astype("float32")
                            cached = True
                except Exception:
                    mat = None
            if mat is None:
                try:
                    chunks = []
                    total = len(texts)
                    t_enc = time.perf_counter()
                    for i in range(0, total, batch):
                        chunks.append(_encode(texts[i:i + batch]))
                        _emit({"event": "progress", "done": min(i + batch, total),
                               "total": total})
                    encode_s = round(time.perf_counter() - t_enc, 2)
                    if chunks:
                        mat = np.vstack(chunks).astype("float32")
                    else:
                        mat = np.zeros((0, max(dim, 1)), dtype="float32")
                    if index_path and h:
                        try:
                            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(index_path,
                                                vectors=mat.astype("float16"),
                                                hash=np.str_(h), model=np.str_(model_key))
                        except Exception as e:
                            _emit({"event": "warn",
                                   "message": f"索引缓存写入失败: {e}"})
                except Exception as e:
                    _emit({"event": "error", "message": f"索引编码失败: {e}"})
                    continue
            state["hash"], state["mat"] = h, mat
            state["count"] = int(mat.shape[0]) if mat is not None else 0
            _emit({"event": "built", "count": state["count"],
                   "dim": int(mat.shape[1]) if mat is not None and mat.size else dim,
                   "cached": cached, "encode_s": encode_s})
            continue

        if op == "query":
            parts = dict(req.get("parts", {}) or {})
            k = max(1, int(req.get("top_k") or 64))
            if state["mat"] is None or not state["mat"].size:
                _emit({"event": "matches", "matches": []})
                continue
            segs = [(n, t) for n, t in parts.items() if t]
            if not segs:
                _emit({"event": "matches", "matches": []})
                continue
            try:
                q = _encode([t for _, t in segs])
                sims = q @ state["mat"].T                      # (S, N)
                best_seg = np.argmax(sims, axis=0)
                best = sims[best_seg, np.arange(sims.shape[1])]
                order = np.argsort(-best)[:k]
                matches = [{"idx": int(i), "score": round(float(best[i]), 4),
                            "seg": segs[int(best_seg[i])][0]} for i in order]
                _emit({"event": "matches", "matches": matches})
            except Exception as e:
                _emit({"event": "error", "message": f"查询失败: {e}"})
            continue

        _emit({"event": "error", "message": f"未知操作: {op}"})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            _emit({"event": "error", "message": f"worker 异常退出: {e}"})
        except Exception:
            pass
        sys.exit(1)
