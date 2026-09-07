"""st_embed_worker.py — ST 嵌入 worker（在 st-embedding venv 内运行）。

"""
from __future__ import annotations

import json
import os
import sys
import time


def _emit(obj) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _load_model(model_path: str, device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_path, device=device)


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as f:
        params = json.load(f)
    model_path = str(params["model_path"])
    device = str(params.get("device", "auto") or "auto")
    index_path = str(params.get("index_path", "") or "")
    batch = 32

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    t0 = time.time()
    try:
        model = _load_model(model_path, device)
    except Exception as e:
        _emit({"event": "error", "message": f"模型加载失败: {e}"})
        return
    try:
        dim = int(model.get_sentence_embedding_dimension() or 0)
        has_query_prompt = "query" in (getattr(model, "prompts", None) or {})
    except Exception:
        dim, has_query_prompt = 0, False
    _emit({"event": "ready", "dim": dim, "device": str(getattr(model, "device", device)),
           "query_prompt": has_query_prompt, "load_s": round(time.time() - t0, 1)})

    state = {"hash": None, "mat": None, "count": 0}

    def _encode(texts, is_query):
        kwargs = {}
        if is_query and has_query_prompt:
            kwargs["prompt_name"] = "query"  # Qwen3-Embedding 查询侧 instruction
        return model.encode(list(texts), batch_size=batch,
                            normalize_embeddings=True, **kwargs)

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
            mat, cached = None, False
            if index_path and h and os.path.isfile(index_path):
                try:
                    import numpy as np
                    z = np.load(index_path, allow_pickle=False)
                    if str(z["hash"]) == h and str(z["model"]) == model_path:
                        v = z["vectors"]
                        if v.shape[0] == len(texts):
                            mat = v.astype("float32")
                            cached = True
                except Exception:
                    mat = None
            if mat is None:
                try:
                    import numpy as np
                    chunks = []
                    total = len(texts)
                    t_enc = time.perf_counter()
                    for i in range(0, total, batch):
                        chunks.append(_encode(texts[i:i + batch], is_query=False))
                        _emit({"event": "progress", "done": min(i + batch, total),
                               "total": total})
                    encode_s = round(time.perf_counter() - t_enc, 2)
                    if chunks:
                        mat = np.vstack(chunks).astype("float32")
                    else:
                        mat = np.zeros((0, max(dim, 1)), dtype="float32")
                    if index_path and h:
                        try:
                            os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
                            np.savez_compressed(index_path,
                                                vectors=mat.astype("float16"),
                                                hash=np.str_(h), model=np.str_(model_path))
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
                   "cached": cached,
                   "encode_s": None if cached else encode_s})
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
                import numpy as np
                q = _encode([t for _, t in segs], is_query=True).astype("float32")
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
