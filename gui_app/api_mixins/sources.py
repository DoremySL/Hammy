"""SourcesMixin — 多源管理（添加/移除文件夹和零散文件）+ 触发扫描。"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import discovery
from ..constants import PROBE_POOL_WORKERS
from ..workspace_paths import SIMILAR_CACHE_FILE, stable_id
from ..workspace_store import (
    NO_WRITE,
    add_adhoc_files,
    add_root,
    clear_sources,
    read_json,
    remove_adhoc,
    remove_root,
    update_json,
)
from batch_rename import dedup

_similar_lock = threading.Lock()
_similar_stop: Optional[threading.Event] = None   # 当前相似扫描的取消事件；None = 无扫描


def _fmt_size(size: int) -> str:
    """字节 → 人类可读大小。"""
    if size >= 1024 * 1024 * 1024:
        return f"{size / 1024 / 1024 / 1024:.2f} GB"
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} B"


class SourcesMixin:
    """多源管理相关 API。"""

    # ── 多源管理 ──

    def add_sources(self, paths: List[str]) -> Dict[str, Any]:
        """追加源（文件夹或文件）。返回更新后的 manifest + 扫描结果。"""
        # 防御性：前端可能传非 list
        if not isinstance(paths, (list, tuple)):
            return {"error": "paths 必须是列表"}
        try:
            for p in paths:
                if not isinstance(p, str) or not p:
                    continue
                if Path(p).is_dir():
                    add_root(p)
                elif Path(p).is_file():
                    add_adhoc_files([p])
        except Exception as e:
            return {"error": f"添加源失败: {e}"}
        return self.scan()

    def remove_source(self, path: str, is_adhoc: bool) -> Dict[str, Any]:
        """从 manifest 移除一个源。"""
        if is_adhoc:
            remove_adhoc(path)
        else:
            remove_root(path)
        return self.scan()

    def clear_sources(self) -> Dict[str, Any]:
        """清空全部源（manifest 的 roots + adhoc_files）并重新扫描。"""
        clear_sources()
        return self.scan()

    # ── 去重 ──

    def find_duplicates(self) -> Dict[str, Any]:
        """扫描当前待处理列表，返回内容重复的分组（仅检测，不移动）：
        {"groups": [{"keep", "remove", "size", "size_str"}], "total_remove": int}
        """
        pending = discovery.collect_all()
        groups = dedup.find_duplicates(pending)
        out: List[Dict[str, Any]] = []
        total_remove = 0
        for g in groups:
            keep_p = Path(g["keep"])
            remove_items = []
            for rp in g["remove"]:
                rp_path = Path(rp)
                remove_items.append({
                    "path": rp,
                    "name": rp_path.name,
                    "dir": str(rp_path.parent),
                })
            out.append({
                "keep": {
                    "path": g["keep"],
                    "name": keep_p.name,
                    "dir": str(keep_p.parent),
                },
                "remove": remove_items,
                "size": g["size"],
                "size_str": _fmt_size(g["size"]),
            })
            total_remove += len(remove_items)
        return {"groups": out, "total_remove": total_remove}

    def find_similar_versions(self, mode: str = "fast") -> Dict[str, Any]:
        """第二阶段去重：检测同内容不同版本（不同编码/分辨率/带短头尾）。
        时长预筛 → 分阶段指纹（window0 粗判 → 存活对补全窗）→ 对齐聚组。"""
        global _similar_stop
        from batch_rename import similarity as sim
        from batch_rename.collector import is_mpeg_ts
        from batch_rename.utils import path_stat
        mode_cfg = sim.SCAN_MODES.get(mode) or sim.FAST
        if not _similar_lock.acquire(blocking=False):
            return {"busy": True, "error": "已有相似扫描正在进行"}
        stop = threading.Event()
        _similar_stop = stop
        try:
            paths = discovery.collect_all()
            with ThreadPoolExecutor(PROBE_POOL_WORKERS) as ex:
                infos = list(ex.map(discovery.probe_video, paths))
            discovery.prune_probe_cache(paths)
            metas = {p: i for p, i in zip(paths, infos) if i and i.get("duration")}
            order = sorted(metas, key=lambda p: metas[p]["duration"])
            aspect = {p: sim.aspect_ratio(metas[p].get("resolution", "")) for p in order}
            pairs = self._similar_pairs(order, metas, aspect, mode_cfg)
            if not pairs:
                return {"groups": []}

            fp_cache = read_json(SIMILAR_CACHE_FILE, {})
            new_fps: Dict[str, Dict] = {}
            session: Dict[str, Dict[int, List]] = {}

            def get_fp(p: str, idxs) -> Dict[int, List]:
                key = os.path.normcase(os.path.normpath(p))
                try:
                    st = path_stat(p)
                    mtime, size = int(st.st_mtime), st.st_size
                except OSError:
                    mtime, size = 0, 0
                ent = fp_cache.get(key)
                ent_ok = bool(ent) and ent.get("mode") == mode \
                    and ent.get("mtime") == mtime and ent.get("size") == size
                have = dict(session.get(key, {}))
                if ent_ok:
                    for k, v in ent.get("windows", {}).items():
                        have.setdefault(int(k), [tuple(f) for f in v])
                is_ts = is_mpeg_ts(p)
                need = [0] if is_ts or metas[p]["duration"] <= mode_cfg.short_sec else list(idxs)
                miss = [i for i in need if not have.get(i)]
                if miss:
                    frames = sim.fingerprint_windows(
                        p, metas[p]["duration"], is_ts, miss, stop, mode=mode_cfg)
                    session.setdefault(key, {}).update(frames)
                    have.update(frames)
                    # 从 have（会话已抽 + 磁盘已有 + 本次新抽）全量合并，
                    # 避免阶段 2 补抽时把阶段 1 已抽的中间窗覆盖丢弃
                    base = {"mode": mode, "mtime": mtime, "size": size,
                            "windows": {str(i): [list(f) for f in fl]
                                        for i, fl in have.items()}}
                    new_fps[key] = base
                return {i: have.get(i, []) for i in need}

            cand = sorted({p for pr in pairs for p in pr})
            single = {p for p in cand
                      if is_mpeg_ts(p) or metas[p]["duration"] <= mode_cfg.short_sec}
            fps1, accepted, surv = self._similar_phase1(
                cand, pairs, single, get_fp, metas, mode_cfg)
            fpsf, accepted2, full_edges = self._similar_phase2(
                surv, pairs, get_fp, metas, mode_cfg)
            accepted += accepted2
            # 写入时同步裁剪：只保留当前库内文件的条目（去重移走/删除的条目清掉，
            # 否则缓存只增不删，库越大每次扫描读 JSON 越慢）；无变化时返回 NO_WRITE 不落盘
            cache_keys = {os.path.normcase(os.path.normpath(p)) for p in paths}

            def _sync_cache(d: Dict) -> Any:
                changed = bool(new_fps)
                if cache_keys:
                    for k in [k for k in d if k not in cache_keys]:
                        del d[k]
                        changed = True
                if new_fps:
                    d.update(new_fps)
                return d if changed else NO_WRITE

            update_json(SIMILAR_CACHE_FILE, _sync_cache, default_factory=dict)
            if not accepted:
                return {"groups": []}
            return {"groups": self._similar_finalize(
                metas, cand, accepted, fps1, fpsf, mode_cfg, full_edges)}
        finally:
            _similar_stop = None
            _similar_lock.release()

    def stop_similar_scan(self) -> Dict[str, Any]:
        """请求中止进行中的相似扫描（不取锁，扫描中被调用才能生效）。"""
        ev = _similar_stop
        if ev is not None:
            ev.set()
            return {"ok": True, "stopped": True}
        return {"ok": True, "stopped": False}

    @staticmethod
    def _similar_pairs(order: List[str], metas: Dict[str, Dict],
                       aspect: Dict[str, float], mode_cfg) -> List[Tuple[str, str]]:
        """预筛出待比对对：时长差在容差内且宽高比接近。"""
        from batch_rename import similarity as sim
        pairs = []
        for i in range(len(order)):
            # 容差按对中较短时长取（快速模式分段）；order 升序、i 固定则容差固定，break 剪枝仍成立
            tol = mode_cfg.dur_tol_for(metas[order[i]]["duration"])
            for j in range(i + 1, len(order)):
                if metas[order[j]]["duration"] - metas[order[i]]["duration"] > tol:
                    break
                if abs(aspect[order[i]] - aspect[order[j]]) > sim.ASPECT_TOL:
                    continue
                pairs.append((order[i], order[j]))
        return pairs

    @staticmethod
    def _similar_phase1(cand: List[str], pairs, single, get_fp, metas, mode_cfg):
        """阶段 1：并行抽中间窗粗判；单段模式（ts/短视频）一次即全量，直接定案。
        返回 (fps1, accepted, surv)。"""
        from batch_rename import similarity as sim

        def seg1(fd):
            return fd.get(sim.PHASE1_WINDOW) or fd.get(0) or []

        with ThreadPoolExecutor(min(8, os.cpu_count() or 4)) as ex:
            fps1 = dict(zip(cand, ex.map(
                lambda p: get_fp(p, [sim.PHASE1_WINDOW]), cand)))
        accepted: List[Tuple[str, str, Any]] = []
        surv = set()
        for a, b in pairs:
            da, db = metas[a]["duration"], metas[b]["duration"]
            fa, fb = seg1(fps1[a]), seg1(fps1[b])
            if not fa or not fb:
                continue
            # 帧级预筛：窗中间帧最小汉明超阈值直接剔除，省掉 δ 探针
            if sim.pre_dist(fa, fb) > sim.PRE_FILTER_DIST:
                continue
            r = sim.align([fa], [fb], db - da, da, db, quick=True, mode=mode_cfg)
            if not r:
                continue
            if a in single or b in single:
                accepted.append((a, b, r))
                continue
            surv.add(a)
            surv.add(b)
        return fps1, accepted, surv

    @staticmethod
    def _similar_phase2(surv, pairs, get_fp, metas, mode_cfg):
        """阶段 2：存活者并行补全窗细对齐。返回 (fpsf, accepted, full_edges)，
        full_edges 按文件对索引，供终验复用已算对齐结果。"""
        from batch_rename import similarity as sim
        fpsf: Dict[str, Dict[int, List]] = {}
        accepted: List[Tuple[str, str, Any]] = []
        full_edges: Dict[frozenset, Any] = {}
        if surv:
            with ThreadPoolExecutor(min(8, os.cpu_count() or 4)) as ex:
                fpsf = dict(zip(sorted(surv), ex.map(
                    lambda p: get_fp(p, mode_cfg.windows), sorted(surv))))
            for a, b in pairs:
                if a not in surv or b not in surv:
                    continue
                da, db = metas[a]["duration"], metas[b]["duration"]
                fa, fb = fpsf[a], fpsf[b]
                r2 = sim.align([fa[i] for i in sorted(fa) if fa[i]],
                               [fb[i] for i in sorted(fb) if fb[i]], db - da, da, db,
                               mode=mode_cfg)
                if r2:
                    accepted.append((a, b, r2))
                    full_edges[frozenset((a, b))] = r2
        return fpsf, accepted, full_edges

    @staticmethod
    def _similar_finalize(metas, cand, accepted, fps1, fpsf, mode_cfg,
                          full_edges) -> List[Dict[str, Any]]:
        """聚组后终验：每个成员与 keep 实际对齐（accepted 边直接复用结果），
        防止 A≈B≈C 桥接链误并；有成员被踢时用保留边重聚并重算 keep/extras。"""
        from batch_rename import similarity as sim
        all_fp = {p: fpsf.get(p) or fps1.get(p) for p in cand
                  if fpsf.get(p) or fps1.get(p)}
        groups = []
        for g in sim.cluster_groups(metas, accepted):
            members = g["paths"]
            keep = g["keep"]
            fk = {i: v for i, v in all_fp[keep].items() if v}
            ok_set = {keep}
            for m in members:
                if m == keep:
                    continue
                r = full_edges.get(frozenset((m, keep)))
                if r is None:
                    fm = {i: v for i, v in all_fp[m].items() if v}
                    r = sim.align([fm[i] for i in sorted(fm)],
                                  [fk[i] for i in sorted(fk)],
                                  metas[keep]["duration"] - metas[m]["duration"],
                                  metas[m]["duration"], metas[keep]["duration"],
                                  mode=mode_cfg)
                if r:
                    ok_set.add(m)
            if len(ok_set) < len(members):
                sub_edges = [(a, b, r2) for a, b, r2 in accepted
                             if a in ok_set and b in ok_set]
                for g2 in sim.cluster_groups(metas, sub_edges):
                    if keep in g2["paths"]:
                        g = g2
                        break
            items = []
            for p in g["paths"]:
                m = metas[p]
                pp = Path(p)
                items.append({
                    "id": stable_id(p), "path": p, "name": pp.name, "dir": str(pp.parent),
                    "size": m.get("size", 0), "size_str": _fmt_size(m.get("size", 0)),
                    "resolution": m.get("resolution", ""), "codec": m.get("codec", ""),
                    "duration": m.get("duration", 0),
                    "audio_codec": m.get("audio_codec", ""), "has_audio": m.get("has_audio", False),
                })
            groups.append({"items": items, "keep": g["keep"], "extras": g["extras"]})
        return groups

    def confirm_dedup(self, remove_paths: List[str]) -> Dict[str, Any]:
        """执行去重：将指定文件移入 _duplicates/ 子目录，然后重新扫描。"""
        if not isinstance(remove_paths, (list, tuple)) or not remove_paths:
            return {"error": "没有需要移除的文件"}
        moved, errors = dedup.move_to_duplicates(list(remove_paths))
        result = self.scan()
        result["dedup_moved"] = len(moved)
        result["dedup_failed"] = len(errors)
        return result

    # ── 扫描 ──

    def scan(self) -> Dict[str, Any]:
        """全量扫描。"""
        return discovery.scan_all()
