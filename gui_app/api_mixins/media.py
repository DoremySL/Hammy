"""MediaMixin — 视频探针、缩略图、NFO 读取/导出、还原、失败文件移回。"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import discovery, runner
from batch_rename.dedup import DUPLICATES_DIR
from ..workspace_paths import (
    PIXAI_TAGS_FILE,
    THUMB_DIR,
    WHISPER_SRT_DIR,
    WHISPER_TRANSCRIPTS_FILE,
    nfo_path,
    stable_id,
)
from ..workspace_store import (
    append_history_entry,
    get_history_by_id,
    read_json,
    remove_adhoc,
    update_adhoc_path,
    write_json,
)
from ..workspace_service import find_cached_nfo, restore_video, restore_videos


def _recycle_files(paths: List[str]) -> Tuple[List[str], List[str]]:
    """移入系统回收站（SHFileOperationW，支持批量），返回 (成功路径, 错误列表)。"""
    errors: List[str] = []
    targets: List[str] = []
    for p in paths:
        if Path(p).is_file():
            targets.append(p)
        else:
            errors.append(f"{Path(p).name}: 文件不存在")
    if not targets:
        return [], errors
    if os.name != "nt":
        return [], errors + [f"{Path(p).name}: 仅 Windows 支持回收站" for p in targets]

    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND), ("wFunc", ctypes.c_uint),
            ("pFrom", ctypes.c_wchar_p), ("pTo", ctypes.c_wchar_p),
            ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", ctypes.c_wchar_p),
        ]

    # pFrom 为 \0 分隔、双 \0 结尾的路径列表；c_wchar_p 会截断，需用缓冲区
    buf = ctypes.create_unicode_buffer("\u0000".join(targets) + "\u0000\u0000")
    op = SHFILEOPSTRUCTW()
    op.wFunc = 3  # FO_DELETE
    op.pFrom = ctypes.cast(buf, ctypes.c_wchar_p)
    op.fFlags = 0x40 | 0x10 | 0x4 | 0x400  # ALLOWUNDO | NOCONFIRMATION | SILENT | NOERRORUI
    res = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    moved = [p for p in targets if not Path(p).exists()]
    if res != 0 or op.fAnyOperationsAborted:
        errors += [f"{Path(p).name}: 移入回收站失败" for p in targets if Path(p).exists()]
    return moved, errors


class MediaMixin:
    """媒体资源相关 API。"""

    # ── 探针/缩略图 ──

    def get_probe(self, path: str, vid: str) -> Dict[str, Any]:
        """同步探针（前端按需请求单个视频元数据）。"""
        info = discovery.probe_video(path)
        discovery.flush_probe_cache()
        return {"id": vid, "path": path, "info": info}

    def get_thumb(self, path: str, vid: str) -> Optional[str]:
        """同步获取缩略图 data URL（带缓存）。"""
        return discovery.generate_thumbnail(path, vid)

    # ── NFO / 详情 ──

    def get_nfo(self, vid: str) -> Dict[str, Any]:
        """读取缓存的 NFO（按 id）。同时附带 history.original_name 供前端区分显示。"""
        hist = get_history_by_id(vid)
        result: Dict[str, Any] = {"ok": False, "id": vid}
        if hist:
            result["file_original_name"] = hist.get("original_name", "")
            result["processed_at"] = hist.get("processed_at", 0)

        # 查找缓存 NFO（按 <stable_id>.nfo 命名）
        cached_nfo = find_cached_nfo(vid)
        if cached_nfo is None:
            # 兜底：尝试读视频同目录的 NFO
            if hist:
                vp = hist.get("new_path") or hist.get("original_path") or ""
                if vp:
                    sibling = Path(vp).with_suffix(".nfo")
                    if sibling.exists():
                        cached_nfo = sibling
        if cached_nfo is None or not cached_nfo.exists():
            result["error"] = "NFO 不存在"
            return result
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(str(cached_nfo))
            root = tree.getroot()
            result["ok"] = True
            for tag in ("title", "originaltitle", "plot", "runtime", "premiered", "year"):
                node = root.find(tag)
                result[tag] = node.text if node is not None and node.text else ""
            result["tags"] = [t.text for t in root.findall("tag") if t.text]
            return result
        except Exception as e:
            result["error"] = str(e)
            return result

    def export_nfo(self, vid: str) -> Dict[str, Any]:
        """把缓存 NFO 复制到视频目录。"""
        return runner.export_nfo(vid)

    def export_nfo_batch(self, vids: List[str]) -> Dict[str, Any]:
        return runner.export_nfo_batch(vids)

    def generate_posters(self, vids: List[str]) -> Dict[str, Any]:
        """按 history thumb_time 生成 <视频名>-poster.jpg 全分辨率海报（媒体服务器用）。"""
        ok_count, errors = 0, []
        for vid in vids:
            try:
                p = (get_history_by_id(vid) or {}).get("new_path", "")
                if not p or not Path(p).is_file():
                    errors.append(f"{Path(p).name if p else vid}: 视频不存在")
                    continue
                data = discovery.generate_poster(p, vid)
                if not data:
                    errors.append(f"{Path(p).name}: 抽帧失败")
                    continue
                target = Path(p).with_name(f"{Path(p).stem}-poster.jpg")
                tmp = target.with_suffix(".jpg.tmp")
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, target)
                ok_count += 1
            except Exception as e:
                errors.append(f"{vid}: {e}")
        return {"ok": True, "ok_count": ok_count, "failed_count": len(errors), "errors": errors}

    # ── 还原 ──

    def restore(self, vid: str) -> Dict[str, Any]:
        return restore_video(vid)

    def restore_batch(self, vids: List[str]) -> Dict[str, Any]:
        """批量还原多个视频的原文件名。"""
        return restore_videos(vids)

    def move_failed_out(self, paths: List[str]) -> Dict[str, Any]:
        """把排除视频移出 _failed/_duplicates 目录回到上级目录（重新作为待处理）。"""
        from batch_rename.utils import to_long_path
        from batch_rename.naming import resolve_collision
        moved, errors, old_dirs = 0, [], []
        for fp in paths:
            try:
                p = Path(fp)
                if not p.is_file():
                    errors.append(f"{p.name}: 文件不存在")
                    continue
                if p.parent.name.lower() not in ("_failed", DUPLICATES_DIR):
                    errors.append(f"{p.name}: 不在排除目录中")
                    continue
                dest_dir = p.parent.parent
                src_long = to_long_path(str(p))
                dest, status = resolve_collision(dest_dir, p.stem, p.suffix, src_long)
                if status == "error":
                    errors.append(f"{p.name}: 上级目录存在同名冲突")
                    continue
                if status == "skipped":
                    continue
                os.replace(src_long, to_long_path(str(dest)))
                moved += 1
                old_dirs.append(str(p.parent))
            except Exception as e:
                errors.append(f"{Path(fp).name}: {e}")
        discovery.prune_excluded_dirs(old_dirs)
        try:
            scan = discovery.scan_all()
        except Exception as e:
            scan = {"error": str(e)}
        return {"ok": True, "moved": moved, "errors": errors, "scan": scan}

    def move_to_recycle(self, paths: List[str]) -> Dict[str, Any]:
        """把排除视频移入系统回收站。"""
        moved, errors = _recycle_files(paths)
        old_dirs = [str(Path(fp).parent) for fp in moved]
        discovery.prune_excluded_dirs(old_dirs)
        try:
            scan = discovery.scan_all()
        except Exception as e:
            scan = {"error": str(e)}
        return {"ok": True, "moved": len(moved), "errors": errors, "scan": scan}

    # ── 播放 / 导出到文件夹 ──

    def play_video(self, path: str) -> Dict[str, Any]:
        """使用系统默认播放器打开视频。"""
        try:
            if not Path(path).is_file():
                return {"ok": False, "error": "文件不存在"}
            os.startfile(path)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def export_to_folder(self, items: List[Dict[str, str]], dest: str,
                         with_nfo: bool = False) -> Dict[str, Any]:
        """将视频移动到目标文件夹。"""
        from batch_rename.utils import to_long_path
        from batch_rename.naming import resolve_collision

        dest_dir = Path(dest)
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return {"ok": False, "error": f"无法创建目标目录: {e}"}

        # 语音转录开启时，除 NFO 外还一并带走视频同目录的导出字幕
        from ..config_store import load_whisper_config
        whisper_on = load_whisper_config().get("enabled", False)

        moved, nfo_count, sub_count, poster_count, errors = 0, 0, 0, 0, []
        exported_paths: List[str] = []

        for item in items:
            vid = item.get("id", "")
            fp = item.get("path", "")
            try:
                p = Path(fp)
                if not p.is_file():
                    errors.append(f"{p.name}: 文件不存在")
                    continue
                src_long = to_long_path(str(p))
                target, status = resolve_collision(dest_dir, p.stem, p.suffix, src_long)
                if status == "error":
                    errors.append(f"{p.name}: 目标目录存在同名冲突")
                    continue
                if status == "skipped":
                    continue
                os.replace(src_long, to_long_path(str(target)))
                moved += 1
                exported_paths.append(fp)

                if with_nfo:
                    nfo_exported = False
                    # 优先：视频同目录的 sibling NFO（移动）
                    sibling = p.with_suffix(".nfo")
                    if sibling.is_file():
                        nfo_target = target.with_suffix(".nfo")
                        try:
                            os.replace(to_long_path(str(sibling)),
                                       to_long_path(str(nfo_target)))
                            nfo_exported = True
                        except Exception:
                            pass
                    # 其次：从缓存复制
                    if not nfo_exported and vid:
                        cached = nfo_path(vid)
                        if cached.exists():
                            nfo_target = target.with_suffix(".nfo")
                            try:
                                shutil.copy2(str(cached), str(nfo_target))
                                nfo_exported = True
                            except Exception:
                                pass
                    if nfo_exported:
                        nfo_count += 1

                # 字幕导出：whisper 转录开启时，随视频一起移动同目录字幕
                if with_nfo and whisper_on:
                    for suf in (".srt", ".zh.srt"):
                        sub = p.with_suffix(suf)
                        if sub.is_file():
                            try:
                                os.replace(to_long_path(str(sub)),
                                           to_long_path(str(target.with_suffix(suf))))
                                sub_count += 1
                            except Exception:
                                pass

                # 海报（<stem>-poster.jpg）随附件一起移动
                if with_nfo:
                    poster = p.with_name(f"{p.stem}-poster.jpg")
                    if poster.is_file():
                        try:
                            os.replace(to_long_path(str(poster)),
                                       to_long_path(str(target.with_name(f"{target.stem}-poster.jpg"))))
                            poster_count += 1
                        except Exception:
                            pass

            except Exception as e:
                errors.append(f"{Path(fp).name}: {e}")

        for fp in exported_paths:
            try:
                remove_adhoc(fp)
            except Exception:
                pass

        try:
            scan = discovery.scan_all()
        except Exception as e:
            scan = {"error": str(e)}

        return {"ok": True, "moved": moved, "nfo_count": nfo_count,
                "sub_count": sub_count, "poster_count": poster_count,
                "errors": errors, "scan": scan}

    # ── 手动重命名 ──

    def manual_rename(self, items: List[Dict[str, str]], mode: str, text: str,
                      text2: str = "", dry_run: bool = True,
                      use_regex: bool = False, match_all: bool = True,
                      case_mode: str = "", match_subtitles: bool = False) -> Dict[str, Any]:
        """手动批量重命名视频（添加前缀/后缀、删除、替换文件名主干）。"""
        from batch_rename.naming import (apply_manual_transform, resolve_collision,
                                         rename_video, match_subtitle_files)
        from batch_rename.utils import to_long_path, path_exists

        if not isinstance(items, list) or not items:
            return {"ok": False, "error": "没有选中文件"}
        if mode == "remove" and not text.strip():
            return {"ok": False, "error": "请输入要删除的文本"}
        if use_regex and mode in ("remove", "replace") and text.strip():
            try:
                re.compile(text)
            except re.error as e:
                return {"ok": False, "error": f"正则表达式错误: {e}"}

        previews: List[Dict[str, Any]] = []
        errors: List[str] = []
        renamed = skipped = sub_renamed = 0
        # 编号模板计数器：replace 且替换串含 ${...} 时，仅结果发生变化的文件
        # 按选中顺序消耗序号（无变化的文件不编号，保证 S1E01、S1E02 连续）
        counter = 0
        # 批次内预留的目标文件名（按目录分组，小写归一）：
        # dry_run 逐文件跑时磁盘上还没有新文件，若两个文件变换后同名，
        # 不预留的话两者都会预览成同一个新名，但执行时第二个会变成 _N——
        # 预留后预览与执行结果一致。
        planned: Dict[str, set] = {}
        # 匹配字幕：仅当开启且所有视频同目录（跨目录时前端不显示按钮，这里兜底忽略）
        sub_map: Dict[int, List[Tuple[str, str]]] = {}
        if match_subtitles:
            dirs = {os.path.normcase(os.path.dirname(it.get("path", ""))) for it in items}
            if len(dirs) == 1 and dirs != {os.path.normcase("")}:
                d = Path(items[0]["path"]).parent
                sub_paths = [str(p) for p in d.glob("*.ass") if p.is_file()]
                for vi, sl in match_subtitle_files(
                        [it.get("path", "") for it in items], sub_paths):
                    sub_map[vi] = sl

        for idx, item in enumerate(items):
            fp = item.get("path", "")
            vid = item.get("id", "")
            p = Path(fp)
            if not p.is_file():
                errors.append(f"{p.name}: 文件不存在")
                continue
            if mode == "replace" and "${" in text2:
                # 先不带编号算一次判断是否变化，变化才消耗序号并展开模板
                new_stem, err = apply_manual_transform(p.stem, mode, text, text2, use_regex,
                                                   None, match_all, case_mode)
                if not err and new_stem != p.stem:
                    counter += 1
                    new_stem, err = apply_manual_transform(
                        p.stem, mode, text, text2, use_regex, counter, match_all, case_mode)
            else:
                new_stem, err = apply_manual_transform(p.stem, mode, text, text2, use_regex,
                                                   None, match_all, case_mode)
            if err:
                # dry_run 时错误也返回预览行（原名 + note 原因），预览不因整批失败而清空
                if dry_run:
                    previews.append({"id": vid, "path": fp, "name": p.name,
                                     "new_name": p.name, "changed": False, "note": err})
                else:
                    errors.append(f"{p.name}: {err}")
                continue
            if new_stem == p.stem:
                skipped += 1
                previews.append({"id": vid, "path": fp, "name": p.name,
                                 "new_name": p.name, "changed": False, "note": ""})
                # 视频未变化时字幕只展示配对关系（不改名），供空输入确认匹配结果
                for sp, lang in sub_map.get(idx, []):
                    sub_path = Path(sp)
                    previews.append({"id": "", "path": sp, "name": sub_path.name,
                                     "new_name": sub_path.name, "changed": False,
                                     "note": "", "kind": "sub"})
                continue
            if dry_run:
                target, status = resolve_collision(p.parent, new_stem, p.suffix,
                                                   to_long_path(str(p)))
                if status == "error":
                    previews.append({"id": vid, "path": fp, "name": p.name,
                                     "new_name": p.name, "changed": False,
                                     "note": "同名冲突过多"})
                    continue
                if status == "skipped":
                    previews.append({"id": vid, "path": fp, "name": p.name,
                                     "new_name": p.name, "changed": False,
                                     "note": "仅大小写变化无法重命名"})
                    continue
                # 批次内撞名：继续递增 _N 后缀，直到与磁盘和本批次都不冲突
                reserved = planned.setdefault(os.path.normcase(str(p.parent)), set())
                if target.name.lower() in reserved:
                    bumped = None
                    for bump_i in range(1, 101):
                        cand = p.parent / f"{new_stem}_{bump_i}{p.suffix}"
                        if cand.name.lower() not in reserved and not path_exists(cand):
                            bumped = cand
                            break
                    if bumped is None:
                        previews.append({"id": vid, "path": fp, "name": p.name,
                                         "new_name": p.name, "changed": False,
                                         "note": "同名冲突过多"})
                        continue
                    target = bumped
                reserved.add(target.name.lower())
                previews.append({"id": vid, "path": fp, "name": p.name,
                                 "new_name": target.name, "changed": True,
                                 "note": "" if target.name == f"{new_stem}{p.suffix}"
                                 else f"同名，将重命名为 {target.name}"})
                # 配对字幕条目（新名 = 视频新名 + 语言标记 + .ass）
                for sp, lang in sub_map.get(idx, []):
                    sub_path = Path(sp)
                    sub_name = f"{target.stem}.{lang}.ass" if lang else f"{target.stem}.ass"
                    if sub_path.name.lower() == sub_name.lower():
                        continue
                    note = ""
                    if path_exists(sub_path.parent / sub_name):
                        note = "目标已存在，将跳过"
                    previews.append({"id": "", "path": sp, "name": sub_path.name,
                                     "new_name": sub_name, "changed": True,
                                     "note": note, "kind": "sub"})
                continue
            errs: List[str] = []
            new_path, rstatus = rename_video(str(p), new_stem, None, errs)
            if rstatus == "skipped":
                skipped += 1
                continue
            if rstatus != "ok":
                errors.append(f"{p.name}: {errs[-1] if errs else '重命名失败'}")
                continue
            renamed += 1
            _sync_after_manual_rename(vid, fp, new_path)
            # 配对字幕跟随改名（视频成功后执行；失败跳过报错，不影响视频结果）
            for sp, lang in sub_map.get(idx, []):
                sub_path = Path(sp)
                sub_name = f"{Path(new_path).stem}.{lang}.ass" if lang \
                    else f"{Path(new_path).stem}.ass"
                if sub_path.name.lower() == sub_name.lower():
                    continue
                if path_exists(sub_path.parent / sub_name):
                    errors.append(f"{sub_path.name}: 目标 {sub_name} 已存在，字幕未改名")
                    continue
                try:
                    os.replace(to_long_path(str(sub_path)),
                               to_long_path(str(sub_path.parent / sub_name)))
                except OSError as e:
                    errors.append(f"{sub_path.name}: {e}")
                    continue
                sub_renamed += 1

        if dry_run:
            return {"ok": True, "items": previews, "errors": errors,
                    "sub_total": sum(len(sl) for sl in sub_map.values())}
        try:
            scan = discovery.scan_all()
        except Exception as e:
            scan = {"error": str(e)}
        return {"ok": renamed > 0, "renamed": renamed, "skipped": skipped,
                "sub_renamed": sub_renamed,
                "sub_total": sum(len(sl) for sl in sub_map.values()),
                "errors": errors, "scan": scan}


def _sync_after_manual_rename(vid: str, old_path: str, new_path: str) -> None:
    """手动重命名后的数据同步：manifest 路径、history、缓存 id 迁移。"""
    update_adhoc_path(old_path, new_path)
    hist = get_history_by_id(vid) if vid else None
    old_vid = (hist or {}).get("id") or vid or stable_id(old_path)
    if hist:
        hist["new_path"] = new_path
        hist["new_name"] = Path(new_path).name
        append_history_entry(hist)
    new_vid = stable_id(new_path)
    if old_vid == new_vid:
        return
    tp = THUMB_DIR / f"{old_vid}.jpg"
    if tp.exists():
        try:
            os.replace(tp, THUMB_DIR / f"{new_vid}.jpg")
        except OSError:
            pass
    srt = WHISPER_SRT_DIR / f"{old_vid}.srt"
    if srt.exists():
        try:
            os.replace(srt, WHISPER_SRT_DIR / f"{new_vid}.srt")
        except OSError:
            pass
    for fpath in (WHISPER_TRANSCRIPTS_FILE, PIXAI_TAGS_FILE):
        data = read_json(fpath, {})
        if isinstance(data, dict) and old_vid in data:
            data[new_vid] = data.pop(old_vid)
            write_json(fpath, data)
