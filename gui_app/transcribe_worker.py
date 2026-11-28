"""转写 worker：在转写 venv 内按 `python transcribe_worker.py params.json` 执行。

参数经 JSON 文件传入，stdout 逐行回传 JSON（log/idx/done，解析方见
transcribe_cpp._run_batch_script）。业务全局量由 main() 赋值，import 无副作用。"""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading

# 超长区段分块上限（VAD 缺失回退成整段区段时生效）
FAMILY_CHUNK_SEC = {'moss': 1800}
VAD_MIN_SPEECH_SEC = 0.3  # 总语音时长低于此值视为无语音


def plog(msg):
    print(json.dumps({'log': msg}, ensure_ascii=False), flush=True)


def _make_vad():
    # Silero VAD 会话；未配置/失败返回 None（整段推理）
    if not vad_model_path:
        return None
    try:
        return tc.VadSession(tc.Model(vad_model_path, backend='cpu'))
    except Exception as e:
        plog(f'Silero VAD 不可用（{str(e)[:80]}），整段推理')
        return None


def _vad_regions(vad, audio, sr):
    # 返回 [(start_sec, end_sec)]；无语音返回 []；VAD 不可用返回整段单一区段
    whole = [(0.0, len(audio) / sr)] if len(audio) else []
    if vad is None:
        return whole
    try:
        # 单段 ≤30s 送模型，长段在停顿处预切
        res = vad.run(audio, min_silence_ms=2000, speech_pad_ms=300,
                      max_speech_ms=29000)
        segs = [(s.start_sample / sr, s.end_sample / sr) for s in res.segments]
        if segs and sum(b - a for a, b in segs) >= VAD_MIN_SPEECH_SEC:
            return segs
        return []
    except Exception as e:
        plog(f'Silero VAD 推理失败（{str(e)[:80]}），本次整段推理')
        return whole


def _join_cjk(parts):
    # 边界任一侧为 CJK/全角则直接拼接，否则补空格
    out = ''
    for t in parts:
        if t and out and out[-1].isascii() and t[0].isascii():
            out += ' '
        out += t
    return out


def _run_region(sess, seg_audio):
    # 单次区段推理；OutputTruncated（短区段解码超限）对半切分重试
    try:
        return [(sess.run(seg_audio, **run_kw), 0)]
    except tc.OutputTruncated:
        if len(seg_audio) < 8000:
            return []
        mid = len(seg_audio) // 2
        return (_run_region(sess, seg_audio[:mid])
                + [(r, o + mid) for r, o in _run_region(sess, seg_audio[mid:])])


def _windowed_text(sess, audio, sr, regions, win, stop_at=0):
    # 区段内分窗推理拼接文本；stop_at>0 时累计达标提前结束
    texts, acc = [], 0
    for a, b in regions:
        pos = a
        while pos < b - 0.1:
            end = min(pos + win, b)
            for r, _off in _run_region(sess, audio[int(pos * sr): int(end * sr)]):
                t = _join_cjk(s.text.strip() for s in (r.segments or [])
                              if s.text.strip()) or (r.text or '').strip()
                if t:
                    texts.append(t)
                    acc += len(t) + 1
                if stop_at and acc >= stop_at:
                    return _join_cjk(texts)
            pos = end
    return _join_cjk(texts)


def _srt_ts(ms):
    h, rem = divmod(int(ms) // 1000, 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d},{int(ms) % 1000:03d}'


def _to_srt(segments, offset_ms=0, idx=1, max_ms=0):
    lines = []
    for seg in segments:
        t = (getattr(seg, 'text', '') or '').strip()
        if not t:
            continue
        t0 = seg.t0_ms + offset_ms
        t1 = seg.t1_ms + offset_ms
        if max_ms:
            if t0 >= max_ms:
                continue
            t1 = min(t1, max_ms)
        lines.append(str(idx))
        lines.append(f'{_srt_ts(t0)} --> {_srt_ts(t1)}')
        lines.append(t)
        lines.append('')
        idx += 1
    return '\n'.join(lines), idx


def _extract_wav(vp, out_wav):
    cmd = [ffmpeg, '-y', '-v', 'error', '-i', vp,
           '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', out_wav]
    # 不闪控制台窗
    r = subprocess.run(cmd, capture_output=True, text=True,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if r.returncode != 0:
        raise RuntimeError(f'ffmpeg 提取音频失败: {(r.stderr or "")[-300:]}')


def transcribe_one(sess, vad, i, vp):
    # 临时目录由父进程持有，残留由父进程清理
    td = os.path.join(params['work_dir'], 'v%d' % i)
    os.makedirs(td, exist_ok=True)
    try:
        full = os.path.join(td, 'a.wav')
        _extract_wav(vp, full)
        audio, sr = sf.read(full, dtype='float32', always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur = len(audio) / sr
        if mode == 'text':
            regions = _vad_regions(vad, audio, sr)
            if not regions:
                # 静音/底噪：不送推理防幻觉
                entry = {'ok': True, 'text': '', 'srt': '', 'language': '',
                         'duration': round(dur, 1), 'error': None, 'silent': True}
            else:
                entry = {'ok': True,
                         'text': _windowed_text(sess, audio, sr, regions,
                                                float(params['inject_win_sec']),
                                                max_chars),
                         'srt': '', 'language': '',
                         'duration': round(dur, 1), 'error': None}
        else:
            regions = _vad_regions(vad, audio, sr)
            if not regions:
                # 静音/底噪：跳过推理防幻觉
                entry = {'ok': True, 'srt': '', 'text': '', 'language': '',
                         'duration': round(dur, 1), 'error': None, 'silent': True}
            elif not has_ts:
                # 无逐句时间戳：分窗转写文本合成单条字幕
                body = _windowed_text(sess, audio, sr, regions,
                                      float(params.get('inject_win_sec') or 30.0))
                srt = (f'1\n{_srt_ts(0)} --> {_srt_ts(int(dur * 1000))}\n{body}\n'
                       if body else '')
                entry = {'ok': True, 'srt': srt, 'language': '', 'text': '',
                         'duration': round(dur, 1), 'error': None}
            else:
                # 逐 VAD 区段出 SRT，超长区段按家族上限分块
                chunk = FAMILY_CHUNK_SEC.get(family) or dur
                max_ms = int(dur * 1000)
                parts, lang, idx = [], '', 1
                for a, b in regions:
                    pos = a
                    while pos < b - 0.1:
                        end = min(pos + chunk, b)
                        p0 = int(pos * sr)
                        for r, off in _run_region(sess, audio[p0: int(end * sr)]):
                            part, idx = _to_srt(r.segments,
                                                offset_ms=int((p0 + off) * 1000 / sr),
                                                idx=idx, max_ms=max_ms)
                            parts.append(part)
                            lang = lang or (getattr(r, 'language', '') or '')
                        pos = end
                srt = '\n'.join(p for p in parts if p)
                entry = {'ok': True, 'srt': srt, 'language': lang, 'text': '',
                         'duration': round(dur, 1), 'error': None}
    except Exception as e:
        entry = {'ok': False, 'srt': '', 'text': '', 'language': '', 'duration': 0, 'error': str(e)}
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print(json.dumps({'idx': i, **entry}, ensure_ascii=False), flush=True)
    return entry


def main():
    global params, video_paths, model_path, family, has_ts, language
    global ffmpeg, site_packages, max_chars, vad_model_path, mode
    global run_kw, tc, sf

    params_file = sys.argv[1]
    with open(params_file, 'r', encoding='utf-8') as f:
        params = json.load(f)

    video_paths = params['video_paths']
    model_path = params['model_path']
    family = params.get('family', 'whisper')
    has_ts = bool(params.get('timestamps', True))  # 无逐句时间戳 → SRT 合成整段单条字幕
    language = params.get('language', '') or None
    ffmpeg = params.get('ffmpeg', '')
    site_packages = params.get('venv_site_packages', '')
    max_chars = int(params.get('max_chars', 0) or 0)  # 注入模式的提前停止阈值
    vad_model_path = params.get('vad_model', '')  # 官方 Silero VAD GGUF（空=整段推理）
    mode = params.get('mode', 'srt')  # srt=字幕转写 / text=注入文本
    if not vad_model_path:
        plog('VAD 模型未配置，本次跳过静音判断')

    # venv 库优先于脚本目录，防同名模块遮蔽
    if site_packages:
        sys.path.insert(0, site_packages)
    import transcribe_cpp as tc
    import soundfile as sf

    # 按后端优先级初始化，失败逐级回退；并发各路独立 Model
    n_workers = max(1, int(params.get('workers', 2) or 2))
    backends = ['vulkan', 'cpu']
    models = []
    for _ in range(n_workers):
        model = None
        for be in backends:
            try:
                model = tc.Model(model_path, backend=be)
                break
            except Exception as e:
                plog(f'{be} 初始化失败: {str(e).splitlines()[0]}')
        if model is None:
            sys.exit(1)
        models.append(model)

    # whisper/moss 出 SRT；sensevoice 出纯文本
    run_kw = {}
    if family in ('whisper', 'moss'):
        run_kw['timestamps'] = 'segment'
    if family == 'moss':
        run_kw['diarize'] = 'on'
    if language:
        run_kw['language'] = language

    # K 路并发领任务，结果按 idx 归位
    jobs = queue.Queue()
    for i, vp in enumerate(video_paths):
        jobs.put((i, vp))
    per_video = [None] * len(video_paths)

    def _worker(sess, vad):
        while True:
            try:
                i, vp = jobs.get_nowait()
            except queue.Empty:
                return
            per_video[i] = transcribe_one(sess, vad, i, vp)

    threads = [threading.Thread(target=_worker,
                                args=(m.session(), _make_vad()))
               for m in models]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 最终汇总行
    print(json.dumps({'done': True, 'per_video': per_video}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
