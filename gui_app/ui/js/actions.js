/* ════════════════════════════════════════════════════════════
   处理
   ════════════════════════════════════════════════════════════ */
async function startProcessing(paths, label) {
  state.processing = true;
  state.gpuBusy = true;
  $('#btn-start').disabled = true;
  $('#btn-stop').classList.add('active');
  $('#btn-stop').classList.remove('done');
  $('#log').innerHTML = '';
  $('#prog-bar').style.width = '0%';
  $('#prog-bar').className = 'active';
  $('#prog-num').textContent = '处理中…';
  gotoLog();
  updateMiniProg();
  toast(label);
  const reset = () => {
    state.processing = false;
    state.gpuBusy = false;
    $('#btn-stop').classList.remove('active');
    updateStats();
  };
  try {
    const res = await apiCall('process', paths);
    if (!res || !res.ok) {
      toast(res && res.error ? res.error : '启动失败', 'err');
      reset();
    }
  } catch (e) {
    toast('启动失败: ' + ((e && e.message) || e), 'err');
    reset();
  }
}

$('#btn-start').addEventListener('click', async () => {
  if (state.processing || state.llamaPendingLaunch || !state.pending.length) return;
  if (state.llamaIntegration && !state.aiConnected) {
    const ok = await ensureLlamaReady();
    if (!ok) return;
  }
  if (state.processing || !state.pending.length || !state.aiConnected) return;
  const paths = state.pending.map(v => v.path);
  await startProcessing(paths, '开始处理…');
});
$('#btn-stop').addEventListener('click', async () => {
  if (state.installing) {
    try {
      const res = await apiCall('stop_install');
      if (res && res.ok) {
        toast(res.message || '正在停止安装…', 'info');
      } else {
        toast((res && res.error) || '无运行中的安装', 'dim');
      }
    } catch (e) {
      toast('停止失败: ' + ((e && e.message) || e), 'err');
    }
    return;
  }
  if (state.gpuBusy && !state.processing) {
    try {
      const res = await apiCall('stop_gpu_task');
      if (res && res.ok) {
        toast(res.message || '正在停止，进行中的步骤完成后生效', 'info');
      } else {
        state.gpuBusy = false;
        $('#btn-stop').classList.remove('active');
        updateStartBtn();
        toast((res && res.error) || '无运行中的任务', 'dim');
      }
    } catch (e) {
      state.gpuBusy = false;
      $('#btn-stop').classList.remove('active');
      updateStartBtn();
      toast('停止失败: ' + ((e && e.message) || e), 'err');
    }
    return;
  }
  try {
    const res = await apiCall('stop');
    if (res && res.ok) {
      toast('已发送停止请求');
    } else {
      state.processing = false;
      state.gpuBusy = false;
      $('#btn-stop').classList.remove('active');
      updateStats();
      toast((res && res.error) || '无运行中的任务', 'dim');
    }
  } catch (e) {
    state.processing = false;
    state.gpuBusy = false;
    $('#btn-stop').classList.remove('active');
    updateStats();
    toast('停止失败: ' + ((e && e.message) || e), 'err');
  }
});

async function processSelected() {
  if (state.processing || state.gpuBusy) return;
  if (!state.llamaIntegration && !state.aiConnected) return;
  const pendingSel = [...state.selected]
    .map(id => findVideoById(id))
    .filter(v => v && v.status === 'pending');
  if (!pendingSel.length) { toast('未选中待处理视频', 'err'); return; }
  if (state.llamaIntegration && !state.aiConnected) {
    const ok = await ensureLlamaReady();
    if (!ok) return;
  }
  const paths = pendingSel.map(v => v.path);
  await startProcessing(paths, `开始处理 ${pendingSel.length} 个视频…`);
}

async function moveOut() {
  const sel = [...state.selected]
    .map(id => findVideoById(id))
    .filter(v => v && (v.status === 'failed' || v.status === 'duplicate'));
  if (!sel.length) { toast('未选中排除视频', 'err'); return; }
  toast(`正在移回 ${sel.length} 个视频…`);
  const paths = sel.map(v => v.path);
  const r = await callApi('move_failed_out', paths);
  if (!r) return;
  if (r.ok) {
    if (r.scan && !r.scan.error) loadFromResult(r.scan);
    const errN = (r.errors || []).length;
    toast(errN ? `移回完成，${errN} 个失败` : `已将 ${r.moved} 个视频移回上级目录`, errN ? 'err' : 'ok', true);
  } else {
    toast(r && r.error ? r.error : '移回失败', 'err', true);
  }
}
async function recycleExcluded(ids) {
  const n = ids.length;
  if (!n) return;
  if (!await showConfirm(`确定把 ${n} 个视频移入回收站吗？\n\n之后如需恢复，可从系统回收站还原。`, { okText: '移入回收站', danger: true })) return;
  toast(`正在移入回收站 ${n} 个视频…`);
  const paths = ids.map(id => findVideoById(id)).filter(Boolean).map(v => v.path);
  const r = await callApi('move_to_recycle', paths);
  if (!r) return;
  if (r.ok) {
    if (r.scan && !r.scan.error) loadFromResult(r.scan);
    const errN = (r.errors || []).length;
    toast(errN ? `移入完成，${errN} 个失败` : `已将 ${r.moved} 个视频移入回收站`, errN ? 'err' : 'ok', true);
  } else {
    toast(r && r.error ? r.error : '移入回收站失败', 'err', true);
  }
}

/* ════════════════════════════════════════════════════════════
   导出 NFO / 还原 / 资源管理器
   ════════════════════════════════════════════════════════════ */
async function exportNfo(vid) {
  toast('正在导出…');
  const r = await callApi('export_nfo', vid);
  if (!r) return;
  toast(r.ok ? r.message : '导出失败: ' + (r.message || ''), r.ok ? 'ok' : 'err', true);
}
async function exportNfoBatch(vids) {
  if (!vids.length) return;
  toast(`正在导出 ${vids.length} 个…`);
  const r = await callApi('export_nfo_batch', vids);
  if (!r) return;
  toast(`导出成功 ${r.ok_count} 个${r.failed_count ? '，失败 ' + r.failed_count + ' 个' : ''}`, r.failed_count ? 'err' : 'ok', true);
}
async function generatePosters(ids) {
  toast(`正在生成 ${ids.length} 张缩略图…`);
  const r = await callApi('generate_posters', ids);
  if (!r) return;
  toast(`已生成 ${r.ok_count} 个${r.failed_count ? '，失败 ' + r.failed_count + ' 个' : ''}`, r.failed_count ? 'err' : 'ok', true);
}
async function restoreVideo(vid) {
  if (!await showConfirm('确定还原该视频的初始文件名吗？\n\n还原后：\n• 视频会恢复为初始文件名\n• 视频目录里已导出的 NFO 会被删除\n• 已处理记录会被清除', { okText: '还原' })) return;
  const r = await callApi('restore', vid);
  if (!r) return;
  if (r.ok) {
    toast(r.message, 'ok');
    const sr = await callApi('scan');
    if (sr) loadFromResult(sr);
  } else {
    toast('还原失败: ' + (r.message || ''), 'err');
  }
}
async function restoreBatch(vids) {
  if (!vids.length) return;
  if (!await showConfirm(`确定批量还原 ${vids.length} 个视频的初始文件名吗？\n\n还原后：\n• 视频会恢复为初始文件名\n• 已导出的 NFO 会被删除\n• 已处理记录会被清除`, { okText: '还原' })) return;
  toast(`正在还原 ${vids.length} 个…`);
  const r = await callApi('restore_batch', vids);
  if (!r) return;
  if (r.ok_count != null) {
    toast(`已还原成功 ${r.ok_count} 个${r.failed_count ? '，失败 ' + r.failed_count + ' 个' : ''}`, r.failed_count ? 'err' : 'ok', true);
    const sr = await callApi('scan');
    if (sr) loadFromResult(sr);
  } else {
    toast('批量还原失败', 'err', true);
  }
}
async function openInExplorer(path) {
  await callApi('open_in_explorer', path);
}

/* ════════════════════════════════════════════════════════════
   导出到文件夹
   ════════════════════════════════════════════════════════════ */
async function exportToFolder(withNfo) {
  const sel = [...state.selected].map(id => findVideoById(id)).filter(Boolean);
  if (!sel.length) { toast('未选中视频', 'err'); return; }
  const folders = await callApi('pick_folders');
  if (!folders || !folders.length) return;
  const dest = folders[0];
  const items = sel.map(v => ({ id: v.id, path: v.path }));
  toast(`正在导出 ${items.length} 个视频…`);
  const r = await callApi('export_to_folder', items, dest, withNfo);
  if (!r) return;
  if (r.ok) {
    if (r.scan && !r.scan.error) loadFromResult(r.scan);
    const errN = (r.errors || []).length;
    let msg = `已导出 ${r.moved} 个视频`;
    if (withNfo && r.nfo_count) {
      const total = r.nfo_count + (r.sub_count || 0) + (r.poster_count || 0);
      msg += ` + ${total} 个附属文件`;
    }
    if (errN) msg += `，${errN} 个失败`;
    toast(msg, errN ? 'err' : 'ok', true);
  } else {
    toast(r && r.error ? r.error : '导出失败', 'err', true);
  }
}

async function withGpuBusy(progText, apiFn) {
  state.gpuBusy = true;
  updateStartBtn();
  $('#btn-stop').classList.add('active');
  $('#btn-stop').classList.remove('done');
  gotoLog();
  $('#prog-bar').style.width = '0%';
  $('#prog-bar').className = 'active';
  $('#prog-num').textContent = progText;
  try {
    return await apiFn();
  } finally {
    state.gpuBusy = false;
    updateStartBtn();
    $('#btn-stop').classList.remove('active');
  }
}

/* ════════════════════════════════════════════════════════════
   PixAI Tagger
   ════════════════════════════════════════════════════════════ */
async function detectIpTags(ids) {
  if (state.gpuBusy) { toast('GPU 正忙，请等待当前操作完成', 'err'); return; }
  const videos = ids.map(id => findVideoById(id)).filter(v => v && v.status === 'pending');
  if (!videos.length) { toast('未选中待处理视频', 'err'); return; }
  const items = videos.map(v => [v.id, v.path]);
  const r = await withGpuBusy(`正在分析 ${videos.length} 个视频的IP信息…`, () => callApi('detect_ip_tags', items));
  if (!r) { $('#prog-bar').className = ''; $('#prog-num').textContent = ''; return; }
  if (r.results) {
    for (const [vid, d] of Object.entries(r.results)) {
      if (d.error) continue;
      state.pixaiAnimeIds.delete(vid);
      state.pixaiRealIds.delete(vid);
      state.pixaiUncertainIds.delete(vid);
      state.pixaiTaggedIds.delete(vid);
      if (d.is_anime === true) state.pixaiAnimeIds.add(vid);
      else if (d.is_anime === false) state.pixaiRealIds.add(vid);
      else if (d.is_anime === null) state.pixaiUncertainIds.add(vid);
      if ((d.character_tags && d.character_tags.length) || (d.ip_tags && d.ip_tags.length)) {
        state.pixaiTaggedIds.add(vid);
      }
    }
  }
  const stopped = r.error === '已停止';
  if (r.ok || stopped) {
    const okN = r.ok_count || 0;
    const realN = r.real_count || 0;
    $('#prog-bar').style.width = '100%';
    $('#prog-bar').className = 'done';
    $('#prog-num').textContent = stopped ? `已停止：完成 ${okN}/${r.total}` : `分析完成：成功 ${okN}/${r.total}`;
    let msg = stopped ? `IP分析已停止: 成功 ${okN}/${r.total}` : `IP分析完成: 成功 ${okN}/${r.total}`;
    if (realN > 0) msg += `（其中 ${realN} 个非二次元作品）`;
    toast(msg, okN > 0 ? 'ok' : 'err', true);
    refreshGridData();
  } else {
    $('#prog-bar').className = '';
    $('#prog-num').textContent = '分析失败';
    toast('IP分析失败: ' + (r.error || ''), 'err', true);
  }
}

/* ════════════════════════════════════════════════════════════
   Faster-Whisper
   ════════════════════════════════════════════════════════════ */
async function detectSpeech(ids) {
  if (state.gpuBusy) { toast('GPU 正忙，请等待当前操作完成', 'err'); return; }
  const videos = ids.map(id => findVideoById(id)).filter(v => v && v.path);
  if (!videos.length) { toast('未选中有效视频', 'err'); return; }
  const items = videos.map(v => [v.id, v.path]);
  const r = await withGpuBusy(`正在转录 ${videos.length} 个视频…`, () => callApi('detect_speech', items));
  if (!r) { $('#prog-bar').className = ''; $('#prog-num').textContent = ''; return; }
  if (r.results) {
    for (const [vid, d] of Object.entries(r.results)) {
      if (!d.error) state.whisperTranscribedIds.add(vid);
    }
  }
  const stopped = r.error === '已停止';
  if (r.ok || stopped) {
    const okN = r.ok_count || 0;
    $('#prog-bar').style.width = '100%';
    $('#prog-bar').className = 'done';
    $('#prog-num').textContent = stopped ? `已停止：完成 ${okN}/${r.total}` : `转录完成：成功 ${okN}/${r.total}`;
    toast(`语音转录${stopped ? '已停止' : '完成'}: 成功 ${okN}/${r.total}`, okN > 0 ? 'ok' : 'err', true);
    refreshGridData();
  } else {
    $('#prog-bar').className = '';
    $('#prog-num').textContent = '转录失败';
    toast('语音转录失败: ' + (r.error || ''), 'err', true);
  }
}

async function ensureTranscribed(ids) {
  const videos = ids.map(id => findVideoById(id)).filter(v => v && v.path);
  const need = videos.filter(v => !state.whisperTranscribedIds.has(v.id));
  if (need.length) {
    toast(`其中 ${need.length} 个尚未转录，先自动转录…`, 'info');
    await detectSpeech(need.map(v => v.id));
    return need.every(v => state.whisperTranscribedIds.has(v.id));
  }
  return true;
}

async function exportSrt(ids) {
  if (state.gpuBusy) { toast('GPU 正忙，请等待当前操作完成', 'err'); return; }
  const videos = ids.map(id => findVideoById(id)).filter(v => v && v.path);
  if (!videos.length) { toast('未选中有效视频', 'err'); return; }
  if (!(await ensureTranscribed(ids))) { toast('转录未完成，已取消导出', 'warn'); return; }
  const items = videos.map(v => [v.id, v.path]);
  toast(`正在导出 ${items.length} 个字幕…`);
  const r = await callApi('export_srt', items);
  if (r && r.ok) {
    toast(`已导出 ${r.exported} 个 SRT 字幕`, 'ok');
  } else {
    toast('导出失败: ' + ((r && r.errors && r.errors[0]) || ''), 'err');
  }
}

async function exportSrtTranslated(ids) {
  const videos = ids.map(id => findVideoById(id)).filter(v => v && v.path);
  if (!videos.length) { toast('未选中有效视频', 'err'); return; }
  await ensureTranscribed(ids);
  if (state.llamaIntegration && !state.aiConnected) {
    toast(state.llamaStarting
      ? '本地推理服务启动中，请稍候再试'
      : '本地推理服务未运行，请先在本地推理页启动服务', 'err');
    return;
  }
  if (state.gpuBusy) { toast('GPU 正忙，请等待当前操作完成', 'err'); return; }
  const items = videos.map(v => [v.id, v.path]);
  const r = await withGpuBusy(`正在翻译 ${items.length} 个字幕…`, () => callApi('export_srt_translated', items));
  if (r && (r.ok || r.cancelled)) {
    const done = r.exported || 0;
    $('#prog-bar').style.width = '100%';
    $('#prog-bar').className = 'done';
    $('#prog-num').textContent = r.cancelled ? `已停止：完成 ${done}/${items.length}` : `翻译完成：${done}/${items.length}`;
    toast(r.cancelled ? `翻译已停止: 成功 ${done}/${items.length}` : `翻译完成: 成功 ${done}/${items.length}`, done > 0 ? 'ok' : 'err');
  } else {
    $('#prog-bar').className = '';
    $('#prog-num').textContent = '翻译失败';
    toast('翻译失败: ' + ((r && r.errors && r.errors[0]) || ''), 'err');
  }
}

/* ════════════════════════════════════════════════════════════
   右键菜单
   ════════════════════════════════════════════════════════════ */
function showContextMenu(e, v) {
  if (!state.selected.has(v.id)) {
    state.selected = new Set([v.id]);
    state.selAnchor = v.id;
    state.primaryId = v.id;
    refreshSelectionUI();
  }
  const m = $('#ctxmenu');
  const selIds = [...state.selected];
  const multi = selIds.length > 1;

  const groups = [];
  const gBasic = [
    { label: '在资源管理器中打开', fn: () => openInExplorer(v.path) },
    { label: '使用默认播放器播放', fn: () => callApi('play_video', v.path) },
  ];
  const gRename = [];
  if (!state.processing && !state.gpuBusy) {
    gRename.push({ label: `手动重命名（${selIds.length} 个）`, fn: () => openRenameDialog() });
  }
  if (gRename.length) groups.push(gRename);
  const gExport = [
    { label: `导出到…（${selIds.length} 个）`, fn: () => exportToFolder(false) },
  ];
  if (state.view === 'processed') {
    gExport.push({ label: `导出到…（含附属文件，${selIds.length} 个）`, fn: () => exportToFolder(true) });
  }
  groups.push(gBasic, gExport);

  const gView = [];
  if (state.view === 'pending') {
    const pendingIds = selIds.filter(id => {
      const vv = findVideoById(id);
      return vv && vv.status === 'pending';
    });
    if (pendingIds.length) {
      const noProcess = !(state.llamaIntegration || state.aiConnected) || state.processing || state.gpuBusy;
      gView.push({ label: `处理选中视频（${pendingIds.length} 个）`, fn: () => processSelected(), disabled: noProcess });
    }
  }
  const gPixai = [];
  if (state.view === 'pending' && state.pixaiTaggerEnabled) {
    const pendingIds2 = selIds.filter(id => {
      const vv = findVideoById(id);
      return vv && vv.status === 'pending';
    });
    if (pendingIds2.length) {
      gPixai.push({ label: `获取IP信息（${pendingIds2.length} 个）`, fn: () => detectIpTags(pendingIds2), disabled: state.gpuBusy });
    }
  }
  if (gPixai.length) groups.push(gPixai);
  const gWhisper = [];
  if (state.whisperEnabled && (state.view === 'pending' || state.view === 'processed')) {
    const viewStatus = state.view;
    const matchIds = selIds.filter(id => {
      const vv = findVideoById(id);
      return vv && vv.status === viewStatus;
    });
    if (matchIds.length) {
      gWhisper.push({ label: `语音转录（${matchIds.length} 个）`, fn: () => detectSpeech(matchIds), disabled: state.gpuBusy });
      gWhisper.push({ label: `导出字幕 SRT（${matchIds.length} 个）`, fn: () => exportSrt(matchIds), disabled: state.gpuBusy });
      gWhisper.push({ label: `导出字幕并翻译为中文（${matchIds.length} 个）`, fn: () => exportSrtTranslated(matchIds), disabled: state.gpuBusy || (!state.llamaIntegration && !state.aiConnected) });
    }
  }
  if (gWhisper.length) groups.push(gWhisper);
  if (state.view === 'failed') {
    const exIds = selIds.filter(id => {
      const vv = findVideoById(id);
      return vv && (vv.status === 'failed' || vv.status === 'duplicate');
    });
    if (exIds.length) {
      gView.push({ label: `移回上级目录（${exIds.length} 个）`, fn: () => moveOut() });
      gView.push({ label: `移入回收站（${exIds.length} 个）`, fn: () => recycleExcluded(exIds), danger: true });
    }
  }
  if (gView.length) groups.push(gView);

  const gNfo = [];
  if (multi) {
    const processedIds = selIds.filter(id => {
      const vv = findVideoById(id);
      return vv && vv.status === 'processed';
    });
    if (processedIds.length) {
      gNfo.push({ label: `导出 NFO（${processedIds.length} 个）`, fn: () => exportNfoBatch(processedIds) });
      if (state.thumbOptimize) {
        gNfo.push({ label: `生成缩略图（${processedIds.length} 个）`, fn: () => generatePosters(processedIds) });
      }
      gNfo.push({ label: `还原初始文件名（${processedIds.length} 个）`, fn: () => restoreBatch(processedIds) });
    }
  } else if (v.status === 'processed') {
    gNfo.push({ label: '导出 NFO 到视频目录', fn: () => exportNfo(v.id) });
    if (state.thumbOptimize) {
      gNfo.push({ label: '生成缩略图', fn: () => generatePosters([v.id]) });
    }
    gNfo.push({ label: '还原初始文件名', fn: () => restoreVideo(v.id) });
  }
  if (gNfo.length) groups.push(gNfo);

  const allItems = [];
  m.innerHTML = groups.map(g => {
    const html = g.map(it => {
      const i = allItems.length;
      allItems.push(it);
      return `<button data-i="${i}"${it.disabled ? ' disabled' : ''}>${it.label}</button>`;
    }).join('');
    return html;
  }).join('<hr>');
  m.querySelectorAll('button').forEach(b => {
    const it = allItems[Number(b.dataset.i)];
    if (it.disabled) return;
    b.onclick = () => { it.fn(); hideContextMenu(); };
  });

  positionCtxMenu(m, e);
}
function positionCtxMenu(m, e) {
  m.style.display = 'block';
  const mw = m.offsetWidth, mh = m.offsetHeight;
  let x = e.clientX, y = e.clientY;
  if (x + mw > window.innerWidth) x = window.innerWidth - mw - 4;
  if (y + mh > window.innerHeight) y = window.innerHeight - mh - 4;
  m.style.left = Math.max(0, x) + 'px';
  m.style.top = Math.max(0, y) + 'px';
}
function hideContextMenu() { $('#ctxmenu').style.display = 'none'; }
document.addEventListener('click', e => {
  if (!e.target.closest('#ctxmenu')) hideContextMenu();
});
document.addEventListener('contextmenu', e => {
  if (!e.target.closest('.card') && !e.target.closest('.pt-item')) { hideContextMenu(); }
});
