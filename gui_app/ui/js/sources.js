/* ════════════════════════════════════════════════════════════
   源清单渲染
   ════════════════════════════════════════════════════════════ */
function renderSources() {
  const bar = $('#sourceBar');
  const prevScroll = $('.src-scroll');
  const prevLeft = prevScroll ? prevScroll.scrollLeft : 0;
  bar.innerHTML = '';
  if (!state.roots.length && !state.adhoc_files.length) {
    setSourceBar(false);
    return;
  }
  const wrap = document.createElement('div');
  wrap.className = 'src-wrap';
  const scroll = document.createElement('div');
  scroll.className = 'src-scroll';
  for (const r of state.roots) {
    const span = document.createElement('span');
    span.className = 'src' + (state.sourceFilter === r ? ' active' : '');
    span.dataset.tip = r + '（点击筛选该源）';
    span.innerHTML = `${icon('folder-open')} <span class="nm" data-tip="${esc(r)}">${esc(shortPath(r))}</span> <span class="x" tabindex="0" aria-label="移除">×</span>`;
    span.querySelector('.x').onclick = (e) => { e.stopPropagation(); removeSource(r, false); };
    span.addEventListener('click', () => toggleSourceFilter(r));
    scroll.appendChild(span);
  }
  for (const f of state.adhoc_files) {
    const span = document.createElement('span');
    span.className = 'src' + (state.sourceFilter === f ? ' active' : '');
    span.dataset.tip = f + '（点击筛选该源）';
    span.innerHTML = `${icon('file')} <span class="nm" data-tip="${esc(f)}">${esc(shortPath(f))}</span> <span class="x" tabindex="0" aria-label="移除">×</span>`;
    span.querySelector('.x').onclick = (e) => { e.stopPropagation(); removeSource(f, true); };
    span.addEventListener('click', () => toggleSourceFilter(f));
    scroll.appendChild(span);
  }
  wrap.appendChild(scroll);
  const fadeL = document.createElement('div');
  fadeL.className = 'src-fade left';
  const fadeR = document.createElement('div');
  fadeR.className = 'src-fade right';
  wrap.appendChild(fadeL);
  wrap.appendChild(fadeR);
  bar.appendChild(wrap);
  scroll.addEventListener('scroll', updateSrcFades, { passive: true });
  const actions = document.createElement('div');
  actions.className = 'src-actions';
  const dd = document.createElement('button');
  dd.className = 'src-action';
  dd.innerHTML = '清理重复';
  dd.dataset.tip = '扫描并移除重复视频';
  dd.onclick = findDuplicates;
  actions.appendChild(dd);
  const fx = document.createElement('button');
  fx.className = 'src-action';
  fx.innerHTML = '格式修复';
  fx.dataset.tip = '补全无后缀视频，待处理流式封装视频转 MP4';
  fx.onclick = openFormatFix;
  fx.classList.toggle('has-fix', (state.stream || []).length > 0);
  actions.appendChild(fx);
  const clr = document.createElement('button');
  clr.className = 'src-action danger';
  clr.innerHTML = icon('trash', '14px') + ' 清空源';
  clr.dataset.tip = '从源列表移除，不删除磁盘文件和重命名记录';
  clr.onclick = clearSources;
  actions.appendChild(clr);
  bar.appendChild(actions);
  const scNew = $('.src-scroll');
  if (scNew && prevLeft) scNew.scrollLeft = prevLeft;
  updateSrcFades();
}

function updateSrcFades() {
  const sc = $('.src-scroll');
  const wrap = $('.src-wrap');
  if (!sc || !wrap) return;
  const overflow = sc.scrollWidth - sc.clientWidth;
  wrap.classList.toggle('fade-r', overflow > 2 && sc.scrollLeft < overflow - 2);
  wrap.classList.toggle('fade-l', sc.scrollLeft > 2);
}
$('#sourceBar').addEventListener('wheel', (e) => {
  const sc = $('.src-scroll');
  if (!sc || sc.scrollWidth - sc.clientWidth <= 0) return;
  e.preventDefault();
  sc.scrollLeft += e.deltaY + e.deltaX;
}, { passive: false });
window.addEventListener('resize', updateSrcFades);
function toggleSourceFilter(path) {
  state.sourceFilter = (state.sourceFilter === path) ? null : path;
  renderSources();
  renderGrid();
}
function shortPath(p) {
  if (!p) return '';
  const parts = p.replace(/\\/g, '/').split('/').filter(Boolean);
  if (parts.length <= 3) return p;
  return '.../' + parts.slice(-2).join('/');
}
async function removeSource(path, isAdhoc) {
  const r = await callApi('remove_source', path, isAdhoc);
  if (r) loadFromResult(r, true);
}

/* ════════════════════════════════════════════════════════════
   添加源
   ════════════════════════════════════════════════════════════ */
async function addFolder() {
  const r = await callApi('pick_folders');
  if (!r || !r.length) return;
  toastBusy('正在扫描…');
  const result = await callApi('add_sources', r);
  if (!result) return;
  if (result.error) { loadFromResult(result); return; }
  loadFromResult(result);
  const total = (result.pending || []).length + (result.processed || []).length + (result.failed || []).length;
  const hasFailed = r.some(p => p.replace(/\\/g, '/').split('/').filter(Boolean).pop().toLowerCase() === '_failed');
  const ni = result.nfo_import;
  const nfoMsg = ni && ni.imported ? `，从 NFO 恢复 ${ni.imported} 条重命名记录` : '';
  if (total === 0) {
    toast(hasFailed ? '已添加，但 _failed 目录内的视频已自动排除' : '未扫描到视频', 'err', true);
  } else {
    toast('已添加文件夹' + nfoMsg, 'ok', true);
  }
}
async function addFiles() {
  const r = await callApi('pick_files');
  if (!r || !r.length) return;
  toastBusy('正在扫描…');
  const result = await callApi('add_sources', r);
  if (!result) return;
  if (result.error) { loadFromResult(result); return; }
  loadFromResult(result);
  const total = (result.pending || []).length + (result.processed || []).length + (result.failed || []).length;
  const ni = result.nfo_import;
  const nfoMsg = ni && ni.imported ? `，从 NFO 恢复 ${ni.imported} 条重命名记录` : '';
  if (total === 0) {
    toast('未扫描到视频', 'err', true);
  } else {
    toast(`已添加 ${r.length} 个文件${nfoMsg}`, 'ok', true);
  }
}
$('#btn-add-folder').addEventListener('click', addFolder);
$('#btn-add-files').addEventListener('click', addFiles);
$('#emptyAdd').addEventListener('click', addFolder);

/* ════════════════════════════════════════════════════════════
   刷新 / 清空 / 源栏切换
   ════════════════════════════════════════════════════════════ */
$('#btn-refresh').addEventListener('click', async () => {
  if (!state.roots.length && !state.adhoc_files.length) return;
  toastBusy('正在刷新…');
  const r = await callApi('scan');
  if (!r) return;
  loadFromResult(r);
  toast('已刷新', 'ok', true);
});
async function clearSources() {
  if (!await showConfirm('将移除已添加的文件夹/文件。\n\n重命名记录仍保留，可在「设置 → 工作区」查看。\n\n确定清空所有源吗？', { okText: '清空' })) return;
  const r = await callApi('clear_sources');
  if (!r) return;
  state.hasAutoSwitched = false;
  loadFromResult(r);
  toast('已清空所有源', 'ok');
}
function setSourceBar(open) {
  $('#sourceBar').style.display = open ? '' : 'none';
  $('#folderPath').classList.toggle('expanded', open);
  if (open) updateSrcFades();
}
$('#folderPath').addEventListener('click', () => {
  setSourceBar($('#sourceBar').style.display === 'none');
});

/* ════════════════════════════════════════════════════════════
   去重
   ════════════════════════════════════════════════════════════ */
let _dedupGroups = [];
let _dedupStage = 'identical';
let _dedupGen = 0;
let _dedupMode = 'fast';
let _dedupScanning = false;   // 相似扫描进行中（关闭弹窗时据此请求后端中止）

function dedupByPath() {
  const m = new Map();
  for (const v of [...state.pending, ...state.processed, ...state.failed]) m.set(v.path, v);
  return m;
}

async function findDuplicates() {
  if (!state.pending.length && !state.processed.length) {
    $('#dedupBg').classList.add('show');
    showDedupInterlude('没有可参与比对的视频', false);
    return;
  }
  const gen = ++_dedupGen;
  $('#dedupBg').classList.add('show');
  renderDedupScanning('正在扫描重复视频…');
  const r = await callApi('find_duplicates');
  if (gen !== _dedupGen) return;
  if (!r) { showDedupInterlude('扫描失败，请重试', false); return; }
  const groups = r.groups || [];
  if (!groups.length) {
    showDedupInterlude('未发现完全相同的重复视频');
    return;
  }
  const byPath = dedupByPath();
  _dedupStage = 'identical';
  _dedupGroups = groups.map(g => ({
    kind: 'identical',
    sizeStr: g.size_str,
    items: [
      ...(g.keeps || []).map(k => ({ path: k.path, name: k.name, keep: true })),
      ...(g.remove || []).map(rm => ({ path: rm.path, name: rm.name, keep: false })),
    ].map(it => ({ ...it, video: byPath.get(it.path) || null })),
  }));
  renderDedup();
}

function renderDedup() {
  const body = $('#dedupBody');
  body.innerHTML = '';
  for (const g of _dedupGroups) {
    const box = document.createElement('div');
    box.className = 'dedup-group';
    const cards = document.createElement('div');
    cards.className = 'dedup-cards';
    box.appendChild(cards);
    body.appendChild(box);
    for (const it of g.items) {
      const card = makeDedupCard(it);
      cards.appendChild(card);
      attachDedupThumb(card.querySelector('.dedup-thumb'), it.video, it);
    }
    const size = document.createElement('div');
    size.className = 'dedup-gsize';
    size.textContent = g.kind === 'identical'
      ? `完全相同 · ${g.items.length} 份 · 单文件 ${g.sizeStr}`
      : `同内容不同版本 · ${g.items.length} 份`;
    box.appendChild(size);
  }
  $('#dedupOk').style.display = '';
  updateDedupSub();
}

function makeDedupCard(it) {
  const v = it.video;
  const card = document.createElement('div');
  card.className = 'dedup-card ' + (it.keep ? 'keep' : 'rm');
  card.innerHTML =
    `<div class="dedup-thumb"><div class="ph">${icon('clapper', '22px')}</div>` +
    `<span class="badge">${it.keep ? '保留' : '移除'}</span>` +
    (v && v.status === 'processed' ? '<span class="badge proc">已处理</span>' : '') +
    `</div>` +
    `<div class="dedup-meta">${dedupMetaHtml(it)}</div>`;
  card.onclick = () => {
    it.keep = !it.keep;
    card.classList.toggle('keep', it.keep);
    card.classList.toggle('rm', !it.keep);
    card.querySelector('.badge').textContent = it.keep ? '保留' : '移除';
    updateDedupSub();
  };
  card.oncontextmenu = (e) => showDedupCtxMenu(e, it);
  if (v && (!v.resolution || !v.duration || !v.codec || !v.audio_codec)) {
    loadProbe(v).then(() => {
      if (card.isConnected) card.querySelector('.dedup-meta').innerHTML = dedupMetaHtml(it);
    });
  }
  return card;
}

function showDedupCtxMenu(e, it) {
  e.preventDefault();
  e.stopPropagation();
  const m = $('#ctxmenu');
  m.innerHTML =
    '<button data-i="0">使用默认播放器播放</button>' +
    '<button data-i="1">在资源管理器中查看</button>';
  m.querySelector('[data-i="0"]').onclick = () => { callApi('play_video', it.path); hideContextMenu(); };
  m.querySelector('[data-i="1"]').onclick = () => { openInExplorer(it.path); hideContextMenu(); };
  positionCtxMenu(m, e);
}

function dedupMetaHtml(it) {
  const v = it.video || {};
  const dotIdx = it.name.lastIndexOf('.');
  const ext = dotIdx > 0 ? it.name.slice(dotIdx + 1).toUpperCase() : '';
  const res = it.resolution || v.resolution;
  const codec = it.codec || v.codec;
  const dur = it.duration || v.duration;
  const parts = [];
  if (res) parts.push(res);
  if (ext) parts.push(ext);
  if (codec) parts.push(codec.toUpperCase());
  const tail = [];
  if (dur) tail.push(fmtDur(dur));
  if (it.size_str) tail.push(it.size_str);
  const ac = it.audio_codec || v.audio_codec;
  if (ac) tail.push(ac.toUpperCase());
  else if (it.has_audio === false || v.has_audio === false) tail.push('无音频');
  const durLine = tail.join(' · ') || '—';
  const mtLine = parts.join(' · ') || '—';
  return `<div class="stem dur-line" data-tip="${esc(it.path)}">${esc(durLine)}</div>` +
    `<div class="stem mt-line" data-tip="${esc(it.path)}">${esc(mtLine)}</div>`;
}

function updateDedupSub() {
  let rm = 0;
  for (const g of _dedupGroups) for (const it of g.items) if (!it.keep) rm++;
  $('#dedupSub').textContent = `${_dedupGroups.length} 组 · 将移除 ${rm} 个文件`;
}

function showDedupInterlude(msg, withScanBtn = true) {
  _dedupStage = 'interlude';
  const body = $('#dedupBody');
  body.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'dedup-interlude';
  let inner = `<div>${esc(msg)}</div>`;
  if (withScanBtn) {
    inner +=
      `<div class="mode-switch">
        <button class="pill clickable" data-mode="fast">快速</button>
        <span class="pill-sep"></span>
        <button class="pill clickable" data-mode="normal">常规</button>
        <span class="pill-sep"></span>
        <button class="pill clickable" data-mode="extreme">极慢</button>
      </div>
      <button class="btn primary" id="dedup-scan-similar">扫描相似视频</button>
      <div class="mode-hint"></div>`;
  }
  div.innerHTML = inner;
  body.appendChild(div);
  if (withScanBtn) {
    const hintMap = {
      fast: '粗略比对时长几乎相同的视频，筛出同一视频的相似版本',
      normal: '比对时长相差 ≤15s的视频，筛出带剪辑软件开头结尾的相似版本',
      extreme: '比对时长相差 ≤35s，视频文件较多时不推荐使用',
    };
    const setMode = (m) => {
      _dedupMode = m;
      div.querySelectorAll('.mode-switch button')
        .forEach(b => b.classList.toggle('active', b.dataset.mode === m));
      div.querySelector('.mode-hint').textContent = hintMap[m];
    };
    div.querySelectorAll('.mode-switch button').forEach(b => b.onclick = () => setMode(b.dataset.mode));
    setMode(_dedupMode);
    const sw = div.querySelector('.mode-switch');
    const scanBtn = $('#dedup-scan-similar');
    if (sw && scanBtn) scanBtn.style.width = sw.offsetWidth + 'px';
    $('#dedup-scan-similar').onclick = scanSimilar;
  }
  $('#dedupOk').style.display = 'none';
  $('#dedupSub').textContent = '';
}

function renderDedupScanning(msg) {
  const body = $('#dedupBody');
  body.innerHTML = '<div class="dedup-interlude"><div class="dedup-spinner"></div>' +
    `<div>${esc(msg || '正在扫描…')}</div></div>`;
  $('#dedupOk').style.display = 'none';
  $('#dedupSub').textContent = '';
}

async function scanSimilar() {
  const gen = ++_dedupGen;
  renderDedupScanning('正在扫描相似视频…（关闭窗口可中止）');
  _dedupScanning = true;
  const r = await callApi('find_similar_versions', _dedupMode);
  _dedupScanning = false;
  if (gen !== _dedupGen) return;
  if (r && r.busy) { showDedupInterlude('已有扫描在进行中，请稍候'); return; }
  const groups = (r && r.groups) || [];
  if (!groups.length) { showDedupInterlude('未发现相似视频'); return; }
  const byPath = dedupByPath();
  _dedupStage = 'similar';
  _dedupGroups = groups.map(g => ({
    kind: 'similar',
    items: g.items.map(it =>
      ({ ...it, keep: (g.keeps || []).includes(it.path), video: byPath.get(it.path) || null })),
  }));
  renderDedup();
}

const _dedupThumbLimit = createLimiter(6);
function attachDedupThumb(thumb, v, it) {
  const id = (v && v.id) || (it && it.id);
  if (!id) return;
  const cached = state.thumbCache.get(id);
  if (cached) { setDedupThumbImg(thumb, cached); return; }
  _dedupThumbLimit(() => thumb.isConnected ? apiCall('get_thumb', it.path, id) : Promise.resolve(null)).then(url => {
    if (!url) return;
    state.thumbCache.set(id, url);
    if (thumb.isConnected) setDedupThumbImg(thumb, url);
  }).catch(() => {});
}

function setDedupThumbImg(thumb, url) {
  if (thumb.querySelector('img')) return;
  const img = document.createElement('img');
  img.alt = '';
  img.onload = () => {
    const ph = thumb.querySelector('.ph');
    if (ph) ph.remove();
    applyPortraitThumb(thumb, img, url);
  };
  img.src = url;
  thumb.insertBefore(img, thumb.firstChild);
}

function closeDedup() {
  _dedupGen++;
  if (_dedupScanning) {
    _dedupScanning = false;
    apiCall('stop_similar_scan').catch(() => {});
  }
  $('#dedupBg').classList.remove('show');
  _dedupGroups = [];
}

async function confirmDedup() {
  const paths = [];
  for (const g of _dedupGroups) for (const it of g.items) if (!it.keep) paths.push(it.path);
  if (!paths.length) {
    showDedupInterlude('已全部保留，未移除任何文件');
    return;
  }
  const gen = ++_dedupGen;
  renderDedupScanning('正在移除重复视频…');
  const r = await callApi('confirm_dedup', paths);
  if (gen !== _dedupGen) return;
  if (!r) { showDedupInterlude('移除失败，请重试', false); return; }
  if (r.error) { showDedupInterlude('移除失败: ' + r.error, false); return; }
  loadFromResult(r, true);
  const moved = r.dedup_moved != null ? r.dedup_moved : paths.length;
  const failed = r.dedup_failed || 0;
  const cleaned = r.dedup_history_removed || 0;
  const cleanMsg = cleaned ? `，清理 ${cleaned} 条重命名记录` : '';
  if (_dedupStage === 'identical') {
    showDedupInterlude(`已移除 ${moved} 个完全相同副本` + (failed ? `，失败 ${failed} 个` : '') + cleanMsg);
  } else {
    showDedupInterlude(`已移除 ${moved} 个重复视频` + (failed ? `，失败 ${failed} 个` : '') + cleanMsg);
  }
}

$('#btn-closededup').addEventListener('click', closeDedup);
$('#dedupCancel').addEventListener('click', closeDedup);
$('#dedupOk').addEventListener('click', confirmDedup);
$('#dedupBg').addEventListener('click', (e) => { if (e.target === $('#dedupBg')) closeDedup(); });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && $('#dedupBg').classList.contains('show')) closeDedup();
});

/* ════════════════════════════════════════════════════════════
   格式修复
   ════════════════════════════════════════════════════════════ */
let _fixGen = 0;
let _fixItems = [];     // 可修复项 {path,name,dir,size,size_str,ok,ext,reason,codecs,convert,checked}
let _fixBusy = false;

function openFormatFix() {
  const gen = ++_fixGen;
  $('#fixBg').classList.add('show');
  _fixBusy = true;
  renderFixScanning('正在扫描可修复项…');
  callApi('scan_fixable').then(r => {
    _fixBusy = false;
    if (gen !== _fixGen) return;
    if (!r) { renderFixInterlude('扫描失败，请重试'); return; }
    if (r.error) { renderFixInterlude(r.error); return; }
    _fixItems = (r.items || []).map(it => ({ ...it, checked: !!it.ok }));
    renderFix();
  });
}

function renderFix() {
  const body = $('#fixBody');
  body.innerHTML = '';
  const group = document.createElement('div');
  group.className = 'dedup-group';
  const title = document.createElement('div');
  title.className = 'fix-group-title';
  title.textContent = `可修复文件（${_fixItems.length}）`;
  group.appendChild(title);
  if (!_fixItems.length) {
    const empty = document.createElement('div');
    empty.className = 'fix-empty';
    empty.textContent = '未发现可修复文件';
    group.appendChild(empty);
  }
  for (const it of _fixItems) group.appendChild(makeFixRow(it));
  body.appendChild(group);
  $('#fixOk').textContent = '开始修复';
  updateFixSub();
}

function makeFixRow(it) {
  const row = document.createElement('div');
  row.className = 'fix-row' + (it.ok ? (it.checked ? ' on' : '') : ' off');
  if (it.ok) {
    row.onclick = () => {
      it.checked = !it.checked;
      row.classList.toggle('on', it.checked);
      updateFixSub();
    };
  }
  row.innerHTML =
    '<span class="fix-check"></span>' +
    `<span class="fix-name" data-tip="${esc(it.path)}">${esc(it.name)}</span>` +
    (it.ok
      ? (it.convert
          ? `<span class="fix-info">${esc(it.size_str || '')}</span>` +
            '<span class="fix-arrow">→ .mp4</span>'
          : `<span class="fix-info">${esc((it.codecs || '').toUpperCase())}</span>` +
            `<span class="fix-arrow">→ ${esc(it.ext)}</span>`)
      : `<span class="fix-why">${esc(it.reason || '无法识别')}</span>`);
  return row;
}

function updateFixSub() {
  const sel = _fixItems.filter(it => it.ok && it.checked);
  const n = sel.filter(it => !it.convert).length;
  const t = sel.filter(it => it.convert).length;
  const parts = [];
  if (n) parts.push(`补全 ${n} 个后缀`);
  if (t) parts.push(`转换 ${t} 个视频`);
  $('#fixSub').textContent = parts.join(' · ') || '未选择修复项';
  $('#fixOk').disabled = _fixBusy || !parts.length;
}

function renderFixScanning(msg) {
  $('#fixBody').innerHTML = '<div class="dedup-interlude"><div class="dedup-spinner"></div>' +
    `<div>${esc(msg || '正在处理…')}</div></div>`;
  $('#fixSub').textContent = '';
  $('#fixOk').disabled = true;
}

function renderFixInterlude(msg) {
  $('#fixBody').innerHTML = `<div class="dedup-interlude"><div>${esc(msg)}</div></div>`;
  $('#fixSub').textContent = '';
  $('#fixOk').disabled = true;
}

async function confirmFix() {
  if (_fixBusy) return;
  const sel = _fixItems.filter(it => it.ok && it.checked);
  if (!sel.length) {
    toast('未选择任何修复项', 'err');
    renderFix();
    return;
  }
  const gen = ++_fixGen;
  _fixBusy = true;
  const renamedList = [];
  const skipList = [];
  const renameItems = sel.filter(it => !it.convert);
  if (renameItems.length) {
    renderFixScanning('正在补全后缀…');
    const r = await callApi('apply_renames', renameItems.map(it => it.path));
    if (gen !== _fixGen) return;
    if (!r || r.error) {
      _fixBusy = false;
      renderFixInterlude('补全失败: ' + ((r && r.error) || '请重试'));
      return;
    }
    renamedList.push(...(r.fix_renamed || []));
    skipList.push(...(r.fix_skipped || []));
    loadFromResult(r, true);
  }
  const queue = sel.filter(it => it.convert);
  const converted = [];
  const skippedTs = [];
  const fails = [];
  let mp4Cnt = 0, mkvCnt = 0;
  for (let i = 0; i < queue.length; i++) {
    if (gen !== _fixGen) return;
    renderFixScanning(`正在转换 ${i + 1}/${queue.length}：${queue[i].name}`);
    const r = await callApi('convert_ts', queue[i].path);
    if (gen !== _fixGen) return;
    if (r && r.ok) {
      converted.push(queue[i].name);
      if (r.container === 'mp4') mp4Cnt++; else mkvCnt++;
    } else if (r && r.skipped) {
      skippedTs.push({ name: queue[i].name, reason: r.reason || '已跳过' });
    } else {
      fails.push({ name: queue[i].name, reason: (r && r.reason) || '转换失败' });
    }
  }
  const rr = await callApi('scan');
  _fixBusy = false;
  if (gen !== _fixGen) {
    if (rr) loadFromResult(rr, true);
    return;
  }
  if (rr) loadFromResult(rr, true);
  renderFixResult(renamedList, skipList, converted, mp4Cnt, mkvCnt, skippedTs, fails);
}

function renderFixResult(renamedList, skipList, converted, mp4Cnt, mkvCnt, skippedTs, fails) {
  const body = $('#fixBody');
  body.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'dedup-interlude';
  const lines = [];
  if (renamedList.length) lines.push(`已补全 ${renamedList.length} 个后缀`);
  if (converted.length) {
    lines.push(`已转换 ${converted.length} 个流式封装视频（MP4 ${mp4Cnt}${mkvCnt ? ` · MKV ${mkvCnt}` : ''}）`);
  }
  if (!lines.length) lines.push('未执行任何修复');
  let html = `<div>${lines.map(esc).join('，')}</div>`;
  const failRows = [...skipList.map(s => `${s.name}：${s.reason}`),
                    ...skippedTs.map(s => `${s.name}：${s.reason}`),
                    ...fails.map(f => `${f.name}：${f.reason}`)];
  if (failRows.length) {
    html += '<div class="fix-fails">' +
      failRows.map(x => `<div>${esc(x)}</div>`).join('') + '</div>';
  }
  div.innerHTML = html;
  body.appendChild(div);
  $('#fixSub').textContent = '完成';
  $('#fixOk').disabled = true;
}

function closeFix() {
  const wasBusy = _fixBusy;
  _fixGen++;
  _fixBusy = false;
  $('#fixBg').classList.remove('show');
  _fixItems = [];
  if (wasBusy) {
    apiCall('cancel_format_fix').catch(() => {});
    callApi('scan').then(r => { if (r) loadFromResult(r, true); });
  }
}

$('#btn-closefix').addEventListener('click', closeFix);
$('#fixCancel').addEventListener('click', closeFix);
$('#fixOk').addEventListener('click', confirmFix);
$('#fixBg').addEventListener('click', (e) => { if (e.target === $('#fixBg')) closeFix(); });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && $('#fixBg').classList.contains('show')) closeFix();
});
