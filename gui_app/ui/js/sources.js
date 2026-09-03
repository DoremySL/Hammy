/* ════════════════════════════════════════════════════════════
   源清单渲染
   ════════════════════════════════════════════════════════════ */
function renderSources() {
  const bar = $('#sourceBar');
  bar.innerHTML = '';
  if (!state.roots.length && !state.adhoc_files.length) {
    bar.innerHTML = '<span class="drop-hint">点击右上方「添加文件夹 / 添加文件」按钮添加源</span>';
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
    span.innerHTML = `${icon('folder-open')} <span class="nm" data-tip="${esc(r)}">${esc(shortPath(r))}</span> <span class="x" data-tip="移除" tabindex="0">×</span>`;
    span.querySelector('.x').onclick = (e) => { e.stopPropagation(); removeSource(r, false); };
    span.addEventListener('click', () => toggleSourceFilter(r));
    scroll.appendChild(span);
  }
  for (const f of state.adhoc_files) {
    const span = document.createElement('span');
    span.className = 'src' + (state.sourceFilter === f ? ' active' : '');
    span.dataset.tip = f + '（点击筛选该源）';
    span.innerHTML = `${icon('file')} <span class="nm" data-tip="${esc(f)}">${esc(shortPath(f))}</span> <span class="x" data-tip="移除" tabindex="0">×</span>`;
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
  const clr = document.createElement('button');
  clr.className = 'src-action danger';
  clr.innerHTML = icon('trash', '14px') + ' 清空源';
  clr.dataset.tip = '移除所有已添加的文件夹/文件';
  clr.onclick = clearSources;
  bar.appendChild(clr);
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
  if (r) loadFromResult(r);
}

/* ════════════════════════════════════════════════════════════
   添加源
   ════════════════════════════════════════════════════════════ */
async function addFolder() {
  const r = await callApi('pick_folders');
  if (!r || !r.length) return;
  toast('正在扫描…', null, false, 0);
  const result = await callApi('add_sources', r);
  if (!result) return;
  if (result.error) { loadFromResult(result); return; }
  loadFromResult(result);
  const total = (result.pending || []).length + (result.processed || []).length + (result.failed || []).length;
  const hasFailed = r.some(p => p.replace(/\\/g, '/').split('/').filter(Boolean).pop().toLowerCase() === '_failed');
  if (total === 0) {
    toast(hasFailed ? '已添加，但 _failed 目录内的视频已自动排除' : '未扫描到视频', 'err', true);
  } else {
    toast('已添加文件夹', 'ok', true);
  }
}
async function addFiles() {
  const r = await callApi('pick_files');
  if (!r || !r.length) return;
  toast('正在扫描…', null, false, 0);
  const result = await callApi('add_sources', r);
  if (!result) return;
  if (result.error) { loadFromResult(result); return; }
  loadFromResult(result);
  const total = (result.pending || []).length + (result.processed || []).length + (result.failed || []).length;
  if (total === 0) {
    toast('未扫描到视频', 'err', true);
  } else {
    toast(`已添加 ${r.length} 个文件`, 'ok', true);
  }
}
$('#btn-add-folder').addEventListener('click', addFolder);
$('#btn-add-files').addEventListener('click', addFiles);

/* ════════════════════════════════════════════════════════════
   刷新 / 清空 / 源栏切换
   ════════════════════════════════════════════════════════════ */
$('#btn-refresh').addEventListener('click', async () => {
  if (!state.roots.length && !state.adhoc_files.length) return;
  toast('正在刷新…');
  const r = await callApi('scan');
  if (!r) return;
  loadFromResult(r);
  toast('已刷新', 'ok', true);
});
async function clearSources() {
  if (!await showConfirm('将移除已添加的文件夹/文件。\n\n已处理记录仍保留，可在「设置 → 工作区」查看。\n\n确定清空所有源吗？', { okText: '清空' })) return;
  const r = await callApi('clear_sources');
  if (!r) return;
  state.hasAutoSwitched = false;
  loadFromResult(r);
  toast('已清空所有源', 'ok');
}
$('#folderPath').addEventListener('click', () => {
  const bar = $('#sourceBar');
  const show = bar.style.display === 'none';
  bar.style.display = show ? '' : 'none';
  $('#folderPath').classList.toggle('expanded', show);
  if (show) updateSrcFades();
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
  if (!state.pending.length) {
    $('#dedupBg').classList.add('show');
    showDedupInterlude('待处理列表为空', false);
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
    items: [g.keep, ...g.remove].map((it, i) =>
      ({ path: it.path, name: it.name, keep: i === 0, video: byPath.get(it.path) || null })),
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
    for (const it of g.items) cards.appendChild(makeDedupCard(it));
    const size = document.createElement('div');
    size.className = 'dedup-gsize';
    size.textContent = g.kind === 'identical'
      ? `完全相同 · ${g.items.length} 份 · 单文件 ${g.sizeStr}`
      : `同内容不同版本 · ${g.items.length} 份`;
    box.appendChild(cards);
    box.appendChild(size);
    body.appendChild(box);
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
    `<span class="badge">${it.keep ? '保留' : '移除'}</span></div>` +
    `<div class="dedup-meta">${dedupMetaHtml(it)}</div>`;
  card.onclick = () => {
    it.keep = !it.keep;
    card.classList.toggle('keep', it.keep);
    card.classList.toggle('rm', !it.keep);
    card.querySelector('.badge').textContent = it.keep ? '保留' : '移除';
    updateDedupSub();
  };
  card.oncontextmenu = (e) => showDedupCtxMenu(e, it);
  attachDedupThumb(card.querySelector('.dedup-thumb'), v, it);
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
    '<button data-i="1">在资源管理器中打开</button>';
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
      ({ ...it, keep: it.path === g.keep, video: byPath.get(it.path) || null })),
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
  loadFromResult(r);
  const moved = r.dedup_moved != null ? r.dedup_moved : paths.length;
  const failed = r.dedup_failed || 0;
  if (_dedupStage === 'identical') {
    showDedupInterlude(`已移除 ${moved} 个完全相同副本` + (failed ? `，失败 ${failed} 个` : ''));
  } else {
    showDedupInterlude(`已移除 ${moved} 个重复视频` + (failed ? `，失败 ${failed} 个` : ''));
  }
}

$('#btn-dedup').addEventListener('click', findDuplicates);
$('#btn-closededup').addEventListener('click', closeDedup);
$('#dedupCancel').addEventListener('click', closeDedup);
$('#dedupOk').addEventListener('click', confirmDedup);
$('#dedupBg').addEventListener('click', (e) => { if (e.target === $('#dedupBg')) closeDedup(); });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && $('#dedupBg').classList.contains('show')) closeDedup();
});
