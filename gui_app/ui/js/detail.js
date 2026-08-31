/* ════════════════════════════════════════════════════════════
   详情
   ════════════════════════════════════════════════════════════ */
let _detailSeq = 0;

async function showDetail(v) {
  if (v.status === 'pending' && (state.whisperEnabled || state.pixaiTaggerEnabled)) {
    showPendingDetail(v);
    return;
  }
  const seq = ++_detailSeq;
  switchTab('detail');
  const d = $('#detail');
  d.innerHTML = '<div class="empty">加载中…</div>';
  let data;
  try { data = await apiCall('get_nfo', v.id); }
  catch (e) { if (seq === _detailSeq) d.innerHTML = '<div class="empty">读取 NFO 失败</div>'; return; }

  if (seq !== _detailSeq) return;
  if (!data || !data.ok) {
    d.innerHTML = `<div class="empty">该视频暂无 NFO${data && data.error ? '（' + esc(data.error) + '）' : ''}</div>`;
    return;
  }
  const tags = (data.tags || []).map(t =>
    `<button class="chip" data-tag="${esc(t)}">${esc(t)}</button>`).join('')
    || '<span class="filemeta">无标签</span>';
  d.innerHTML = `
    <div class="row">
      <h4>标签（点击筛选同类视频）</h4>
      <div class="chips">${tags}</div>
    </div>
    <div class="row">
      <h4>简介</h4>
      <div class="plot">${esc(data.plot || '—')}</div>
    </div>`;
  d.querySelectorAll('.chip').forEach(ch =>
    ch.addEventListener('click', () => onTagClick(ch.dataset.tag)));
  if (state.primaryId === v.id && state.selected.size === 1) {
    const si = $('#selectedInfo');
    if (si) {
      const title = data.title || v.title || v.name || '';
      const orig = data.file_original_name || v.original_name || '';
      let txt = title;
      if (orig && orig !== title) txt += ` ｜ 初始文件名: ${orig}`;
      txt += ` ｜ 路径: ${v.path || ''}`;
      si.style.display = '';
      si.textContent = txt;
      si.dataset.tip = txt;
    }
  }
}

async function showPendingDetail(v) {
  const seq = ++_detailSeq;
  switchTab('detail');
  const d = $('#detail');
  d.innerHTML = '<div class="empty">加载中…</div>';

  let pixai = null, whisper = null;
  const [p, w] = await Promise.all([
    state.pixaiTaggerEnabled ? apiCall('get_pixai_tags', v.id).catch(() => null) : null,
    state.whisperEnabled ? apiCall('get_whisper_transcript', v.id).catch(() => null) : null,
  ]);
  if (seq !== _detailSeq) return;
  pixai = p; whisper = w;

  const pixaiDone = !!(pixai && pixai.ok);
  const showPixai = pixaiDone &&
    ((pixai.character_tags && pixai.character_tags.length) || (pixai.ip_tags && pixai.ip_tags.length));
  const hasWhisper = whisper && whisper.ok && whisper.text;

  if (!showPixai && !hasWhisper) {
    const hints = [];
    if (state.pixaiTaggerEnabled && !pixaiDone) hints.push('右键「获取IP信息」');
    if (state.whisperEnabled) hints.push('右键「语音转录」');
    d.innerHTML = `<div class="empty">该视频暂无分析数据` +
      (hints.length ? `<br/><span style="font-size:11px;color:var(--faint)">${hints.join(' / ')}</span>` : '') +
      `</div>`;
    return;
  }

  let html = '';
  if (showPixai) {
    const charTags = (pixai.character_tags || []).map(t =>
      `<button class="chip" data-web-tag="${esc(t.name)}">${esc(t.name)} <small>${t.score != null ? (t.score * 100).toFixed(0) : '—'}%</small></button>`).join('')
      || '<span class="filemeta">未识别到角色</span>';
    const ipTags = (pixai.ip_tags || []).map(t =>
      `<button class="chip" data-web-tag="${esc(t.name)}">${esc(t.name)}</button>`).join('')
      || '<span class="filemeta">未识别到IP</span>';
    html += `
    <div class="row">
      <h4>角色标签（置信度 &gt; 阈值）</h4>
      <div class="chips">${charTags}</div>
    </div>
    <div class="row">
      <h4>IP / 版权标签</h4>
      <div class="chips">${ipTags}</div>
    </div>`;
  }
  if (hasWhisper) {
    const lang = whisper.language ? ` <small style="color:var(--faint)">(${esc(whisper.language)})</small>` : '';
    html += `
    <div class="row">
      <h4>语音转录${lang}</h4>
      <div class="transcript-box">${esc(whisper.text)}</div>
    </div>`;
  }
  d.innerHTML = html;
  d.querySelectorAll('.chip[data-web-tag]').forEach(ch =>
    ch.addEventListener('click', () => onWebTagClick(ch.dataset.webTag)));
}

async function onWebTagClick(name) {
  if (!await showConfirm(`使用搜索引擎搜索「${name}」？`)) return;
  try {
    const r = await apiCall('open_bing_search', name);
    if (!r || !r.ok) toast('打开浏览器失败: ' + ((r && r.error) || '未知错误'), 'err');
  } catch (e) {
    toast('打开浏览器失败: ' + ((e && e.message) || e), 'err');
  }
}

function onTagClick(tag) {
  setDropdownValue($('#searchMode'), 'tags');
  $('#searchInput').value = tag;
  $('#searchInput').placeholder = '空格分隔多关键词';
  state.search = tag;
  state.searchMode = 'tags';
  $('#searchWidget').classList.add('has-text');
  $('#searchInput').focus();
  if (state.view !== 'processed') {
    switchView('processed');
  } else {
    renderGrid();
  }
  updateStatusCount();
}

/* ════════════════════════════════════════════════════════════
   搜索栏交互
   ════════════════════════════════════════════════════════════ */
initDropdown($('#searchMode'), val => {
  state.searchMode = val;
  renderGrid();
  updateStatusCount();
});
let _searchTimer;
$('#searchInput').addEventListener('input', e => {
  clearTimeout(_searchTimer);
  _searchTimer = setTimeout(() => {
    state.search = e.target.value;
    $('#searchWidget').classList.toggle('has-text', !!e.target.value);
    renderGrid();
    updateStatusCount();
  }, 150);
});
$('#searchInput').addEventListener('focus', () => {
  $('#searchInput').placeholder = '空格分隔多关键词';
  if (state.search) updateStatusCount();
});
$('#searchInput').addEventListener('blur', () => {
  if (!$('#searchInput').value) $('#searchInput').placeholder = '搜索…';
});
$('#btn-clear-search').addEventListener('click', () => {
  state.search = '';
  $('#searchInput').value = '';
  $('#searchWidget').classList.remove('has-text');
  $('#searchInput').placeholder = '空格分隔多关键词';
  renderGrid();
  updateStatusCount();
  $('#searchInput').focus();
});
function updateStatusCount() {
  const el = $('#selCount');
  if (state.search) {
    el.textContent = `匹配 ${currentList().length} 个`;
  } else if (state.selected.size > 0) {
    el.textContent = `已选 ${state.selected.size} 个`;
  } else {
    el.textContent = '';
  }
}
