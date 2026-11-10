/* ════════════════════════════════════════════════════════════
   详情
   ════════════════════════════════════════════════════════════ */
let _detailSeq = 0;

function showDetail(v, quiet) {
  if (v.status === 'pending' && (state.transcribeEnabled || state.taggerEnabled)) {
    showPendingDetail(v, quiet);
    return;
  }
  ++_detailSeq;
  if (!quiet) switchTab('detail');
  const d = $('#detail');
  const tags = (v.tags || []).map(t =>
    `<button class="chip" data-tag="${esc(t)}">${esc(t)}</button>`).join('')
    || '<span class="filemeta">无标签</span>';
  d.innerHTML = `
    <div class="row">
      <div class="chips">${tags}</div>
    </div>
    <div class="row">
      <div class="plot">${esc(v.plot || '—')}</div>
    </div>`;
  d.querySelectorAll('.chip').forEach(ch => {
    ch.addEventListener('click', () => onTagClick(ch.dataset.tag));
    ch.addEventListener('contextmenu', e => showChipMenu(e, ch.dataset.tag));
  });
  if (state.primaryId === v.id && state.selected.size === 1) {
    const si = $('#selectedInfo');
    if (si) {
      const txt = v.path || '';
      si.style.display = '';
      si.textContent = txt;
      si.dataset.tip = txt;
    }
  }
}

async function showPendingDetail(v, quiet) {
  const seq = ++_detailSeq;
  if (!quiet) switchTab('detail');
  const d = $('#detail');
  d.innerHTML = '<div class="empty">加载中…</div>';

  let tagger = null, transcript = null;
  const [p, w] = await Promise.all([
    state.taggerEnabled ? apiCall('get_tagger_tags', v.id).catch(() => null) : null,
    state.transcribeEnabled ? apiCall('get_transcribe_transcript', v.id).catch(() => null) : null,
  ]);
  if (seq !== _detailSeq) return;
  tagger = p; transcript = w;

  const taggerDone = !!(tagger && tagger.ok);
  const showTagger = taggerDone &&
    ((tagger.character_tags && tagger.character_tags.length) || (tagger.ip_tags && tagger.ip_tags.length));
  const hasTranscript = transcript && transcript.ok && transcript.text;

  if (!showTagger && !hasTranscript) {
    const hints = [];
    if (state.taggerEnabled && !taggerDone) hints.push('右键「生成辅助分类标签」');
    if (state.transcribeEnabled) hints.push('右键「语音转录」');
    d.innerHTML = `<div class="empty">该视频暂无分析数据` +
      (hints.length ? `<br/><span style="font-size:11px;color:var(--faint)">${hints.join(' / ')}</span>` : '') +
      `</div>`;
    return;
  }

  let html = '';
  if (showTagger) {
    // data-tagger-tag=英文映射键；data-zh=库中文名
    const tagChip = t =>
      `<button class="chip" data-tagger-tag="${esc(t.name)}"${t.zh ? ` data-zh="${esc(t.zh)}"` : ''}>${esc(t.zh ? `${t.zh} / ${t.name}` : t.name)} <small>${t.score != null ? (t.score * 100).toFixed(0) : '—'}%</small></button>`;
    const charTags = (tagger.character_tags || []).map(tagChip).join('')
      || '<span class="filemeta">未识别到角色</span>';
    const ipTags = (tagger.ip_tags || []).map(tagChip).join('')
      || '<span class="filemeta">未识别到IP</span>';
    html += `
    <div class="row">
      <h4>角色标签</h4>
      <div class="chips">${charTags}</div>
    </div>
    <div class="row">
      <h4>IP / 版权标签</h4>
      <div class="chips">${ipTags}</div>
    </div>`;
  }
  if (hasTranscript) {
    const lang = transcript.language ? ` <small style="color:var(--faint)">(${esc(transcript.language)})</small>` : '';
    html += `
    <div class="row">
      <h4>语音转录${lang}</h4>
      <div class="transcript-box">${esc(transcript.text)}</div>
    </div>`;
  }
  d.innerHTML = html;
  d.querySelectorAll('.chip[data-tagger-tag]').forEach(ch => {
    ch.addEventListener('click', () => onTaggerTagClick(ch.dataset.taggerTag));
    // 右键：映射态=编辑标签；降级态=加入标签检索（related 预填英文名）
    const mapped = !!ch.dataset.zh;
    ch.addEventListener('contextmenu', e => showChipMenu(e, ch.dataset.taggerTag,
      { webSearch: true, edit: mapped, asRelated: !mapped,
        group: mapped ? '' : 'Hammy-tagger' }));
  });
}

async function webSearchTag(name) {
  try {
    const r = await apiCall('open_bing_search', name);
    if (!r || !r.ok) toast('打开浏览器失败: ' + ((r && r.error) || '未知错误'), 'err');
  } catch (e) {
    toast('打开浏览器失败: ' + ((e && e.message) || e), 'err');
  }
}

function onTagClick(tag) {
  setDropdownValue($('#searchMode'), 'tags');
  expandSearch();
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
}

function onTaggerTagClick(name) {
  setDropdownValue($('#searchMode'), 'tagger');
  expandSearch();
  $('#searchInput').value = name;
  $('#searchInput').placeholder = '空格分隔多关键词';
  state.search = name;
  state.searchMode = 'tagger';
  $('#searchWidget').classList.add('has-text');
  $('#searchInput').focus();
  if (state.view !== 'pending') {
    switchView('pending');
  } else {
    renderGrid();
  }
}

/* ════════════════════════════════════════════════════════════
   搜索栏交互
   ════════════════════════════════════════════════════════════ */
/* 搜索模式按视图启停：标签/标题/简介仅已处理视图；角色/IP 仅待处理视图且启用 tagger */
function syncSearchModes() {
  const dd = $('#searchMode');
  if (!dd) return;
  const inProcessed = state.view === 'processed';
  const showTagger = state.view === 'pending' && state.taggerEnabled;
  ['tags', 'title', 'plot'].forEach(m => setDropdownDisabled(dd, m, !inProcessed));
  setDropdownDisabled(dd, 'tagger', !showTagger);
  const mode = state.searchMode || 'all';
  const ok = mode === 'all' || mode === 'filename'
    || (inProcessed && ['tags', 'title', 'plot'].includes(mode))
    || (showTagger && mode === 'tagger');
  if (!ok) {
    state.searchMode = 'all';
    setDropdownValue(dd, 'all');
  }
}

function expandSearch() {
  const w = $('#searchWidget');
  if (!w) return;
  w.classList.add('open');
  const inp = $('#searchInput');
  if (!inp || document.activeElement === inp) return;
  inp.focus();
  if (document.activeElement !== inp) requestAnimationFrame(() => inp.focus());
}
function collapseSearch(force) {
  const w = $('#searchWidget');
  if (!w) return;
  const inp = $('#searchInput');
  if (!force && inp && inp.value) return;
  w.classList.remove('open');
}
$('#btn-search-toggle').addEventListener('click', (e) => {
  e.stopPropagation();
  if ($('#searchWidget').classList.contains('open')) collapseSearch(true);
  else expandSearch();
});
document.addEventListener('click', (e) => {
  if (!$('#searchWidget').classList.contains('open')) return;
  if (e.target.closest('#searchWidget')) return;
  collapseSearch();
});
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape' || !$('#searchWidget').classList.contains('open')) return;
  if (document.querySelector('.modal-bg.show,.confirm-bg.show,#dedupBg.show')) return;
  collapseSearch(true);
});
initDropdown($('#searchMode'), val => {
  state.searchMode = val;
  renderGrid();
});
let _searchTimer;
$('#searchInput').addEventListener('input', e => {
  clearTimeout(_searchTimer);
  _searchTimer = setTimeout(() => {
    state.search = e.target.value;
    $('#searchWidget').classList.toggle('has-text', !!e.target.value);
    renderGrid();
  }, 150);
});
$('#searchInput').addEventListener('focus', () => {
  $('#searchInput').placeholder = '空格分隔多关键词';
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
  $('#searchInput').focus();
});
