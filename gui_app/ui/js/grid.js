/* ════════════════════════════════════════════════════════════
   列表加载
   ════════════════════════════════════════════════════════════ */
function loadFromResult(result, preserveScroll) {
  if (result.error) { toast('扫描出错: ' + result.error, 'err'); return; }
  state.pending = result.pending || [];
  state.processed = result.processed || [];
  state.failed = result.failed || [];
  state.roots = result.roots || [];
  state.adhoc_files = result.adhoc_files || [];
  state.sourceFilter = null;
  state.search = '';
  $('#searchInput').value = '';
  const allIds = new Set([...state.pending, ...state.processed, ...state.failed].map(x => x.id));
  state.selected = new Set([...state.selected].filter(id => allIds.has(id)));
  if (state.selAnchor && !allIds.has(state.selAnchor)) state.selAnchor = null;
  if (state.primaryId && !allIds.has(state.primaryId)) state.primaryId = null;
  for (const id of [...state.thumbCache.keys()]) {
    if (!allIds.has(id)) state.thumbCache.delete(id);
  }
  renderSources();
  if (!state.hasAutoSwitched) {
    const total = state.pending.length + state.processed.length + state.failed.length;
    if (total > 0) {
      state.hasAutoSwitched = true;
      if (state.pending.length) state.view = 'pending';
      else if (state.processed.length) state.view = 'processed';
      else state.view = 'failed';
      $$('.pill.clickable[data-view]').forEach(p =>
        p.classList.toggle('active', p.dataset.view === state.view));
    }
  }
  if (preserveScroll) refreshGridData();
  else renderGrid();
  updateStats();
  updateSelectedInfo();
}

function currentList() {
  let list;
  if (state.view === 'processed') list = state.processed;
  else if (state.view === 'failed') list = state.failed;
  else list = state.pending;
  if (state.search) {
    const kws = state.search.toLowerCase().split(/\s+/).filter(Boolean);
    if (kws.length) {
      const mode = state.searchMode || 'all';
      list = list.filter(v => {
        if (mode === 'tags') {
          const tags = (v.tags || []).map(t => t.toLowerCase());
          return kws.every(k => tags.includes(k));
        }
        let hay;
        if (mode === 'filename') {
          hay = (v.name || '').toLowerCase();
        } else if (mode === 'title') {
          hay = (v.title || '').toLowerCase();
        } else if (mode === 'plot') {
          hay = (v.plot || '').toLowerCase();
        } else {
          if (state.view === 'processed') {
            hay = ((v.title || '') + ' ' + (v.plot || '') + ' ' + ((v.tags || []).join(' ')) + ' ' + (v.name || '')).toLowerCase();
          } else {
            hay = (v.name || '').toLowerCase();
          }
        }
        return kws.every(k => hay.includes(k));
      });
    }
  }
  if (state.sourceFilter) {
    const sf = state.sourceFilter.toLowerCase().replace(/[\\/]+$/, '');
    list = list.filter(v => {
      const vp = (v.path || '').toLowerCase();
      return vp === sf || vp.startsWith(sf + '\\') || vp.startsWith(sf + '/');
    });
  }
  return _applySort(list);
}

/* ════════════════════════════════════════════════════════════
   排序
   ════════════════════════════════════════════════════════════ */
const _NAME_CMP = new Intl.Collator('zh-CN', { numeric: true, sensitivity: 'base' });

const _dirOf = v => {
  const p = v.path || v.name || '';
  return p.slice(0, p.length - (v.name || '').length);
};

function _markDirBands(list) {
  let band = 0, prev = null;
  for (const v of list) {
    const d = _dirOf(v);
    if (d !== prev) { band = 1 - band; prev = d; }
    v._dirBand = band;
  }
}

function _sortCmp(mode) {
  switch (mode) {
    case 'dir':
      return (a, b) => _NAME_CMP.compare(_dirOf(a), _dirOf(b))
        || _NAME_CMP.compare(a.name, b.name);
    case 'name_asc':       return (a, b) => _NAME_CMP.compare(a.name, b.name);
    case 'name_desc':      return (a, b) => _NAME_CMP.compare(b.name, a.name);
    case 'mtime_desc':     return (a, b) => (b.mtime || 0) - (a.mtime || 0);
    case 'mtime_asc':      return (a, b) => (a.mtime || 0) - (b.mtime || 0);
    case 'size_desc':      return (a, b) => (b.size || 0) - (a.size || 0);
    case 'size_asc':       return (a, b) => (a.size || 0) - (b.size || 0);
    case 'duration_desc':  return (a, b) => (b.duration || 0) - (a.duration || 0);
    case 'duration_asc':   return (a, b) => (a.duration || 0) - (b.duration || 0);
    case 'processed_desc': return (a, b) => (b.processed_at || 0) - (a.processed_at || 0);
    default:               return null;
  }
}

function _typeRank(v, realFirst) {
  if (state.pixaiRealIds.has(v.id)) return realFirst ? 0 : 2;
  if (state.pixaiAnimeIds.has(v.id)) return realFirst ? 2 : 0;
  if (state.pixaiUncertainIds.has(v.id)) return 1;
  return 3;
}

function _applySort(list) {
  const mode = state.sortBy || 'default';
  if (mode === 'default' || list.length < 2) return list;
  const arr = list.slice();
  if (mode === 'type_real' || mode === 'type_anime') {
    const realFirst = mode === 'type_real';
    arr.sort((a, b) => _typeRank(a, realFirst) - _typeRank(b, realFirst)
      || _NAME_CMP.compare(a.name, b.name));
  } else {
    const cmp = _sortCmp(mode);
    if (!cmp) return list;
    arr.sort((a, b) => cmp(a, b) || _NAME_CMP.compare(a.name, b.name));
    if (mode === 'dir') _markDirBands(arr);
  }
  return arr;
}

function _sortDim(mode) {
  if (!mode || mode === 'default') return 'default';
  if (mode === 'type_real' || mode === 'type_anime') return 'type';
  return mode.split('_')[0];
}

function _sortLabelFor(mode) {
  switch (mode) {
    case 'name_asc':       return '文件名 ↑';
    case 'name_desc':      return '文件名 ↓';
    case 'dir':            return '目录分组';
    case 'mtime_desc':     return '修改时间 ↓';
    case 'mtime_asc':      return '修改时间 ↑';
    case 'size_desc':      return '文件大小 ↓';
    case 'size_asc':       return '文件大小 ↑';
    case 'duration_desc':  return '视频时长 ↓';
    case 'duration_asc':   return '视频时长 ↑';
    case 'processed_desc': return '处理时间 ↓';
    case 'type_real':      return '类型：非二次元优先';
    case 'type_anime':     return '类型：二次元优先';
    default:               return '默认顺序';
  }
}

const _SORT_DEFAULT_MODE = {
  name: 'name_asc', mtime: 'mtime_desc', size: 'size_desc',
  duration: 'duration_desc', processed: 'processed_desc', type: 'type_real',
  dir: 'dir',
};

function _toggleSortMode(dim, cur) {
  if (dim === 'type') return cur === 'type_real' ? 'type_anime' : 'type_real';
  if (dim === 'dir') return 'dir';
  return cur === dim + '_asc' ? dim + '_desc' : dim + '_asc';
}

function syncSortLabel() {
  const dd = $('#sortSel');
  if (!dd) return;
  const dim = _sortDim(state.sortBy);
  dd.querySelectorAll('.dd-opt').forEach(o =>
    o.classList.toggle('active', o.dataset.value === dim));
  dd.querySelector('.dd-label').textContent = _sortLabelFor(state.sortBy);
}

function updateSortDropdown() {
  const dd = $('#sortSel');
  if (!dd) return;
  const showType = !!state.pixaiTaggerEnabled && state.view === 'pending';
  const showProcessed = state.view === 'processed';
  dd.querySelectorAll('.dd-opt').forEach(o => {
    const v = o.dataset.value;
    let hidden = false;
    if (v === 'type') hidden = !showType;
    else if (v === 'duration' || v === 'processed') hidden = !showProcessed;
    o.style.display = hidden ? 'none' : '';
  });
  const curOpt = dd.querySelector(`.dd-opt[data-value="${_sortDim(state.sortBy)}"]`);
  if (!curOpt || curOpt.style.display === 'none') {
    state.sortBy = 'default';
  }
  syncSortLabel();
}

/* ════════════════════════════════════════════════════════════
   虚拟滚动引擎
   ════════════════════════════════════════════════════════════ */
const _VS_BUFFER_ROWS = 5;
const _VS_GAP = 14;
const _VS_MIN_COL = 170;
const _VS_META_H = 54;

let _vsList = [];
let _vsCols = 1;
let _vsRowH = 100;
let _vsStartRow = -1;
let _vsEndRow = -1;
let _thumbLoader = null;
let _scrollRAF = null;
let _cardMap = new Map();
let _layoutDirty = true;
let _lastWrapW = 0;

function _calcLayout() {
  const wrap = $('#gridWrap');
  const w = wrap.clientWidth - 28;
  if (w === _lastWrapW && !_layoutDirty) return;
  _lastWrapW = w;
  _layoutDirty = false;
  _vsCols = Math.max(1, Math.floor((w + _VS_GAP) / (_VS_MIN_COL + _VS_GAP)));
  const colW = (w - (_vsCols - 1) * _VS_GAP) / _vsCols;
  const thumbH = colW * 9 / 16;
  _vsRowH = thumbH + _VS_META_H + _VS_GAP;
  if (_vsList.length) {
    const totalRows = Math.ceil(_vsList.length / _vsCols);
    $('#gridInner').style.height = (totalRows * _vsRowH) + 'px';
  }
}

function _markLayoutDirty() { _layoutDirty = true; }

function _calibrateRowH() {
  const grid = $('#grid');
  const cards = grid.querySelectorAll('.card');
  if (cards.length < 2) return;
  let maxH = 0;
  const sampleN = Math.min(cards.length, 3);
  for (let i = 0; i < sampleN; i++) {
    const h = cards[i].getBoundingClientRect().height;
    if (h > maxH) maxH = h;
  }
  const calibrated = maxH + _VS_GAP;
  if (Math.abs(calibrated - _vsRowH) > 1) {
    _vsRowH = calibrated;
    const totalRows = Math.ceil(_vsList.length / _vsCols);
    $('#gridInner').style.height = (totalRows * _vsRowH) + 'px';
    $('#grid').style.transform = `translateY(${_vsStartRow * _vsRowH}px)`;
  }
}

function _renderVisible() {
  const wrap = $('#gridWrap');
  const totalRows = Math.ceil(_vsList.length / _vsCols);
  const scrollTop = wrap.scrollTop;
  const viewH = wrap.clientHeight;

  let startRow = Math.floor(scrollTop / _vsRowH) - _VS_BUFFER_ROWS;
  let endRow = Math.ceil((scrollTop + viewH) / _vsRowH) + _VS_BUFFER_ROWS;
  startRow = Math.max(0, startRow);
  endRow = Math.min(totalRows, endRow);

  if (startRow === _vsStartRow && endRow === _vsEndRow) return;
  _vsStartRow = startRow;
  _vsEndRow = endRow;

  const startIdx = startRow * _vsCols;
  const endIdx = Math.min(endRow * _vsCols, _vsList.length);

  $('#grid').style.transform = `translateY(${startRow * _vsRowH}px)`;

  if (!_thumbLoader) {
    _thumbLoader = new ThumbLoader(6, (v) => {
      if (!v.duration || !v.resolution || !v.codec) loadProbe(v);
    });
  }

  const newIds = new Set();
  for (let i = startIdx; i < endIdx; i++) newIds.add(_vsList[i].id);

  for (const [id, card] of _cardMap) {
    if (!newIds.has(id)) {
      card.remove();
      _cardMap.delete(id);
    }
  }

  const grid = $('#grid');
  const newCards = [];
  for (let i = startIdx; i < endIdx; i++) {
    const v = _vsList[i];
    if (_cardMap.has(v.id)) continue;
    const card = makeCard(v);
    _cardMap.set(v.id, card);
    newCards.push([card, v]);
  }
  if (newCards.length) {
    const ordered = [];
    for (let i = startIdx; i < endIdx; i++) {
      const c = _cardMap.get(_vsList[i].id);
      if (c) ordered.push(c);
    }
    grid.replaceChildren(...ordered);
    for (const [card, v] of newCards) _thumbLoader.observe(card, v);
  }
}

function renderGrid() {
  updateSortDropdown();
  _vsList = currentList();
  const hasSources = state.roots.length || state.adhoc_files.length;
  $('#gridWrap').style.display = (_vsList.length || hasSources) ? 'block' : 'none';
  $('#emptyState').style.display = (_vsList.length || hasSources) ? 'none' : 'flex';

  if (!_vsList.length) {
    $('#grid').innerHTML = '';
    $('#gridInner').style.height = '0';
    if (_thumbLoader) _thumbLoader.destroy();
    _thumbLoader = null;
    _cardMap.clear();
    return;
  }
  _markLayoutDirty();
  _calcLayout();
  $('#gridWrap').scrollTop = 0;
  _vsStartRow = -1; _vsEndRow = -1;
  if (_thumbLoader) _thumbLoader.destroy();
  _thumbLoader = null;
  _cardMap.clear();
  $('#grid').innerHTML = '';
  _renderVisible();
  _calibrateRowH();
  const grid = $('#grid');
  grid.classList.remove('fade-in');
  void grid.offsetWidth;
  grid.classList.add('fade-in');
}

function refreshGridData() {
  updateSortDropdown();
  _vsList = currentList();
  _vsStartRow = -1; _vsEndRow = -1;
  if (_thumbLoader) _thumbLoader.destroy();
  _thumbLoader = null;
  _cardMap.clear();
  $('#grid').innerHTML = '';
  _markLayoutDirty();
  _calcLayout();
  _renderVisible();
}

function removeFromGrid(id) {
  const idx = _vsList.findIndex(v => v.id === id);
  if (idx >= 0) _vsList.splice(idx, 1);
  if (state.sortBy === 'dir') _markDirBands(_vsList);
  const card = _cardMap.get(id);
  if (card) { card.remove(); _cardMap.delete(id); }
  state.thumbCache.delete(id);
  const totalRows = Math.ceil(_vsList.length / _vsCols);
  $('#gridInner').style.height = (totalRows * _vsRowH) + 'px';
  if (!_vsList.length) {
    renderGrid();
    return;
  }
  _vsStartRow = -1; _vsEndRow = -1;
  _renderVisible();
}

$('#gridWrap').addEventListener('scroll', () => {
  if (_scrollRAF) return;
  _scrollRAF = requestAnimationFrame(() => {
    _scrollRAF = null;
    _renderVisible();
  });
}, { passive: true });

window.addEventListener('resize', () => {
  if (!_vsList.length) return;
  _markLayoutDirty();
  _calcLayout();
  _vsStartRow = -1; _vsEndRow = -1;
  _renderVisible();
});

/* ════════════════════════════════════════════════════════════
   缩略图懒加载
   ════════════════════════════════════════════════════════════ */
class ThumbLoader {
  constructor(maxInflight = 6, onVisible = null) {
    this._maxInflight = maxInflight;
    this._inflight = 0;
    this._queue = [];
    this._loading = new Set();
    this._cardVideoMap = new Map();
    this._observer = null;
    this._onVisible = onVisible;
  }

  observe(card, video) {
    const cached = state.thumbCache.get(video.id);
    if (cached) this._attachThumb(card, cached, true);
    else {
      const thumb = card.querySelector('.thumb');
      if (thumb) thumb.classList.add('loading');
    }
    this._cardVideoMap.set(video.id, video);
    this._ensureObserver();
    this._observer.observe(card);
  }

  destroy() {
    if (this._observer) { this._observer.disconnect(); this._observer = null; }
    this._cardVideoMap.clear();
    this._queue.length = 0;
    this._inflight = 0;
    this._loading.clear();
  }

  _ensureObserver() {
    if (this._observer) return;
    this._observer = new IntersectionObserver((entries, obs) => {
      for (const en of entries) {
        if (!en.isIntersecting) continue;
        const card = en.target;
        obs.unobserve(card);
        const v = this._cardVideoMap.get(card.dataset.id);
        if (!v) continue;
        this._queue.push(v);
        this._pump();
        if (this._onVisible) this._onVisible(v, card);
      }
    }, { root: $('#gridWrap'), rootMargin: '200px', threshold: 0.01 });
  }

  _pump() {
    while (this._inflight < this._maxInflight && this._queue.length) {
      const v = this._queue.shift();
      this._inflight++;
      this._loadThumb(v).finally(() => { this._inflight--; this._pump(); });
    }
  }

  async _loadThumb(v) {
    if (this._loading.has(v.id)) return;
    const cached = state.thumbCache.get(v.id);
    if (cached) {
      const card = $('#grid').querySelector(`.card[data-id="${v.id}"]`);
      if (card) this._attachThumb(card, cached, true);
      return;
    }
    this._loading.add(v.id);
    try {
      const url = await apiCall('get_thumb', v.path, v.id);
      if (!url) {
        const card = $('#grid').querySelector(`.card[data-id="${v.id}"]`);
        if (card) { const t = card.querySelector('.thumb'); if (t) t.classList.remove('loading'); }
        return;
      }
      state.thumbCache.set(v.id, url);
      const card = $('#grid').querySelector(`.card[data-id="${v.id}"]`);
      if (!card) return;
      this._attachThumb(card, url);
    } catch (e) {
      const card = $('#grid').querySelector(`.card[data-id="${v.id}"]`);
      if (card) { const t = card.querySelector('.thumb'); if (t) t.classList.remove('loading'); }
    } finally { this._loading.delete(v.id); }
  }

  _attachThumb(card, url, instant) {
    const thumb = card.querySelector('.thumb');
    if (!thumb || thumb.querySelector('img')) return;
    const img = document.createElement('img');
    img.alt = '';
    if (instant) {
      img.classList.add('loaded');
      img.src = url;
      thumb.classList.remove('loading');
      const ph = thumb.querySelector('.ph');
      if (ph) ph.remove();
    } else {
      img.onload = () => {
        img.classList.add('loaded');
        thumb.classList.remove('loading');
        const ph = thumb.querySelector('.ph');
        if (ph) ph.remove();
      };
      img.src = url;
    }
    thumb.insertBefore(img, thumb.firstChild);
  }
}

/* ════════════════════════════════════════════════════════════
   卡片创建
   ════════════════════════════════════════════════════════════ */
function makeCard(v) {
  const card = document.createElement('div');
  card.className = 'card' + (state.selected.has(v.id) ? ' selected' : '')
    + (state.sortBy === 'dir' && v._dirBand ? ' band' : '');
  card.tabIndex = 0;
  card.setAttribute('role', 'button');
  card.setAttribute('aria-label', v.name);
  card.dataset.id = v.id;
  card.dataset.path = v.path;
  card._video = v;
  const dotIdx = v.name.lastIndexOf('.');
  const displayName = dotIdx > 0 ? v.name.slice(0, dotIdx) : v.name;
  const ext = dotIdx > 0 ? v.name.slice(dotIdx + 1).toUpperCase() : '';
  v._ext = ext;
  const meta = _cardMeta(v);
  const durBadge = v.duration ? `<span class="dur-badge">${fmtDur(v.duration)}</span>` : '';
  const isReal = state.pixaiRealIds.has(v.id);
  const isAnime = state.pixaiAnimeIds.has(v.id);
  const isUnc = state.pixaiUncertainIds.has(v.id);
  let pixaiBadge = '';
  if (isReal) pixaiBadge += '<span class="tag-badge real" data-tip="预筛分类为非二次元作品">REAL</span>';
  if (isAnime) pixaiBadge += '<span class="tag-badge anime" data-tip="分类为二次元作品">ANIME</span>';
  if (isUnc) pixaiBadge += '<span class="tag-badge unc" data-tip="分类不确定（二次元/非二次元无法判定）">UNC</span>';
  if (state.pixaiTaggedIds.has(v.id)) pixaiBadge += '<span class="tag-badge pixai" data-tip="已获取角色/IP标签">IP</span>';
  const whisperBadge = state.whisperTranscribedIds.has(v.id) ? '<span class="tag-badge whisper" data-tip="已获取语音转录">CC</span>' : '';
  const badges = (pixaiBadge || whisperBadge) ? `<div class="badges">${pixaiBadge}${whisperBadge}</div>` : '';
  card.innerHTML = `
    <div class="thumb"><div class="ph">${icon('clapper','26px')}</div>${badges}</div>
    <div class="meta">
      <div class="name" data-tip="${esc(v.name)}" data-tip-trunc>${esc(displayName)}</div>
      <div class="stem">${esc(meta) || '—'}</div>
    </div>${durBadge}`;
  return card;
}

function _cardMeta(v) {
  const parts = [];
  if (v.resolution) parts.push(v.resolution);
  if (v._ext) parts.push(v._ext);
  if (v.codec) parts.push(v.codec.toUpperCase());
  return parts.join(' · ');
}

const _probeLimit = createLimiter(4);
const _probeInflight = new Map();
function loadProbe(v) {
  if (_probeInflight.has(v.id)) return _probeInflight.get(v.id);
  const p = _probeLimit(() => apiCall('get_probe', v.path, v.id)).then(r => {
    if (r && r.info) {
      const info = r.info;
      v.duration = info.duration || 0;
      if (info.size != null) v.size = info.size;
      v.resolution = info.resolution || '';
      v.codec = info.codec || '';
      v.audio_codec = info.audio_codec || '';
      v.has_audio = !!info.has_audio;
      const card = $('#grid').querySelector(`.card[data-id="${v.id}"]`);
      if (card) {
        const stem = card.querySelector('.stem');
        if (stem) stem.textContent = _cardMeta(v) || '—';
        if (v.duration) {
          let badge = card.querySelector('.dur-badge');
          if (!badge) {
            badge = document.createElement('span');
            badge.className = 'dur-badge';
            card.appendChild(badge);
          }
          badge.textContent = fmtDur(v.duration);
        }
      }
    }
  }).catch(() => {}).finally(() => _probeInflight.delete(v.id));
  _probeInflight.set(v.id, p);
  return p;
}

/* ════════════════════════════════════════════════════════════
   事件委托
   ════════════════════════════════════════════════════════════ */
const _EMPTY_DETAIL = '<div class="empty">点击「已处理」中的视频查看详情</div>';
$('#grid').addEventListener('click', e => {
  const card = e.target.closest('.card');
  if (!card || !card._video) return;
  selectCard(card._video, e);
});
$('#grid').addEventListener('dblclick', e => {
  const card = e.target.closest('.card');
  if (!card || !card._video) return;
  const v = card._video;
  if (!state.selected.has(v.id)) {
    state.selected = new Set([v.id]);
    state.selAnchor = v.id;
    state.primaryId = v.id;
    refreshSelectionUI();
  }
  if (v.status === 'processed') { showDetail(v); }
  else { callApi('play_video', v.path); }
});
$('#grid').addEventListener('contextmenu', e => {
  const card = e.target.closest('.card');
  if (!card || !card._video) return;
  e.preventDefault();
  showContextMenu(e, card._video);
});
$('#grid').addEventListener('keydown', e => {
  if (e.key !== 'Enter' && e.key !== ' ') return;
  const card = e.target.closest('.card');
  if (!card || !card._video) return;
  e.preventDefault();
  selectCard(card._video, e);
});

/* ════════════════════════════════════════════════════════════
   选择逻辑
   ════════════════════════════════════════════════════════════ */
function findVideoById(id) {
  for (const arr of [state.processed, state.failed, state.pending]) {
    const v = arr.find(x => x.id === id);
    if (v) return v;
  }
  return null;
}
function updateSelectedInfo() {
  const el = $('#selectedInfo');
  if (!el) return;
  const n = state.selected.size;
  if (n === 0) { el.textContent = ''; el.style.display = 'none'; return; }
  el.style.display = '';
  if (n > 1) { el.textContent = `已选 ${n} 个视频`; return; }
  const v = findVideoById(state.primaryId);
  if (v) {
    const title = v.title || v.name || '';
    const orig = v.original_name || '';
    let txt = title;
    if (orig && orig !== title) txt += ` ｜ 初始文件名: ${orig}`;
    txt += ` ｜ 路径: ${v.path || ''}`;
    el.textContent = txt;
    el.dataset.tip = txt;
  } else { el.textContent = ''; el.dataset.tip = ''; }
}
function refreshSelectionUI() {
  $('#grid').querySelectorAll('.card').forEach(c =>
    c.classList.toggle('selected', state.selected.has(c.dataset.id)));
  updateSelectedInfo();
  updateStatusCount();
  updateSelToolbar();
}
function selectCard(v, ev) {
  ev = ev || {};
  const id = v.id;
  const shiftRange = ev.shiftKey && !ev.ctrlKey && !ev.metaKey && state.selAnchor != null;
  const plain = !(ev.ctrlKey || ev.metaKey) && !shiftRange;
  if (plain && ev.detail > 1) return;
  state.selAnchor = pickSelection(state.selected, state.selAnchor, id, ev, _vsList.map(x => x.id));
  state.primaryId = shiftRange ? id : state.selAnchor;
  refreshSelectionUI();
  if (state.selected.size === 0) {
    $('#detail').innerHTML = _EMPTY_DETAIL;
  } else if (state.selected.size === 1 && v.status === 'processed'
             && !$('#pane-bottom').classList.contains('collapsed')) {
    showDetail(v);
  } else if (state.selected.size === 1 && v.status === 'pending'
             && (state.pixaiTaggerEnabled || state.whisperEnabled)
             && !$('#pane-bottom').classList.contains('collapsed')) {
    showPendingDetail(v);
  } else if (v.status !== 'processed') {
    $('#detail').innerHTML = _EMPTY_DETAIL;
  }
}
document.addEventListener('keydown', e => {
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a') {
    if (_vsList.length) {
      e.preventDefault();
      state.selected = new Set(_vsList.map(x => x.id));
      state.primaryId = _vsList[_vsList.length - 1].id;
      refreshSelectionUI();
    }
  }
});

/* ════════════════════════════════════════════════════════════
   选区操作工具栏
   ════════════════════════════════════════════════════════════ */
function updateSelToolbar() {
  const tb = $('#selToolbar');
  if (!tb) return;
  const n = state.selected.size;
  const view = state.view;
  $('#tb-process').style.display = view === 'pending' ? '' : 'none';
  $('#tb-export-nfo').style.display = view === 'processed' ? '' : 'none';
  $('#tb-restore').style.display = view === 'processed' ? '' : 'none';
  $('#tb-move-out').style.display = view === 'failed' ? '' : 'none';
  const allSel = _vsList.length > 0 && _vsList.every(x => state.selected.has(x.id));
  const selBtn = $('#tb-select-all');
  selBtn.disabled = !_vsList.length;
  selBtn.dataset.tbTip = allSel ? '取消全选' : '全选 (Ctrl+A)';
  selBtn.querySelector('use').setAttribute('href', allSel ? '#ic-deselect-all' : '#ic-select-all');
  $('#tb-export').disabled = !n;
  $('#tb-process').disabled = !n || !state.aiConnected || state.processing;
  $('#tb-export-nfo').disabled = !n;
  $('#tb-restore').disabled = !n;
  $('#tb-move-out').disabled = !n;
}
$('#tb-select-all').onclick = () => {
  if (!_vsList.length) return;
  const allSel = _vsList.every(x => state.selected.has(x.id));
  if (allSel) {
    _vsList.forEach(x => state.selected.delete(x.id));
    if (!state.selected.size) {
      state.selAnchor = null;
      state.primaryId = null;
      $('#detail').innerHTML = _EMPTY_DETAIL;
    }
  } else {
    state.selected = new Set(_vsList.map(x => x.id));
    state.primaryId = _vsList[_vsList.length - 1].id;
  }
  refreshSelectionUI();
};
$('#tb-export').onclick = () => exportToFolder(false);
$('#tb-process').onclick = () => processSelected();
$('#tb-export-nfo').onclick = () => {
  const ids = [...state.selected].filter(id => {
    const v = findVideoById(id);
    return v && v.status === 'processed';
  });
  if (ids.length) exportNfoBatch(ids);
};
$('#tb-restore').onclick = () => {
  const ids = [...state.selected].filter(id => {
    const v = findVideoById(id);
    return v && v.status === 'processed';
  });
  if (ids.length) restoreBatch(ids);
};
$('#tb-move-out').onclick = () => moveFailedOut();

initDropdown($('#sortSel'), dim => {
  const cur = state.sortBy || 'default';
  let mode;
  if (dim === 'default') mode = 'default';
  else if (_sortDim(cur) === dim) mode = _toggleSortMode(dim, cur);
  else mode = _SORT_DEFAULT_MODE[dim] || 'default';
  state.sortBy = mode;
  syncSortLabel();
  renderGrid();
  updateStatusCount();
});

/* ════════════════════════════════════════════════════════════
   视图切换
   ════════════════════════════════════════════════════════════ */
function switchView(view) {
  state.view = view;
  state.selected = new Set();
  state.selAnchor = null;
  state.primaryId = null;
  $$('.pill.clickable[data-view]').forEach(p =>
    p.classList.toggle('active', p.dataset.view === view));
  const noMeta = view !== 'processed';
  ['tags', 'title', 'plot'].forEach(m => {
    setDropdownDisabled($('#searchMode'), m, noMeta);
  });
  if (noMeta && ['tags', 'title', 'plot'].includes(state.searchMode)) {
    state.searchMode = 'all';
    setDropdownValue($('#searchMode'), 'all');
  }
  renderGrid();
  renderSources();
  updateStats();
  updateStatusCount();
  updateSelectedInfo();
  updateSelToolbar();
  $('#detail').innerHTML = _EMPTY_DETAIL;
}
function switchTab(tab) {
  $$('.tab[data-tab]').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  $('#body-progress').classList.toggle('active', tab === 'progress');
  $('#body-detail').classList.toggle('active', tab === 'detail');
  $('#pane-bottom').classList.remove('collapsed');
}
function gotoLog() {
  switchTab('progress');
  const log = $('#log');
  requestAnimationFrame(() => {
    log.scrollTop = log.scrollHeight;
    requestAnimationFrame(() => { log.scrollTop = log.scrollHeight; });
  });
}
$('#tab-progress').addEventListener('click', () => switchTab('progress'));
$('#tab-detail').addEventListener('click', () => switchTab('detail'));
$('#btn-collapse').addEventListener('click', () => {
  $('#pane-bottom').classList.toggle('collapsed');
});
function updateMiniProg() {
  const el = $('#miniProg'), arc = $('#miniProgArc');
  const box = $('#miniProgBox');
  if (!el) return;
  if (state.hfDownloading) {
    box.classList.add('show', 'dl');
    if (arc) arc.style.strokeDashoffset =
      (71.98 * (1 - Math.min(100, state.hfDlPct) / 100)).toFixed(2);
    const mpTxt = $('#miniProgPct');
    if (mpTxt) mpTxt.textContent = Math.round(state.hfDlPct);
    el.dataset.tip = `正在后台下载模型… ${Math.round(state.hfDlPct)}%（点击查看）`;
    return;
  }
  box.classList.remove('show', 'dl');
  el.dataset.tip = '';
}
let _collapseAccum = 0, _collapseTimer = null;
$('.pane.top').addEventListener('wheel', (e) => {
  if ($('#pane-bottom').classList.contains('collapsed')) return;
  if (e.target.closest('#sourceBar')) return;
  _collapseAccum += Math.abs(e.deltaY);
  clearTimeout(_collapseTimer);
  _collapseTimer = setTimeout(() => { _collapseAccum = 0; }, 500);
  if (_collapseAccum >= _vsRowH * 5) {
    $('#pane-bottom').classList.add('collapsed');
    _collapseAccum = 0;
  }
}, { passive: true });
function updateStats() {
  $('#stat-pending').textContent = state.pending.length;
  $('#stat-processed').textContent = state.processed.length;
  $('#stat-failed').textContent = state.failed.length;
  const _sp = { pending: state.pending.length, processed: state.processed.length, failed: state.failed.length };
  $$('.stats .pill[data-view]').forEach(p =>
    p.classList.toggle('has-items', (_sp[p.dataset.view] || 0) > 0));
  updateStartBtn();
  const fp = $('#folderPath');
  const hasSrc = !!(state.roots.length || state.adhoc_files.length);
  $('#btn-dedup').style.display = hasSrc ? '' : 'none';
  if (hasSrc) {
    fp.style.display = '';
    const parts = [];
    if (state.roots.length) parts.push(`${icon('folder-open')} ${state.roots.length} 个文件夹`);
    if (state.adhoc_files.length) parts.push(`${icon('paperclip')} ${state.adhoc_files.length} 个文件`);
    fp.innerHTML = parts.join('  ·  ') + ' ' + ddArrow('chevron');
  } else {
    fp.style.display = 'none';
  }
}
