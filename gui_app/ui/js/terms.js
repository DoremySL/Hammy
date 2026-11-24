/* ════════════════════════════════════════════════════════════
   设置 Modal - 词条检索页（双栏管理器，修改即时写盘）
   ════════════════════════════════════════════════════════════ */
function tmNormItems(items) {
  return (items || []).map(x => ({
    keyword: x.keyword || '',
    description: x.description || '',
    related: x.related || '',
    group: x.group || '',
  }));
}

function tmAllGroups(items) {
  return [...new Set(items.map(x => x.group).filter(Boolean))].sort();
}

async function persistTermLibrary() {
  const r = await callApi('save_term_library', state.tmItems, state.tmMode, [...state.tmDisabled]);
  if (!r || !r.ok) {
    toast('保存失败: ' + ((r && r.error) || '未知错误'), 'err');
    return false;
  }
  return true;
}

async function tmLeaveGuard() {
  if (!state.tmDirty) return true;
  if (!await showConfirm('当前词条有未保存的修改，确定丢弃？', { okText: '丢弃修改' })) return false;
  state.tmDirty = false;
  return true;
}

function renderTermsTab(lib) {
  state.tmItems = tmNormItems(lib.items);
  state.tmMode = ['off', 'on', 'enhanced'].includes(lib.mode) ? lib.mode : 'off';
  state.tmDisabled = new Set(lib.disabled_groups || []);
  state.tmSelected = new Set();
  state.tmSelAnchor = null;
  state.tmSearch = '';
  state.tmGroupFilter = '';
  state.tmEditIdx = -1;
  state.tmAddMode = true;
  state.tmDirty = false;
  const body = $('#modal-body');
  body.innerHTML = `
    <div class="tag-mgr">
      <div class="tag-cols">
        <div class="tag-left">
          <div class="tag-left-top">
            <span id="tm-mode-seg" class="mode-switch">
              <button type="button" class="pill clickable${state.tmMode === 'off' ? ' active' : ''}" data-mode="off" data-tip="词条库不参与分析">关闭</button>
              <span class="pill-sep"></span>
              <button type="button" class="pill clickable${state.tmMode === 'on' ? ' active' : ''}" data-mode="on" data-tip="按文件名与角色词条做字面检索，命中词条注入首轮提示词，只跑一轮">开启</button>
              <span class="pill-sep"></span>
              <button type="button" class="pill clickable${state.tmMode === 'enhanced' ? ' active' : ''}" data-mode="enhanced" data-tip="在开启档基础上，第二轮用嵌入模型按首轮输出语义检索候选词条让AI修订；未装嵌入模型时等同开启档">增强</button>
            </span>
            <div class="tag-seg">
              <button type="button" id="btn-tm-import">导入</button>
              <button type="button" id="btn-tm-export">导出</button>
            </div>
          </div>
          <div class="search-widget tm-search-widget" id="tmSearchWidget">
            <div class="dd search-mode" id="tm-group-dd"></div>
            <span class="search-sep"></span>
            <input type="text" id="tm-search" placeholder="搜索词条、描述或关联词…" spellcheck="false"/>
            <button type="button" id="btn-tm-clear-search" aria-label="清除"><svg class="ic"><use href="#ic-close"></use></svg></button>
          </div>
          <div class="tag-list-wrap"><div class="tm-list" id="tm-list"></div></div>
        </div>
        <div class="tag-right" id="tm-editor"></div>
      </div>
    </div>`;
  $$('#tm-mode-seg .pill').forEach(p => p.addEventListener('click', async () => {
    if (state.tmMode === p.dataset.mode) return;
    state.tmMode = p.dataset.mode;
    $$('#tm-mode-seg .pill').forEach(x => x.classList.toggle('active', x.dataset.mode === state.tmMode));
    await persistTermLibrary();
  }));
  let _tmSearchTimer = null;
  $('#tm-search').addEventListener('input', e => {
    const v = e.target.value;
    $('#tmSearchWidget').classList.toggle('has-text', !!v);
    clearTimeout(_tmSearchTimer);
    _tmSearchTimer = setTimeout(() => {
      state.tmSearch = v.trim().toLowerCase();
      renderTermRows();
    }, 150);
  });
  $('#btn-tm-clear-search').addEventListener('click', () => {
    clearTimeout(_tmSearchTimer);
    state.tmSearch = '';
    $('#tm-search').value = '';
    $('#tmSearchWidget').classList.remove('has-text');
    renderTermRows();
    $('#tm-search').focus();
  });
  /* 列表事件委托 + 滚动驱动窗口渲染 */
  const list = $('#tm-list');
  list.addEventListener('click', e => {
    const row = e.target.closest('.tm-item');
    if (row && list.contains(row)) tmRowClick(e, Number(row.dataset.i));
  });
  list.addEventListener('contextmenu', e => {
    const row = e.target.closest('.tm-item');
    if (!row || !list.contains(row)) return;
    tmRowMenu(e, Number(row.dataset.i));
  });
  let _tmScrollRaf = null;
  list.addEventListener('scroll', () => {
    if (_tmScrollRaf) return;
    _tmScrollRaf = requestAnimationFrame(() => { _tmScrollRaf = null; tmRenderWindow(); });
  }, { passive: true });
  tmRenderGroupDd();
  tmBindGroupDd($('#tm-group-dd'));
  $('#btn-tm-import').onclick = importTerms;
  $('#btn-tm-export').onclick = exportTerms;
  renderTermRows();
  tmRenderEditor();
}

function tmGroupDdRow(val, label, group) {
  const off = state.tmDisabled.has(group);
  return `<div class="dd-opt${state.tmGroupFilter === val ? ' active' : ''}${off ? ' grp-off' : ''}"` +
    ` data-value="${esc(val)}" data-toggle-group="${esc(group)}" data-tip="右键启用/停用该分组">` +
    `<span class="tm-g-label">${esc(label)}</span><span class="tm-g-dot${off ? ' off' : ''}"></span></div>`;
}

function tmGroupDdOptions() {
  return ['<div class="dd-opt' + (state.tmGroupFilter === '' ? ' active' : '') + '" data-value="">全部词条</div>',
    tmGroupDdRow('__default__', '默认分组', ''),
    ...tmAllGroups(state.tmItems).map(g => tmGroupDdRow(g, g, g)),
    '<div class="dd-hint">右键分组行启用/停用组控制是否参与检索</div>'].join('');
}

function tmRenderGroupDd() {
  const dd = $('#tm-group-dd');
  if (!dd) return;
  const curLabel = state.tmGroupFilter === '' ? '全部词条'
    : state.tmGroupFilter === '__default__' ? '默认分组' : state.tmGroupFilter;
  dd.innerHTML = `<button class="dd-btn" type="button"><span class="dd-label">${esc(curLabel)}</span>${ddArrow()}</button>
    <div class="dd-panel">${tmGroupDdOptions()}</div>`;
}

function tmBindGroupDd(dd) {
  if (!dd) return;
  dd.addEventListener('click', e => {
    if (e.target.closest('.dd-btn')) {
      e.stopPropagation();
      const wasOpen = dd.classList.contains('open');
      $$('.dd.open').forEach(d => d.classList.remove('open'));
      if (!wasOpen) {
        _positionPanel(dd);
        dd.classList.add('open');
      }
      return;
    }
    const opt = e.target.closest('.dd-opt');
    if (!opt || !dd.contains(opt) || opt.classList.contains('disabled')) return;
    e.stopPropagation();
    state.tmGroupFilter = opt.dataset.value;
    dd.querySelector('.dd-label').textContent =
      opt.dataset.value === '' ? '全部词条'
        : opt.dataset.value === '__default__' ? '默认分组' : opt.dataset.value;
    dd.querySelectorAll('.dd-opt').forEach(o => o.classList.toggle('active', o === opt));
    dd.classList.remove('open');
    renderTermRows();
  });
  dd.addEventListener('contextmenu', e => {
    const opt = e.target.closest('.dd-opt[data-toggle-group]');
    if (!opt || !dd.contains(opt)) return;
    e.preventDefault();
    e.stopPropagation();
    tmToggleGroup(opt.dataset.toggleGroup);
  });
}

async function tmToggleGroup(g) {
  const old = new Set(state.tmDisabled);
  const next = new Set(old);
  next.has(g) ? next.delete(g) : next.add(g);
  state.tmDisabled = next;
  if (!await persistTermLibrary()) {
    state.tmDisabled = old;
    return;
  }
  const dd = $('#tm-group-dd');
  if (dd) dd.querySelector('.dd-panel').innerHTML = tmGroupDdOptions();
  renderTermRows();
}

function tmFilteredIndices() {
  const q = state.tmSearch;
  return state.tmItems.map((it, i) => ({ it, i })).filter(({ it }) => {
    if (state.tmGroupFilter === '__default__' && it.group) return false;
    if (state.tmGroupFilter && state.tmGroupFilter !== '__default__' && it.group !== state.tmGroupFilter) return false;
    if (!q) return true;
    return it.keyword.toLowerCase().includes(q)
      || it.description.toLowerCase().includes(q)
      || it.related.toLowerCase().includes(q);
  }).map(x => x.i);
}

/* 列表虚拟窗口渲染：行高恒定（首帧行高校正），仅绘制可视区±缓冲 */
let _tmRowH = 34;
let _tmWin = { start: -1, end: -1 };

function renderTermRows() {
  const list = $('#tm-list');
  if (!list) return;
  state.tmIdxs = tmFilteredIndices();
  if (!state.tmItems.length) {
    list.innerHTML = '<div class="tm-list-empty">暂无词条，在右侧输入关键词后回车即可添加。</div>';
    return;
  }
  if (!state.tmIdxs.length) {
    list.innerHTML = '<div class="tm-list-empty">没有匹配的词条。</div>';
    return;
  }
  _tmWin = { start: -1, end: -1 };
  list.innerHTML = '<div id="tm-list-spacer"><div id="tm-list-win"></div></div>';
  tmRenderWindow();
}

function tmRenderWindow() {
  const list = $('#tm-list'), win = $('#tm-list-win');
  if (!win) return;
  const idxs = state.tmIdxs;
  // 先定 spacer 高，再读 scrollTop：否则内容为空时布局会把滚动位置卡到 0
  $('#tm-list-spacer').style.height = (idxs.length * _tmRowH) + 'px';
  const start = Math.max(0, Math.floor(list.scrollTop / _tmRowH) - 10);
  const end = Math.min(idxs.length, start + Math.ceil(list.clientHeight / _tmRowH) + 20);
  if (start === _tmWin.start && end === _tmWin.end) return;
  _tmWin = { start, end };
  win.style.transform = `translateY(${start * _tmRowH}px)`;
  win.innerHTML = idxs.slice(start, end)
    .map((i, p) => tmRowHtml(i, idxs, p + start)).join('');
  const h = win.firstElementChild && win.firstElementChild.offsetHeight;
  if (h && Math.abs(h - _tmRowH) > 0.5) {  // 首帧行高校正
    _tmRowH = h;
    $('#tm-list-spacer').style.height = (idxs.length * _tmRowH) + 'px';
    win.style.transform = `translateY(${start * _tmRowH}px)`;
  }
}

function tmRowHtml(i, idxs, p) {
  const it = state.tmItems[i];
  const sel = state.tmSelected.has(i);
  let runCls = '';
  if (sel) {
    const prev = p > 0 && state.tmSelected.has(idxs[p - 1]);
    const next = p + 1 < idxs.length && state.tmSelected.has(idxs[p + 1]);
    runCls = prev && next ? ' sel-mid' : next ? ' sel-start' : prev ? ' sel-end' : '';
  }
  const sub = it.description || it.related;
  return `<div class="tm-item${sel ? ' selected' : ''}${runCls}${i === state.tmEditIdx && !state.tmAddMode ? ' editing' : ''}${state.tmDisabled.has(it.group) ? ' grp-off' : ''}" data-i="${i}">` +
    `<span class="tm-kw">${esc(it.keyword)}</span>` +
    (sub ? `<span class="tm-desc" data-tip="${esc(sub)}">${esc(sub)}</span>` : '') +
    `</div>`;
}

function tmRowClick(e, idx) {
  if (e.ctrlKey || e.metaKey || e.shiftKey) {
    state.tmSelAnchor = pickSelection(state.tmSelected, state.tmSelAnchor, idx, e);
    renderTermRows();
    return;
  }
  tmOpenEdit(idx);
}

function tmRowMenu(e, idx) {
  e.preventDefault();
  e.stopPropagation();
  const multi = state.tmSelected.has(idx) && state.tmSelected.size > 1;
  const delList = multi ? [...state.tmSelected] : [idx];
  const m = $('#ctxmenu');
  const items = [{ label: multi ? `删除选中 (${delList.length})` : '删除', fn: () => tmDeleteIndices(delList) }];
  m.innerHTML = items.map((it, i) => `<button data-i="${i}">${it.label}</button>`).join('');
  m.querySelectorAll('button').forEach((b, i) => { b.onclick = () => { items[i].fn(); hideContextMenu(); }; });
  positionCtxMenu(m, e);
}

async function tmDeleteIndices(delList) {
  const before = state.tmItems;
  const delSet = new Set(delList);
  if (!await showConfirm(`确定删除选中的 ${delList.length} 个词条？`, { okText: '删除' })) return;
  if (delSet.has(state.tmEditIdx) && !state.tmAddMode && !await tmLeaveGuard()) return;
  state.tmItems = before.filter((_, i) => !delSet.has(i));
  if (!await persistTermLibrary()) {
    state.tmItems = before;
    return;
  }
  state.tmSelected.clear();
  state.tmSelAnchor = null;
  if (state.tmAddMode) {
    tmRenderEditor();
  } else if (delSet.has(state.tmEditIdx)) {
    state.tmEditIdx = -1;
    state.tmAddMode = true;
    tmRenderEditor();
  } else {
    const shift = [...delSet].filter(i => i < state.tmEditIdx).length;
    state.tmEditIdx -= shift;
  }
  renderTermRows();
}

async function tmOpenEdit(idx) {
  if (!state.tmAddMode && state.tmEditIdx === idx) return;
  if (!await tmLeaveGuard()) return;
  state.tmAddMode = false;
  state.tmEditIdx = idx;
  state.tmSelected.clear();
  state.tmSelAnchor = idx;
  state.tmDirty = false;
  renderTermRows();
  tmRenderEditor();
}

async function tmSetAddMode(guard = true) {
  if (guard && !await tmLeaveGuard()) return;
  state.tmAddMode = true;
  state.tmEditIdx = -1;
  state.tmSelected.clear();
  state.tmSelAnchor = null;
  state.tmDirty = false;
  renderTermRows();
  tmRenderEditor(true);
}

function tmGroupOptions(cur) {
  const groups = tmAllGroups(state.tmItems);
  if (cur && !groups.includes(cur)) groups.push(cur);
  return ['<div class="dd-opt' + (cur ? '' : ' active') + '" data-value="">默认分组</div>',
    ...groups.map(g => `<div class="dd-opt${g === cur ? ' active' : ''}" data-value="${esc(g)}">${esc(g)}</div>`)].join('');
}

function tmSyncEditorGroupDd(group) {
  const dd = $('#tm-f-group');
  if (!dd) return;
  dd.querySelector('.dd-panel').innerHTML = tmGroupOptions(group);
  setDropdownValue(dd, group);
}

function tmFieldsHtml(it, { readonlyRelated = false } = {}) {
  return `
    <div class="field field-desc">
      <label data-tip="将连同关键词一起发送给AI">描述</label>
      <textarea id="tm-f-desc" rows="3">${esc(it.description || '')}</textarea>
    </div>
    <div class="field">
      <label data-tip="字面检索与向量检索的匹配文本${readonlyRelated ? '；tagger 条目固定为识别的英文名' : '；tagger 分组须为唯一英文标签名'}">关联</label>
      <input type="text" id="tm-f-related" value="${esc(it.related || '')}" spellcheck="false"${readonlyRelated ? ' readonly' : ''}/>
    </div>
    <div class="field">
      <label>分组</label>
      <div class="tag-ed-group-row">
        <div class="dd" id="tm-f-group">
          <button class="dd-btn" type="button"><span class="dd-label">${esc(it.group || '默认分组')}</span>${ddArrow()}</button>
          <div class="dd-panel">${tmGroupOptions(it.group || '')}</div>
        </div>
        <input type="text" id="tm-f-newgroup" class="tag-ed-newgroup is-hidden" placeholder="新分组名" spellcheck="false"/>
        <button type="button" class="btn tag-ed-newgroup-btn" id="tm-newgroup-toggle">＋ 新建分组</button>
        <button type="button" class="btn tag-ed-newgroup-cancel is-hidden" id="tm-newgroup-cancel" title="取消新建分组">✕</button>
      </div>
    </div>`;
}

function tmBindGroupRow(root, onChange) {
  initDropdown(root.querySelector('#tm-f-group'), () => {
    const ng = root.querySelector('#tm-f-newgroup');
    if (ng && ng.value) ng.value = '';
    if (onChange) onChange();
  });
  const ngInput = root.querySelector('#tm-f-newgroup');
  const ngToggle = root.querySelector('#tm-newgroup-toggle');
  const ngCancel = root.querySelector('#tm-newgroup-cancel');
  ngToggle.onclick = () => {
    ngInput.classList.remove('is-hidden');
    ngCancel.classList.remove('is-hidden');
    ngToggle.classList.add('is-hidden');
    ngInput.focus();
  };
  ngCancel.onclick = () => {
    ngInput.value = '';
    ngInput.classList.add('is-hidden');
    ngCancel.classList.add('is-hidden');
    ngToggle.classList.remove('is-hidden');
  };
}

function tmMarkDirty(container) {
  if (!(container.id === 'tm-editor' || container.closest('#tm-editor'))) return;
  state.tmDirty = true;
}

function tmCollectFields(container) {
  const g = s => container.querySelector(s);
  const newGroup = ((g('#tm-f-newgroup') && g('#tm-f-newgroup').value) || '').trim();
  return {
    keyword: ((g('#tm-f-kw') && g('#tm-f-kw').value) || '').trim(),
    description: ((g('#tm-f-desc') && g('#tm-f-desc').value) || '').trim(),
    related: ((g('#tm-f-related') && g('#tm-f-related').value) || '').trim(),
    group: newGroup || (g('#tm-f-group') ? getDropdownValue(g('#tm-f-group')) : ''),
  };
}

function tmRenderEditor(focusKw) {
  const ed = $('#tm-editor');
  if (!ed) return;
  let isAdd = state.tmAddMode || state.tmEditIdx < 0;
  if (!isAdd && !state.tmItems[state.tmEditIdx]) {
    state.tmAddMode = isAdd = true;
    state.tmEditIdx = -1;
  }
  const it = isAdd
    ? { keyword: '', description: '', related: '',
        group: (state.tmGroupFilter && state.tmGroupFilter !== '__default__') ? state.tmGroupFilter : '' }
    : state.tmItems[state.tmEditIdx];
  ed.innerHTML = `
    <div class="tag-ed-head">
      <div class="tag-ed-title-wrap">
        <input type="text" id="tm-f-kw" class="tag-ed-title" value="${esc(it.keyword || '')}"
               placeholder="输入关键词…" spellcheck="false"/>
      </div>
      <div class="tag-ed-actions">
        ${isAdd
          ? '<button class="btn primary" id="tm-add">新建</button>'
          : `<button class="btn ghost" id="tm-to-add">切换至新建</button>
             <span class="tag-ed-actions-sep"></span>
             <button class="btn danger-ghost" id="tm-del">删除</button>
             <button class="btn primary" id="tm-save">保存</button>`}
      </div>
    </div>
    <div class="tag-ed-body" id="tm-ed-body">${tmFieldsHtml(it)}</div>`;

  const body = ed.querySelector('#tm-ed-body');
  tmBindGroupRow(body, () => tmMarkDirty(body));
  const markDirty = () => tmMarkDirty(body);
  ed.oninput = markDirty;
  ed.onchange = markDirty;


  const kwEl = ed.querySelector('#tm-f-kw');
  kwEl.addEventListener('keydown', e => {
    if (e.key !== 'Enter') return;
    e.preventDefault();
    if (isAdd) tmApplyAdd();
    else ed.querySelector('#tm-f-desc').focus();
  });
  ed.onkeydown = e => {
    if (e.key !== 'Escape' || isAdd) return;
    e.stopPropagation();
    tmLeaveGuard().then(ok => { if (ok) tmSetAddMode(false); });
  };
  if (isAdd) {
    ed.querySelector('#tm-add').onclick = tmApplyAdd;
    if (focusKw) kwEl.focus();
  } else {
    ed.querySelector('#tm-to-add').onclick = () => tmSetAddMode(true);
    ed.querySelector('#tm-del').onclick = () => tmDeleteIndices([state.tmEditIdx]);
    ed.querySelector('#tm-save').onclick = tmApplyEdit;
  }
}


function tmDupCheck(kw, excludeIdx) {
  return state.tmItems.some((x, i) => i !== excludeIdx && x.keyword.toLowerCase() === kw.toLowerCase());
}

async function tmApplyAdd() {
  const f = tmCollectFields($('#tm-editor'));
  if (!f.keyword) { $('#tm-editor #tm-f-kw').focus(); return; }
  if (tmDupCheck(f.keyword, -1)) { toast('词条已存在: ' + f.keyword, 'err'); return; }
  if (tmRelatedDupCheck(state.tmItems, f.group, f.related, -1)) {
    toast('该英文名在 tagger 分组已存在: ' + f.related, 'err'); return;
  }
  const item = { keyword: f.keyword, description: f.description, related: f.related, group: f.group };
  state.tmItems.unshift(item);
  if (!await persistTermLibrary()) {
    state.tmItems.shift();
    return;
  }
  state.tmDirty = false;
  if (state.tmGroupFilter && state.tmGroupFilter !== '__default__' && state.tmGroupFilter !== f.group) {
    state.tmGroupFilter = f.group;
  }
  tmSyncEditorGroupDd(f.group);
  tmRenderGroupDd();
  renderTermRows();
  const ed = $('#tm-editor');
  ['tm-f-kw', 'tm-f-desc', 'tm-f-related'].forEach(id => { const el = ed.querySelector('#' + id); if (el) el.value = ''; });
  ed.querySelector('#tm-f-kw').focus();
  toast(`已添加: ${f.keyword}`, 'ok');
}

async function tmApplyEdit() {
  const idx = state.tmEditIdx;
  if (idx < 0 || !state.tmItems[idx]) return;
  const f = tmCollectFields($('#tm-editor'));
  if (!f.keyword) { $('#tm-editor #tm-f-kw').focus(); return; }
  if (tmDupCheck(f.keyword, idx)) { toast('词条已存在: ' + f.keyword, 'err'); return; }
  if (tmRelatedDupCheck(state.tmItems, f.group, f.related, idx)) {
    toast('该英文名在 tagger 分组已存在: ' + f.related, 'err'); return;
  }
  const old = { ...state.tmItems[idx] };
  state.tmItems[idx] = { keyword: f.keyword, description: f.description, related: f.related, group: f.group };
  if (!await persistTermLibrary()) {
    state.tmItems[idx] = old;
    return;
  }
  state.tmDirty = false;
  const ed = $('#tm-editor');
  const ng = ed && ed.querySelector('#tm-f-newgroup');
  if (ng) {
    ng.value = '';
    ng.classList.add('is-hidden');
    ed.querySelector('#tm-newgroup-toggle').classList.remove('is-hidden');
    ed.querySelector('#tm-newgroup-cancel').classList.add('is-hidden');
  }
  tmSyncEditorGroupDd(f.group);
  tmRenderGroupDd();
  renderTermRows();
  toast(`已保存: ${f.keyword}`, 'ok');
}

async function importTerms() {
  const res = await callApi('import_term_library');
  if (!res || res.cancelled) return;
  if (!res.ok) { toast('导入失败: ' + (res.error || ''), 'err'); return; }
  const incoming = tmNormItems(res.items);
  const existing = new Set(state.tmItems.map(x => x.keyword.toLowerCase()));
  const existingRel = new Set(state.tmItems
    .filter(x => TAGGER_TERM_GROUPS.includes(x.group)).map(x => (x.related || '').toLowerCase()));
  let relSkipped = 0;
  const fresh = [];
  for (const it of incoming) {
    const key = it.keyword.toLowerCase();
    if (existing.has(key)) continue;
    existing.add(key);
    if (TAGGER_TERM_GROUPS.includes(it.group)) {
      const rel = (it.related || '').toLowerCase();
      if (rel && existingRel.has(rel)) { relSkipped++; continue; }
      existingRel.add(rel);
    }
    fresh.push(it);
  }
  if (!fresh.length) { toast('没有可导入的新词条（关键词均已存在）', 'err'); return; }
  const skipped = incoming.length - fresh.length;
  if (state.tmItems.length &&
      !await showConfirm(`将追加 ${fresh.length} 个新词条${skipped ? `（跳过已存在 ${skipped} 个）` : ''}，确定继续？`, { okText: '追加导入' })) return;
  const before = state.tmItems;
  state.tmItems = [...fresh, ...before];
  if (!await persistTermLibrary()) {
    state.tmItems = before;
    return;
  }
  state.tmSelected.clear();
  state.tmSelAnchor = null;
  state.tmEditIdx = -1;
  state.tmAddMode = true;
  state.tmDirty = false;
  tmRenderGroupDd();
  renderTermRows();
  tmRenderEditor();
  refreshTaggerSearchCache();
  toast(`已导入 ${fresh.length} 个新词条` +
    (relSkipped ? `，另跳过 ${relSkipped} 个英文名重复的 tagger 条目` : ''), 'ok');
}

async function exportTerms() {
  const res = await callApi('export_term_library', tmNormItems(state.tmItems));
  if (!res || res.cancelled) return;
  if (res.ok) toast('已导出到: ' + (res.path || ''), 'ok');
  else toast('导出失败: ' + (res.error || ''), 'err');
}

/* ── 主界面标签 chip 右键菜单与「加入词条检索 / 编辑词条」弹窗 ── */
function showChipMenu(e, name, { webSearch = false, asRelated = false, edit = false, group = '', taggerDel = null } = {}) {
  e.preventDefault();
  e.stopPropagation();
  const items = [];
  if (webSearch) items.push({ label: '网页搜索', fn: () => webSearchTag(name) });
  items.push(edit
    ? { label: '编辑词条', fn: () => openTaggerTermEdit(name) }
    : { label: '加入词条检索', fn: () => openTermAddToSearch(name, asRelated, group) });
  if (taggerDel) items.push({ label: '删除此标签', fn: () => deleteTaggerTag(taggerDel.vid, taggerDel.group, name) });
  const m = $('#ctxmenu');
  m.innerHTML = items.map((it, i) => `<button data-i="${i}">${it.label}</button>`).join('');
  m.querySelectorAll('button').forEach((b, i) => { b.onclick = () => { items[i].fn(); hideContextMenu(); }; });
  positionCtxMenu(m, e);
}

const TAGGER_TERM_GROUPS = ['Hammy-tagger', 'Hammy-tagger-G'];

function tmRelatedDupCheck(items, group, related, excludeIdx) {
  /* tagger 英文名组内唯一（重复会造成映射歧义） */
  if (!TAGGER_TERM_GROUPS.includes(group) || !related) return false;
  return items.some((x, i) =>
    i !== excludeIdx && TAGGER_TERM_GROUPS.includes(x.group) &&
    (x.related || '').toLowerCase() === related.toLowerCase());
}

async function openTermLibraryEditor(prefill, targetIdx = -1) {
  const cur = await callApi('get_term_library');
  if (!cur) return;
  const items = tmNormItems(cur.items);
  const isEdit = targetIdx >= 0;
  const bg = document.createElement('div');
  bg.className = 'tm-editor-bg';
  bg.innerHTML = `
    <div class="tm-editor tm-term-editor">
      <div class="tag-ed-head">
        <div class="tag-ed-title-wrap">
          <input type="text" id="tm-f-kw" class="tag-ed-title" value="${esc(prefill.keyword || '')}"
                 placeholder="输入关键词…" spellcheck="false"/>
        </div>
        <div class="tag-ed-actions">
          <button class="btn ghost" id="tm-add-cancel">取消</button>
          <button class="btn primary" id="tm-add-save">保存</button>
        </div>
      </div>
      <div class="tag-ed-body" id="tm-add-body">${tmFieldsHtml(prefill, { readonlyRelated: isEdit })}</div>
    </div>`;
  document.body.appendChild(bg);
  const root = bg.querySelector('.tm-editor');
  tmBindGroupRow(root, null);
  const kwEl = root.querySelector('#tm-f-kw');
  kwEl.focus();
  if (!isEdit && prefill.keyword) kwEl.select();
  bg.addEventListener('keydown', e => {
    if (e.key === 'Escape') { e.stopPropagation(); bg.remove(); }
  });
  kwEl.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); bg.querySelector('#tm-add-save').click(); }
  });
  bg.querySelector('#tm-add-cancel').onclick = () => bg.remove();
  bg.querySelector('#tm-add-save').onclick = async () => {
    const f = tmCollectFields(root);
    if (isEdit) f.related = items[targetIdx].related;  // 英文映射键只读
    if (!f.keyword) { kwEl.focus(); return; }
    if (items.some((x, i) => i !== targetIdx && x.keyword.toLowerCase() === f.keyword.toLowerCase())) {
      toast('词条已存在: ' + f.keyword, 'err');
      kwEl.focus(); kwEl.select();
      return;
    }
    if (tmRelatedDupCheck(items, f.group, f.related, targetIdx)) {
      toast('该英文名在 tagger 分组已存在: ' + f.related, 'err');
      return;
    }
    const entry = { keyword: f.keyword, description: f.description, related: f.related, group: f.group };
    if (isEdit) items.splice(targetIdx, 1, entry);
    else items.unshift(entry);
    const saveBtn = bg.querySelector('#tm-add-save');
    saveBtn.disabled = true;
    const r = await callApi('save_term_library', items, cur.mode || 'off', cur.disabled_groups || []);
    if (r && r.ok) {
      bg.remove();
      toast(isEdit ? `已保存: ${f.keyword}` : `已加入词条检索: ${f.keyword}`, 'ok');
      refreshTaggerSearchCache();
    } else {
      saveBtn.disabled = false;
      toast('保存失败: ' + ((r && r.error) || '未知错误'), 'err');
    }
  };
}

async function openTermAddToSearch(name, asRelated, group = '') {
  return openTermLibraryEditor(asRelated
    ? { keyword: '', description: '', related: name, group }
    : { keyword: name, description: '', related: '', group });
}

async function openTaggerTermEdit(enName) {
  const cur = await callApi('get_term_library');
  if (!cur) return;
  const items = tmNormItems(cur.items);
  const idx = items.findIndex(x => TAGGER_TERM_GROUPS.includes(x.group) && x.related === enName);
  if (idx < 0) { await openTermAddToSearch(enName, true, 'Hammy-tagger-G'); return; }
  return openTermLibraryEditor({ ...items[idx] }, idx);
}

async function deleteTaggerTag(vid, group, name) {
  const r = await callApi('delete_tagger_tag', vid, group, name);
  if (!r || !r.ok) { toast('删除失败: ' + ((r && r.error) || '未知错误'), 'err'); return; }
  await refreshTaggerBadges();
  refreshTaggerSearchCache();
  toast(`已删除: ${name}`, 'ok');
}

/* 编辑后刷新待处理搜索缓存与详情 */
async function refreshTaggerSearchCache() {
  try {
    const m = await apiCall('get_tagger_all_tags');
    if (m) state.taggerTagNames = m;
  } catch (e) { /* 失败沿用旧缓存 */ }
  if (typeof refreshGridData === 'function') refreshGridData();
  if (typeof renderDetailForSelection === 'function' && typeof findVideoById === 'function') {
    const v = state.primaryId ? findVideoById(state.primaryId) : null;
    if (v && v.status === 'pending' && state.selected.size === 1 &&
        (state.taggerEnabled || state.transcribeEnabled)) renderDetailForSelection(v);
  }
}
