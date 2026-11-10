/* ════════════════════════════════════════════════════════════
   设置 Modal - 标签检索页（双栏管理器，修改即时写盘）
   ════════════════════════════════════════════════════════════ */
function ptNormItems(items) {
  return (items || []).map(x => ({
    keyword: x.keyword || '',
    description: x.description || '',
    related: x.related || '',
    group: x.group || '',
  }));
}

function ptAllGroups(items) {
  return [...new Set(items.map(x => x.group).filter(Boolean))].sort();
}

async function persistTagLibrary() {
  const r = await callApi('save_tag_library', state.ptItems, state.ptMode, [...state.ptDisabled]);
  if (!r || !r.ok) {
    toast('保存失败: ' + ((r && r.error) || '未知错误'), 'err');
    return false;
  }
  return true;
}

async function ptLeaveGuard() {
  if (!state.ptDirty) return true;
  if (!await showConfirm('当前标签有未保存的修改，确定丢弃？', { okText: '丢弃修改' })) return false;
  state.ptDirty = false;
  return true;
}

function renderTagsTab(pt) {
  state.ptItems = ptNormItems(pt.items);
  state.ptMode = ['off', 'on', 'enhanced'].includes(pt.mode) ? pt.mode : 'off';
  state.ptDisabled = new Set(pt.disabled_groups || []);
  state.ptSelected = new Set();
  state.ptSelAnchor = null;
  state.ptSearch = '';
  state.ptGroupFilter = '';
  state.ptEditIdx = -1;
  state.ptAddMode = true;
  state.ptDirty = false;
  const body = $('#modal-body');
  body.innerHTML = `
    <div class="tag-mgr">
      <div class="tag-cols">
        <div class="tag-left">
          <div class="tag-left-top">
            <span id="pt-mode-seg" class="mode-switch">
              <button type="button" class="pill clickable${state.ptMode === 'off' ? ' active' : ''}" data-mode="off" data-tip="标签库不参与分析">关闭</button>
              <span class="pill-sep"></span>
              <button type="button" class="pill clickable${state.ptMode === 'on' ? ' active' : ''}" data-mode="on" data-tip="按文件名与角色标签做字面检索，命中标签注入首轮提示词，只跑一轮">开启</button>
              <span class="pill-sep"></span>
              <button type="button" class="pill clickable${state.ptMode === 'enhanced' ? ' active' : ''}" data-mode="enhanced" data-tip="在开启档基础上，第二轮用嵌入模型按首轮输出语义检索候选标签让AI修订；未装嵌入模型时等同开启档">增强</button>
            </span>
            <div class="tag-seg">
              <button type="button" id="btn-pt-import">导入</button>
              <button type="button" id="btn-pt-export">导出</button>
            </div>
          </div>
          <div class="search-widget pt-search-widget" id="ptSearchWidget">
            <div class="dd search-mode" id="pt-group-dd"></div>
            <span class="search-sep"></span>
            <input type="text" id="pt-search" placeholder="搜索标签、描述或关联词…" spellcheck="false"/>
            <button type="button" id="btn-pt-clear-search" aria-label="清除"><svg class="ic"><use href="#ic-close"></use></svg></button>
          </div>
          <div class="tag-list-wrap"><div class="pt-list" id="pt-list"></div></div>
        </div>
        <div class="tag-right" id="pt-editor"></div>
      </div>
    </div>`;
  $$('#pt-mode-seg .pill').forEach(p => p.addEventListener('click', async () => {
    if (state.ptMode === p.dataset.mode) return;
    state.ptMode = p.dataset.mode;
    $$('#pt-mode-seg .pill').forEach(x => x.classList.toggle('active', x.dataset.mode === state.ptMode));
    await persistTagLibrary();
  }));
  let _ptSearchTimer = null;
  $('#pt-search').addEventListener('input', e => {
    const v = e.target.value;
    $('#ptSearchWidget').classList.toggle('has-text', !!v);
    clearTimeout(_ptSearchTimer);
    _ptSearchTimer = setTimeout(() => {
      state.ptSearch = v.trim().toLowerCase();
      renderTagRows();
    }, 150);
  });
  $('#btn-pt-clear-search').addEventListener('click', () => {
    clearTimeout(_ptSearchTimer);
    state.ptSearch = '';
    $('#pt-search').value = '';
    $('#ptSearchWidget').classList.remove('has-text');
    renderTagRows();
    $('#pt-search').focus();
  });
  /* 列表事件委托 + 滚动驱动窗口渲染 */
  const list = $('#pt-list');
  list.addEventListener('click', e => {
    const row = e.target.closest('.pt-item');
    if (row && list.contains(row)) ptRowClick(e, Number(row.dataset.i));
  });
  list.addEventListener('contextmenu', e => {
    const row = e.target.closest('.pt-item');
    if (!row || !list.contains(row)) return;
    ptRowMenu(e, Number(row.dataset.i));
  });
  let _ptScrollRaf = null;
  list.addEventListener('scroll', () => {
    if (_ptScrollRaf) return;
    _ptScrollRaf = requestAnimationFrame(() => { _ptScrollRaf = null; ptRenderWindow(); });
  }, { passive: true });
  ptRenderGroupDd();
  ptBindGroupDd($('#pt-group-dd'));
  $('#btn-pt-import').onclick = importTags;
  $('#btn-pt-export').onclick = exportTags;
  renderTagRows();
  ptRenderEditor();
}

function ptGroupDdRow(val, label, group) {
  const off = state.ptDisabled.has(group);
  return `<div class="dd-opt${state.ptGroupFilter === val ? ' active' : ''}${off ? ' grp-off' : ''}"` +
    ` data-value="${esc(val)}" data-toggle-group="${esc(group)}" data-tip="右键启用/停用该分组">` +
    `<span class="pt-g-label">${esc(label)}</span><span class="pt-g-dot${off ? ' off' : ''}"></span></div>`;
}

function ptGroupDdOptions() {
  return ['<div class="dd-opt' + (state.ptGroupFilter === '' ? ' active' : '') + '" data-value="">全部标签</div>',
    ptGroupDdRow('__default__', '默认分组', ''),
    ...ptAllGroups(state.ptItems).map(g => ptGroupDdRow(g, g, g)),
    '<div class="dd-hint">右键分组行启用/停用组控制是否参与检索</div>'].join('');
}

function ptRenderGroupDd() {
  const dd = $('#pt-group-dd');
  if (!dd) return;
  const curLabel = state.ptGroupFilter === '' ? '全部标签'
    : state.ptGroupFilter === '__default__' ? '默认分组' : state.ptGroupFilter;
  dd.innerHTML = `<button class="dd-btn" type="button"><span class="dd-label">${esc(curLabel)}</span>${ddArrow()}</button>
    <div class="dd-panel">${ptGroupDdOptions()}</div>`;
}

function ptBindGroupDd(dd) {
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
    state.ptGroupFilter = opt.dataset.value;
    dd.querySelector('.dd-label').textContent =
      opt.dataset.value === '' ? '全部标签'
        : opt.dataset.value === '__default__' ? '默认分组' : opt.dataset.value;
    dd.querySelectorAll('.dd-opt').forEach(o => o.classList.toggle('active', o === opt));
    dd.classList.remove('open');
    renderTagRows();
  });
  dd.addEventListener('contextmenu', e => {
    const opt = e.target.closest('.dd-opt[data-toggle-group]');
    if (!opt || !dd.contains(opt)) return;
    e.preventDefault();
    e.stopPropagation();
    ptToggleGroup(opt.dataset.toggleGroup);
  });
}

async function ptToggleGroup(g) {
  const old = new Set(state.ptDisabled);
  const next = new Set(old);
  next.has(g) ? next.delete(g) : next.add(g);
  state.ptDisabled = next;
  if (!await persistTagLibrary()) {
    state.ptDisabled = old;
    return;
  }
  const dd = $('#pt-group-dd');
  if (dd) dd.querySelector('.dd-panel').innerHTML = ptGroupDdOptions();
  renderTagRows();
}

function ptFilteredIndices() {
  const q = state.ptSearch;
  return state.ptItems.map((it, i) => ({ it, i })).filter(({ it }) => {
    if (state.ptGroupFilter === '__default__' && it.group) return false;
    if (state.ptGroupFilter && state.ptGroupFilter !== '__default__' && it.group !== state.ptGroupFilter) return false;
    if (!q) return true;
    return it.keyword.toLowerCase().includes(q)
      || it.description.toLowerCase().includes(q)
      || it.related.toLowerCase().includes(q);
  }).map(x => x.i);
}

/* 列表虚拟窗口渲染：行高恒定（首帧行高校正），仅绘制可视区±缓冲 */
let _ptRowH = 34;
let _ptWin = { start: -1, end: -1 };

function renderTagRows() {
  const list = $('#pt-list');
  if (!list) return;
  state.ptIdxs = ptFilteredIndices();
  if (!state.ptItems.length) {
    list.innerHTML = '<div class="pt-list-empty">暂无标签，在右侧输入关键词后回车即可添加。</div>';
    return;
  }
  if (!state.ptIdxs.length) {
    list.innerHTML = '<div class="pt-list-empty">没有匹配的标签。</div>';
    return;
  }
  _ptWin = { start: -1, end: -1 };
  list.innerHTML = '<div id="pt-list-spacer"><div id="pt-list-win"></div></div>';
  ptRenderWindow();
}

function ptRenderWindow() {
  const list = $('#pt-list'), win = $('#pt-list-win');
  if (!win) return;
  const idxs = state.ptIdxs;
  // 先定 spacer 高，再读 scrollTop：否则内容为空时布局会把滚动位置卡到 0
  $('#pt-list-spacer').style.height = (idxs.length * _ptRowH) + 'px';
  const start = Math.max(0, Math.floor(list.scrollTop / _ptRowH) - 10);
  const end = Math.min(idxs.length, start + Math.ceil(list.clientHeight / _ptRowH) + 20);
  if (start === _ptWin.start && end === _ptWin.end) return;
  _ptWin = { start, end };
  win.style.transform = `translateY(${start * _ptRowH}px)`;
  win.innerHTML = idxs.slice(start, end)
    .map((i, p) => ptRowHtml(i, idxs, p + start)).join('');
  const h = win.firstElementChild && win.firstElementChild.offsetHeight;
  if (h && Math.abs(h - _ptRowH) > 0.5) {  // 首帧行高校正
    _ptRowH = h;
    $('#pt-list-spacer').style.height = (idxs.length * _ptRowH) + 'px';
    win.style.transform = `translateY(${start * _ptRowH}px)`;
  }
}

function ptRowHtml(i, idxs, p) {
  const it = state.ptItems[i];
  const sel = state.ptSelected.has(i);
  let runCls = '';
  if (sel) {
    const prev = p > 0 && state.ptSelected.has(idxs[p - 1]);
    const next = p + 1 < idxs.length && state.ptSelected.has(idxs[p + 1]);
    runCls = prev && next ? ' sel-mid' : next ? ' sel-start' : prev ? ' sel-end' : '';
  }
  const sub = it.description || it.related;
  return `<div class="pt-item${sel ? ' selected' : ''}${runCls}${i === state.ptEditIdx && !state.ptAddMode ? ' editing' : ''}${state.ptDisabled.has(it.group) ? ' grp-off' : ''}" data-i="${i}">` +
    `<span class="pt-kw">${esc(it.keyword)}</span>` +
    (sub ? `<span class="pt-desc" data-tip="${esc(sub)}">${esc(sub)}</span>` : '') +
    `</div>`;
}

function ptRowClick(e, idx) {
  if (e.ctrlKey || e.metaKey || e.shiftKey) {
    state.ptSelAnchor = pickSelection(state.ptSelected, state.ptSelAnchor, idx, e);
    renderTagRows();
    return;
  }
  ptOpenEdit(idx);
}

function ptRowMenu(e, idx) {
  e.preventDefault();
  e.stopPropagation();
  const multi = state.ptSelected.has(idx) && state.ptSelected.size > 1;
  const delList = multi ? [...state.ptSelected] : [idx];
  const m = $('#ctxmenu');
  const items = [{ label: multi ? `删除选中 (${delList.length})` : '删除', fn: () => ptDeleteIndices(delList) }];
  m.innerHTML = items.map((it, i) => `<button data-i="${i}">${it.label}</button>`).join('');
  m.querySelectorAll('button').forEach((b, i) => { b.onclick = () => { items[i].fn(); hideContextMenu(); }; });
  positionCtxMenu(m, e);
}

async function ptDeleteIndices(delList) {
  const before = state.ptItems;
  const delSet = new Set(delList);
  if (!await showConfirm(`确定删除选中的 ${delList.length} 个标签？`, { okText: '删除' })) return;
  if (delSet.has(state.ptEditIdx) && !state.ptAddMode && !await ptLeaveGuard()) return;
  state.ptItems = before.filter((_, i) => !delSet.has(i));
  if (!await persistTagLibrary()) {
    state.ptItems = before;
    return;
  }
  state.ptSelected.clear();
  state.ptSelAnchor = null;
  if (state.ptAddMode) {
    ptRenderEditor();
  } else if (delSet.has(state.ptEditIdx)) {
    state.ptEditIdx = -1;
    state.ptAddMode = true;
    ptRenderEditor();
  } else {
    const shift = [...delSet].filter(i => i < state.ptEditIdx).length;
    state.ptEditIdx -= shift;
  }
  renderTagRows();
}

async function ptOpenEdit(idx) {
  if (!state.ptAddMode && state.ptEditIdx === idx) return;
  if (!await ptLeaveGuard()) return;
  state.ptAddMode = false;
  state.ptEditIdx = idx;
  state.ptSelected.clear();
  state.ptSelAnchor = idx;
  state.ptDirty = false;
  renderTagRows();
  ptRenderEditor();
}

async function ptSetAddMode(guard = true) {
  if (guard && !await ptLeaveGuard()) return;
  state.ptAddMode = true;
  state.ptEditIdx = -1;
  state.ptSelected.clear();
  state.ptSelAnchor = null;
  state.ptDirty = false;
  renderTagRows();
  ptRenderEditor(true);
}

function ptGroupOptions(cur) {
  const groups = ptAllGroups(state.ptItems);
  if (cur && !groups.includes(cur)) groups.push(cur);
  return ['<div class="dd-opt' + (cur ? '' : ' active') + '" data-value="">默认分组</div>',
    ...groups.map(g => `<div class="dd-opt${g === cur ? ' active' : ''}" data-value="${esc(g)}">${esc(g)}</div>`)].join('');
}

function ptSyncEditorGroupDd(group) {
  const dd = $('#pt-f-group');
  if (!dd) return;
  dd.querySelector('.dd-panel').innerHTML = ptGroupOptions(group);
  setDropdownValue(dd, group);
}

function ptBuildFields(container, it, opts = {}) {
  const readonlyRelated = !!opts.readonlyRelated;
  container.innerHTML = `
    <div class="field"><label>关键词</label><input type="text" id="pt-f-kw" value="${esc(it.keyword || '')}" spellcheck="false"/></div>
    <div class="field"><label>描述 <span class="pt-lbl-hint">将连同关键词一起发送给AI</span></label><textarea id="pt-f-desc" rows="6">${esc(it.description || '')}</textarea></div>
    <div class="field"><label>关联 <span class="pt-lbl-hint">字面检索与向量检索的匹配文本${readonlyRelated ? '；tagger 条目固定为识别的英文名' : '；Hammy-tagger 组须为唯一英文标签名'}</span></label><input type="text" id="pt-f-related" value="${esc(it.related || '')}" spellcheck="false"${readonlyRelated ? ' readonly' : ''}/></div>
    <div class="tag-ed-group">
      <div class="field"><label>分组</label>
        <div class="dd" id="pt-f-group"><button class="dd-btn" type="button"><span class="dd-label">${esc(it.group || '默认分组')}</span>${ddArrow()}</button>
          <div class="dd-panel">${ptGroupOptions(it.group || '')}</div></div>
      </div>
      <div class="field"><label>新分组</label><input type="text" id="pt-f-newgroup" placeholder="输入即新建分组" spellcheck="false"/></div>
    </div>`;
  initDropdown(container.querySelector('#pt-f-group'), () => {
    const ng = container.querySelector('#pt-f-newgroup');
    if (ng && ng.value) ng.value = '';
    ptMarkDirty(container);
  });
  ['input', 'change'].forEach(ev => container.addEventListener(ev, e => {
    if (e.target.closest('#pt-editor, .pt-editor')) ptMarkDirty(container);
  }));
}

function ptMarkDirty(container) {
  if (container.id === 'pt-editor' || container.closest('#pt-editor')) state.ptDirty = true;
}

function ptCollectFields(container) {
  const g = s => container.querySelector(s);
  const newGroup = ((g('#pt-f-newgroup') && g('#pt-f-newgroup').value) || '').trim();
  return {
    keyword: ((g('#pt-f-kw') && g('#pt-f-kw').value) || '').trim(),
    description: ((g('#pt-f-desc') && g('#pt-f-desc').value) || '').trim(),
    related: ((g('#pt-f-related') && g('#pt-f-related').value) || '').trim(),
    group: newGroup || (g('#pt-f-group') ? getDropdownValue(g('#pt-f-group')) : ''),
  };
}

function ptRenderEditor(focusKw) {
  const ed = $('#pt-editor');
  if (!ed) return;
  let isAdd = state.ptAddMode || state.ptEditIdx < 0;
  if (!isAdd && !state.ptItems[state.ptEditIdx]) {
    state.ptAddMode = isAdd = true;
    state.ptEditIdx = -1;
  }
  const it = isAdd
    ? { keyword: '', description: '', related: '',
        group: (state.ptGroupFilter && state.ptGroupFilter !== '__default__') ? state.ptGroupFilter : '' }
    : state.ptItems[state.ptEditIdx];
  ed.innerHTML = `
    <div class="tag-ed-body" id="pt-ed-body"></div>
    <div class="tag-ed-foot">
      <span class="pt-ed-help">检索将使用关键词或关联词匹配</span>
      ${isAdd
        ? '<button class="btn primary" id="pt-add">新建</button>'
        : `<button class="btn" id="pt-to-add">切换至新建</button>
           <button class="btn danger" id="pt-del">删除</button>
           <button class="btn primary" id="pt-save">保存</button>`}
    </div>`;
  ptBuildFields(ed.querySelector('#pt-ed-body'), it);
  const kwEl = ed.querySelector('#pt-f-kw');
  kwEl.addEventListener('keydown', e => {
    if (e.key !== 'Enter') return;
    e.preventDefault();
    if (isAdd) ptApplyAdd();
    else ed.querySelector('#pt-f-desc').focus();
  });
  ed.onkeydown = e => {
    if (e.key !== 'Escape' || isAdd) return;
    e.stopPropagation();
    ptLeaveGuard().then(ok => { if (ok) ptSetAddMode(false); });
  };
  if (isAdd) {
    ed.querySelector('#pt-add').onclick = ptApplyAdd;
    if (focusKw) kwEl.focus();
  } else {
    ed.querySelector('#pt-to-add').onclick = () => ptSetAddMode(true);
    ed.querySelector('#pt-del').onclick = () => ptDeleteIndices([state.ptEditIdx]);
    ed.querySelector('#pt-save').onclick = ptApplyEdit;
  }
}

function ptDupCheck(kw, excludeIdx) {
  return state.ptItems.some((x, i) => i !== excludeIdx && x.keyword.toLowerCase() === kw.toLowerCase());
}

async function ptApplyAdd() {
  const f = ptCollectFields($('#pt-editor'));
  if (!f.keyword) { $('#pt-editor #pt-f-kw').focus(); return; }
  if (ptDupCheck(f.keyword, -1)) { toast('标签已存在: ' + f.keyword, 'err'); return; }
  if (ptRelatedDupCheck(state.ptItems, f.group, f.related, -1)) {
    toast('该英文名在 Hammy-tagger 组已存在: ' + f.related, 'err'); return;
  }
  const item = { keyword: f.keyword, description: f.description, related: f.related, group: f.group };
  state.ptItems.unshift(item);
  if (!await persistTagLibrary()) {
    state.ptItems.shift();
    return;
  }
  state.ptDirty = false;
  if (state.ptGroupFilter && state.ptGroupFilter !== '__default__' && state.ptGroupFilter !== f.group) {
    state.ptGroupFilter = f.group;
  }
  ptSyncEditorGroupDd(f.group);
  ptRenderGroupDd();
  renderTagRows();
  const ed = $('#pt-editor');
  ['pt-f-kw', 'pt-f-desc', 'pt-f-related'].forEach(id => { const el = ed.querySelector('#' + id); if (el) el.value = ''; });
  ed.querySelector('#pt-f-kw').focus();
  toast(`已添加: ${f.keyword}`, 'ok');
}

async function ptApplyEdit() {
  const idx = state.ptEditIdx;
  if (idx < 0 || !state.ptItems[idx]) return;
  const f = ptCollectFields($('#pt-editor'));
  if (!f.keyword) { $('#pt-editor #pt-f-kw').focus(); return; }
  if (ptDupCheck(f.keyword, idx)) { toast('标签已存在: ' + f.keyword, 'err'); return; }
  if (ptRelatedDupCheck(state.ptItems, f.group, f.related, idx)) {
    toast('该英文名在 Hammy-tagger 组已存在: ' + f.related, 'err'); return;
  }
  const old = { ...state.ptItems[idx] };
  state.ptItems[idx] = { keyword: f.keyword, description: f.description, related: f.related, group: f.group };
  if (!await persistTagLibrary()) {
    state.ptItems[idx] = old;
    return;
  }
  state.ptDirty = false;
  ptSyncEditorGroupDd(f.group);
  ptRenderGroupDd();
  renderTagRows();
  toast(`已保存: ${f.keyword}`, 'ok');
}

async function importTags() {
  const res = await callApi('import_tag_library');
  if (!res || res.cancelled) return;
  if (!res.ok) { toast('导入失败: ' + (res.error || ''), 'err'); return; }
  const incoming = ptNormItems(res.items);
  const existing = new Set(state.ptItems.map(x => x.keyword.toLowerCase()));
  const existingRel = new Set(state.ptItems
    .filter(x => x.group === TAGGER_LIB_GROUP).map(x => (x.related || '').toLowerCase()));
  let relSkipped = 0;
  const fresh = [];
  for (const it of incoming) {
    const key = it.keyword.toLowerCase();
    if (existing.has(key)) continue;
    existing.add(key);
    if (it.group === TAGGER_LIB_GROUP) {
      const rel = (it.related || '').toLowerCase();
      if (rel && existingRel.has(rel)) { relSkipped++; continue; }
      existingRel.add(rel);
    }
    fresh.push(it);
  }
  if (!fresh.length) { toast('没有可导入的新标签（关键词均已存在）', 'err'); return; }
  const skipped = incoming.length - fresh.length;
  if (state.ptItems.length &&
      !await showConfirm(`将追加 ${fresh.length} 个新标签${skipped ? `（跳过已存在 ${skipped} 个）` : ''}，确定继续？`, { okText: '追加导入' })) return;
  const before = state.ptItems;
  state.ptItems = [...fresh, ...before];
  if (!await persistTagLibrary()) {
    state.ptItems = before;
    return;
  }
  state.ptSelected.clear();
  state.ptSelAnchor = null;
  state.ptEditIdx = -1;
  state.ptAddMode = true;
  state.ptDirty = false;
  ptRenderGroupDd();
  renderTagRows();
  ptRenderEditor();
  refreshTaggerSearchCache();
  toast(`已导入 ${fresh.length} 个新标签` +
    (relSkipped ? `，另跳过 ${relSkipped} 个英文名重复的 Hammy-tagger 条目` : ''), 'ok');
}

async function exportTags() {
  const res = await callApi('export_tag_library', ptNormItems(state.ptItems));
  if (!res || res.cancelled) return;
  if (res.ok) toast('已导出到: ' + (res.path || ''), 'ok');
  else toast('导出失败: ' + (res.error || ''), 'err');
}

/* ── 主界面标签 chip 右键菜单与「加入标签检索 / 编辑标签」弹窗 ── */
function showChipMenu(e, name, { webSearch = false, asRelated = false, edit = false, group = '' } = {}) {
  e.preventDefault();
  e.stopPropagation();
  const items = [];
  if (webSearch) items.push({ label: '网页搜索', fn: () => webSearchTag(name) });
  items.push(edit
    ? { label: '编辑标签', fn: () => openTaggerTagEdit(name) }
    : { label: '加入标签检索', fn: () => openTagAddToSearch(name, asRelated, group) });
  const m = $('#ctxmenu');
  m.innerHTML = items.map((it, i) => `<button data-i="${i}">${it.label}</button>`).join('');
  m.querySelectorAll('button').forEach((b, i) => { b.onclick = () => { items[i].fn(); hideContextMenu(); }; });
  positionCtxMenu(m, e);
}

const TAGGER_LIB_GROUP = 'Hammy-tagger';

function ptRelatedDupCheck(items, group, related, excludeIdx) {
  /* tagger 英文名组内唯一（重复会造成映射歧义） */
  if (group !== TAGGER_LIB_GROUP || !related) return false;
  return items.some((x, i) =>
    i !== excludeIdx && x.group === TAGGER_LIB_GROUP &&
    (x.related || '').toLowerCase() === related.toLowerCase());
}

async function openTagLibraryEditor(prefill, targetIdx = -1) {
  const cur = await callApi('get_tag_library');
  if (!cur) return;
  const items = ptNormItems(cur.items);
  const isEdit = targetIdx >= 0;
  const bg = document.createElement('div');
  bg.className = 'pt-editor-bg';
  bg.innerHTML = `
    <div class="pt-editor">
      <h3>${isEdit ? '编辑标签' : '加入标签检索'}</h3>
      <div class="tag-ed-body" id="pt-add-body"></div>
      <div class="pt-editor-foot">
        <button class="btn" id="pt-add-cancel">取消</button>
        <button class="btn primary" id="pt-add-save">保存</button>
      </div>
    </div>`;
  document.body.appendChild(bg);
  const body = bg.querySelector('#pt-add-body');
  ptBuildFields(body, prefill, { readonlyRelated: isEdit });
  const kwEl = body.querySelector('#pt-f-kw');
  kwEl.focus();
  if (!isEdit && prefill.keyword) kwEl.select();
  bg.addEventListener('keydown', e => {
    if (e.key === 'Escape') { e.stopPropagation(); bg.remove(); }
  });
  kwEl.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); bg.querySelector('#pt-add-save').click(); }
  });
  bg.querySelector('#pt-add-cancel').onclick = () => bg.remove();
  bg.querySelector('#pt-add-save').onclick = async () => {
    const f = ptCollectFields(body);
    if (isEdit) f.related = items[targetIdx].related;  // 英文映射键只读
    if (!f.keyword) { kwEl.focus(); return; }
    if (items.some((x, i) => i !== targetIdx && x.keyword.toLowerCase() === f.keyword.toLowerCase())) {
      toast('标签已存在: ' + f.keyword, 'err');
      kwEl.focus(); kwEl.select();
      return;
    }
    if (ptRelatedDupCheck(items, f.group, f.related, targetIdx)) {
      toast('该英文名在 Hammy-tagger 组已存在: ' + f.related, 'err');
      return;
    }
    const entry = { keyword: f.keyword, description: f.description, related: f.related, group: f.group };
    if (isEdit) items.splice(targetIdx, 1, entry);
    else items.unshift(entry);
    const saveBtn = bg.querySelector('#pt-add-save');
    saveBtn.disabled = true;
    const r = await callApi('save_tag_library', items, cur.mode || 'off', cur.disabled_groups || []);
    if (r && r.ok) {
      bg.remove();
      toast(isEdit ? `已保存: ${f.keyword}` : `已加入标签检索: ${f.keyword}`, 'ok');
      refreshTaggerSearchCache();
    } else {
      saveBtn.disabled = false;
      toast('保存失败: ' + ((r && r.error) || '未知错误'), 'err');
    }
  };
}

async function openTagAddToSearch(name, asRelated, group = '') {
  return openTagLibraryEditor(asRelated
    ? { keyword: '', description: '', related: name, group }
    : { keyword: name, description: '', related: '', group });
}

async function openTaggerTagEdit(enName) {
  const cur = await callApi('get_tag_library');
  if (!cur) return;
  const items = ptNormItems(cur.items);
  const idx = items.findIndex(x => x.group === TAGGER_LIB_GROUP && x.related === enName);
  if (idx < 0) { await openTagAddToSearch(enName, true, TAGGER_LIB_GROUP); return; }
  return openTagLibraryEditor({ ...items[idx] }, idx);
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
