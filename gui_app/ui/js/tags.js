/* ════════════════════════════════════════════════════════════
   设置 Modal - 标签检索页（双栏管理器，修改即时写盘）
   ════════════════════════════════════════════════════════════ */
const _PT_MODE_HINT = {
  off: '标签库不参与分析：不写入提示词，也不做检索。开启后会降低处理速度，增加 token 消耗。',
  on: '全部标签一次性写入提示词，一次分析完成；标签很多时会撑大提示词、引入幻觉，拖慢分析，适合小标签库。',
  enhanced: '先做一轮不带标签的分析，再按结果从标签库挑出候选标签，让 AI 第二轮修订，适合大标签库。',
};

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

async function persistPriorityTags() {
  const r = await callApi('save_priority_tags', state.ptItems, state.ptMode);
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
      <div class="tag-head">
        <span id="pt-mode-seg" class="mode-switch">
          <button type="button" class="pill clickable${state.ptMode === 'off' ? ' active' : ''}" data-mode="off">关闭</button>
          <span class="pill-sep"></span>
          <button type="button" class="pill clickable${state.ptMode === 'on' ? ' active' : ''}" data-mode="on">开启</button>
          <span class="pill-sep"></span>
          <button type="button" class="pill clickable${state.ptMode === 'enhanced' ? ' active' : ''}" data-mode="enhanced">增强</button>
        </span>
        <span class="help" style="margin:0" id="pt-mode-hint">${_PT_MODE_HINT[state.ptMode]}</span>
        <span class="tag-count" id="pt-count"></span>
      </div>
      <div class="tag-cols">
        <div class="tag-left">
          <input type="text" id="pt-search" placeholder="搜索标签、描述或关联词…" spellcheck="false"/>
          <div class="tag-left-bar">
            <div class="dd" id="pt-group-dd"></div>
            <div class="tag-seg">
              <button type="button" id="btn-pt-import">导入</button>
              <button type="button" id="btn-pt-export">导出</button>
            </div>
          </div>
          <div class="pt-list" id="pt-list"></div>
        </div>
        <div class="tag-right" id="pt-editor"></div>
      </div>
    </div>`;
  $$('#pt-mode-seg .pill').forEach(p => p.addEventListener('click', async () => {
    if (state.ptMode === p.dataset.mode) return;
    state.ptMode = p.dataset.mode;
    $$('#pt-mode-seg .pill').forEach(x => x.classList.toggle('active', x.dataset.mode === state.ptMode));
    const hint = $('#pt-mode-hint');
    if (hint) hint.textContent = _PT_MODE_HINT[state.ptMode] || '';
    await persistPriorityTags();
  }));
  $('#pt-search').addEventListener('input', e => {
    state.ptSearch = e.target.value.trim().toLowerCase();
    renderTagRows();
  });
  ptRenderGroupDd();
  $('#btn-pt-import').onclick = importTags;
  $('#btn-pt-export').onclick = exportTags;
  renderTagRows();
  ptRenderEditor();
}

function ptRenderGroupDd() {
  const dd = $('#pt-group-dd');
  if (!dd) return;
  const groups = ptAllGroups(state.ptItems);
  const opts = ['<div class="dd-opt active" data-value="">全部标签</div>',
    `<div class="dd-opt" data-value="__default__">默认分组</div>`,
    ...groups.map(g => `<div class="dd-opt" data-value="${esc(g)}">${esc(g)}</div>`)].join('');
  const curLabel = state.ptGroupFilter === '' ? '全部标签'
    : state.ptGroupFilter === '__default__' ? '默认分组' : state.ptGroupFilter;
  dd.innerHTML = `<button class="dd-btn" type="button"><span class="dd-label">${esc(curLabel)}</span>${ddArrow()}</button>
    <div class="dd-panel">${opts}</div>`;
  initDropdown(dd, v => {
    state.ptGroupFilter = v;
    dd.querySelector('.dd-label').textContent =
      v === '' ? '全部标签' : v === '__default__' ? '默认分组' : v;
    renderTagRows();
  });
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

function renderTagRows() {
  const list = $('#pt-list');
  if (!list) return;
  const idxs = ptFilteredIndices();
  if (!state.ptItems.length) {
    list.innerHTML = '<div class="pt-list-empty">暂无标签，在右侧输入关键词后回车即可添加。</div>';
  } else if (!idxs.length) {
    list.innerHTML = '<div class="pt-list-empty">没有匹配的标签。</div>';
  } else {
    list.innerHTML = idxs.map(i => {
      const it = state.ptItems[i];
      const sub = it.description || it.related;
      return `<div class="pt-item${state.ptSelected.has(i) ? ' selected' : ''}${i === state.ptEditIdx && !state.ptAddMode ? ' editing' : ''}" data-i="${i}">` +
        `<span class="pt-kw">${esc(it.keyword)}</span>` +
        (sub ? `<span class="pt-desc" data-tip="${esc(sub)}">${esc(sub)}</span>` : '') +
        `</div>`;
    }).join('');
    list.querySelectorAll('.pt-item').forEach(el => {
      const i = Number(el.dataset.i);
      el.addEventListener('click', e => ptRowClick(e, i));
      el.addEventListener('contextmenu', e => ptRowMenu(e, i));
    });
  }
  const cnt = $('#pt-count');
  if (cnt) cnt.textContent = state.ptItems.length ? `共 ${state.ptItems.length} 个` : '';
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
  if (!await persistPriorityTags()) {
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
  state.ptSelAnchor = null;
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
  ptRenderEditor();
}

function ptGroupOptions(cur) {
  const groups = ptAllGroups(state.ptItems);
  if (cur && !groups.includes(cur)) groups.push(cur);
  return ['<div class="dd-opt' + (cur ? '' : ' active') + '" data-value="">默认分组</div>',
    ...groups.map(g => `<div class="dd-opt${g === cur ? ' active' : ''}" data-value="${esc(g)}">${esc(g)}</div>`)].join('');
}

function ptBuildFields(container, it) {
  container.innerHTML = `
    <div class="field"><label>关键词</label><input type="text" id="pt-f-kw" value="${esc(it.keyword || '')}" spellcheck="false"/></div>
    <div class="field"><label>描述 <span class="pt-lbl-hint">增强模式下作为候选标签的判断依据</span></label><textarea id="pt-f-desc" rows="6">${esc(it.description || '')}</textarea></div>
    <div class="field"><label>关联 <span class="pt-lbl-hint">不开启嵌入模型时的字面检索，逗号分隔；向量检索同样生效</span></label><input type="text" id="pt-f-related" value="${esc(it.related || '')}" spellcheck="false"/></div>
    <div class="tag-ed-group">
      <div class="field"><label>分组 <span class="pt-lbl-hint">仅供显示，不参与检索</span></label>
        <div class="dd" id="pt-f-group"><button class="dd-btn" type="button"><span class="dd-label">${esc(it.group || '默认分组')}</span>${ddArrow()}</button>
          <div class="dd-panel">${ptGroupOptions(it.group || '')}</div></div>
      </div>
      <div class="field"><label>新分组</label><input type="text" id="pt-f-newgroup" placeholder="输入即新建分组" spellcheck="false"/></div>
    </div>`;
  initDropdown(container.querySelector('#pt-f-group'), () => ptMarkDirty(container));
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

function ptRenderEditor() {
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
    kwEl.focus();
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
  if (!f.keyword) { $('#pt-f-kw').focus(); return; }
  if (ptDupCheck(f.keyword, -1)) { toast('标签已存在: ' + f.keyword, 'err'); return; }
  const item = { keyword: f.keyword, description: f.description, related: f.related, group: f.group };
  state.ptItems.unshift(item);
  if (!await persistPriorityTags()) {
    state.ptItems.shift();
    return;
  }
  state.ptDirty = false;
  if (state.ptGroupFilter && state.ptGroupFilter !== '__default__' && state.ptGroupFilter !== f.group) {
    state.ptGroupFilter = f.group;
  }
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
  if (!f.keyword) { $('#pt-f-kw').focus(); return; }
  if (ptDupCheck(f.keyword, idx)) { toast('标签已存在: ' + f.keyword, 'err'); return; }
  const old = { ...state.ptItems[idx] };
  state.ptItems[idx] = { keyword: f.keyword, description: f.description, related: f.related, group: f.group };
  if (!await persistPriorityTags()) {
    state.ptItems[idx] = old;
    return;
  }
  state.ptDirty = false;
  ptRenderGroupDd();
  renderTagRows();
  toast(`已保存: ${f.keyword}`, 'ok');
}

function collectTagsData() {
  return { mode: state.ptMode || 'off', items: ptNormItems(state.ptItems) };
}

async function importTags() {
  const res = await callApi('import_priority_tags');
  if (!res || res.cancelled) return;
  if (!res.ok) { toast('导入失败: ' + (res.error || ''), 'err'); return; }
  if (state.ptItems.length &&
      !await showConfirm(`导入将覆盖当前 ${state.ptItems.length} 个标签，确定继续？`, { okText: '覆盖导入' })) return;
  const before = state.ptItems;
  state.ptItems = ptNormItems(res.items);
  if (!await persistPriorityTags()) {
    state.ptItems = before;
    return;
  }
  state.ptEditIdx = -1;
  state.ptAddMode = true;
  state.ptDirty = false;
  ptRenderGroupDd();
  renderTagRows();
  ptRenderEditor();
  toast(`已导入 ${state.ptItems.length} 个标签`, 'ok');
}

async function exportTags() {
  const d = collectTagsData();
  const res = await callApi('export_priority_tags', d.mode, d.items);
  if (!res || res.cancelled) return;
  if (res.ok) toast('已导出到: ' + (res.path || ''), 'ok');
  else toast('导出失败: ' + (res.error || ''), 'err');
}

/* ── 主界面标签 chip「加入标签检索」弹窗（与右栏编辑器同套字段） ── */
async function openTagAddToSearch(name) {
  const cur = await callApi('get_priority_tags');
  if (!cur) return;
  const items = ptNormItems(cur.items);
  const groups = ptAllGroups(items);
  const bg = document.createElement('div');
  bg.className = 'pt-editor-bg';
  bg.innerHTML = `
    <div class="pt-editor">
      <h3>加入标签检索</h3>
      <div class="tag-ed-body" id="pt-add-body"></div>
      <div class="pt-editor-foot">
        <button class="btn" id="pt-add-cancel">取消</button>
        <button class="btn primary" id="pt-add-save">保存</button>
      </div>
    </div>`;
  document.body.appendChild(bg);
  const body = bg.querySelector('#pt-add-body');
  ptBuildFields(body, { keyword: name, description: '', related: '', group: '' });
  if (!groups.length) body.querySelector('#pt-f-group').closest('.field').style.display = 'none';
  const kwEl = body.querySelector('#pt-f-kw');
  kwEl.focus(); kwEl.select();
  bg.addEventListener('keydown', e => {
    if (e.key === 'Escape') { e.stopPropagation(); bg.remove(); }
  });
  kwEl.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); bg.querySelector('#pt-add-save').click(); }
  });
  bg.querySelector('#pt-add-cancel').onclick = () => bg.remove();
  bg.querySelector('#pt-add-save').onclick = async () => {
    const f = ptCollectFields(body);
    if (!f.keyword) { kwEl.focus(); return; }
    if (items.some(x => x.keyword.toLowerCase() === f.keyword.toLowerCase())) {
      toast('标签已存在: ' + f.keyword, 'err'); kwEl.focus(); kwEl.select();
      return;
    }
    items.unshift({ keyword: f.keyword, description: f.description, related: f.related, group: f.group });
    const saveBtn = bg.querySelector('#pt-add-save');
    saveBtn.disabled = true;
    const r = await callApi('save_priority_tags', items, cur.mode || 'off');
    if (r && r.ok) {
      bg.remove();
      toast(`已加入标签检索: ${f.keyword}`, 'ok');
    } else {
      saveBtn.disabled = false;
      toast('保存失败: ' + ((r && r.error) || '未知错误'), 'err');
    }
  };
}
