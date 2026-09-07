/* ════════════════════════════════════════════════════════════
   设置 Modal - 标签检索页
   ════════════════════════════════════════════════════════════ */
const _PT_MODE_HINT = {
  off: '标签库不参与分析：不写入提示词，也不做检索。',
  on: '全部标签一次性写入提示词，一次分析完成；标签很多时会撑大提示词、引入幻觉，拖慢分析，适合小标签库。',
  enhanced: '先做一轮不带标签的分析，再按结果从标签库挑出候选标签，让 AI 第二轮修订，适合大标签库。',
};

function renderTagsTab(pt) {
  state.ptItems = (pt.items || []).map(x => ({ keyword: x.keyword, description: x.description || '' }));
  state.ptMode = ['off', 'on', 'enhanced'].includes(pt.mode) ? pt.mode : 'off';
  state.ptSelected = new Set();
  state.ptSelAnchor = null;
  const body = $('#modal-body');
  body.innerHTML = `
    <div class="group">
      <div class="field" style="flex-direction:row;align-items:center;gap:12px;flex-wrap:wrap">
        <span id="pt-mode-seg" class="mode-switch">
          <button type="button" class="pill clickable${state.ptMode === 'off' ? ' active' : ''}" data-mode="off">关闭</button>
          <span class="pill-sep"></span>
          <button type="button" class="pill clickable${state.ptMode === 'on' ? ' active' : ''}" data-mode="on">开启</button>
          <span class="pill-sep"></span>
          <button type="button" class="pill clickable${state.ptMode === 'enhanced' ? ' active' : ''}" data-mode="enhanced">增强</button>
        </span>
        <span class="help" style="margin:0" id="pt-mode-hint">${_PT_MODE_HINT[state.ptMode]}</span>
      </div>
    </div>
    <div class="group">
      <div class="pt-quick" style="margin-bottom:0">
        <input type="text" id="pt-quick-kw" placeholder="输入关键词，Enter 快速添加"/>
        <button class="btn sm" id="btn-pt-quick-add">添加</button>
      </div>
    </div>
    <div class="group">
      <div class="pt-tools">
        <button class="btn sm" id="btn-pt-batch">快速创建</button>
        <button class="btn sm" id="btn-pt-import">导入</button>
        <button class="btn sm" id="btn-pt-export">导出</button>
        <button class="btn sm" id="btn-pt-save-list">保存</button>
        <span class="pt-count" id="pt-count"></span>
      </div>
      <div class="pt-list" id="pt-list"></div>
    </div>`;
  renderTagRows();
  $$('#pt-mode-seg .pill').forEach(p => p.addEventListener('click', () => {
    state.ptMode = p.dataset.mode;
    $$('#pt-mode-seg .pill').forEach(x => x.classList.toggle('active', x.dataset.mode === state.ptMode));
    const hint = $('#pt-mode-hint');
    if (hint) hint.textContent = _PT_MODE_HINT[state.ptMode] || '';
  }));
  $('#btn-pt-quick-add').onclick = quickAddTag;
  $('#pt-quick-kw').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); quickAddTag(); } });
  $('#btn-pt-batch').onclick = batchCreateTags;
  $('#btn-pt-import').onclick = importTags;
  $('#btn-pt-export').onclick = exportTags;
  $('#btn-pt-save-list').onclick = saveTagsListOnly;
}

function renderTagRows() {
  const list = $('#pt-list');
  if (!list) return;
  if (!state.ptItems.length) {
    list.innerHTML = '<div class="empty">暂无标签，请在上方输入关键词添加。</div>';
  } else {
    list.innerHTML = state.ptItems.map((it, i) =>
      `<div class="pt-item${state.ptSelected.has(i) ? ' selected' : ''}" data-i="${i}">` +
      `<span class="pt-kw">${esc(it.keyword)}</span>` +
      `<span class="pt-desc" data-tip="${esc(it.description)}">${it.description ? esc(it.description) : '<i class="pt-nodesc">（无描述）</i>'}</span>` +
      `</div>`).join('');
    list.querySelectorAll('.pt-item').forEach(el => {
      const i = Number(el.dataset.i);
      el.addEventListener('click', e => selectTag(i, e));
      el.addEventListener('contextmenu', e => showTagContextMenu(e, i));
    });
  }
  const cnt = $('#pt-count'); if (cnt) cnt.textContent = state.ptItems.length ? `共 ${state.ptItems.length} 个` : '';
}

function selectTag(idx, e) {
  state.ptSelAnchor = pickSelection(state.ptSelected, state.ptSelAnchor, idx, e);
  $$('#pt-list .pt-item').forEach(el => {
    el.classList.toggle('selected', state.ptSelected.has(Number(el.dataset.i)));
  });
}

function showTagContextMenu(e, idx) {
  e.preventDefault();
  e.stopPropagation();
  const m = $('#ctxmenu');
  const delN = (state.ptSelected.has(idx) && state.ptSelected.size > 1) ? state.ptSelected.size : 0;
  const items = [
    { label: '编辑', fn: () => openTagEditor(idx) },
    { label: delN ? `删除选中 (${delN})` : '删除', fn: () => deleteTagAt(idx) },
  ];
  m.innerHTML = items.map((it, i) => `<button data-i="${i}">${it.label}</button>`).join('<hr>');
  m.querySelectorAll('button').forEach((b, i) => { b.onclick = () => { items[i].fn(); hideContextMenu(); }; });
  positionCtxMenu(m, e);
}

async function deleteTagAt(idx) {
  if (state.ptSelected.has(idx) && state.ptSelected.size > 1) {
    if (!await showConfirm(`确定删除选中的 ${state.ptSelected.size} 个标签？`, { okText: '删除' })) return;
    [...state.ptSelected].sort((a, b) => b - a).forEach(i => state.ptItems.splice(i, 1));
  } else {
    state.ptItems.splice(idx, 1);
  }
  state.ptSelected = new Set(); state.ptSelAnchor = null;
  renderTagRows();
}

function quickAddTag() {
  const inp = $('#pt-quick-kw'); if (!inp) return;
  const kw = inp.value.trim();
  if (!kw) { inp.focus(); return; }
  if (state.ptItems.some(x => x.keyword.toLowerCase() === kw.toLowerCase())) {
    toast('标签已存在: ' + kw, 'err'); inp.focus(); return;
  }
  state.ptItems.push({ keyword: kw, description: '' });
  state.ptSelected = new Set(); state.ptSelAnchor = null;
  inp.value = ''; inp.focus();
  renderTagRows();
}

const _PT_NOISE = new Set(['更多', '全部', '其他', '首页', '查看全部']);
function batchCreateTags() {
  const bg = document.createElement('div');
  bg.className = 'pt-editor-bg';
  bg.innerHTML = `
    <div class="pt-editor">
      <h3>快速创建标签</h3>
      <div class="field"><label class="hint-label">粘贴分类文本（自动按空格、换行、逗号、竖线分割）</label>
        <textarea id="pt-batch-input" rows="8" placeholder="番剧  电影  国创  电视剧…"></textarea></div>
      <div class="pt-editor-foot">
        <button class="btn" id="pt-batch-cancel">取消</button>
        <button class="btn primary" id="pt-batch-ok">确定</button>
      </div>
    </div>`;
  document.body.appendChild(bg);
  const ta = $('#pt-batch-input'); ta.focus();
  $('#pt-batch-cancel').onclick = () => bg.remove();
  bg.addEventListener('keydown', e => {
    if (e.key === 'Escape') { e.stopPropagation(); bg.remove(); }
  });
  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); doBatch(); }
  });
  $('#pt-batch-ok').onclick = doBatch;
  function doBatch() {
    const raw = ta.value;
    const tokens = raw.split(/[\s,，|｜、]+/).map(s => s.trim()).filter(Boolean);
    const existing = new Set(state.ptItems.map(x => x.keyword.toLowerCase()));
    const added = [];
    for (const t of tokens) {
      if (_PT_NOISE.has(t)) continue;
      if (existing.has(t.toLowerCase())) continue;
      existing.add(t.toLowerCase());
      added.push({ keyword: t, description: '' });
    }
    if (!added.length) { toast('未识别到新标签（可能全部重复或为噪音词）', 'err'); return; }
    state.ptItems.push(...added);
    state.ptSelected = new Set(); state.ptSelAnchor = null;
    bg.remove();
    renderTagRows();
    toast(`已添加 ${added.length} 个标签（点「保存」生效）`, 'ok');
  }
}

function openTagEditor(idx) {
  const cur = state.ptItems[idx];
  const bg = document.createElement('div');
  bg.className = 'pt-editor-bg';
  bg.innerHTML = `
    <div class="pt-editor">
      <h3>编辑标签</h3>
      <div class="field"><label>关键词</label><input type="text" id="pt-ed-kw" value="${esc(cur.keyword)}"/></div>
      <div class="field"><label>描述（可选，说明该标签的适用场景）</label><textarea id="pt-ed-desc" rows="5">${esc(cur.description)}</textarea></div>
      <div class="pt-editor-foot">
        <button class="btn" id="pt-ed-cancel">取消</button>
        <button class="btn primary" id="pt-ed-save">保存</button>
      </div>
    </div>`;
  document.body.appendChild(bg);
  const kwEl = $('#pt-ed-kw'); kwEl.focus();
  $('#pt-ed-cancel').onclick = () => bg.remove();
  bg.addEventListener('keydown', e => {
    if (e.key === 'Escape') { e.stopPropagation(); bg.remove(); }
  });
  $('#pt-ed-save').onclick = () => {
    const kw = kwEl.value.trim();
    if (!kw) { kwEl.focus(); return; }
    const dup = state.ptItems.some((x, i) => i !== idx && x.keyword.toLowerCase() === kw.toLowerCase());
    if (dup) { toast('标签已存在: ' + kw, 'err'); kwEl.focus(); kwEl.select(); return; }
    const desc = $('#pt-ed-desc').value.trim();
    state.ptItems[idx] = { keyword: kw, description: desc };
    state.ptSelected = new Set(); state.ptSelAnchor = null;
    bg.remove();
    renderTagRows();
  };
}

function showChipMenu(e, name) {
  e.preventDefault();
  e.stopPropagation();
  const m = $('#ctxmenu');
  const items = [{ label: '加入标签检索', fn: () => openTagAddToSearch(name) }];
  m.innerHTML = items.map((it, i) => `<button data-i="${i}">${it.label}</button>`).join('');
  m.querySelectorAll('button').forEach((b, i) => { b.onclick = () => { items[i].fn(); hideContextMenu(); }; });
  positionCtxMenu(m, e);
}

async function openTagAddToSearch(name) {
  const bg = document.createElement('div');
  bg.className = 'pt-editor-bg';
  bg.innerHTML = `
    <div class="pt-editor">
      <h3>加入标签检索</h3>
      <div class="field"><label>关键词</label><input type="text" id="pt-add-kw" value="${esc(name)}"/></div>
      <div class="field"><label>描述（可选，说明该标签的适用场景）</label><textarea id="pt-add-desc" rows="5">${esc(name)}</textarea></div>
      <div class="pt-editor-foot">
        <button class="btn" id="pt-add-cancel">取消</button>
        <button class="btn primary" id="pt-add-save">保存</button>
      </div>
    </div>`;
  document.body.appendChild(bg);
  const kwEl = $('#pt-add-kw'); kwEl.focus(); kwEl.select();
  $('#pt-add-cancel').onclick = () => bg.remove();
  bg.addEventListener('keydown', e => {
    if (e.key === 'Escape') { e.stopPropagation(); bg.remove(); }
  });
  kwEl.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); $('#pt-add-save').click(); }
  });
  $('#pt-add-save').onclick = async () => {
    const kw = kwEl.value.trim();
    if (!kw) { kwEl.focus(); return; }
    const desc = $('#pt-add-desc').value.trim();
    const saveBtn = $('#pt-add-save'); saveBtn.disabled = true;
    try {
      const cur = await apiCall('get_priority_tags');
      const items = ((cur && cur.items) || []).map(x => ({ keyword: x.keyword, description: x.description || '' }));
      if (items.some(x => x.keyword.toLowerCase() === kw.toLowerCase())) {
        toast('标签已存在: ' + kw, 'err');
        saveBtn.disabled = false; kwEl.focus(); kwEl.select();
        return;
      }
      items.push({ keyword: kw, description: desc });
      const r = await apiCall('save_priority_tags', items, (cur && cur.mode) || 'off');
      if (r && r.ok) {
        bg.remove();
        toast(`已加入标签检索: ${kw}（设置→标签检索 中可管理）`, 'ok');
      } else {
        saveBtn.disabled = false;
        toast('保存失败: ' + ((r && r.error) || '未知错误'), 'err');
      }
    } catch (e) {
      saveBtn.disabled = false;
      toast('保存失败: ' + ((e && e.message) || e), 'err');
    }
  };
}

function collectTagsData() {
  const mode = state.ptMode || 'off';
  const items = state.ptItems
    .map(x => ({ keyword: (x.keyword || '').trim(), description: (x.description || '').trim() }))
    .filter(x => x.keyword);
  return { mode, items };
}

async function importTags() {
  const res = await callApi('import_priority_tags');
  if (!res || res.cancelled) return;
  if (!res.ok) { toast('导入失败: ' + (res.error || ''), 'err'); return; }
  if (state.ptItems.length && !await showConfirm(`导入将覆盖当前 ${state.ptItems.length} 个标签，确定继续？`, { okText: '覆盖导入' })) return;
  state.ptItems = res.items.map(x => ({ keyword: x.keyword, description: x.description || '' }));
  state.ptSelected = new Set(); state.ptSelAnchor = null;
  renderTagRows();
  toast(`已导入 ${res.items.length} 个标签（点「保存」生效）`, 'ok');
}

async function exportTags() {
  const d = collectTagsData();
  const res = await callApi('export_priority_tags', d.mode, d.items);
  if (!res || res.cancelled) return;
  if (res.ok) toast('已导出到: ' + (res.path || ''), 'ok');
  else toast('导出失败: ' + (res.error || ''), 'err');
}

async function saveTagsListOnly() {
  const d = collectTagsData();
  const res = await callApi('save_priority_tags', d.items, d.mode);
  if (!res) return;
  if (res.ok) {
    toast(`标签列表已保存（模式: ${{ off: '关闭', on: '开启', enhanced: '增强' }[d.mode] || d.mode}）`, 'ok');
  } else toast('保存失败: ' + (res.error || ''), 'err');
}
