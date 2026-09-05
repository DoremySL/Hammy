/* ════════════════════════════════════════════════════════════
   设置 Modal - 标签增强页
   ════════════════════════════════════════════════════════════ */
function renderTagsTab(pt) {
  state.ptItems = (pt.items || []).map(x => ({ keyword: x.keyword, description: x.description || '' }));
  state.ptLoadedEnabled = !!pt.enabled;
  state.ptSelected = new Set();
  state.ptSelAnchor = null;
  const body = $('#modal-body');
  body.innerHTML = `
    <div class="group">
      <div class="field" style="flex-direction:row;align-items:center;gap:12px">
        <label class="switch"><input type="checkbox" id="pt-enabled" ${pt.enabled ? 'checked' : ''}/> 启用标签增强</label>
        <span class="help" style="margin:0">占位功能，后续会大改，目前仅简单把全部关键词拼入 prompt，不宜过多否则模型容易产生幻觉。</span>
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
    </div>
    <div class="group">
      <button class="disclosure" id="pt-preview-toggle"><svg class="ic" style="width:12px;height:12px"><use href="#ic-play"/></svg> 预览：将注入提示词的段落</button>
      <div class="prompt-preview" id="pt-preview" style="display:none"></div>
    </div>`;
  renderTagRows();
  refreshPtPreview();
  $('#btn-pt-quick-add').onclick = quickAddTag;
  $('#pt-quick-kw').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); quickAddTag(); } });
  $('#btn-pt-batch').onclick = batchCreateTags;
  $('#btn-pt-import').onclick = importTags;
  $('#btn-pt-export').onclick = exportTags;
  $('#btn-pt-save-list').onclick = saveTagsListOnly;
  $('#pt-enabled').addEventListener('change', refreshPtPreview);
  const ptToggle = $('#pt-preview-toggle');
  const ptPreview = $('#pt-preview');
  ptToggle.onclick = () => {
    const shown = ptPreview.style.display !== 'none';
    ptPreview.style.display = shown ? 'none' : '';
    ptToggle.classList.toggle('open', !shown);
  };
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
  renderTagRows(); refreshPtPreview();
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
  renderTagRows(); refreshPtPreview();
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
    renderTagRows(); refreshPtPreview();
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
    const desc = $('#pt-ed-desc').value.trim();
    state.ptItems[idx] = { keyword: kw, description: desc };
    state.ptSelected = new Set(); state.ptSelAnchor = null;
    bg.remove();
    renderTagRows(); refreshPtPreview();
  };
}

function collectTagsData() {
  const enabled = $('#pt-enabled') ? $('#pt-enabled').checked : false;
  const items = state.ptItems
    .map(x => ({ keyword: (x.keyword || '').trim(), description: (x.description || '').trim() }))
    .filter(x => x.keyword);
  return { enabled, items };
}

let _ptPreviewTimer = null, _ptPreviewSeq = 0;
async function refreshPtPreview() {
  const pv = $('#pt-preview'); if (!pv) return;
  clearTimeout(_ptPreviewTimer);
  const seq = ++_ptPreviewSeq;
  _ptPreviewTimer = setTimeout(async () => {
    const d = collectTagsData();
    try {
      const res = await apiCall('preview_priority_tags', d.enabled, d.items);
      if (seq !== _ptPreviewSeq) return;
      const el = $('#pt-preview'); if (!el) return;
      el.textContent = (res && res.section) ? res.section : '（未启用或列表为空，不会注入任何内容）';
    } catch (e) {
      if (seq !== _ptPreviewSeq) return;
      const el = $('#pt-preview'); if (!el) return;
      el.textContent = '预览失败：' + String(e);
    }
  }, 250);
}

async function importTags() {
  const res = await callApi('import_priority_tags');
  if (!res || res.cancelled) return;
  if (!res.ok) { toast('导入失败: ' + (res.error || ''), 'err'); return; }
  if (state.ptItems.length && !await showConfirm(`导入将覆盖当前 ${state.ptItems.length} 个标签，确定继续？`, { okText: '覆盖导入' })) return;
  state.ptItems = res.items.map(x => ({ keyword: x.keyword, description: x.description || '' }));
  state.ptSelected = new Set(); state.ptSelAnchor = null;
  const en = $('#pt-enabled'); if (en) en.checked = !!res.enabled;
  renderTagRows(); refreshPtPreview();
  toast(`已导入 ${res.items.length} 个标签（点「保存」生效）`, 'ok');
}

async function exportTags() {
  const d = collectTagsData();
  const res = await callApi('export_priority_tags', d.enabled, d.items);
  if (!res || res.cancelled) return;
  if (res.ok) toast('已导出到: ' + (res.path || ''), 'ok');
  else toast('导出失败: ' + (res.error || ''), 'err');
}

async function saveTagsListOnly() {
  const d = collectTagsData();
  const res = await callApi('save_priority_tags', state.ptLoadedEnabled, d.items);
  if (!res) return;
  if (res.ok) {
    toast('标签列表已保存', 'ok');
  } else toast('保存失败: ' + (res.error || ''), 'err');
}
