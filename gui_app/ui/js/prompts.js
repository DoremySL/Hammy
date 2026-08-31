/* ════════════════════════════════════════════════════════════
   设置 Modal - 提示词预设页
   ════════════════════════════════════════════════════════════ */
let _pvTimer = null, _pvSeq = 0;

function renderPromptsTab(presets, activePreset) {
  const body = $('#modal-body');
  const isCustom = activePreset.id.startsWith('custom_');
  const isActive = activePreset.id === state.active_preset_id;
  const canDelete = isCustom && !isActive;
  const canSave = isCustom;
  body.innerHTML = `
    <div class="preset-bar">
      <input type="hidden" id="preset-editing-id" value="${esc(activePreset.id)}"/>
      <div class="dd" id="preset-select">
        <button class="dd-btn"><span class="dd-label">${esc(activePreset.name)}</span>${ddArrow()}</button>
        <div class="dd-panel">
          ${presets.map(p => `<div class="dd-opt${p.id === activePreset.id ? ' active' : ''}${p.id === state.active_preset_id ? ' is-current' : ''}" data-value="${esc(p.id)}">${esc(p.name)}</div>`).join('')}
        </div>
      </div>
      <span class="preset-bar-actions">
        <button class="btn sm" id="btn-activate-preset">启用</button>
        <button class="btn sm" id="btn-save-preset" ${canSave ? '' : 'disabled'} data-tip="${canSave ? '保存对该预设的修改' : '内置预设不可保存，请「保存为新预设」'}">保存</button>
        <button class="btn sm" id="btn-save-as-preset">保存为新预设</button>
        <button class="btn sm" id="btn-delete-preset" ${canDelete ? '' : 'disabled'}>删除</button>
      </span>
    </div>

    <div class="group">
      <h3><span class="tip-text" data-tip="定义 AI 的角色与行为准则，作用于所有字段的生成。">系统提示词（System Prompt）</span></h3>
      <div class="field"><textarea id="preset-system_prompt" data-field="system_prompt" rows="5">${esc(activePreset.system_prompt || '')}</textarea></div>
    </div>
    <div class="group">
      <h3><span class="tip-text" data-tip="分别告诉 AI 如何生成 plot、title、tags 三个字段。">字段引导文案</span></h3>
      <div class="grid-2">
        <div class="field"><label>plot 描述要求</label><textarea data-field="plot_guidance">${esc(activePreset.fields.plot_guidance)}</textarea></div>
        <div class="field"><label>title 字数/格式要求</label><textarea data-field="title_guidance">${esc(activePreset.fields.title_guidance)}</textarea></div>
      </div>
      <div class="field" style="margin-top:14px"><label>tags 维度要求</label><textarea data-field="tags_dim">${esc(activePreset.fields.tags_dim)}</textarea></div>
    </div>
    <div class="group">
      <h3><span class="tip-text" data-tip="给 AI 一个高质量范例，输出格式与风格会更稳定。">示例值</span></h3>
      <div class="field"><label>plot 示例</label><textarea data-field="plot_example">${esc(activePreset.fields.plot_example)}</textarea></div>
      <div class="field" style="margin-top:14px"><label>tags 示例</label><input type="text" data-field="tags_example" value="${esc(activePreset.fields.tags_example)}"/></div>
      <div class="field" style="margin-top:14px"><label>title 示例</label><input type="text" data-field="title_example" value="${esc(activePreset.fields.title_example)}"/></div>
    </div>
    <div class="group">
      <button class="disclosure" id="prompt-preview-toggle"><svg class="ic" style="width:12px;height:12px"><use href="#ic-play"/></svg> 预览：AI 收到的引导内容</button>
      <div class="prompt-preview" id="prompt-preview" style="display:none"><span class="readonly-tag">自动生成</span>...</div>
    </div>
  `;

  const pvToggle = $('#prompt-preview-toggle');
  const pvContent = $('#prompt-preview');
  pvToggle.onclick = () => {
    const shown = pvContent.style.display !== 'none';
    pvContent.style.display = shown ? 'none' : '';
    pvToggle.classList.toggle('open', !shown);
  };

  function updatePreview() {
    clearTimeout(_pvTimer);
    const seq = ++_pvSeq;
    _pvTimer = setTimeout(() => {
      const fields = {};
      body.querySelectorAll('[data-field]').forEach(el => {
        if (el.dataset.field !== 'system_prompt') fields[el.dataset.field] = el.value;
      });
      apiCall('preview_prompt', fields).then(r => {
        if (seq !== _pvSeq) return;
        const pv = $('#prompt-preview');
        if (!pv) return;
        pv.innerHTML = '<span class="readonly-tag">自动生成</span>\n' + esc(r.prompt);
      }).catch(() => {
        const pv = $('#prompt-preview');
        if (pv) pv.innerHTML = '<span class="readonly-tag">自动生成</span> 预览失败';
      });
    }, 250);
  }
  body.querySelectorAll('[data-field]').forEach(el => el.addEventListener('input', updatePreview));
  updatePreview();

  initDropdown($('#preset-select'), async (val) => {
    const p = await callApi('get_preset', val);
    if (p && !p.error) renderPromptsTab(presets, p);
  });
  $('#btn-activate-preset').onclick = async () => {
    const sel = getDropdownValue($('#preset-select'));
    const r = await callApi('set_active_preset', sel);
    if (!r) return;
    if (r.ok) {
      toast('已启用', 'ok');
      state.active_preset_id = sel;
      $$('#preset-select .dd-opt').forEach(o => o.classList.toggle('is-current', o.dataset.value === sel));
    }
    else toast(r.error || '失败', 'err');
  };
  $('#btn-save-as-preset').onclick = async () => {
    const name = await showPrompt('新预设将包含当前编辑的全部内容。', {
      title: '保存为新预设',
      defaultValue: activePreset.name + ' 副本',
      okText: '保存',
    });
    if (!name) return;
    const data = collectPresetData(name);
    data.id = '';
    const res = await callApi('save_preset', data);
    if (!res) return;
    if (res.ok) {
      toast('已保存为新预设', 'ok');
      await callApi('set_active_preset', res.id);
      state.active_preset_id = res.id;
      const presets2 = await callApi('list_presets');
      const p = await callApi('get_preset', res.id);
      if (presets2 && p) renderPromptsTab(presets2, p);
    } else toast('保存失败: ' + (res.error || ''), 'err');
  };
  $('#btn-save-preset').onclick = async () => {
    const res = await callApi('save_preset', collectPresetData(activePreset.name));
    if (!res) return;
    if (res.ok) {
      toast('预设已保存', 'ok');
      await callApi('set_active_preset', res.id);
      state.active_preset_id = res.id;
      const presets2 = await callApi('list_presets');
      const p = await callApi('get_preset', res.id);
      if (presets2 && p) renderPromptsTab(presets2, p);
    } else toast('保存失败: ' + (res.error || ''), 'err');
  };
  $('#btn-delete-preset').onclick = async () => {
    const sel = getDropdownValue($('#preset-select'));
    if (sel === 'default') { toast('内置预设不允许删除', 'err'); return; }
    if (sel === state.active_preset_id) { toast('已启用的预设不允许删除，请先启用其他预设', 'err'); return; }
    if (!await showConfirm('确定删除该预设？', { okText: '删除' })) return;
    const r = await callApi('delete_preset', sel);
    if (!r) return;
    if (r.ok) {
      const presets2 = await callApi('list_presets');
      const active2 = await callApi('get_active_preset');
      if (presets2 && active2) renderPromptsTab(presets2, active2.preset);
      toast('已删除', 'ok');
    } else toast(r.error || '失败', 'err');
  };
}

function collectPresetData(name) {
  const preset = {
    id: $('#preset-editing-id') ? $('#preset-editing-id').value : '',
    name: name || '未命名',
    system_prompt: $('#preset-system_prompt').value,
    fields: {},
  };
  $$('#modal-body [data-field]').forEach(el => {
    if (el.dataset.field === 'system_prompt') return;
    preset.fields[el.dataset.field] = el.value;
  });
  return preset;
}
