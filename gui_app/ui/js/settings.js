/* ════════════════════════════════════════════════════════════
   设置 Modal
   ════════════════════════════════════════════════════════════ */
let _settingsSeq = 0;

async function ensureLlamaStateKnown() {
  if (state.llamaSynced) return;
  try { syncLlamaState(await apiCall('get_llama_status')); }
  catch (e) {  }
}

async function openSettings(tab) {
  await ensureLlamaStateKnown();
  if (tab === 'ai' || tab === undefined || tab === null) {
    const toLlama = state.llamaEnabled && state.llamaIntegration;
    state.settings_tab = toLlama ? 'llama' : 'config';
    state.settings_scroll_to = (tab === 'ai' && !toLlama) ? 'ai' : null;
  } else if (['config', 'prompts', 'tags', 'experimental', 'llama', 'workspace'].includes(tab)) {
    state.settings_tab = tab;
  } else {
    state.settings_tab = 'config';
  }
  $$('#settings-tabs .tab').forEach(x =>
    x.classList.toggle('active', x.dataset.settingsTab === state.settings_tab));
  $('#modal').classList.add('show');
  renderSettings();
}
function closeSettings() {
  state.settingsOpen.clear();
  state.settingsScroll = 0;
  $('#modal-body').innerHTML = '';
  $('#modal').classList.remove('show');
}
$('#btn-settings').addEventListener('click', () => openSettings());
$('#btn-closemodal').addEventListener('click', closeSettings);
$('#btn-cancelcfg').addEventListener('click', closeSettings);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && $('#modal').classList.contains('show')) closeSettings();
});

$$('#settings-tabs .tab').forEach(t => {
  t.addEventListener('click', () => {
    state.settings_tab = t.dataset.settingsTab;
    state.settings_scroll_to = null;
    $$('#settings-tabs .tab').forEach(x => x.classList.toggle('active', x === t));
    renderSettings();
  });
});

async function renderSettings() {
  const seq = ++_settingsSeq;
  const body = $('#modal-body');
  const foot = $('#modal-foot-note');
  state.settingsScroll = body.scrollTop;
  let cfg;
  if (state.settings_tab === 'config') {
    try {
      cfg = await apiCall('get_config');
    } catch (e) {
      if (seq !== _settingsSeq) return;
      body.innerHTML = '<div class="empty">' + icon('warning') + ' 加载失败：' + esc(String(e).slice(0, 100)) +
        '<br/><br/><button class="btn">重试</button></div>';
      body.querySelector('button').onclick = () => renderSettings();
      return;
    }
    if (seq !== _settingsSeq) return;
  }
  try {
    switch (state.settings_tab) {
      case 'config':
        foot.textContent = '修改后点击应用生效。';
        renderConfigTab(cfg);
        break;
      case 'prompts':
        foot.textContent = '编辑后点击「应用」生效。';
        await renderPromptsTabWrapper();
        break;
      case 'tags':
        foot.textContent = '编辑后点击「应用」生效。';
        await renderTagsTabWrapper();
        break;
      case 'experimental':
        foot.textContent = '总开关需点「应用」才生效。';
        await renderExperimentalTabWrapper();
        break;
      case 'llama':
        foot.textContent = '参数修改后点「应用」保存生效；启动/停止/聊天界面为即时操作。';
        await renderLlamaTabWrapper();
        break;
      case 'workspace':
        foot.textContent = '';
        await renderWorkspaceTabWrapper();
        break;
    }
  } catch (e) {
    if (seq !== _settingsSeq) return;
    body.innerHTML = '<div class="empty">' + icon('warning') + ' 渲染失败：' + esc(String(e).slice(0, 200)) +
      '<br/><br/><button class="btn">重试</button></div>';
    body.querySelector('button').onclick = () => renderSettings();
  }
  if (seq !== _settingsSeq) return;
  updateSaveButtonState();
  updateLlamaTabVisibility();
  if (state.settings_tab === 'config') {
    animateModalBody();
    body.scrollTop = state.settingsScroll;
  }
}

function updateLlamaTabVisibility() {
  const t = document.querySelector('#settings-tabs .tab[data-settings-tab="llama"]');
  if (t) t.style.display = state.llamaEnabled ? '' : 'none';
}

function animateModalBody() {
  const body = $('#modal-body');
  body.classList.remove('fade-in');
  void body.offsetWidth;
  body.classList.add('fade-in');
}

const CFG_FIELDS_BASIC = [
  { group: 'AI 服务（基础）', noTitle: true, cols: 4, fields: [
    { sec: 'ai', key: 'model', label: '模型名称', type: 'text', span: 2, help: '需支持多模态', default: 'model' },
    { sec: 'ai', key: 'base_url', label: 'API 地址', type: 'url', span: 2, help: 'OpenAI 兼容 /v1', default: 'http://localhost:8080/v1' },
    { sec: 'ai', key: 'api_key', label: 'API 密钥', type: 'password', span: 2, help: '本地 AI 服务通常无需填写', default: 'not-needed' },
    { sec: 'ai', key: 'ai_workers', label: '并发数', type: 'number', min: 1, help: '同时发起的 AI 请求数，越大越快但更占内存/显存；开启本地推理集成后由「并发线程 -np」接管', default: 4 },
  ]},
];
const CFG_FIELDS_AI_ADVANCED = [
  { group: 'AI 服务（进阶）', noTitle: true, cols: 3, fields: [
    { sec: 'ai', key: 'max_tokens', label: '最大生成长度', type: 'number', min: 1, help: '单次 AI 输出的 Token 上限，改动提示词或开启思考时可按需加大', default: 3000 },
    { sec: 'ai', key: 'temperature', label: '温度', type: 'number', min: 0, max: 2, step: 0.1, help: '采样随机性，越高越有创意但可能偏离主题，越低越稳定', default: 0.6 },
    { sec: 'ai', key: 'top_p', label: 'Top-p', type: 'number', min: 0, max: 1, step: 0.05, help: '核采样阈值，与温度共同控制输出多样性', default: 0.8 },
    { sec: 'ai', key: 'retry_times', label: '重试次数', type: 'number', min: 0, help: '请求失败后的重试次数，重试之间会自动等待', default: 2 },
    { sec: 'ai', key: 'ai_timeout', label: '超时(秒)', type: 'number', min: 1, help: '单次 AI 请求的超时时间（秒），超时按失败重试', default: 300 },
    { sec: 'ai', key: 'enforce_json_mode', label: 'JSON 模式', type: 'bool', help: '确保输出合法 JSON' },
  ]},
];
const CFG_FIELDS_PROCESSING = [
  { group: '视频抽帧', noTitle: true, cols: 4, fields: [
    { sec: 'video', key: 'sampling_points', label: '采样点位', type: 'number', min: 1, help: '均匀分布的关键帧取样位置数；采样点位*每点帧数=抽取的关键帧数量', default: 5 },
    { sec: 'video', key: 'frames_per_point', label: '每点帧数', type: 'number', min: 1, help: '每个点位取连续关键帧数，增大此项可给AI提供连续的画面信息', default: 3 },
    { sec: 'video', key: 'frame_max_side', label: '长边像素', type: 'number', min: 64, help: '抽帧图片长边上限，仅缩小不放大', default: 640 },
    { sec: 'video', key: 'frame_time_tags', label: '时间标签', type: 'select', options: [['1', '添加时间标签'], ['2', '添加并用于优化缩略图'], ['0', '不添加标签']], default: '0', help: '每张截图前添加时间戳，帮助模型理解画面时间顺序；开启用于优化缩略图将提示模型给出最符合视频主题的截图时间戳，对模型能力有要求' },
  ]},
  { group: '命名格式', noTitle: true, fields: [
    { sec: 'naming', key: 'include_date', label: '日期前缀', type: 'bool', help: '文件名前添加日期，降低重名几率' },
    { sec: 'naming', key: 'include_original', label: '初始文件名后缀', type: 'bool', help: '文件名末尾追加初始文件名，降低重名几率' },
  ]},
  { group: 'GUI 选项', noTitle: true, fields: [
    { sec: '__gui', key: 'nfo_auto_export', label: '自动输出 NFO 至视频目录', type: 'bool', help: '开启后处理时直接写入视频目录；关闭则先存工作区，导出时再复制' },
    { sec: '__gui', key: 'force_animation', label: '默认启用动画', type: 'bool', help: '忽略系统「减弱动态效果」设置' },
  ]},
];

function collectConfigData() {
  const data = { ai: {}, video: {}, naming: {} };
  $$('#modal-body [data-key]').forEach(el => {
    if (el.disabled) return;
    const sec = el.dataset.sec, key = el.dataset.key;
    let v;
    if (el.classList && el.classList.contains('dd')) v = el.dataset.bool ? Number(el.dataset.value) : el.dataset.value;
    else if (el.type === 'checkbox') v = el.checked;
    else if (el.type === 'number') {
      if (el.value === '') return;
      v = Number(el.value);
      const min = el.min !== '' ? Number(el.min) : undefined;
      const max = el.max !== '' ? Number(el.max) : undefined;
      if (min !== undefined && v < min) v = min;
      if (max !== undefined && v > max) v = max;
    }
    else v = el.value;
    if (sec === '__gui') data[key] = v;
    else data[sec][key] = v;
  });
  return data;
}

let aiTestToken = 0;

async function saveConfigAndTest() {
  const btn = $('#btn-save-and-test');
  if (!btn) return;
  if (btn.dataset.testing === '1') {
    aiTestToken++;
    btn.disabled = false;
    btn.dataset.testing = '';
    btn.textContent = '应用并测试连接';
    toast('已停止连接测试', 'ok');
    return;
  }
  const token = ++aiTestToken;
  btn.disabled = true; btn.textContent = '应用中…';
  try {
    const data = collectConfigData();
    const res = await apiCall('save_config', data);
    if (!res || !res.ok) {
      toast('应用失败: ' + ((res && res.error) || ''), 'err');
      return;
    }
    state.thumbOptimize = Number((data.video || {}).frame_time_tags) === 2;
    btn.textContent = '正在测试连接…';
    btn.disabled = false;
    btn.dataset.testing = '1';
    const r = await withTimeout(apiCall('check_connection'), 20000);
    if (token !== aiTestToken) return;
    btn.dataset.testing = '';
    const addr = r.base_url ? '（' + r.base_url + '）' : '';
    toast(r.ok ? '已应用 · ' + r.message : '已应用 · 连接失败: ' + (r.message || '') + addr, r.ok ? 'ok' : 'err');
    updateConnectionUI(r);
  } catch (e) {
    if (token !== aiTestToken) return;
    if (btn.dataset.testing) btn.dataset.testing = '';
    toast('测试失败: ' + e, 'err');
  } finally {
    if (token === aiTestToken && btn.isConnected) {
      btn.disabled = false; btn.textContent = '应用并测试连接';
      btn.dataset.testing = '';
    }
  }
}

function renderConfigTab(cfg) {
  const body = $('#modal-body');
  body.innerHTML = '';
  const aiMasked = !!(state.llamaEnabled && state.llamaIntegration);
  for (const g of CFG_FIELDS_BASIC) {
    const gd = renderCfgGroup(g, cfg);
    const btnField = document.createElement('div');
    btnField.className = 'field';
    btnField.style.justifyContent = 'flex-end';
    const saveTestBtn = document.createElement('button');
    saveTestBtn.className = 'btn';
    saveTestBtn.id = 'btn-save-and-test';
    saveTestBtn.textContent = '应用并测试连接';
    saveTestBtn.style.width = '100%';
    saveTestBtn.onclick = saveConfigAndTest;
    btnField.appendChild(saveTestBtn);
    gd.querySelector('.grid-4').appendChild(btnField);
    if (aiMasked) {
      const w = gd.querySelector('input[data-key="ai_workers"]');
      if (w) {
        w.value = Math.max(1, Number(state.llamaParallel) || 1);
        w.disabled = true;
      }
      const wrap = document.createElement('div');
      wrap.className = 'ai-mask-wrap';
      wrap.appendChild(gd);
      const mask = document.createElement('div');
      mask.className = 'ai-mask';
      mask.innerHTML =
        '<div class="ai-mask-title">' + icon('warning') + ' 已优先使用本地推理</div>' +
        '<div class="ai-mask-sub">AI 请求改走本地服务，基础连接设置暂不生效，更多AI参数（温度、Top-p、生成长度等）依然生效。</div>';
      wrap.appendChild(mask);
      body.appendChild(wrap);
    } else {
      body.appendChild(gd);
    }
  }
  for (const g of CFG_FIELDS_PROCESSING) {
    body.appendChild(renderCfgGroup(g, cfg));
  }
  const advWrap = document.createElement('div');
  advWrap.className = 'group';
  const advToggle = document.createElement('button');
  advToggle.className = 'disclosure';
  advToggle.innerHTML = '<svg class="ic" style="width:12px;height:12px"><use href="#ic-play"/></svg> 更多AI参数';
  const advContent = document.createElement('div');
  advContent.style.display = state.settingsOpen.has('cfg-adv') ? 'block' : 'none';
  advContent.style.marginTop = '10px';
  if (state.settingsOpen.has('cfg-adv')) advToggle.classList.add('open');
  advToggle.onclick = () => {
    const shown = advContent.style.display !== 'none';
    advContent.style.display = shown ? 'none' : 'block';
    advToggle.classList.toggle('open', !shown);
    if (shown) state.settingsOpen.delete('cfg-adv');
    else state.settingsOpen.add('cfg-adv');
  };
  advWrap.appendChild(advToggle);
  advWrap.appendChild(advContent);
  body.appendChild(advWrap);
  for (const g of CFG_FIELDS_AI_ADVANCED) {
    advContent.appendChild(renderCfgGroup(g, cfg));
  }
  if (state.settings_scroll_to === 'ai') {
    setTimeout(() => {
      const target = body.querySelector('.group');
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        target.classList.add('highlight-pulse');
        target.addEventListener('animationend', () => {
          target.classList.remove('highlight-pulse');
        }, { once: true });
      }
      state.settings_scroll_to = null;
    }, 100);
  }
}

function renderCfgGroup(g, cfg) {
  const gd = document.createElement('div');
  gd.className = 'group';
  gd.innerHTML = g.noTitle ? '' : `<h3>${g.group}</h3>`;
  const grid = document.createElement('div');
  grid.className = g.cols === 4 ? 'grid-4' : (g.cols === 3 ? 'grid-3' : 'grid-2');
  for (const f of g.fields) {
    let val;
    if (f.sec === '__gui') val = cfg[f.key];
    else val = cfg[f.sec] ? cfg[f.sec][f.key] : '';
    const field = document.createElement('div');
    field.className = 'field';
    if (f.full) field.style.gridColumn = '1 / -1';
    else if (f.span) field.style.gridColumn = `span ${f.span}`;
    if (f.type === 'bool') {
      const on = (val === undefined || val === null || val === '') ? !!f.default : val;
      const lbl = f.help ? `<span class="tip-text" data-tip="${esc(f.help)}">${f.label}</span>` : f.label;
      field.innerHTML = `<label style="visibility:hidden" aria-hidden="true">.</label>` +
        `<label class="switch" style="flex:1"><input type="checkbox" data-sec="${f.sec}" data-key="${f.key}" ${on ? 'checked' : ''}/> ${lbl}</label>`;
    } else if (f.type === 'select') {
      const raw = (val === undefined || val === null || val === '') ? f.default : val;
      const cur = (raw === true || raw === '1') ? '1' : (raw === false || raw === '0') ? '0' : String(raw);
      const opts = (f.options || []).map(([v, t]) =>
        `<div class="dd-opt${cur === String(v) ? ' active' : ''}" data-value="${esc(v)}">${esc(t)}</div>`).join('');
      const curLabel = ((f.options || []).find(([v]) => cur === String(v)) || [])[1] || '';
      field.innerHTML = `<label ${f.help ? `data-tip="${esc(f.help)}"` : ''}>${f.label}</label>` +
        `<div class="dd" data-bool="1" data-sec="${f.sec}" data-key="${f.key}" data-value="${esc(cur)}">` +
        `<button class="dd-btn" type="button"><span class="dd-label">${esc(curLabel)}</span>${ddArrow()}</button>` +
        `<div class="dd-panel">${opts}</div></div>`;
      const dd = field.querySelector('.dd');
      initDropdown(dd, v => { dd.dataset.value = v; });
    } else {
      const ph = f.default != null ? ` placeholder="${esc(String(f.default))}"` : '';
      const a = `type="${f.type}" data-sec="${f.sec}" data-key="${f.key}" value="${esc(val == null ? '' : val)}"` +
        (f.min != null ? ` min="${f.min}"` : '') + (f.max != null ? ` max="${f.max}"` : '') + (f.step != null ? ` step="${f.step}"` : '') + ph;
      if (f.type === 'password') {
        field.innerHTML = `<label ${f.help ? `data-tip="${esc(f.help)}"` : ''}>${f.label}</label><div class="input-wrap"><input ${a}/><button type="button" class="eye-btn" data-tip="显示/隐藏">${icon('eye')}</button></div>`;
      } else {
        field.innerHTML = `<label ${f.help ? `data-tip="${esc(f.help)}"` : ''}>${f.label}</label><input ${a}/>`;
      }
    }
    grid.appendChild(field);
  }
  gd.appendChild(grid);
  gd.querySelectorAll('.eye-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const inp = btn.parentElement.querySelector('input');
      const show = inp.type === 'password';
      inp.type = show ? 'text' : 'password';
      btn.innerHTML = icon(show ? 'eye-off' : 'eye');
    });
  });
  return gd;
}

async function renderTabWithLoading(bodyEl, fetchFn, renderFn, retryFn) {
  const seq = _settingsSeq;
  try {
    const data = await fetchFn();
    if (seq !== _settingsSeq) return;
    renderFn(data);
    animateModalBody();
    bodyEl.scrollTop = state.settingsScroll;
  } catch (e) {
    if (seq !== _settingsSeq) return;
    bodyEl.innerHTML = '<div class="empty">' + icon('warning') + ' 加载失败：' + esc(String(e).slice(0, 100)) +
      '<br/><br/><button class="btn">重试</button></div>';
    bodyEl.querySelector('button').onclick = retryFn;
  }
}

async function renderPromptsTabWrapper() {
  const body = $('#modal-body');
  await renderTabWithLoading(body,
    async () => {
      const presets = await apiCall('list_presets');
      const active = await apiCall('get_active_preset');
      state.active_preset_id = active.preset.id;
      return { presets, preset: active.preset };
    },
    (data) => renderPromptsTab(data.presets, data.preset),
    () => renderSettings()
  );
}

async function renderWorkspaceTabWrapper() {
  const body = $('#modal-body');
  await renderTabWithLoading(body,
    () => apiCall('get_workspace_stats'),
    (stats) => renderWorkspaceTab(stats),
    () => renderSettings()
  );
}

async function renderTagsTabWrapper() {
  const body = $('#modal-body');
  await renderTabWithLoading(body,
    () => apiCall('get_priority_tags'),
    (pt) => renderTagsTab(pt),
    () => renderSettings()
  );
}

$('#btn-savecfg').addEventListener('click', async () => {
  try {
    if (state.settings_tab === 'config') {
      const data = collectConfigData();
      const res = await apiCall('save_config', data);
      if (res && res.ok) {
        state.thumbOptimize = Number((data.video || {}).frame_time_tags) === 2;
        toast('配置已应用', 'ok');
        if ('force_animation' in data) applyForceAnimation(data.force_animation !== false);
        if (state.settings_tab === 'config') checkConnection();
        renderSettings();
      } else toast('应用失败: ' + ((res && res.error) || ''), 'err');
    } else if (state.settings_tab === 'prompts') {
      const editingId = $('#preset-editing-id') ? $('#preset-editing-id').value : '';
      if (!editingId.startsWith('custom_')) {
        toast('内置模板不可保存，请「保存为新模板」', 'err');
        return;
      }
      const nameLabel = $('#preset-select .dd-label');
      const res = await apiCall('save_preset', collectPresetData(nameLabel ? nameLabel.textContent : ''));
      if (res && res.ok) {
        toast('模板已保存', 'ok');
        await apiCall('set_active_preset', res.id);
        state.active_preset_id = res.id;
        renderSettings();
      } else toast('保存失败: ' + ((res && res.error) || ''), 'err');
    } else if (state.settings_tab === 'tags') {
      const d = collectTagsData();
      const res2 = await apiCall('save_priority_tags', d.enabled, d.items);
      if (res2 && res2.ok) {
        toast('标签检索已保存', 'ok');
        renderSettings();
      } else toast('保存失败: ' + ((res2 && res2.error) || ''), 'err');
    } else if (state.settings_tab === 'experimental') {
      if (!$('#pixai-frames') || !$('#whisper-vad') || !$('#llama-enable-toggle')) {
        toast('页面尚未加载完成，请稍候再试', 'err');
        return;
      }
      let thr = Number($('#pixai-threshold').value) || 0.9;
      if (thr > 1) thr = 0.99;
      if (thr < 0.5) thr = 0.5;
      const wv = Number($('#whisper-workers').value);
      const expData = { experimental: {
        pixai_classify: $('#pixai-classify').checked,
        pixai_frames: Math.max(1, Number($('#pixai-frames').value) || 15),
        pixai_short_side: Math.max(64, Number($('#pixai-short-side').value) || 448),
        pixai_crop_square: $('#pixai-crop-square').checked,
        pixai_crop_portrait: $('#pixai-crop-portrait').checked,
        pixai_threshold: thr,
        pixai_tagger_enabled: $('#pixai-enable-toggle').checked,
        whisper_vad: $('#whisper-vad').checked,
        whisper_language: $('#whisper-language').value.trim(),
        whisper_max_chars: Math.max(100, Number($('#whisper-max-chars').value) || 800),
        whisper_inject_timestamps: $('#whisper-inject-ts').checked,
        whisper_batch: $('#whisper-batch').checked,
        whisper_workers: Math.min(16, Math.max(0, Number.isFinite(wv) ? wv : 4)),
        whisper_enabled: $('#whisper-enable-toggle').checked,
      }};
      const wmv = getDropdownValue($('#whisper-model-dd'));
      if (wmv) expData.experimental.whisper_model = wmv;
      const llamaToggle = $('#llama-enable-toggle');
      let llamaError = null;
      if (llamaToggle) {
        const ll = {
          models_dir: (($('#llama-models-dir') && $('#llama-models-dir').value) || '').trim(),
          auto_run: !!($('#llama-autorun') && $('#llama-autorun').checked),
          integrate: !!($('#llama-integrate') && $('#llama-integrate').checked),
          show_logs: !!($('#llama-showlogs') && $('#llama-showlogs').checked),
        };
        await apiCall('set_llama_config', ll);
        const resEn = await apiCall('set_llama_enabled', llamaToggle.checked);
        if (resEn && resEn.ok === false) {
          llamaError = resEn.error || '停止本地推理服务出错';
        }
      }
      let ok = true;
      if (Object.keys(expData.experimental).length) {
        const res3 = await apiCall('save_config', expData);
        ok = !!(res3 && res3.ok);
      }
      if (ok) {
        if (llamaError) {
          toast('扩展功能参数已保存（但 llama 停用失败: ' + llamaError + '）', 'err');
        } else {
          toast('扩展功能参数已保存', 'ok');
        }
        state.llamaSynced = false;
        await ensureLlamaStateKnown();
        renderSettings();
        updateSortDropdown();
      } else toast('保存失败', 'err');
    } else if (state.settings_tab === 'llama') {
      const res4 = await apiCall('set_llama_config', _collectLlamaParams());
      if (res4 && res4.ok) {
        toast('本地推理参数已保存', 'ok');
        renderSettings();
      } else toast('保存失败: ' + ((res4 && res4.error) || ''), 'err');
    } else {
      toast('此页面无需保存', 'dim');
    }
  } catch (e) {
    toast('保存失败: ' + ((e && e.message) || e), 'err');
  }
});

function updateSaveButtonState() {
  const btnSave = $('#btn-savecfg');
  if (state.settings_tab === 'workspace') {
    btnSave.disabled = true;
    btnSave.textContent = '无需保存';
  } else {
    btnSave.disabled = false;
    btnSave.textContent = '应用';
  }
}
