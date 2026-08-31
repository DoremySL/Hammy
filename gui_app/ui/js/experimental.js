/* ════════════════════════════════════════════════════════════
   设置 Modal - 扩展功能页
   ════════════════════════════════════════════════════════════ */

async function renderExperimentalTabWrapper() {
  const body = $('#modal-body');
  await renderTabWithLoading(body,
    async () => {
      const st = await apiCall('get_experimental_status');
      syncLlamaState(st.llama);
      state.pixaiTaggerEnabled = !!(st.pixai && st.pixai.enabled);
      state.whisperEnabled = !!(st.whisper && st.whisper.enabled);
      return { cfg: st.cfg, uvStatus: st.uv, llamaStatus: st.llama, pStatus: st.pixai, wStatus: st.whisper };
    },
    (data) => renderExperimentalPage(data.cfg, data.uvStatus, data.llamaStatus, data.pStatus, data.wStatus),
    () => renderSettings()
  );
}

function syncLlamaState(llamaStatus) {
  state.llamaSynced = true;
  state.llamaEnabled = !!(llamaStatus && llamaStatus.enabled);
  state.llamaIntegration = !!(llamaStatus && llamaStatus.enabled && llamaStatus.config && llamaStatus.config.integrate);
  state.llamaRunning = !!(llamaStatus && llamaStatus.running);
  state.llamaStarting = !!(llamaStatus && llamaStatus.starting);
  if (state.llamaIntegration) {
    state.pillMode = 'llama';
    state.aiConnected = state.llamaRunning && !state.llamaStarting;
    updateLlamaPill(llamaStatus);
    updateStartBtn();
  } else if (state.pillMode === 'llama') {
    state.pillMode = 'conn';
    checkConnection();
  }
  if (typeof updateLlamaTabVisibility === 'function') updateLlamaTabVisibility();
}

function renderExperimentalPage(cfg, uvStatus, llamaStatus, pStatus, wStatus) {
  const body = $('#modal-body');
  const exp = (cfg && cfg.experimental) || {};
  body.innerHTML = `
    <div class="exp-page">
      <div class="exp-intro">安装占用大量硬盘空间，运行时消耗较高的硬件资源，仅推荐显存≥6GB的N卡用户尝试</div>
      ${_renderLlamaSection(llamaStatus)}
      ${_renderPixaiSection(exp, pStatus)}
      ${_renderWhisperSection(exp, wStatus)}
      ${_renderUvCard(uvStatus)}
    </div>
  `;

  _bindManageButtons(uvStatus, llamaStatus, pStatus, wStatus);
  _bindLlamaEvents(llamaStatus);
  _bindPixaiEvents(pStatus);
  _bindWhisperEvents(wStatus);
}

function _statusBadge(st) {
  if (st.ready) {
    if (st.cls_model_exists === false) return `<span class="exp-badge warn">${icon('warning')} 已就绪（预筛模型未下载）</span>`;
    return `<span class="exp-badge ok">${icon('check')} 已安装就绪</span>`;
  }
  if (st.venv_exists && !st.model_exists) return `<span class="exp-badge warn">${icon('warning')} 依赖已装，模型未下载</span>`;
  if (st.dir_exists) return `<span class="exp-badge warn">${icon('warning')} 安装不完整</span>`;
  return '<span class="exp-badge dim">未安装</span>';
}

function _installBtns(prefix, st) {
  const install = st.ready ? '' :
    `<button class="ws-btn fixed-lead" id="btn-${prefix}-install">安装</button>`;
  return `${install}<button class="ws-btn danger" id="btn-${prefix}-remove" ${!st.dir_exists ? 'disabled' : ''}>卸载</button>`;
}

function _bindCardToggle(prefix, enabled) {
  const panel = $(`#${prefix}-cfg-panel`);
  const card = panel && panel.closest('.exp-card');
  if (!card) return;
  card.addEventListener('click', (e) => {
    if (!enabled || e.target.closest('button, .switch, .exp-cfg-panel')) return;
    const shown = panel.style.display !== 'none';
    panel.style.display = shown ? 'none' : 'block';
    if (shown) state.settingsOpen.delete(`${prefix}-cfg`);
    else state.settingsOpen.add(`${prefix}-cfg`);
  });
}

function _renderUvCard(uvStatus) {
  const uv = uvStatus || {};
  const uvBadge = uv.installed
    ? `<span class="exp-badge ok">${icon('check')} ${uv.in_runtime ? '程序目录' : '系统 PATH'}已安装</span>`
    : '<span class="exp-badge dim">未安装（首次安装模块时自动下载）</span>';
  return `
    <div class="exp-card exp-card-static">
      <div class="exp-card-head">
        <span class="exp-master-ic" data-tip="UV 包管理工具">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 3.4 19.8 7.7v8.6L12 20.6l-7.8-4.3V7.7Z" stroke-width="1.8"/>
            <path d="m4.5 7.9 7.5 4.1 7.5-4.1" stroke-width="1.4" opacity=".75"/>
            <path d="M12 12v8.4" stroke-width="1.4" opacity=".75"/>
          </svg>
        </span>
        <div class="exp-head-main">
          <div class="exp-title-row"><strong>UV 包管理工具</strong>${uvBadge}</div>
          <div class="exp-desc">转录 / 标签获取模块的依赖安装工具，清理或卸载不影响已装模块。</div>
        </div>
        <div class="exp-head-actions">
          <button class="ws-btn" id="btn-uv-clean" ${uv.installed ? '' : 'disabled'}>清理缓存</button>
          <button class="ws-btn danger" id="btn-uv-uninstall" ${uv.installed ? '' : 'disabled'}>卸载</button>
        </div>
      </div>
    </div>`;
}

function _bindManageButtons(uvStatus, llamaStatus, pStatus, wStatus) {
  const installBtn = $('#btn-llama-install');
  if (installBtn && !llamaStatus.ready) installBtn.addEventListener('click', () => showLlamaCudaDialog());

  const upgradeBtn = $('#btn-llama-upgrade');
  if (upgradeBtn) upgradeBtn.addEventListener('click', () => upgradeLlama());

  const removeBtn = $('#btn-llama-remove');
  if (removeBtn && llamaStatus.dir_exists) removeBtn.addEventListener('click', () => removeLlama());

  const pInstallBtn = $('#btn-pixai-install');
  if (pInstallBtn && !pStatus.ready) pInstallBtn.addEventListener('click', async () => {
    const sel = await showMirrorSelectDialog({ api: 'get_pixai_mirrors', title: '选择镜像站 — PixAI Tagger' });
    if (sel) startPixaiInstall(sel.pytorch || 'nju-cu128', sel.pypi || 'nju');
  });

  const pRemoveBtn = $('#btn-pixai-remove');
  if (pRemoveBtn && pStatus.dir_exists) pRemoveBtn.addEventListener('click', async () => {
    if (!await showConfirm('确定删除 pixai-tagger 吗？\n\n将删除所有已安装的依赖和模型文件。')) return;
    pRemoveBtn.disabled = true; pRemoveBtn.textContent = '删除中…';
    const r = await apiCall('remove_pixai_tagger');
    if (r && r.ok) {
      state.pixaiTaggerEnabled = false; state.pixaiTaggedIds.clear(); state.pixaiRealIds.clear(); state.pixaiAnimeIds.clear(); state.pixaiUncertainIds.clear();
      refreshGridData(); toast(r.message || '已删除', 'ok'); renderSettings();
    } else { toast('删除失败: ' + ((r && r.error) || ''), 'err'); pRemoveBtn.disabled = false; pRemoveBtn.textContent = '卸载'; }
  });

  const wInstallBtn = $('#btn-whisper-install');
  if (wInstallBtn && !wStatus.ready) wInstallBtn.addEventListener('click', async () => {
    const sel = await showMirrorSelectDialog({
      api: 'get_whisper_mirrors', title: '选择镜像站与模型 — Faster-Whisper',
      showModelSelect: true, models: (wStatus && wStatus.models) || [],
    });
    if (sel) startWhisperInstall(sel.pypi || 'nju', sel.model || 'v3-turbo');
  });

  const wRemoveBtn = $('#btn-whisper-remove');
  if (wRemoveBtn && wStatus.dir_exists) wRemoveBtn.addEventListener('click', async () => {
    if (!await showConfirm('确定删除 faster-whisper 吗？\n\n将删除所有已安装的依赖和模型文件。')) return;
    wRemoveBtn.disabled = true; wRemoveBtn.textContent = '删除中…';
    const r = await apiCall('remove_whisper');
    if (r && r.ok) {
      state.whisperEnabled = false; state.whisperTranscribedIds.clear();
      refreshGridData(); toast(r.message || '已删除', 'ok'); renderSettings();
    } else { toast('删除失败: ' + ((r && r.error) || ''), 'err'); wRemoveBtn.disabled = false; wRemoveBtn.textContent = '卸载'; }
  });

  const uvCleanBtn = $('#btn-uv-clean');
  if (uvCleanBtn && uvStatus && uvStatus.installed) uvCleanBtn.addEventListener('click', async () => {
    if (!await showConfirm('确定清理 UV 包缓存吗？\n\n清理后下次安装模块需重新下载依赖。')) return;
    uvCleanBtn.disabled = true; uvCleanBtn.textContent = '清理中…';
    const r = await apiCall('clean_uv_cache');
    if (r && r.ok) { toast(r.message || '已清理', 'ok'); renderSettings(); }
    else { toast('清理失败: ' + ((r && r.error) || ''), 'err'); uvCleanBtn.disabled = false; uvCleanBtn.textContent = '清理缓存'; }
  });

  const uvUninstallBtn = $('#btn-uv-uninstall');
  if (uvUninstallBtn && uvStatus && uvStatus.installed) uvUninstallBtn.addEventListener('click', async () => {
    if (!await showConfirm('确定卸载 UV 包管理工具吗？\n\n将删除 UV-Tool 目录（含包缓存）。已安装模块仍可正常运行，下次安装模块时会自动重新下载。')) return;
    uvUninstallBtn.disabled = true; uvUninstallBtn.textContent = '卸载中…';
    const r = await apiCall('uninstall_uv');
    if (r && r.ok) { toast(r.message || '已卸载', 'ok'); renderSettings(); }
    else { toast('卸载失败: ' + ((r && r.error) || ''), 'err'); uvUninstallBtn.disabled = false; uvUninstallBtn.textContent = '卸载'; }
  });
}

/* ════════════════════════════════════════════════════════════
   镜像选择弹窗
   ════════════════════════════════════════════════════════════ */
async function showMirrorSelectDialog(opts) {
  let mirrors;
  try { mirrors = await apiCall(opts.api); }
  catch (e) { toast('获取镜像列表失败', 'err'); return null; }

  const gpu = mirrors.gpu || null;
  const hasPytorch = Array.isArray(mirrors.pytorch) && mirrors.pytorch.length;
  const hasPypi = Array.isArray(mirrors.pypi) && mirrors.pypi.length;

  let gpuBanner = '';
  if (hasPytorch) {
    gpuBanner = (gpu && gpu.has_nvidia)
      ? `<div class="mirror-gpu-banner has-gpu">
          <div><strong>${esc(gpu.gpu_name)}</strong><span class="mirror-gpu-sub">驱动 ${esc(gpu.driver_version)}${gpu.cuda_max ? ` · 支持 CUDA ${esc(gpu.cuda_max)}` : ''} — 已自动推荐 CUDA 版本</span></div>
        </div>`
      : `<div class="mirror-gpu-banner">
          <div><strong>未检测到 NVIDIA 显卡</strong><span class="mirror-gpu-sub">可下载CPU版测试效果，速度较慢请降低采样帧数</span></div>
        </div>`;
  } else if (gpu) {
    gpuBanner = (gpu.has_nvidia)
      ? `<div class="mirror-gpu-banner has-gpu">
          <div><strong>${esc(gpu.gpu_name)}</strong><span class="mirror-gpu-sub">将安装 CUDA 运行时（~500MB），转录使用 GPU 加速</span></div>
        </div>`
      : `<div class="mirror-gpu-banner">
          <div><strong>未检测到 NVIDIA 显卡</strong><span class="mirror-gpu-sub">将跳过 CUDA 运行时（~500MB），转录使用 CPU 模式</span></div>
        </div>`;
  }

  const recommended = (gpu && gpu.recommended) || mirrors.default_pytorch || 'nju-cu128';

  const torchSection = hasPytorch ? `
    <div class="mirror-section">
      <div class="mirror-section-head">
        <strong>PyTorch CUDA 镜像</strong>
        <span class="mirror-section-hint">torch / torchvision 专用索引</span>
      </div>
      <div class="mirror-opts">${mirrors.pytorch.map(m => {
        const isRec = m.id === recommended;
        return `<label class="mirror-opt${isRec ? ' recommended selected' : ''}">
          <input type="radio" name="torch-mirror" value="${esc(m.id)}" ${isRec ? 'checked' : ''}/>
          <span class="mirror-opt-name">${esc(m.name)}${isRec ? '<span class="mirror-rec-badge">推荐</span>' : ''}</span>
          <small class="mirror-opt-url">${esc(m.url)}</small>
        </label>`;
      }).join('')}</div>
    </div>` : '';

  const pypiSection = hasPypi ? `
    <div class="mirror-section">
      <div class="mirror-section-head">
        <strong>通用 PyPI 镜像</strong>
        <span class="mirror-section-hint">常规依赖包索引</span>
      </div>
      <div class="mirror-opts">${mirrors.pypi.map(m => {
        const isDef = m.id === mirrors.default_pypi;
        return `<label class="mirror-opt${isDef ? ' recommended selected' : ''}">
          <input type="radio" name="pypi-mirror" value="${esc(m.id)}" ${isDef ? 'checked' : ''}/>
          <span class="mirror-opt-name">${esc(m.name)}${isDef ? '<span class="mirror-rec-badge">默认</span>' : ''}</span>
          <small class="mirror-opt-url">${esc(m.url)}</small>
        </label>`;
      }).join('')}</div>
    </div>` : '';

  const modelSection = opts.showModelSelect && Array.isArray(opts.models) && opts.models.length ? `
    <div class="mirror-section">
      <div class="mirror-section-head">
        <strong>模型选择</strong>
        <span class="mirror-section-hint">安装时下载所选模型，其余模型可在安装后下载切换</span>
      </div>
      <div class="mirror-opts">${opts.models.map(m => {
        const isRec = !!m.recommended;
        return `<label class="mirror-opt${isRec ? ' recommended selected' : ''}">
          <input type="radio" name="model-select" value="${esc(m.key)}" ${isRec ? 'checked' : ''}/>
          <span class="mirror-opt-name">${esc(m.title)}${isRec ? '<span class="mirror-rec-badge">推荐</span>' : ''}</span>
          <small class="mirror-opt-url">${esc(m.desc || '')} · ${esc(m.size_label || '')}</small>
        </label>`;
      }).join('')}</div>
    </div>` : '';

  const bg = $('#confirmBg');
  const box = bg.querySelector('.confirm-box');
  const origHtml = box.innerHTML;
  box.classList.add('mirror-dialog');
  box.innerHTML = `
    <div class="confirm-title">${esc(opts.title || '选择镜像站')}</div>
    <div class="mirror-body">
      ${gpuBanner}
      ${torchSection}
      ${pypiSection}
      ${modelSection}
    </div>
    <div class="confirm-foot">
      <button class="btn" id="mirrorCancel">取消</button>
      <button class="btn primary" id="mirrorOk">${esc(opts.confirmText || '开始安装')}</button>
    </div>`;
  bg.classList.add('show');

  box.querySelectorAll('.mirror-opt input').forEach(radio => {
    radio.addEventListener('change', () => {
      box.querySelectorAll(`input[name="${radio.name}"]`).forEach(r =>
        r.closest('.mirror-opt').classList.toggle('selected', r.checked));
    });
  });

  return new Promise(resolve => {
    function close(val) {
      bg.classList.remove('show');
      box.classList.remove('mirror-dialog');
      box.innerHTML = origHtml;
      resolve(val);
    }
    $('#mirrorCancel').addEventListener('click', () => close(null));
    $('#mirrorOk').addEventListener('click', () => {
      const torch = box.querySelector('input[name="torch-mirror"]:checked');
      const pypi = box.querySelector('input[name="pypi-mirror"]:checked');
      const model = box.querySelector('input[name="model-select"]:checked');
      close({ pytorch: torch ? torch.value : null, pypi: pypi ? pypi.value : 'nju',
              model: model ? model.value : null });
    });
  });
}

/* ════════════════════════════════════════════════════════════
   llama.cpp
   ════════════════════════════════════════════════════════════ */
function _renderLlamaSection(llamaStatus) {
  const enabled = !!llamaStatus.enabled;
  const cfg = (llamaStatus.config && typeof llamaStatus.config === 'object') ? llamaStatus.config : {};
  const curDir = llamaStatus.models_dir || '';
  const defDir = llamaStatus.default_models_dir || '';
  const badge = llamaStatus.running
    ? `<span class="exp-badge ok">${icon('check')} 运行中 · 端口 ${esc(String(llamaStatus.port || ''))}</span>`
    : (llamaStatus.ready ? `<span class="exp-badge ok">${icon('check')} 已安装就绪</span>`
                         : '<span class="exp-badge dim">未安装</span>');
  return `
    <div class="exp-card ${!enabled ? 'is-disabled' : ''}">
      <div class="exp-card-head">
        <label class="switch master-switch">
          <input type="checkbox" id="llama-enable-toggle" ${enabled ? 'checked' : ''} ${!llamaStatus.ready ? 'disabled' : ''}/>
        </label>
        <div class="exp-head-main" data-tip="点击展开 / 收起配置">
          <div class="exp-title-row"><strong>llama.cpp 本地推理</strong>${badge}</div>
          <div class="exp-desc">本地运行 OpenAI 兼容推理服务，随主程序退出自动停止。</div>
        </div>
        <div class="exp-head-actions">
          ${llamaStatus.ready ? '<button class="ws-btn" id="btn-llama-upgrade">升级引擎</button>' : ''}
          ${_installBtns('llama', llamaStatus)}
        </div>
      </div>
      <div id="llama-cfg-panel" class="exp-cfg-panel" style="display:${state.settingsOpen.has('llama-cfg') ? 'block' : 'none'}">
        <div class="field">
          <label>模型文件夹位置</label>
          <div class="ws-actions" style="justify-content:flex-start;gap:8px;margin-top:0;flex-wrap:nowrap">
            <input type="text" id="llama-models-dir" style="flex:1;min-width:0"
                   value="${esc(cfg.models_dir || '')}"
                   placeholder="${esc(curDir)}"
                   data-tip="留空则使用与 llama.cpp 同级的 models/；悬停可查看当前扫描目录"/>
            <button class="ws-btn" id="btn-llama-pickdir">选择文件夹</button>
            <button class="ws-btn" id="btn-llama-resetdir">恢复默认</button>
          </div>
        </div>
        <div class="exp-switch-row">
          <label class="switch" data-tip="程序启动时自动启动上次成功运行的模型（无记录时回退到设置页选中的模型）"><input type="checkbox" id="llama-autorun" ${cfg.auto_run ? 'checked' : ''}/> 程序启动时自动运行模型</label>
          <label class="switch" data-tip="视频处理、字幕翻译与连接检测统一走本地服务；AI 配置页仅基础连接设置暂不生效，进阶采样参数依然生效。"><input type="checkbox" id="llama-integrate" ${cfg.integrate ? 'checked' : ''}/> 本地推理集成</label>
          <label class="switch" data-tip="开启后 llama-server 输出显示在日志栏，否则在程序终端。"><input type="checkbox" id="llama-showlogs" ${cfg.show_logs ? 'checked' : ''}/> 程序内显示 llama.cpp 日志</label>
        </div>
      </div>
    </div>`;
}

function _bindLlamaEvents(llamaStatus) {
  const llamaToggle = $('#llama-enable-toggle');
  if (llamaToggle) llamaToggle.addEventListener('change', () => {
    toast(llamaToggle.checked ? '已启用 llama.cpp 模块（点击「应用」保存）' : '已禁用 llama.cpp 模块（点击「应用」保存）', 'dim');
  });

  _bindCardToggle('llama', !!llamaStatus.enabled);

  const pickBtn = $('#btn-llama-pickdir');
  if (pickBtn) pickBtn.addEventListener('click', async () => {
    const folders = await callApi('pick_folders');
    if (!folders || !folders.length) return;
    $('#llama-models-dir').value = folders[0];
    toast('已选择模型文件夹（点击「应用」保存）', 'dim');
  });

  const resetBtn = $('#btn-llama-resetdir');
  if (resetBtn) resetBtn.addEventListener('click', () => {
    $('#llama-models-dir').value = llamaStatus.default_models_dir || '';
    toast('已恢复默认模型文件夹（点击「应用」保存）', 'dim');
  });
}

function _renderPixaiSection(exp, pStatus) {
  const enabled = !!state.pixaiTaggerEnabled;
  const ready = !!(pStatus && pStatus.ready);
  return `
    <div class="exp-card ${!enabled ? 'is-disabled' : ''}">
      <div class="exp-card-head">
        <label class="switch master-switch">
          <input type="checkbox" id="pixai-enable-toggle" ${enabled ? 'checked' : ''} ${!ready ? 'disabled' : ''}/>
        </label>
        <div class="exp-head-main" data-tip="点击展开 / 收起配置">
          <div class="exp-title-row"><strong>PixAI Tagger 标签获取</strong>${_statusBadge(pStatus || {})}</div>
          <div class="exp-desc">识别视频角色与作品/IP，辅助 AI 重命名；可预筛跳过非二次元作品。</div>
        </div>
        <div class="exp-head-actions">
          ${enabled ? '<button class="ws-btn" id="btn-pixai-clear-tags">清除标签</button>' : ''}
          ${_installBtns('pixai', pStatus || {})}
        </div>
      </div>
      <div id="pixai-cfg-panel" class="exp-cfg-panel" style="display:${state.settingsOpen.has('pixai-cfg') ? 'block' : 'none'}">
        <div class="exp-input-row">
          <div class="field">
            <label data-tip="每个视频采样的帧数上限（短视频固定头中尾 3 帧）：越多识别越全面，耗时越长">采样帧数</label>
            <input type="number" id="pixai-frames" min="1" value="${exp.pixai_frames || 15}"/>
          </div>
          <div class="field">
            <label data-tip="图片短边像素：越小越快越省资源，越大细节越清楚">短边分辨率</label>
            <input type="number" id="pixai-short-side" min="64" step="16" value="${exp.pixai_short_side || 448}"/>
          </div>
          <div class="field">
            <label data-tip="角色标签的置信度阈值，越高越严格（0.5–0.99）">置信度阈值</label>
            <input type="number" id="pixai-threshold" min="0.5" max="0.99" step="0.01" value="${exp.pixai_threshold || 0.9}"/>
          </div>
        </div>
        <div class="exp-switch-row">
          <label class="switch"><input type="checkbox" id="pixai-classify" ${exp.pixai_classify === true ? 'checked' : ''}/><span class="tip-text" data-tip="开启后非二次元作品跳过标签获取（角标 REAL 无 IP）；关闭则全部视频都获取标签">跳过非二次元作品</span></label>
          <label class="switch" data-tip="横屏视频中心裁剪为正方形后再识别，避免两侧内容挤压变形"><input type="checkbox" id="pixai-crop-square" ${exp.pixai_crop_square === true ? 'checked' : ''}/> 横屏裁剪正方形</label>
          <label class="switch" data-tip="竖屏视频从顶部偏下裁剪为正方形后再识别，避免画面上下拉伸变形"><input type="checkbox" id="pixai-crop-portrait" ${exp.pixai_crop_portrait === true ? 'checked' : ''}/> 竖屏裁剪正方形</label>
        </div>
      </div>
    </div>`;
}

function _bindPixaiEvents(pStatus) {
  const toggle = $('#pixai-enable-toggle');
  if (toggle) toggle.addEventListener('change', () => {
    state.pixaiTaggerEnabled = toggle.checked;
    toast(`PixAI Tagger 开关已${toggle.checked ? '启用' : '禁用'}（点击「应用」保存）`, 'dim');
  });

  _bindCardToggle('pixai', state.pixaiTaggerEnabled);

  const clearBtn = $('#btn-pixai-clear-tags');
  if (clearBtn) clearBtn.addEventListener('click', async () => {
    if (!await showConfirm('确定清除所有已保存的标签数据吗？')) return;
    const r = await apiCall('clear_pixai_tags');
    if (r && r.ok) {
      state.pixaiTaggedIds.clear(); state.pixaiRealIds.clear(); state.pixaiAnimeIds.clear(); state.pixaiUncertainIds.clear();
      refreshGridData(); toast(r.message || '已清除', 'ok');
    } else toast('清除失败', 'err');
  });
}

async function startPixaiInstall(pytorchMirror, pypiMirror) {
  if (state.installing || state.hfDownloading) { toast(state.hfDownloading ? '模型下载进行中，请等待完成' : '已有模块正在安装，请等待完成', 'err'); return; }
  state.installing = true;
  closeSettings();
  gotoLog();
  $('#log').innerHTML = '';
  const logEmpty = $('#logEmpty'); if (logEmpty) logEmpty.remove();
  $('#prog-bar').style.width = '0%';
  $('#prog-bar').className = 'active';
  $('#prog-num').textContent = '正在安装 PixAI Tagger…';
  $('#btn-stop').classList.add('active');
  $('#btn-stop').classList.remove('done');
  toast('开始安装，请查看日志面板…');
  try {
    const r = await apiCall('install_pixai_tagger', pytorchMirror, pypiMirror);
    if (r && r.ok) {
      $('#prog-bar').style.width = '100%'; $('#prog-bar').className = 'done';
      $('#prog-num').textContent = '安装完成';
      await apiCall('set_pixai_tagger_enabled', true);
      state.pixaiTaggerEnabled = true;
      updateSortDropdown();
      toast('PixAI Tagger 安装完成，已自动启用', 'ok');
    } else if (r && r.cancelled) {
      $('#prog-bar').className = ''; $('#prog-num').textContent = '安装已取消';
      toast('安装已取消', 'dim');
    } else {
      $('#prog-bar').className = ''; $('#prog-num').textContent = '安装失败';
      toast('安装失败: ' + ((r && r.error) || '').slice(0, 200), 'err');
    }
  } catch (e) {
    $('#prog-bar').className = ''; $('#prog-num').textContent = '安装失败';
    toast('安装失败: ' + ((e && e.message) || e), 'err');
  } finally {
    state.installing = false;
    $('#btn-stop').classList.remove('active');
    renderSettings();
  }
}

function _renderWhisperSection(exp, wStatus) {
  const enabled = !!state.whisperEnabled;
  const ready = !!(wStatus && wStatus.ready);
  const wModels = (wStatus && wStatus.models) || [];
  const installed = (wStatus && wStatus.installed) || {};
  const current = (wStatus && wStatus.current_model) || 'v3-turbo';
  const hasInstalled = wModels.some(m => installed[m.key]);
  const modelSel = hasInstalled ? `
    <div class="dd whisper-model-dd" id="whisper-model-dd">
      <button class="dd-btn" type="button" data-tip="切换转录模型，点底部「应用」保存">
        <span class="dd-label">${esc((wModels.find(m => m.key === current) || {}).title || '请选择模型')}</span>
        ${ddArrow()}
      </button>
      <div class="dd-panel">
        ${wModels.map(m => {
          const isInst = !!installed[m.key];
          return `<div class="dd-opt${m.key === current ? ' active' : ''}${isInst ? '' : ' disabled'}" data-value="${esc(m.key)}"${isInst ? '' : ` data-tip="${esc(m.repo)}（未下载，点击打开模型下载）"`}>${esc(m.title)}${isInst ? '' : '（未下载）'}</div>`;
        }).join('')}
      </div>
    </div>` : `
    <div class="dd whisper-model-dd is-empty" id="whisper-model-dd">
      <button class="dd-btn" type="button" data-tip="还没有下载任何模型，点击下方选项打开模型下载">
        <span class="dd-label">暂无已下载模型</span>
        ${ddArrow()}
      </button>
      <div class="dd-panel">
        <div class="dd-opt disabled">还没有下载任何模型：<br>点击这里从 hf-mirror.com 下载。</div>
      </div>
    </div>`;
  return `
    <div class="exp-card ${!enabled ? 'is-disabled' : ''}">
      <div class="exp-card-head">
        <label class="switch master-switch">
          <input type="checkbox" id="whisper-enable-toggle" ${enabled ? 'checked' : ''} ${!ready ? 'disabled' : ''}/>
        </label>
        <div class="exp-head-main" data-tip="点击展开 / 收起配置">
          <div class="exp-title-row"><strong>Faster-Whisper 语音转录</strong>${_statusBadge(wStatus || {})}</div>
          <div class="exp-desc">转录视频语音为字幕，辅助 AI 重命名，支持导出与翻译。</div>
        </div>
        <div class="exp-head-actions">
          ${enabled ? '<button class="ws-btn" id="btn-whisper-clear">清除转录</button>' : ''}
          ${_installBtns('whisper', wStatus || {})}
        </div>
      </div>
      <div id="whisper-cfg-panel" class="exp-cfg-panel" style="display:${state.settingsOpen.has('whisper-cfg') ? 'block' : 'none'}">
        <div class="whisper-model-row">
          <div class="field">
            <label data-tip="转录使用的模型；切换后点击底部「应用」生效，点击未下载的模型可直接下载">当前模型</label>
            ${modelSel}
          </div>
          <div class="whisper-switches">
            <label class="switch" data-tip="跳过静音片段再转录，速度更快、更省资源"><input type="checkbox" id="whisper-vad" ${exp.whisper_vad !== false ? 'checked' : ''}/> VAD 过滤静音</label>
            <label class="switch" data-tip="转录文本注入重命名提示词时发送 [MM:SS] 时间标记，AI 可结合台词时间轴理解内容"><input type="checkbox" id="whisper-inject-ts" ${exp.whisper_inject_timestamps ? 'checked' : ''}/> 发送时间戳</label>
            <label class="switch" data-tip="GPU 上显著提速，CPU 反而更慢；会影响字幕时间戳准确性"><input type="checkbox" id="whisper-batch" ${exp.whisper_batch ? 'checked' : ''}/> 批处理模式</label>
          </div>
        </div>
        <div class="grid-3" style="margin-top:10px">
          <div class="field">
            <label data-tip="转录目标语言，留空自动检测；示例：ja（日语）、zh（中文）、en（英文）。ja-1.5B 模型仅支持日语，选择后自动固定为 ja">预期语言</label>
            <input type="text" id="whisper-language" placeholder="自动检测" value="${esc(exp.whisper_language || '')}"/>
          </div>
          <div class="field">
            <label data-tip="同时处理视频的数量；0 = 自动（GPU 4 路 / CPU 1 路）。开启批处理模式时建议设置为 1">并发数</label>
            <input type="number" id="whisper-workers" min="0" max="16" step="1" value="${exp.whisper_workers ?? 4}"/>
          </div>
          <div class="field">
            <label data-tip="注入重命名 AI 提示词的最大字符数">提示词截断</label>
            <input type="number" id="whisper-max-chars" min="100" step="100" value="${exp.whisper_max_chars || 800}"/>
          </div>
        </div>
      </div>
    </div>`;
}

function _bindWhisperEvents(wStatus) {
  const toggle = $('#whisper-enable-toggle');
  if (toggle) toggle.addEventListener('change', () => {
    state.whisperEnabled = toggle.checked;
    toast(`Faster-Whisper 开关已${toggle.checked ? '启用' : '禁用'}（点击「应用」保存）`, 'dim');
  });

  _bindCardToggle('whisper', state.whisperEnabled);

  const modelDD = $('#whisper-model-dd');
  if (modelDD) initDropdown(modelDD, () => toast('模型已选择（点击「应用」保存）', 'dim'));

  if (modelDD) modelDD.querySelectorAll('.dd-opt.disabled').forEach(opt => {
    opt.addEventListener('click', (e) => {
      e.stopPropagation();
      modelDD.classList.remove('open');
      openWhisperModelDialog();
    });
  });

  const clearBtn = $('#btn-whisper-clear');
  if (clearBtn) clearBtn.addEventListener('click', async () => {
    if (!await showConfirm('确定清除所有已保存的语音转录数据吗？')) return;
    const r = await apiCall('clear_whisper_transcripts');
    if (r && r.ok) {
      state.whisperTranscribedIds.clear();
      refreshGridData(); toast(r.message || '已清除', 'ok');
    } else toast('清除失败', 'err');
  });
}

async function startWhisperInstall(pypiMirror, modelKey) {
  if (state.installing || state.hfDownloading) { toast(state.hfDownloading ? '模型下载进行中，请等待完成' : '已有模块正在安装，请等待完成', 'err'); return; }
  state.installing = true;
  closeSettings();
  gotoLog();
  $('#log').innerHTML = '';
  const logEmpty = $('#logEmpty'); if (logEmpty) logEmpty.remove();
  $('#prog-bar').style.width = '0%';
  $('#prog-bar').className = 'active';
  $('#prog-num').textContent = '正在安装 Faster-Whisper…';
  $('#btn-stop').classList.add('active');
  $('#btn-stop').classList.remove('done');
  toast('开始安装，请查看日志面板…');
  try {
    const r = await apiCall('install_whisper', pypiMirror || 'nju', modelKey || 'v3-turbo');
    if (r && r.ok) {
      $('#prog-bar').style.width = '100%'; $('#prog-bar').className = 'done';
      $('#prog-num').textContent = '安装完成';
      await apiCall('set_whisper_enabled', true);
      state.whisperEnabled = true;
      toast('Faster-Whisper 安装完成，已自动启用', 'ok');
    } else if (r && r.cancelled) {
      $('#prog-bar').className = ''; $('#prog-num').textContent = '安装已取消';
      toast('安装已取消', 'dim');
    } else {
      $('#prog-bar').className = ''; $('#prog-num').textContent = '安装失败';
      toast('安装失败: ' + ((r && r.error) || '').slice(0, 200), 'err');
    }
  } catch (e) {
    $('#prog-bar').className = ''; $('#prog-num').textContent = '安装失败';
    toast('安装失败: ' + ((e && e.message) || e), 'err');
  } finally {
    state.installing = false;
    $('#btn-stop').classList.remove('active');
    renderSettings();
  }
}

/* ════════════════════════════════════════════════════════════
   模型下载弹窗（Whisper 模型管理 / llama.cpp HF 搜索下载 共用骨架）
   ════════════════════════════════════════════════════════════ */
function _hfFmtSize(n) {
  if (!n) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = n, i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return v.toFixed(v >= 100 || i === 0 ? 0 : 1) + ' ' + units[i];
}

function _hfFmtCount(n) {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
  return String(n);
}

function _hfFmtDate(iso) {
  if (!iso) return '-';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '-';
  const p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

function _hfBox() {
  const bg = $('#confirmBg');
  return bg ? { bg, box: bg.querySelector('.confirm-box') } : null;
}

function _makeModelDownloader(cfg) {
  const dl = { active: false, done: false, last: null, title: '', origHtml: '' };
  const shown = () => {
    const c = _hfBox();
    return c && c.bg.classList.contains('show') && c.box.classList.contains(cfg.dialogCls) ? c : null;
  };
  function renderProgress(box) {
    const s = dl.last || {};
    const file = s.file ? String(s.file).split('/').pop() : '';
    const pct = Math.round((s.pct || 0) * 100);
    box.innerHTML = `
      <div class="confirm-title">正在下载 ${esc(dl.title)}</div>
      <div class="hf-prog-file" id="dlProgFile">${file || '准备中…'}</div>
      <div class="hf-prog-bar"><div class="hf-prog-fill" id="dlProgFill" style="width:${pct}%"></div></div>
      <div class="hf-prog-num" id="dlProgNum">${s.total
        ? `${_hfFmtSize(s.done)} / ${_hfFmtSize(s.total)}（${pct}%）`
        : (file ? _hfFmtSize(s.done || 0) : '正在连接下载服务器…')}</div>
      <div class="confirm-foot">
        <button class="btn" id="dlBackground">后台下载</button>
        <button class="btn danger" id="dlCancel">取消下载</button>
      </div>`;
    $('#dlBackground').addEventListener('click', () => close(true));
    $('#dlCancel').addEventListener('click', () => {
      apiCall('hf_cancel_download').catch(() => {});
      toast('正在取消下载…', 'info');
    });
  }
  function onProgress(ev) {
    if (ev && dl.active) {
      dl.last = ev;
      if (ev.type === 'file_start') {
        state.hfDlPct = ev.count > 1 ? Math.round((ev.idx - 1) / ev.count * 100) : 0;
      } else if (ev.type === 'progress') {
        state.hfDlPct = ev.count > 1
          ? Math.round(((ev.idx - 1 + (ev.pct || 0)) / ev.count) * 100)
          : Math.round((ev.pct || 0) * 100);
      }
      updateMiniProg();
    }
    const c = shown();
    if (!c) return;
    const t = ev && ev.type;
    const file = ev && ev.file ? String(ev.file).split('/').pop() : '';
    if (t === 'file_start') {
      const pf = $('#dlProgFile'), pn = $('#dlProgNum'), fill = $('#dlProgFill');
      if (pf) pf.textContent = `[${ev.idx}/${ev.count}] ${file}`;
      if (pn) pn.textContent = `正在下载第 ${ev.idx} / ${ev.count} 个文件`;
      if (fill) fill.style.width = '0%';
    } else if (t === 'progress') {
      const pf = $('#dlProgFile'), pn = $('#dlProgNum'), fill = $('#dlProgFill');
      if (pf) pf.textContent = file;
      if (pn) pn.textContent = ev.total
        ? `${_hfFmtSize(ev.done)} / ${_hfFmtSize(ev.total)}（${Math.round(ev.pct * 100)}%）`
        : `${_hfFmtSize(ev.done)}`;
      if (fill) fill.style.width = Math.min(100, Math.round((ev.pct || 0) * 100)) + '%';
    }
  }
  function finish() {
    state.hfDownloading = false;
    state.hfDlPct = 0;
    state.dlKind = null;
    updateMiniProg();
    dl.active = false;
    dl.done = true;
    if (cfg.finishExtra) cfg.finishExtra();
  }
  function showResult(box, r) {
    let title, msg;
    if (!r || r.error) {
      title = r && r.busy ? '无法开始下载' : '下载失败';
      msg = `<div class="hf-msg">${esc((r && r.error) || '未知错误')}</div>`;
    } else if (r.cancelled) {
      title = '下载已取消';
      msg = `<div class="hf-msg">${cfg.cancelledMsg(r)}</div>`;
    } else if (r.failed && r.failed.length) {
      title = '下载完成（部分失败）';
      msg = `<div class="hf-msg">
        <div class="ok">成功 ${r.downloaded} 个文件</div>
        <div class="err">失败 ${r.failed.length} 个文件</div>
        ${r.failed.map(f => `<div class="err">· ${esc(f.file)} — ${esc(f.error)}</div>`).join('')}
      </div>`;
    } else {
      title = '下载完成';
      msg = `<div class="hf-msg">${cfg.successMsg(r)}</div>`;
    }
    box.innerHTML = `
      <div class="confirm-title">${title}</div>
      ${msg}
      <div class="confirm-foot"><button class="btn primary" id="dlResultClose">关闭</button></div>`;
    $('#dlResultClose').addEventListener('click', () => close());
  }
  function onDone(ev) {
    if (!dl.active) return;
    finish();
    const err = !ev || ev.error;
    const cancelled = !!(ev && ev.cancelled);
    const failed = (ev && ev.failed) || [];
    if (err) toast('模型下载失败: ' + ((ev && ev.error) || '未知错误'), 'err');
    else if (cancelled) toast('模型下载已取消，已下载部分保留', 'info');
    else if (failed.length) toast(`模型下载完成: 成功 ${ev.downloaded} 个，失败 ${failed.length} 个`, 'err');
    else toast(`模型下载完成，已下载 ${ev.downloaded} 个模型文件`, 'ok');
    const c = shown();
    if (c) showResult(c.box, ev);
    if (!err && !cancelled) cfg.afterDone();
  }
  function start(box, title, apiPromise) {
    if (state.hfDownloading) { toast('已有模型下载进行中，请等待完成', 'err'); return; }
    if (state.installing) { toast('已有模块正在安装，请等待完成', 'err'); return; }
    dl.active = true;
    dl.done = false;
    dl.last = null;
    dl.title = title;
    state.hfDownloading = true;
    state.hfDlPct = 0;
    state.dlKind = cfg.kind;
    updateMiniProg();
    if (cfg.startExtra) cfg.startExtra();
    renderProgress(box);
    window[cfg.progressKey] = onProgress;
    window[cfg.doneKey] = onDone;
    apiPromise
      .then(r => {
        if (r && r.error && dl.active && !dl.done) {
          finish();
          const c = shown();
          if (c) showResult(c.box, r);
        }
      })
      .catch(e => {
        if (!dl.active || dl.done) return;
        finish();
        const c = shown();
        if (c) showResult(c.box, { error: (e && e.message) || String(e) });
      });
  }
  function close(keepBackground) {
    const c = _hfBox();
    if (!c) return;
    c.box.classList.remove(cfg.dialogCls, 'hf-files-view');
    c.box.innerHTML = dl.origHtml;
    if (keepBackground && dl.active) {
      toast('已转入后台下载，顶栏圆环显示进度', 'dim');
    } else {
      delete window[cfg.progressKey];
      delete window[cfg.doneKey];
      if (state.hfDownloading && state.dlKind === cfg.kind) {
        state.hfDownloading = false; state.hfDlPct = 0; state.dlKind = null; updateMiniProg();
      }
      dl.active = false;
      if (cfg.onClose) cfg.onClose();
    }
    c.bg.classList.remove('show');
  }
  return { dl, renderProgress, start, close };
}

let _wData = null;
const _wDlApi = _makeModelDownloader({
  kind: 'whisper', dialogCls: 'wm-dialog',
  progressKey: '__onWhisperModelProgress', doneKey: '__onWhisperModelDone',
  cancelledMsg: () => '未完成的文件已清理，再次下载将从零开始。',
  successMsg: r => `<div class="ok">已下载 ${r.downloaded} 个文件到模型目录</div>
      <div class="hf-files-hint" style="margin-top:4px">关闭弹窗后，在卡片「当前模型」下拉中选择该模型并点「应用」即可切换。</div>`,
  afterDone: _wRefresh,
  onClose: () => {
    _wData = null;
    if ($('#modal').classList.contains('show')) renderSettings();
  },
});

async function openWhisperModelDialog() {
  const ctx = _hfBox();
  if (!ctx) return;
  const { bg, box } = ctx;
  if (bg.classList.contains('show')) { toast('请先关闭当前弹窗', 'dim'); return; }
  if (state.hfDownloading && state.dlKind !== 'whisper') { toast('已有模型下载进行中，请等待完成', 'err'); return; }
  if (_wDlApi.dl.active) {
    box.classList.add('wm-dialog');
    _wDlApi.renderProgress(box);
    bg.classList.add('show');
    return;
  }
  let st;
  try { st = await apiCall('get_whisper_status'); } catch (e) { st = null; }
  if (!st) { toast('获取模型状态失败', 'err'); return; }
  _wData = { models: st.models || [], installed: st.installed || {}, current: st.current_model || '' };
  _wDlApi.dl.origHtml = box.innerHTML;
  box.classList.add('wm-dialog');
  _wRenderList(box);
  bg.classList.add('show');
}

function _wRenderList(box) {
  box.innerHTML = `
    <div class="confirm-title">Faster-Whisper 模型管理</div>
    <div class="wm-list">${_wData.models.map(m => _wRowHtml(m)).join('')}</div>
    <div class="confirm-foot"><button class="btn" id="wmClose">关闭</button></div>`;
  $('#wmClose').addEventListener('click', () => _wDlApi.close());
  box.querySelectorAll('.wm-dl-btn').forEach(btn =>
    btn.addEventListener('click', () => _wStartDownload(box, btn.dataset.key)));
}

function _wRowHtml(m) {
  const isInst = !!_wData.installed[m.key];
  const isCur = m.key === _wData.current;
  const badge = isInst
    ? `<span class="exp-badge ok">${icon('check')} 已安装${isCur ? ' · 当前使用' : ''}</span>`
    : '';
  const action = isInst ? '' : `
    <button class="ws-btn primary wm-dl-btn" data-key="${esc(m.key)}">${icon('download')} 下载</button>`;
  return `<div class="wm-row">
    <div class="wm-main">
      <div class="wm-name">${esc(m.title)}${m.recommended ? '<span class="mirror-rec-badge">推荐</span>' : ''}</div>
      <div class="wm-desc" title="${esc(m.repo || '')}">${esc(m.desc || '')} · ${esc(m.size_label || '')}${isInst ? '' : ' · ' + esc(m.repo || '')}</div>
    </div>
    <div class="wm-right">${badge}${action}</div>
  </div>`;
}

function _wStartDownload(box, key) {
  const meta = (_wData.models || []).find(m => m.key === key) || { title: '模型' };
  _wDlApi.start(box, meta.title, apiCall('download_whisper_model', key));
}

function _wRefresh() {
  apiCall('get_whisper_status').then(st => {
    if (!st || !_wData) return;
    _wData.installed = st.installed || {};
    _wData.current = st.current_model || '';
    const ctx = _hfBox();
    if (ctx && ctx.bg.classList.contains('show') && ctx.box.classList.contains('wm-dialog')) {
      _wRenderList(ctx.box);
    }
  }).catch(() => {});
}

let _hfData = null;
const _hfDlApi = _makeModelDownloader({
  kind: 'llama', dialogCls: 'hf-dialog',
  progressKey: '__onHfDownloadProgress', doneKey: '__onHfDownloadDone',
  cancelledMsg: r => `成功 ${r.downloaded} 个文件。未完成的文件已清理，再次下载将从零开始。`,
  successMsg: r => `<div class="ok">已下载 ${r.downloaded} 个文件到模型文件夹</div>` +
    (r.dir ? `<div class="hf-files-hint" style="margin-top:4px">${esc(r.dir)}</div>` : ''),
  startExtra: () => {
    const dlBtn = $('#btn-llama-download');
    if (dlBtn) { dlBtn.innerHTML = `${icon('download')} 下载中…`; dlBtn.classList.add('primary'); }
  },
  finishExtra: () => {
    const dlBtn = $('#btn-llama-download');
    if (dlBtn) { dlBtn.innerHTML = `${icon('download')} 下载模型`; dlBtn.classList.remove('primary'); }
  },
  afterDone: () => rescanLlamaModels(),
  onClose: () => { _hfData = null; },
});

async function openHfDownloadDialog() {
  const ctx = _hfBox();
  if (!ctx) return;
  const { bg, box } = ctx;
  if (bg.classList.contains('show')) { toast('请先关闭当前弹窗', 'dim'); return; }
  if (state.hfDownloading && state.dlKind !== 'llama') { toast('已有模型下载进行中，请等待完成', 'err'); return; }
  box.classList.remove('hf-files-view');
  if (_hfDlApi.dl.active) {
    box.classList.add('hf-dialog');
    _hfDlApi.renderProgress(box);
    bg.classList.add('show');
    return;
  }
  _hfData = { repoId: '', gguf: [], mmproj: [], search: null };
  _hfDlApi.dl.origHtml = box.innerHTML;
  box.classList.add('hf-dialog');
  _hfRenderSearch(box);
  bg.classList.add('show');
}

function _hfRenderSearch(box) {
  box.classList.add('hf-files-view');
  box.innerHTML = `
    <button class="hf-dlg-close" id="hfDlgClose" type="button" data-tip="关闭"><svg class="ic"><use href="#ic-close"></use></svg></button>
    <div class="confirm-title">搜索模型 <span style="font-weight:400;font-size:11.5px;color:var(--faint)">hf-mirror.com</span></div>
    <div class="hf-search-row">
      <div class="search-widget hf-search-widget" id="hfSearchWidget">
        <div class="dd search-mode" id="hfSortSel" data-tip="排序方式">
          <button class="dd-btn" type="button"><span class="dd-label">热门</span>${ddArrow()}</button>
          <div class="dd-panel">
            <div class="dd-opt active" data-value="trendingScore">热门</div>
            <div class="dd-opt" data-value="lastModified">更新时间</div>
            <div class="dd-opt" data-value="createdAt">最新创建</div>
            <div class="dd-opt" data-value="downloads">下载量</div>
            <div class="dd-opt" data-value="likes">收藏数</div>
          </div>
        </div>
        <span class="search-sep"></span>
        <input type="text" id="hfSearchInput" placeholder="请输入关键词搜索，如 Qwen3.5 9B GGUF"/>
      </div>
      <button class="btn primary" id="hfSearchBtn">搜索</button>
    </div>
    <div class="hf-results" id="hfResults">
    </div>`;

  $('#hfSearchInput').focus();
  const sortDD = $('#hfSortSel');

  const PAGE_SIZE = 20;
  const page = { kw: '', sort: '', cursor: '', exhausted: false, loading: false, seq: 0 };

  const renderModels = list => list.map(m => {
    const rid = m.id || m.modelId || '?';
    const dl = typeof m.downloads === 'number' ? _hfFmtCount(m.downloads) : '-';
    const gated = m.gated ? '<span class="hf-badge gated">需授权</span>' : '';
    return `<div class="hf-result" data-repo="${esc(rid)}">
      <div class="hf-result-main">${esc(rid)}${gated}<button class="hf-open-btn" type="button" data-tip="查看模型介绍"><svg class="ic"><use href="#ic-external"></use></svg></button></div>
      <div class="hf-result-sub">更新 ${_hfFmtDate(m.lastModified)} · 下载 ${dl} · 收藏 ${typeof m.likes === 'number' ? _hfFmtCount(m.likes) : 0}</div>
    </div>`;
  }).join('');

  const bindResults = () => {
    $('#hfResults').querySelectorAll('.hf-result').forEach(el => {
      el.addEventListener('click', () => _hfOpenFiles(box, el.dataset.repo));
    });
    $('#hfResults').querySelectorAll('.hf-open-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        e.stopPropagation();
        apiCall('hf_open_repo', btn.closest('.hf-result').dataset.repo).catch(() => {});
      });
    });
  };

  const cacheSearch = () => {
    _hfData.search = { kw: page.kw, sort: page.sort, html: $('#hfResults').innerHTML,
                       cursor: page.cursor, exhausted: page.exhausted };
  };

  const maybeFill = () => {
    const el = $('#hfResults');
    if (!page.loading && !page.exhausted && el.scrollHeight <= el.clientHeight + 10) loadMore();
  };

  const doSearch = async () => {
    const kw = $('#hfSearchInput').value.trim();
    const sort = getDropdownValue($('#hfSortSel')) || 'trendingScore';
    const res = $('#hfResults');
    if (!kw) { res.innerHTML = '<div class="hf-hint">请输入搜索关键词</div>'; return; }
    page.kw = kw; page.sort = sort; page.cursor = ''; page.exhausted = false;
    const seq = ++page.seq;
    page.loading = true;
    res.innerHTML = '<div class="hf-hint">搜索中…</div>';
    let r;
    try { r = await apiCall('hf_search_models', kw, sort, PAGE_SIZE, ''); }
    catch (e) {
      if (seq === page.seq) { page.loading = false; res.innerHTML = '<div class="hf-hint">搜索失败，请检查网络后重试</div>'; }
      return;
    }
    if (seq !== page.seq) return;
    page.loading = false;
    if (!r || !r.ok) {
      res.innerHTML = `<div class="hf-hint">${esc((r && r.error) || '搜索失败')}</div>`;
      return;
    }
    const list = r.results || [];
    if (!list.length) { res.innerHTML = '<div class="hf-hint">无匹配结果，换个关键词试试</div>'; page.exhausted = true; return; }
    page.cursor = r.next_cursor || '';
    page.exhausted = !r.has_more;
    res.innerHTML = renderModels(list);
    bindResults();
    cacheSearch();
    maybeFill();
  };

  const loadMore = async () => {
    if (page.loading || page.exhausted || !page.kw) return;
    page.loading = true;
    const seq = ++page.seq;
    const res = $('#hfResults');
    const more = document.createElement('div');
    more.className = 'hf-hint';
    more.textContent = '加载中…';
    res.appendChild(more);
    let r = null;
    try { r = await apiCall('hf_search_models', page.kw, page.sort, PAGE_SIZE, page.cursor); }
    catch (e) { }
    more.remove();
    if (seq !== page.seq) return;
    page.loading = false;
    if (!r || !r.ok) return;
    const list = r.results || [];
    page.cursor = r.next_cursor || '';
    page.exhausted = !r.has_more;
    if (list.length) res.insertAdjacentHTML('beforeend', renderModels(list));
    if (page.exhausted) res.insertAdjacentHTML('beforeend', '<div class="hf-hint">没有更多了</div>');
    if (list.length) bindResults();
    cacheSearch();
    maybeFill();
  };

  if (sortDD) initDropdown(sortDD, () => doSearch());

  $('#hfSearchBtn').addEventListener('click', doSearch);
  $('#hfSearchInput').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
  $('#hfResults').addEventListener('scroll', () => {
    const el = $('#hfResults');
    if (el.scrollTop + el.clientHeight >= el.scrollHeight - 60) loadMore();
  });

  if (_hfData.search) {
    $('#hfSearchInput').value = _hfData.search.kw;
    setDropdownValue(sortDD, _hfData.search.sort);
    page.kw = _hfData.search.kw; page.sort = _hfData.search.sort;
    page.cursor = _hfData.search.cursor || '';
    page.exhausted = !!_hfData.search.exhausted;
    $('#hfResults').innerHTML = _hfData.search.html;
    bindResults();
    maybeFill();
  }
}

async function _hfOpenFiles(box, repoId) {
  box.classList.add('hf-files-view');
  box.innerHTML = `
    <button class="hf-dlg-close" id="hfDlgClose" type="button" data-tip="关闭"><svg class="ic"><use href="#ic-close"></use></svg></button>
    <div class="confirm-title">${esc(repoId)}</div>
    <div class="hf-hint">加载文件列表…</div>`;
  let r;
  try { r = await apiCall('hf_repo_files', repoId); }
  catch (e) { r = null; }
  if (!r || !r.ok) {
    box.innerHTML = `
      <div class="confirm-title">${esc(repoId)}</div>
      <div class="hf-msg">${esc((r && r.error) || '获取文件列表失败')}</div>
      <div class="confirm-foot">
        <button class="btn" id="hfFilesFailBack">返回搜索</button>
        <button class="btn primary" id="hfFilesFailClose">关闭</button>
      </div>`;
    $('#hfFilesFailBack').addEventListener('click', () => _hfRenderSearch(box));
    $('#hfFilesFailClose').addEventListener('click', () => _hfDlApi.close());
    return;
  }
  _hfData.repoId = repoId;
  _hfData.gguf = r.gguf || [];
  _hfData.mmproj = r.mmproj || [];
  _hfRenderFiles(box, r);
}

function _hfRenderFiles(box, r) {
  const col = (title, items) => {
    if (!items.length) {
      return `<div class="hf-col"><div class="hf-col-head">${title}<span class="hf-col-count">0 个</span></div>
        <div class="hf-col-list"><div class="hf-empty-col">无文件</div></div></div>`;
    }
    return `<div class="hf-col"><div class="hf-col-head">${title}<span class="hf-col-count">${items.length} 个</span></div>
      <div class="hf-col-list">${items.map(f => `
        <label class="hf-file-row" data-tip="${esc(f.name)}">
          <input type="checkbox" data-name="${esc(f.name)}"/>
          <span class="hf-file-name">${esc(f.name)}</span>
          <span class="hf-file-size">${_hfFmtSize(f.size)}</span>
        </label>`).join('')}</div></div>`;
  };
  box.innerHTML = `
    <button class="hf-dlg-close" id="hfDlgClose" type="button" data-tip="关闭"><svg class="ic"><use href="#ic-close"></use></svg></button>
    <div class="confirm-title">${esc(_hfData.repoId)}</div>
    <div class="hf-files-hint">保存到：${esc(r.models_dir || '')}</div>
    <div class="hf-files-cols">
      ${col('模型 (.gguf)', _hfData.gguf)}
      ${col('多模态投影 (mmproj)', _hfData.mmproj)}
    </div>
    <div class="hf-dl-summary" id="hfDlSummary"></div>
    <div class="confirm-foot">
      <button class="btn" id="hfFilesBack">返回搜索</button>
      <button class="btn primary" id="hfStartDl">下载选中</button>
    </div>`;

  $('#hfFilesBack').addEventListener('click', () => _hfRenderSearch(box));
  const updateSummary = () => {
    const sel = [...box.querySelectorAll('.hf-file-row input:checked')];
    const size = sel.reduce((s, i) => s + _hfFindSize(i.dataset.name), 0);
    $('#hfDlSummary').textContent = `已选 ${sel.length} 个文件，共 ${_hfFmtSize(size)}`;
  };
  box.querySelectorAll('.hf-file-row input').forEach(i => i.addEventListener('change', updateSummary));
  updateSummary();
  $('#hfStartDl').addEventListener('click', () => {
    const sel = [...box.querySelectorAll('.hf-file-row input:checked')].map(i => i.dataset.name);
    if (!sel.length) { toast('请至少勾选一个文件', 'err'); return; }
    _hfStartDownload(box, sel);
  });
}

function _hfFindSize(name) {
  const f = [..._hfData.gguf, ..._hfData.mmproj].find(x => x.name === name);
  return f ? (f.size || 0) : 0;
}

function _hfStartDownload(box, sel) {
  _hfDlApi.start(box, _hfData.repoId, apiCall('hf_download_models', _hfData.repoId, sel));
}

const LLAMA_DEFAULT_PROXY = 'https://gh-proxy.com/';

async function showLlamaCudaDialog() {
  const bg = $('#confirmBg');
  const box = bg.querySelector('.confirm-box');
  const origHtml = box.innerHTML;
  box.classList.add('mirror-dialog');
  bg.classList.add('show');

  let cleaned = false;
  function cleanup() {
    if (cleaned) return;
    cleaned = true;
    bg.classList.remove('show');
    box.classList.remove('mirror-dialog');
    box.innerHTML = origHtml;
  }

  let method = 'direct';
  let releases = null;
  let fetchErr = '';
  let fetching = false;

  function checkedMethod() {
    const r = box.querySelector('input[name="llama-method"]:checked');
    return r ? r.value : 'direct';
  }
  function proxyValue() {
    const inp = box.querySelector('#llama-proxy-input');
    return (inp && inp.value.trim()) || LLAMA_DEFAULT_PROXY;
  }

  function renderShell() {
    box.innerHTML = `
      <div class="confirm-title">选择构建版本 — llama.cpp</div>
      <div class="mirror-body">
        <div id="llama-build-wrap"></div>
        <div class="mirror-section">
          <div class="mirror-opts">
            <label class="mirror-opt selected">
              <input type="radio" name="llama-method" value="direct" checked/>
              <span class="mirror-opt-name">GitHub 直连<small class="mirror-opt-url">官方 Release，国内访问不稳定</small></span>
            </label>
            <label class="mirror-opt">
              <input type="radio" name="llama-method" value="proxy"/>
              <span class="mirror-opt-name">使用加速代理<small class="mirror-opt-url">使用代理加速站，直连下载受限时可尝试</small></span>
              <input type="text" id="llama-proxy-input" class="mirror-url-input" spellcheck="false"
                     style="display:none" value="${esc(LLAMA_DEFAULT_PROXY)}"
                     placeholder="${esc(LLAMA_DEFAULT_PROXY)}"
                     data-tip="当前代理失效时请自行查找最新可用GitHub加速站地址填入"/>
            </label>
            <label class="mirror-opt">
              <input type="radio" name="llama-method" value="manual"/>
              <span class="mirror-opt-name">手动安装<small class="mirror-opt-url">自行下载解压或编译后放入对应文件夹</small></span>
            </label>
          </div>
        </div>
      </div>
      <div class="confirm-foot">
        <button class="btn" id="llamaCancel">取消</button>
        <button class="btn primary" id="llamaOk">开始安装</button>
      </div>`;

    box.querySelectorAll('input[name="llama-method"]').forEach(radio => {
      radio.addEventListener('change', () => {
        box.querySelectorAll('input[name="llama-method"]').forEach(r =>
          r.closest('.mirror-opt').classList.toggle('selected', r.checked));
        onMethodChange();
      });
    });

    $('#llamaCancel').onclick = cleanup;
    $('#llamaOk').onclick = onOk;
  }

  function onMethodChange() {
    method = checkedMethod();
    const proxyInput = $('#llama-proxy-input');
    const buildWrap = $('#llama-build-wrap');
    if (proxyInput) proxyInput.style.display = method === 'proxy' ? '' : 'none';
    if (method === 'manual') {
      buildWrap.style.display = 'none';
      $('#llamaOk').textContent = '手动安装';
    } else {
      buildWrap.style.display = '';
      $('#llamaOk').textContent = '开始安装';
    }
  }

  async function fetchBuilds() {
    fetching = true;
    fetchErr = '';
    renderBuildLoading();
    let r;
    try { r = await apiCall('get_llama_releases', true); }
    catch (e) { r = null; }
    fetching = false;
    if (cleaned) return;
    if (!r || !r.ok) {
      fetchErr = (r && r.error) || '网络请求异常';
      renderBuildError();
      return;
    }
    if (!(r.builds || []).length) {
      fetchErr = '未找到可用的构建';
      renderBuildError();
      return;
    }
    releases = r;
    renderBuildList();
  }

  function renderBuildLoading() {
    $('#llama-build-wrap').innerHTML = `
      <div class="mirror-loading">
        <div class="spinner"></div>
        <div class="load-hint">正在从官方 GitHub 获取构建列表…</div>
      </div>`;
  }

  function renderBuildError() {
    $('#llama-build-wrap').innerHTML = `
      <div class="mirror-error">
        <div class="err-msg">获取构建列表失败: ${esc(fetchErr)}</div>
        <div class="load-hint">失败时请稍后再试，或改选「手动安装」</div>
        <button class="btn" id="llama-retry">重试</button>
      </div>`;
    $('#llama-retry').onclick = () => fetchBuilds();
  }

  function renderBuildList() {
    const gpu = releases.gpu || {};
    function buildLabel(b) {
      if (!b) return '';
      if (b.type === 'cuda') return 'CUDA ' + b.ver;
      if (b.type === 'vulkan') return 'Vulkan';
      if (b.type === 'cpu') return 'CPU';
      return b.type + (b.ver ? ' ' + b.ver : '');
    }
    const rec = releases.builds.find(b => b.key === releases.recommended_key);
    const noNvidia = !gpu.has_nvidia;
    const autoKey = noNvidia ? ((releases.builds.find(b => b.type === 'cpu') || {}).key || null) : null;
    const gpuBanner = (gpu.has_nvidia)
      ? `<div class="mirror-gpu-banner has-gpu"><div><strong>${esc(gpu.gpu_name)}</strong><span class="mirror-gpu-sub">驱动 ${esc(gpu.driver_version)}${gpu.cuda_max ? ` · 支持 CUDA ${esc(gpu.cuda_max)}` : ''}${releases.recommended_key ? ` — 推荐 ${esc(buildLabel(rec))}` : ''}</span></div></div>`
      : `<div class="mirror-gpu-banner"><div><strong>未检测到 NVIDIA 显卡</strong><span class="mirror-gpu-sub">请尝试选择CPU版本或 Vulkan 版「多数核显独显通用」</span></div></div>`;
    const opts = releases.builds.map(b => {
      const isRec = b.key === releases.recommended_key;
      const checked = (isRec && gpu.has_nvidia) || b.key === autoKey;
      return `<label class="mirror-opt${isRec ? ' recommended' : ''}">
        <input type="radio" name="llama-build" value="${esc(b.key)}" ${checked ? 'checked' : ''}/>
        <span class="mirror-opt-name">${esc(buildLabel(b))}${isRec ? '<span class="mirror-rec-badge">推荐</span>' : ''}</span>
        <small class="mirror-opt-url">llama.cpp ${esc(releases.tag || '')}</small>
      </label>`;
    }).join('');
    $('#llama-build-wrap').innerHTML = `
      ${gpuBanner}
      <div class="mirror-section">
        <div class="mirror-section-head"><strong>可用构建</strong></div>
        <div class="mirror-opts">${opts}</div>
      </div>`;
    box.querySelectorAll('#llama-build-wrap .mirror-opt input').forEach(radio => {
      radio.addEventListener('change', () => {
        box.querySelectorAll('input[name="llama-build"]').forEach(r =>
          r.closest('.mirror-opt').classList.toggle('selected', r.checked));
      });
    });
    box.querySelectorAll('#llama-build-wrap input[name="llama-build"]:checked').forEach(r =>
      r.closest('.mirror-opt').classList.add('selected'));
  }

  function onOk() {
    method = checkedMethod();
    if (method === 'manual') {
      cleanup();
      startLlamaInstall('manual', '');
      return;
    }
    const sel = box.querySelector('input[name="llama-build"]:checked');
    if (!sel) { toast('构建列表尚未就绪，请稍候或点击「重试」加载', 'dim'); return; }
    const proxy = method === 'proxy' ? proxyValue() : '';
    cleanup();
    startLlamaInstall(sel.value, proxy);
  }

  renderShell();
  fetchBuilds();
}

async function startLlamaInstall(cudaVer, proxy) {
  if (state.installing || state.hfDownloading) { toast(state.hfDownloading ? '模型下载进行中，请等待完成' : '已有模块正在安装，请等待完成', 'err'); return; }
  state.installing = true;
  closeSettings();
  gotoLog();
  $('#log').innerHTML = '';
  const logEmpty = $('#logEmpty'); if (logEmpty) logEmpty.remove();
  $('#prog-bar').style.width = '0%';
  $('#prog-bar').className = 'active';
  $('#prog-num').textContent = '正在安装 llama.cpp (' + cudaVer + ')…';
  $('#btn-stop').classList.add('active');
  $('#btn-stop').classList.remove('done');
  toast('开始安装，请查看日志面板…');
  try {
    const r = await apiCall('install_llama', cudaVer, proxy || '');
    if (r && r.ok) {
      $('#prog-bar').style.width = '100%'; $('#prog-bar').className = 'done';
      $('#prog-num').textContent = '安装完成';
      if (r.manual) {
        toast('（手动模式）: 已创建 llama.cpp 文件夹，请自行将对应的编译版本放入其中', 'ok');
      } else {
        toast('llama.cpp 安装完成', 'ok');
      }
    } else if (r && r.cancelled) {
      $('#prog-bar').className = ''; $('#prog-num').textContent = '安装已取消';
      toast('安装已取消', 'dim');
    } else {
      $('#prog-bar').className = ''; $('#prog-num').textContent = '安装失败';
      toast('安装失败: ' + ((r && r.error) || '').slice(0, 200), 'err');
    }
  } catch (e) {
    $('#prog-bar').className = ''; $('#prog-num').textContent = '安装失败';
    toast('安装失败: ' + ((e && e.message) || e), 'err');
  } finally {
    state.installing = false;
    $('#btn-stop').classList.remove('active');
    renderSettings();
  }
}

async function removeLlama() {
  if (!await showConfirm('确定卸载 llama.cpp 吗？\n\n将删除程序目录下的 llama.cpp（含二进制与 cudart）。\n模型文件夹 models/ 不会被删除，可保留复用。')) return;
  const r = await apiCall('remove_llama');
  if (r && r.ok) {
    toast(r.message || '已卸载', 'ok');
    state.llamaEnabled = false;
    updateLlamaTabVisibility();
  } else {
    toast('卸载失败: ' + ((r && r.error) || ''), 'err');
  }
  renderSettings();
}

/* ════════════════════════════════════════════════════════════
   llama.cpp
   ════════════════════════════════════════════════════════════ */
async function launchLlama() {
  const params = _collectLlamaParams();
  const model = params.model;
  const saved = await apiCall('set_llama_config', params);
  if (!saved || saved.ok === false) {
    toast('参数保存失败（' + ((saved && saved.error) || '未知原因') + '），本次仍按界面参数启动', 'err');
  }
  const startBtn = $('#btn-llama-toggle');
  if (startBtn) { startBtn.disabled = true; startBtn.textContent = '启动中…'; }
  toast('正在启动本地推理服务…');
  state.llamaPendingLaunch = true;
  const heroBadge = document.querySelector('.llama-hero-title .exp-badge');
  if (heroBadge) { heroBadge.className = 'exp-badge warn'; heroBadge.innerHTML = icon('refresh') + ' 正在启动'; }
  const heroMeta = document.querySelector('.llama-hero-meta');
  if (heroMeta) heroMeta.textContent = '服务正在启动，模型加载中…';
  if (state.llamaIntegration) {
    updateLlamaPill({ running: false, starting: true, model: model });
    startLlamaPillPolling();
  }
  try {
    const r = await apiCall('launch_llama', model, params);
    if (r && r.ok) {
      toast('llama-server 已启动', 'ok');
      if (!state.llamaIntegration && !state.aiConnected) {
        checkConnection().then(ok => { if (ok) toast('模型加载完成，AI 服务已自动重连', 'ok'); });
      }
    } else if (r && r.cancelled) {
      toast('已停止启动', 'ok');
    } else {
      toast('启动失败: ' + ((r && r.error) || '').slice(0, 200), 'err');
    }
  } catch (e) {
    toast('启动失败: ' + ((e && e.message) || e), 'err');
  } finally {
    state.llamaPendingLaunch = false;
    stopLlamaPillPolling();
    try {
      const st = await apiCall('get_llama_status');
      if (st) syncLlamaState(st);
      if (st && st.ok !== false) {
        if (state.settings_tab === 'llama') renderLlamaTab(st);
        else renderSettings();
      }
    } catch (e) {
      toast('刷新 llama 状态失败: ' + ((e && e.message) || e), 'err');
    }
  }
}

async function ensureLlamaReady() {
  if (state.llamaRunning && !state.llamaStarting) return true;
  state.llamaPendingLaunch = true;
  updateLlamaPill({ running: false, starting: true, model: null });
  startLlamaPillPolling();
  updateStartBtn();
  try {
    const r = await apiCall('ensure_llama_running');
    let st = r ? await apiCall('get_llama_status') : null;
    if (st) syncLlamaState(st);
    if (r && r.ok && st && st.running && !st.starting) return true;
    if (r && r.ok && st && st.starting) {
      st = await waitLlamaSettled();
      if (st) syncLlamaState(st);
      if (st && st.running && !st.starting) return true;
    }
    toast('本地推理服务启动失败: ' + ((r && r.error) || (st && st.launch_failed) || '未知错误'), 'err');
    return false;
  } catch (e) {
    toast('本地推理服务启动失败: ' + ((e && e.message) || e), 'err');
    return false;
  } finally {
    state.llamaPendingLaunch = false;
    stopLlamaPillPolling();
    updateStartBtn();
  }
}

async function waitLlamaSettled(timeoutMs) {
  const limit = timeoutMs || 95000;
  const t0 = Date.now();
  while (Date.now() - t0 < limit) {
    const st = await apiCall('get_llama_status');
    if (st && !st.starting) return st;
    await new Promise(r => setTimeout(r, 1000));
  }
  return null;
}

async function stopLlama() {
  const r = await apiCall('stop_llama');
  toast(r && r.ok ? (r.message || '已停止') : '停止失败', r && r.ok ? 'ok' : 'err');
  try {
    const st = await apiCall('get_llama_status');
    if (st) syncLlamaState(st);
  } catch (e) {  }
  renderSettings();
}

async function upgradeLlama() {
  try {
    const st = await apiCall('get_llama_status');
    if (st) syncLlamaState(st);
  } catch (e) {  }
  if (state.llamaRunning || state.llamaStarting) {
    toast('正在停止本地推理服务…', 'info');
    try {
      await apiCall('stop_llama');
      const st = await apiCall('get_llama_status');
      if (st) syncLlamaState(st);
    } catch (e) {
      toast('停止失败: ' + ((e && e.message) || e) + '，已取消升级', 'err');
      return;
    }
    if (state.llamaRunning || state.llamaStarting) {
      toast('服务未能停止，已取消升级', 'err');
      return;
    }
  }
  showLlamaCudaDialog();
}

/* ════════════════════════════════════════════════════════════
   本地推理页
   ════════════════════════════════════════════════════════════ */
async function renderLlamaTabWrapper() {
  const body = $('#modal-body');
  await renderTabWithLoading(body,
    async () => await apiCall('get_llama_status'),
    (llamaStatus) => renderLlamaTab(llamaStatus),
    () => renderSettings()
  );
}

function renderLlamaTab(llamaStatus) {
  const body = $('#modal-body');
  const edits = body.querySelector('.llama-wrap') ? _collectLlamaParams() : null;
  const cfg = llamaStatus.config || {};
  const d = llamaStatus.defaults || {};
  const models = llamaStatus.models || [];
  const running = !!llamaStatus.running;
  const ready = !!llamaStatus.ready;

  const curModelData = _selectModelData(cfg, models);
  const modelSel = models.length ? `
    <div class="dd llama-model-dd" id="llama-model-dd">
      <button class="dd-btn" data-tip="切换模型">
        <span class="dd-label">${esc(curModelData ? curModelData.name + ' (' + curModelData.size_mb + ' MB)' : '')}</span>
        ${ddArrow()}
      </button>
      <div class="dd-panel">
        ${models.map(m => `<div class="dd-opt${m.path === curModelData.path ? ' active' : ''}" data-value="${esc(m.path)}">${esc(m.name)} (${esc(m.size_mb)} MB)</div>`).join('')}
      </div>
    </div>` : `
    <div class="dd llama-model-dd is-empty" id="llama-model-dd">
      <button class="dd-btn" type="button" data-tip="暂无 .gguf 模型：点「下载模型」在线搜索下载，或将模型文件放入模型文件夹后点「重新扫描」">
        <span class="dd-label">暂无 .gguf 模型</span>
        ${ddArrow()}
      </button>
      <div class="dd-panel">
        <div class="dd-opt disabled">模型文件夹中还没有 .gguf 模型：<br>点右侧「下载模型」从 hf-mirror.com 搜索下载，<br>或将模型文件放入模型文件夹后点「重新扫描」。</div>
      </div>
    </div>`;

  const mcfg = (llamaStatus.model_configs || {})[curModelData ? curModelData.path : ''] || {};
  const pv = (k, fb) => _cfgGet(mcfg, cfg, k, fb);

  const curMmprojs = (curModelData && curModelData.mmprojs) || [];
  const mmprojVal = pv('mmproj', '');
  const mmprojAuto = pv('mmproj_auto', true) !== false;
  const mmprojCpu = pv('no_mmproj_offload', d.no_mmproj_offload) === true;
  const mmprojSelVal = mmprojVal || (mmprojAuto ? (mmprojCpu ? '__auto_cpu__' : '__auto__') : '');

  state.llamaXargs = _xargsNormalize(pv('extra_args', d.extra_args || []));
  state.xargSelected = new Set();
  state.xargSelAnchor = null;

  const curModel = running && llamaStatus.model
    ? String(llamaStatus.model).split(/[\\/]/).pop() : '';

  const ctxMax = (curModelData && curModelData.ctx > 0) ? curModelData.ctx : 65536;
  const nglMax = (curModelData && curModelData.layers > 0) ? curModelData.layers : 999;
  const ctxInit = _clampInt(pv('ctx', d.ctx), 512, ctxMax, d.ctx);
  const nglInit = _clampInt(pv('ngl', d.ngl), 0, nglMax, d.ngl);
  const moeShow = !!(curModelData && curModelData.moe);
  const moeMax = moeShow ? Math.max(1, parseInt(curModelData.layers, 10) || 128) : 128;
  const moeInit = moeShow ? _clampInt(pv('n_cpu_moe', 0), 0, moeMax, 0) : 0;
  const starting = !!llamaStatus.starting || !!state.llamaPendingLaunch;
  const statusMeta = running
    ? `当前模型：<strong>${esc(curModel)}</strong> · PID ${esc(String(llamaStatus.pid))} · 端口 <code>${esc(String(llamaStatus.port || ''))}</code>`
    : starting ? '服务正在启动，模型加载中…'
    : (ready ? '服务未运行' : 'llama.cpp 尚未安装');
  const stateBadge = running
    ? `<span class="exp-badge ok">${icon('check')} 运行中</span>`
    : starting
      ? `<span class="exp-badge warn">${icon('refresh')} 正在启动</span>`
      : (ready
          ? '<span class="exp-badge dim">已停止</span>'
          : `<span class="exp-badge warn">${icon('warning')} 未安装</span>`);

  body.innerHTML = `
  <div class="llama-wrap">
    <!-- 运行状态 Hero 卡 -->
    <div class="llama-hero ${running ? 'is-running' : ''}">
      <div class="llama-hero-left">
        <div class="llama-hero-main">
          <div class="llama-hero-title">本地推理服务 ${stateBadge}</div>
          <div class="llama-hero-meta" data-tip="${running ? esc(curModel) : ''}">${statusMeta}</div>
        </div>
      </div>
      <div class="llama-hero-actions">
        <button class="ws-btn" id="btn-llama-webui" data-tip="用默认浏览器打开 llama.cpp 自带聊天界面">聊天界面</button>
        <button class="ws-btn primary" id="btn-llama-toggle" ${!ready ? 'disabled' : ''} data-tip="${starting ? '模型加载中也可点击停止' : ''}">${running ? '停止服务' : (starting ? '停止启动' : '启动服务')}</button>
      </div>
    </div>

    ${!ready ? '<div class="exp-intro" style="margin-bottom:14px">llama.cpp 尚未安装，请先到「扩展功能」页安装该模块。</div>' : ''}

    <!-- 模型 + 性能参数 -->
    <div class="group">
      <div class="field">
        <label>模型文件 (.gguf)</label>
        <div class="ws-actions" style="justify-content:flex-start;gap:8px;margin-top:0">
          ${modelSel}
          <button class="ws-btn" id="btn-llama-rescan" data-tip="重新扫描模型文件夹">${icon('refresh')}</button>
          <button class="ws-btn${state.hfDownloading ? ' primary' : ''}" id="btn-llama-download" data-tip="${state.hfDownloading ? '正在后台下载模型，点击查看进度' : '从 hf-mirror.com 搜索并下载模型到模型文件夹'}">${icon('download')} ${state.hfDownloading ? '下载中…' : '下载模型'}</button>
        </div>
      </div>
      <div class="llama-sliders" id="llama-sliders-wrap" style="margin-top:14px">
        <div class="slider-row">
          <div class="slider-head">
            <label data-tip="上下文窗口长度：越大可对话内容越多，显存/内存占用越高；上限为该模型的训练上下文">上下文窗口 <code>-c</code></label>
            <input type="number" class="sval" id="llama-ctx-val" value="${ctxInit}" min="512" max="${ctxMax}" step="256"/>
          </div>
          <input type="range" id="llama-ctx-range" min="512" max="${ctxMax}" step="256" value="${ctxInit}"/>
          <input type="hidden" id="llama-ctx" value="${ctxInit}"/>
        </div>
        <div class="slider-row">
          <div class="slider-head">
            <label data-tip="GPU 加载层数：显存不足时调小（如 20~40）">GPU 加载层数 <code>-ngl</code></label>
            <input type="number" class="sval" id="llama-ngl-val" value="${nglInit}" min="0" max="${nglMax}" step="1"/>
          </div>
          <input type="range" id="llama-ngl-range" min="0" max="${nglMax}" step="1" value="${nglInit}"/>
          <input type="hidden" id="llama-ngl" value="${nglInit}"/>
        </div>
        <div class="slider-row" id="llama-ot-row" style="display:${moeShow ? '' : 'none'}">
          <div class="slider-head">
            <label data-tip="MoE 模型专用：把前 N 个专家层权重留在 CPU 以节省显存（0 = 全部上 GPU）">MoE 专家 CPU 卸载 <code>--n-cpu-moe</code></label>
            <input type="number" class="sval" id="llama-moe-val" value="${moeInit}" min="0" max="${moeMax}" step="1"/>
          </div>
          <input type="range" id="llama-moe-range" min="0" max="${moeMax}" step="1" value="${moeInit}"/>
          <input type="hidden" id="llama-moe" value="${moeShow ? moeInit : ''}"/>
        </div>
      </div>
      <div class="grid-3" style="margin-top:10px">
        <div class="field">
          <label data-tip="并发请求数 (parallel)：越大越吃显存">并发线程 <code>-np</code></label>
          <input type="number" id="llama-parallel" value="${pv('parallel', d.parallel)}"/>
        </div>
        <div class="field">
          <label data-tip="KV 缓存量化类型：q8_0 可在几乎无损的情况下显著降低显存占用">KV 量化 <code>--cache-type</code></label>
          <div class="dd llama-kv-dd" id="llama-kv">${_kvDdInner(_kvSelectVal(pv('kv_quant', d.kv_quant || '')))}</div>
        </div>
        <div class="field">
          <label data-tip="统一KV缓存：降低显存占用，大幅提高并发时显存利用率，但会降低预填充速度
禁用提示缓存与检查点：只建议进行连续批处理任务时开启，降低内存占用提高预填充速度">缓存预设</label>
          <div class="dd llama-kv-preset-dd" id="llama-kv-preset">${_kvPresetDdInner(_kvPresetSelectVal(pv('kv_preset', d.kv_preset || '')))}</div>
        </div>
      </div>
      <div class="llama-row" style="margin-top:14px">
        <div class="field">
          <label data-tip="思考深度仅对部分模型生效，批量处理建议关闭思考提高速度">思考控制 <code>--reasoning</code></label>
          <div class="dd llama-reasoning-dd" id="llama-reasoning-dd">${_reasoningDdInner(_reasoningSelectVal(pv('reasoning_mode', d.reasoning_mode || 'off')))}</div>
        </div>
        <div class="field">
          <label data-tip="实现多模态推理的核心配套文件">多模态投影器 <code>mmproj</code></label>
          <div class="dd llama-mmproj-dd" id="llama-mmproj">${_mmprojDdInner(mmprojSelVal, curMmprojs, mmprojVal)}</div>
        </div>
        <div class="field">
          <label data-tip="--spec-type draft-mtp：提高推理生成速度，与并发使用效果差，批量处理时建议关闭；如需其他 MTP 请填写下方额外启动参数">多令牌预测 <code>MTP</code></label>
          <div class="dd llama-spec-dd" id="llama-spec">${_specDdInner(_specSelectVal(pv('spec_draft_n', d.spec_draft_n || '')))}</div>
        </div>
      </div>
    </div>

    <!-- 服务参数 -->
    <div class="group">
      <div class="grid-3">
        <div class="field">
          <label data-tip="监听地址：127.0.0.1 仅本机可访问；0.0.0.0 允许局域网访问">监听地址 <code>--host</code></label>
          <input type="text" id="llama-host" value="${esc(pv('host', d.host))}"/>
        </div>
        <div class="field">
          <label data-tip="服务端口">端口 <code>--port</code></label>
          <input type="number" id="llama-port" value="${pv('port', d.port)}"/>
        </div>
        <div class="field">
          <label data-tip="模型别名：显示在聊天界面标题，便于区分多模型；留空不传 -a">模型别名 <code>-a</code></label>
          <input type="text" id="llama-alias" value="${esc(pv('alias', d.alias || ''))}"/>
        </div>
      </div>
    </div>

    <!-- 进阶参数（默认收起） -->
    <button class="llama-adv-bar" id="llama-adv-toggle">
      <span>进阶参数</span>
      ${ddArrow('chev')}
    </button>
    <div id="llama-adv-panel" style="display:${state.settingsOpen.has('llama-adv') ? 'block' : 'none'};margin-top:12px">
      <div class="grid-3">
        <div class="field">
          <label data-tip="CPU 推理线程数，一般设为物理核心数">推理线程数 <code>-t</code></label>
          <input type="number" id="llama-threads" value="${pv('threads', d.threads)}"/>
        </div>
        <div class="field">
          <label data-tip="批处理线程数：prompt 处理阶段使用的线程数">批处理线程数 <code>-tb</code></label>
          <input type="number" id="llama-tb" value="${pv('threads_batch', d.threads_batch)}"/>
        </div>
        <div class="field">
          <label data-tip="逻辑批大小：影响生成速度与显存占用">逻辑批大小 <code>-b</code></label>
          <input type="number" id="llama-batch" value="${pv('batch', d.batch)}"/>
        </div>
        <div class="field">
          <label data-tip="物理批大小：内部计算微批大小">物理批大小 <code>-ub</code></label>
          <input type="number" id="llama-ubatch" value="${pv('ubatch', d.ubatch)}"/>
        </div>
        <div class="field">
          <label data-tip="最大生成长度 (-1 = 无限)">最大生成长度 <code>-n</code></label>
          <input type="number" id="llama-npredict" value="${pv('npredict', d.npredict)}"/>
        </div>
        <div class="field">
          <label data-tip="请求超时秒数">请求超时秒 <code>--timeout</code></label>
          <input type="number" id="llama-timeout" value="${pv('timeout', d.timeout)}"/>
        </div>
        <div class="field">
          <label data-tip="用于切换模型文件的底层加载策略，从而直接控制推理任务的启动速度、内存占用峰值以及运行时是否会被交换到磁盘">加载模式 <code>--load-mode</code></label>
          <div class="dd llama-loadmode-dd" id="llama-loadmode">${_loadModeDdInner(_loadModeSelectVal(pv('load_mode', d.load_mode)))}</div>
        </div>
        <div class="field">
          <label data-tip="决定单张图片最低使用token数，更高数值可提高识图精度">视觉预算下限 <code>--image-min-tokens</code></label>
          <div class="dd llama-img-tokens-dd" id="llama-image-min">${_imgTokensDdInner(_imgTokensSelectVal(pv('image_min_tokens', d.image_min_tokens)))}</div>
        </div>
        <div class="field">
          <label data-tip="决定单张图片最高使用token数，建议设置低于物理批-ub的数值避免部分模型崩溃">视觉预算上限 <code>--image-max-tokens</code></label>
          <div class="dd llama-img-tokens-dd" id="llama-image-max">${_imgTokensDdInner(_imgTokensSelectVal(pv('image_max_tokens', d.image_max_tokens)))}</div>
        </div>
      </div>
      <div class="field" style="margin-top:10px">
        <label data-tip="追加更多启动参数，值含空格需添加引号">额外启动参数</label>
        <div class="xarg-quick">
          <input type="text" id="llama-extra" placeholder="Enter 添加，添加后可点击选中，Ctrl/Shift 多选，右键编辑或删除"/>
          <button class="ws-btn" id="btn-xarg-add">添加</button>
        </div>
        <div class="xarg-list" id="llama-xargs-list" style="display:none"></div>
      </div>
    </div>
  </div>
  `;
  _bindLlamaTabEvents(llamaStatus);
  _restoreLlamaEdits(edits, llamaStatus);
}

function _restoreLlamaEdits(e, llamaStatus) {
  if (!e) return;
  const cur = ((llamaStatus && llamaStatus.models) || []).find(m => m.path === e.model) || null;
  const modelDD = $('#llama-model-dd');
  if (e.model && modelDD) {
    setDropdownValue(modelDD, e.model);
    _applyModelLimits(cur);
    _renderMmprojDd(e.mmproj_auto ? (e.no_mmproj_offload ? '__auto_cpu__' : '__auto__') : (e.mmproj || ''),
                    (cur && cur.mmprojs) || [], e.mmproj);
  }
  const set = (id, v) => { const el = $(id); if (el && v != null) el.value = v; };
  set('#llama-host', e.host); set('#llama-port', e.port); set('#llama-alias', e.alias);
  set('#llama-threads', e.threads); set('#llama-tb', e.threads_batch);
  set('#llama-batch', e.batch); set('#llama-ubatch', e.ubatch);
  set('#llama-parallel', e.parallel); set('#llama-npredict', e.npredict); set('#llama-timeout', e.timeout);
  _setSliderVal('#llama-ctx-range', '#llama-ctx', '#llama-ctx-val', parseInt(e.ctx, 10));
  _setSliderVal('#llama-ngl-range', '#llama-ngl', '#llama-ngl-val', parseInt(e.ngl, 10));
  if (cur && cur.moe) _setSliderVal('#llama-moe-range', '#llama-moe', '#llama-moe-val', parseInt(e.n_cpu_moe, 10) || 0);
  const dd = (id, v) => { const el = $(id); if (el && v != null) setDropdownValue(el, v); };
  dd('#llama-kv', e.kv_quant); dd('#llama-kv-preset', e.kv_preset);
  dd('#llama-reasoning-dd', e.reasoning_mode); dd('#llama-spec', e.spec_draft_n);
  dd('#llama-loadmode', e.load_mode);
  dd('#llama-image-min', e.image_min_tokens); dd('#llama-image-max', e.image_max_tokens);
  state.llamaXargs = e.extra_args || [];
  renderLlamaXargsList();
  _syncImgTokenLimits();
}

function _clampInt(v, min, max, fallback) {
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
}

function _setValDisplay(el, v) {
  if (!el) return;
  if (el.tagName === 'INPUT') el.value = v; else el.textContent = v;
}

function _wireSlider(rangeId, hiddenId, valId) {
  const range = $(rangeId), hidden = $(hiddenId), val = $(valId);
  if (!range || !hidden || !val) return;
  const apply = () => { hidden.value = range.value; _setValDisplay(val, range.value); };
  range.addEventListener('input', apply);
  val.addEventListener('change', () => {
    const min = parseInt(range.min, 10) || 0;
    const max = parseInt(range.max, 10) || 0;
    const step = parseInt(range.step, 10) || 1;
    let v = parseInt(val.value, 10);
    if (!Number.isFinite(v)) v = parseInt(hidden.value, 10) || min;
    v = Math.min(max, Math.max(min, v));
    v = Math.round((v - min) / step) * step + min;
    range.value = v;
    hidden.value = v;
    _setValDisplay(val, v);
  });
  apply();
}

function _setSliderVal(rangeId, hiddenId, valId, v) {
  const range = $(rangeId), hidden = $(hiddenId), val = $(valId);
  if (!range || typeof v !== 'number' || !Number.isFinite(v)) return false;
  const min = parseInt(range.min, 10) || 0;
  const max = parseInt(range.max, 10) || 0;
  v = Math.min(max, Math.max(min, v));
  range.value = v;
  if (hidden) hidden.value = v;
  _setValDisplay(val, v);
  return true;
}

function _setSliderMax(rangeId, hiddenId, valId, max) {
  const range = $(rangeId);
  if (!range || typeof max !== 'number' || !Number.isFinite(max) || max <= 0) return false;
  range.max = max;
  const valEl = $(valId);
  if (valEl && 'max' in valEl) valEl.max = max;
  const hidden = $(hiddenId);
  const cur = parseInt(hidden ? hidden.value : '0', 10);
  if (Number.isFinite(cur) && cur > max) {
    _setSliderVal(rangeId, hiddenId, valId, max);
  }
  return true;
}

function _xargTokenize(s) {
  const parts = [];
  let cur = '', inQ = false;
  for (const ch of s) {
    if (ch === '"') inQ = !inQ;
    else if (/\s/.test(ch) && !inQ) {
      if (cur) { parts.push(cur); cur = ''; }
    } else cur += ch;
  }
  if (cur) parts.push(cur);
  return parts;
}
function _xargsNormalize(v) {
  if (Array.isArray(v)) {
    return v
      .map(x => Array.isArray(x) ? x.map(y => String(y).trim()).filter(Boolean) : [String(x).trim()].filter(Boolean))
      .filter(g => g.length);
  }
  return [_xargTokenize(String(v || ''))].filter(g => g.length);
}
function _selectXarg(idx, e) {
  state.xargSelAnchor = pickSelection(state.xargSelected, state.xargSelAnchor, idx, e);
  $$('#llama-xargs-list .xarg-item').forEach(el => {
    el.classList.toggle('selected', state.xargSelected.has(Number(el.dataset.i)));
  });
}
function _showXargMenu(e, idx) {
  e.preventDefault();
  e.stopPropagation();
  const m = $('#ctxmenu');
  const delN = (state.xargSelected.has(idx) && state.xargSelected.size > 1) ? state.xargSelected.size : 0;
  const items = [
    { label: '编辑', fn: () => _editXarg(idx) },
    { label: delN ? `删除选中 (${delN})` : '删除', fn: () => _deleteXargAt(idx) },
  ];
  m.innerHTML = items.map((it, i) => `<button data-i="${i}">${it.label}</button>`).join('<hr>');
  m.querySelectorAll('button').forEach((b, i) => { b.onclick = () => { items[i].fn(); hideContextMenu(); }; });
  positionCtxMenu(m, e);
}
function _editXarg(idx) {
  const bg = document.createElement('div');
  bg.className = 'pt-editor-bg';
  bg.innerHTML = `
    <div class="pt-editor">
      <h3>编辑启动参数</h3>
      <div class="field"><label class="hint-label">值含空格需用引号包裹</label><input type="text" id="xarg-ed-input" value="${esc(state.llamaXargs[idx].map(t => /\s/.test(t) ? '"' + t + '"' : t).join(' '))}"/></div>
      <div class="pt-editor-foot">
        <button class="btn" id="xarg-ed-cancel">取消</button>
        <button class="btn primary" id="xarg-ed-save">确定</button>
      </div>
    </div>`;
  document.body.appendChild(bg);
  const input = $('#xarg-ed-input'); input.focus(); input.select();
  $('#xarg-ed-cancel').onclick = () => bg.remove();
  bg.addEventListener('keydown', e => { if (e.key === 'Escape') { e.stopPropagation(); bg.remove(); } });
  input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); save(); } });
  $('#xarg-ed-save').onclick = save;
  function save() {
    const v = input.value.trim();
    if (!v) { input.focus(); return; }
    const group = _xargTokenize(v);
    if (!group.length) return;
    state.llamaXargs[idx] = group;
    bg.remove();
    renderLlamaXargsList();
  }
}
async function _deleteXargAt(idx) {
  if (state.xargSelected.has(idx) && state.xargSelected.size > 1) {
    if (!await showConfirm(`确定删除选中的 ${state.xargSelected.size} 组参数？`, { okText: '删除' })) return;
    [...state.xargSelected].sort((a, b) => b - a).forEach(i => state.llamaXargs.splice(i, 1));
  } else {
    state.llamaXargs.splice(idx, 1);
  }
  state.xargSelected = new Set(); state.xargSelAnchor = null;
  renderLlamaXargsList();
}
function renderLlamaXargsList() {
  const list = $('#llama-xargs-list');
  if (!list) return;
  if (!state.llamaXargs.length) { list.style.display = 'none'; list.innerHTML = ''; return; }
  list.style.display = 'flex';
  list.innerHTML = state.llamaXargs.map((g, i) =>
    `<div class="xarg-item${state.xargSelected.has(i) ? ' selected' : ''}" data-i="${i}">` +
    g.map((x, j) => `<code class="${j > 0 ? 'xarg-val' : ''}">${esc(x)}</code>`).join('') +
    `</div>`).join('');
  list.querySelectorAll('.xarg-item').forEach(el => {
    const i = Number(el.dataset.i);
    el.addEventListener('click', e => _selectXarg(i, e));
    el.addEventListener('contextmenu', e => _showXargMenu(e, i));
  });
}
function addLlamaXarg() {
  const inp = $('#llama-extra');
  if (!inp) return;
  const v = inp.value.trim();
  if (!v) { inp.focus(); return; }
  const group = _xargTokenize(v);
  if (!group.length) { inp.focus(); return; }
  state.llamaXargs.push(group);
  inp.value = ''; inp.focus();
  renderLlamaXargsList();
  const list = $('#llama-xargs-list');
  if (list && list.lastElementChild) {
    list.scrollTop = list.scrollHeight;
    list.scrollIntoView({ block: 'end', behavior: 'smooth' });
  }
}

const _EMPTY_OK_KEYS = new Set(['kv_quant', 'load_mode', 'mmproj', 'kv_preset',
                                'image_min_tokens', 'image_max_tokens', 'spec_draft_n']);
function _cfgGet(mcfg, cfg, k, fb) {
  const ok = v => v !== undefined && v !== null && (_EMPTY_OK_KEYS.has(k) || v !== '');
  const v = ok(mcfg[k]) ? mcfg[k] : cfg[k];
  return ok(v) ? v : fb;
}

function _selectModelData(cfg, models) {
  return models.find(m => m.path === cfg.model)
    || models.find(m => m.path === cfg.last_model)
    || models[0] || null;
}

function _applyModelLimits(model) {
  if (!model) return false;
  const ctxMax = (model.ctx > 0) ? model.ctx : 65536;
  const nglMax = (model.layers > 0) ? model.layers : 999;
  let changed = _setSliderMax('#llama-ctx-range', '#llama-ctx', '#llama-ctx-val', ctxMax);
  changed = _setSliderMax('#llama-ngl-range', '#llama-ngl', '#llama-ngl-val', nglMax) || changed;
  const row = $('#llama-ot-row');
  if (row) {
    const isMoe = !!model.moe;
    if (isMoe) {
      _setSliderMax('#llama-moe-range', '#llama-moe', '#llama-moe-val',
                    Math.max(1, parseInt(model.layers, 10) || 128));
    } else {
      const r = $('#llama-moe-range'), h = $('#llama-moe'), v = $('#llama-moe-val');
      if (r) r.value = 0;
      if (h) h.value = '';
      _setValDisplay(v, '0');
    }
    row.style.display = isMoe ? '' : 'none';
    changed = true;
  }
  return changed;
}

function _applyModelParams(path, cfg, mcfgs, defaults) {
  const d = defaults || {};
  const mcfg = (path && mcfgs && mcfgs[path]) || {};
  const pv = (k, fb) => _cfgGet(mcfg, cfg, k, fb);
  const setVal = (id, v) => {
    const el = $(id);
    if (el && v !== undefined && v !== null) el.value = v;
  };
  setVal('#llama-host', pv('host', d.host));
  setVal('#llama-port', pv('port', d.port));
  setVal('#llama-threads', pv('threads', d.threads));
  setVal('#llama-batch', pv('batch', d.batch));
  setVal('#llama-tb', pv('threads_batch', d.threads_batch));
  setVal('#llama-ubatch', pv('ubatch', d.ubatch));
  setVal('#llama-parallel', pv('parallel', d.parallel));
  setVal('#llama-npredict', pv('npredict', d.npredict));
  setVal('#llama-timeout', pv('timeout', d.timeout));
  setVal('#llama-alias', pv('alias', d.alias || ''));
  state.llamaXargs = _xargsNormalize(pv('extra_args', d.extra_args || []));
  state.xargSelected = new Set();
  state.xargSelAnchor = null;
  renderLlamaXargsList();
  const kvDD = $('#llama-kv');
  if (kvDD) setDropdownValue(kvDD, _kvSelectVal(pv('kv_quant', d.kv_quant || '')));
  const lmDD = $('#llama-loadmode');
  if (lmDD) setDropdownValue(lmDD, _loadModeSelectVal(pv('load_mode', d.load_mode)));
  const imgMinDD = $('#llama-image-min');
  const imgMaxDD = $('#llama-image-max');
  if (imgMinDD) setDropdownValue(imgMinDD, _imgTokensSelectVal(pv('image_min_tokens', d.image_min_tokens)));
  if (imgMaxDD) setDropdownValue(imgMaxDD, _imgTokensSelectVal(pv('image_max_tokens', d.image_max_tokens)));
  _syncImgTokenLimits();
  const setChk = (id, on) => {
    const el = $(id);
    if (el) el.checked = !!on;
  };
  const reasonDD = $('#llama-reasoning-dd');
  if (reasonDD) setDropdownValue(reasonDD, _reasoningSelectVal(pv('reasoning_mode', d.reasoning_mode || 'off')));
  const specDD = $('#llama-spec');
  if (specDD) setDropdownValue(specDD, _specSelectVal(pv('spec_draft_n', d.spec_draft_n || '')));
  const presetDD = $('#llama-kv-preset');
  if (presetDD) setDropdownValue(presetDD, _kvPresetSelectVal(pv('kv_preset', d.kv_preset || '')));
  const moeRow = $('#llama-ot-row');
  if (moeRow && moeRow.style.display !== 'none') {
    _setSliderVal('#llama-moe-range', '#llama-moe', '#llama-moe-val', _clampInt(pv('n_cpu_moe', 0), 0, 99999, 0));
  }
  _setSliderVal('#llama-ctx-range', '#llama-ctx', '#llama-ctx-val', _clampInt(pv('ctx', d.ctx), 512, 999999, d.ctx));
  _setSliderVal('#llama-ngl-range', '#llama-ngl', '#llama-ngl-val', _clampInt(pv('ngl', d.ngl), 0, 99999, d.ngl));
}

function buildDdInner(opts, selVal) {
  const cur = opts.find(([v]) => v === selVal) || opts[0];
  return `<button class="dd-btn" type="button">
      <span class="dd-label">${esc(cur[1])}</span>${ddArrow()}
    </button>
    <div class="dd-panel">
      ${opts.map(([v, t]) => `<div class="dd-opt${v === cur[0] ? ' active' : ''}" data-value="${esc(v)}">${esc(t)}</div>`).join('')}
    </div>`;
}

const KV_TYPES = ['f32', 'f16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1'];
function _kvSelectVal(v) {
  const s = String(v || '').trim().toLowerCase();
  return (s === '' || KV_TYPES.includes(s)) ? s : 'q8_0';
}
const KV_OPT_LABELS = [['', '不启用'], ['f32', 'F32'], ['f16', 'F16'], ['q8_0', 'Q8_0'], ['q4_0', 'Q4_0'], ['q4_1', 'Q4_1'], ['iq4_nl', 'Q4_NL'], ['q5_0', 'Q5_0'], ['q5_1', 'Q5_1']];
const _kvDdInner = sel => buildDdInner(KV_OPT_LABELS, sel);

const KV_PRESET_OPT_LABELS = [['', '默认'], ['kv_unified', '统一KV缓存--kv-unified'], ['disable_reuse', '禁用提示缓存与检查点'], ['kv_unified_disable_reuse', '统一KV+禁用缓存与检查点']];
function _kvPresetSelectVal(v) {
  const s = String(v || '').trim();
  return KV_PRESET_OPT_LABELS.some(([val]) => val === s) ? s : '';
}
const _kvPresetDdInner = sel => buildDdInner(KV_PRESET_OPT_LABELS, sel);

const LOAD_MODES = ['none', 'mmap', 'mmap+mlock', 'mlock', 'dio'];
function _loadModeSelectVal(v) {
  const s = String(v || '').trim().toLowerCase();
  return (s === '' || LOAD_MODES.includes(s)) ? s : 'none';
}
const LOAD_MODE_OPT_LABELS = [['', '自动'], ['none', 'none'], ['mmap', 'mmap'], ['mmap+mlock', 'mmap+mlock'], ['mlock', 'mlock'], ['dio', 'dio']];
const _loadModeDdInner = sel => buildDdInner(LOAD_MODE_OPT_LABELS, sel);

const REASONING_OPT_LABELS = [['on', '启用思考'], ['off', '关闭思考'], ['auto', '自动'],
                              ['low', '思考深度 Low'], ['medium', '思考深度 Medium'],
                              ['xhigh', '思考深度 Xhigh']];
function _reasoningSelectVal(v) {
  const s = String(v || '').trim().toLowerCase();
  return REASONING_OPT_LABELS.some(([val]) => val === s) ? s : 'off';
}
const _reasoningDdInner = sel => buildDdInner(REASONING_OPT_LABELS, sel);

const SPEC_DRAFT_OPTS = [['', '不启用'], ['1', '[启用]草稿令牌数 1'], ['2', '[启用]草稿令牌数 2'],
                         ['3', '[启用]草稿令牌数 3'], ['4', '[启用]草稿令牌数 4'], ['5', '[启用]草稿令牌数 5']];
function _specSelectVal(v) {
  const s = String(v || '').trim();
  return SPEC_DRAFT_OPTS.some(([val]) => val === s) ? s : '';
}
const _specDdInner = sel => buildDdInner(SPEC_DRAFT_OPTS, sel);

const IMG_TOKEN_STR = ['70', '140', '280', '560', '1120'];
const _IMG_TOKEN_OPTS = [['', '自动']].concat(IMG_TOKEN_STR.map(t => [t, t]));
function _imgTokensSelectVal(v) {
  const s = String(v || '').trim();
  return (s === '' || IMG_TOKEN_STR.includes(s)) ? s : '';
}
const _imgTokensDdInner = sel => buildDdInner(_IMG_TOKEN_OPTS, sel);
function _syncImgTokenLimits() {
  const minDD = $('#llama-image-min');
  const maxDD = $('#llama-image-max');
  if (!minDD || !maxDD) return;
  const maxV = getDropdownValue(maxDD);
  const maxN = maxV ? parseInt(maxV, 10) : 0;
  IMG_TOKEN_STR.forEach(t => {
    setDropdownDisabled(minDD, t, !!maxN && parseInt(t, 10) > maxN);
  });
  const minV = getDropdownValue(minDD);
  if (maxN && minV && parseInt(minV, 10) > maxN) setDropdownValue(minDD, maxV);
}

function _mmprojDdInner(selVal, list, extraVal) {
  const baseName = p => String(p).split(/[\\/]/).pop();
  const opts = [['__auto__', '自动检测'], ['__auto_cpu__', '自动检测并使用CPU计算'], ['', '不使用']]
    .concat(list.map(p => [p, baseName(p)]));
  if (extraVal && !list.includes(extraVal)) opts.push([extraVal, '当前值: ' + baseName(extraVal)]);
  return buildDdInner(opts, selVal);
}
function _renderMmprojDd(selVal, list, extraVal) {
  const dd = $('#llama-mmproj');
  if (!dd) return;
  dd.innerHTML = _mmprojDdInner(selVal, list, extraVal);
  initDropdown(dd);
}

function _collectLlamaParams(extra) {
  const sel = $('#llama-model-dd');
  const mmprojEl = $('#llama-mmproj');
  const mmprojSel = mmprojEl ? getDropdownValue(mmprojEl) : '__auto__';
  const kvEl = $('#llama-kv');
  const kvSel = kvEl ? getDropdownValue(kvEl) : '';
  const kvPresetEl = $('#llama-kv-preset');
  const kvPresetSel = kvPresetEl ? getDropdownValue(kvPresetEl) : '';
  const base = {
    model: sel ? getDropdownValue(sel) : '',
    host: $('#llama-host').value,
    port: $('#llama-port').value,
    threads: $('#llama-threads').value,
    threads_batch: $('#llama-tb').value,
    ngl: $('#llama-ngl').value,
    ctx: $('#llama-ctx').value,
    batch: $('#llama-batch').value,
    ubatch: $('#llama-ubatch').value,
    parallel: $('#llama-parallel').value,
    npredict: $('#llama-npredict').value,
    timeout: $('#llama-timeout').value,
    kv_quant: kvSel,
    load_mode: $('#llama-loadmode') ? getDropdownValue($('#llama-loadmode')) : '',
    image_min_tokens: $('#llama-image-min') ? getDropdownValue($('#llama-image-min')) : '',
    image_max_tokens: $('#llama-image-max') ? getDropdownValue($('#llama-image-max')) : '',
    mmproj: (mmprojSel === '__auto__' || mmprojSel === '__auto_cpu__') ? '' : mmprojSel,
    mmproj_auto: mmprojSel === '__auto__' || mmprojSel === '__auto_cpu__',
    no_mmproj_offload: mmprojSel === '__auto_cpu__',
    spec_draft_n: $('#llama-spec') ? getDropdownValue($('#llama-spec')) : '',
    reasoning_mode: $('#llama-reasoning-dd') ? getDropdownValue($('#llama-reasoning-dd')) : 'off',
    kv_preset: kvPresetSel,
    extra_args: state.llamaXargs,
    alias: $('#llama-alias').value.trim(),
    n_cpu_moe: $('#llama-moe').value,
  };
  return Object.assign(base, extra || {});
}

async function rescanLlamaModels(toastMsg) {
  try {
    const r = await apiCall('scan_llama_models');
    if (r && r.ok) {
      if (toastMsg) toast(toastMsg, 'ok');
      const st = await apiCall('get_llama_status');
      if (st && st.ok !== false) {
        renderLlamaTab(st);
        return;
      }
    } else if (r && !r.ok) {
      toast('扫描失败: ' + (r.error || ''), 'err');
    }
  } catch (e) { toast('扫描失败', 'err'); }
  renderSettings();
}

function _bindLlamaTabEvents(llamaStatus) {
  const cfg = llamaStatus.config || {};
  const defaults = llamaStatus.defaults || {};
  const toggleBtn = $('#btn-llama-toggle');
  if (toggleBtn) toggleBtn.addEventListener('click', () => {
    if (state.llamaRunning || state.llamaStarting) stopLlama();
    else if (!state.llamaRunning && !state.llamaStarting) launchLlama();
  });

  const webuiBtn = $('#btn-llama-webui');
  if (webuiBtn) webuiBtn.addEventListener('click', async () => {
    if (!state.llamaRunning) {
      toast(state.llamaStarting ? '服务正在启动中，请稍候…' : '请先启动本地推理服务');
      return;
    }
    const r = await apiCall('open_llama_webui');
    if (r && r.ok) toast('已在默认浏览器打开聊天界面', 'ok');
    else toast((r && r.error) || '打开失败', 'err');
  });

  const rescanBtn = $('#btn-llama-rescan');
  if (rescanBtn) rescanBtn.addEventListener('click', () => rescanLlamaModels('已重新扫描模型'));

  const dlBtn = $('#btn-llama-download');
  if (dlBtn) dlBtn.addEventListener('click', () => openHfDownloadDialog());

  _wireSlider('#llama-ctx-range', '#llama-ctx', '#llama-ctx-val');
  _wireSlider('#llama-ngl-range', '#llama-ngl', '#llama-ngl-val');
  _wireSlider('#llama-moe-range', '#llama-moe', '#llama-moe-val');

  _applyModelLimits(_selectModelData(cfg, llamaStatus.models || []));

  const modelDD = $('#llama-model-dd');
  if (modelDD) {
    initDropdown(modelDD, (path) => {
      if ($('#llama-mmproj')) {
        const m = (llamaStatus.models || []).find(x => x.path === path);
        const list = (m && m.mmprojs) || [];
        const mcfg = (llamaStatus.model_configs || {})[path || ''] || {};
        const pvV = (k, fb) => _cfgGet(mcfg, cfg, k, fb);
        const mVal = pvV('mmproj', '');
        const mAuto = pvV('mmproj_auto', true) !== false;
        const mCpu = pvV('no_mmproj_offload', false) === true;
        const selVal = mVal || (mAuto ? (mCpu ? '__auto_cpu__' : '__auto__') : '');
        _renderMmprojDd(selVal, list, mVal);
        if (list.length) toast(`已切换模型，检测到 ${list.length} 个投影文件`, 'info');
      }
      _applyModelLimits((llamaStatus.models || []).find(x => x.path === path) || null);
      _applyModelParams(path, cfg, llamaStatus.model_configs || {}, defaults);
    });
  }

  const mmprojDD = $('#llama-mmproj');
  if (mmprojDD) initDropdown(mmprojDD);
  const kvDD = $('#llama-kv');
  if (kvDD) initDropdown(kvDD);
  const lmDD = $('#llama-loadmode');
  if (lmDD) initDropdown(lmDD);
  const reasonDD = $('#llama-reasoning-dd');
  if (reasonDD) initDropdown(reasonDD);
  const specDD = $('#llama-spec');
  if (specDD) initDropdown(specDD);
  const presetDD = $('#llama-kv-preset');
  if (presetDD) initDropdown(presetDD);
  const imgMinDD = $('#llama-image-min');
  const imgMaxDD = $('#llama-image-max');
  if (imgMinDD) initDropdown(imgMinDD, () => _syncImgTokenLimits());
  if (imgMaxDD) initDropdown(imgMaxDD, () => _syncImgTokenLimits());
  _syncImgTokenLimits();

  const advToggle = $('#llama-adv-toggle');
  const advPanel = $('#llama-adv-panel');
  if (advToggle && advPanel) {
    if (state.settingsOpen.has('llama-adv')) advToggle.classList.add('open');
    advToggle.addEventListener('click', () => {
      const open = advPanel.style.display !== 'none';
      advPanel.style.display = open ? 'none' : '';
      advToggle.classList.toggle('open', !open);
      if (open) state.settingsOpen.delete('llama-adv');
      else state.settingsOpen.add('llama-adv');
    });
  }

  const xargInput = $('#llama-extra');
  const xargAddBtn = $('#btn-xarg-add');
  if (xargInput) xargInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); addLlamaXarg(); }
  });
  if (xargAddBtn) xargAddBtn.addEventListener('click', addLlamaXarg);
  renderLlamaXargsList();
}



document.addEventListener('click', (e) => {
  if (e.target.closest('#miniProgBox') && state.hfDownloading) {
    if (state.dlKind === 'whisper') openWhisperModelDialog();
    else openHfDownloadDialog();
  }
});

$('#confirmBg').addEventListener('click', e => {
  if (e.target.closest('#hfDlgClose')) {
    if (_hfDlApi.dl.active) _hfDlApi.close(true); else _hfDlApi.close();
  }
});

$('#modal-body').addEventListener('scroll', () => {
  $('#modal-body').classList.toggle('scrolled-past-hero', $('#modal-body').scrollTop > 10);
}, { passive: true });
