/* ════════════════════════════════════════════════════════════
   启动加载态
   ════════════════════════════════════════════════════════════ */
function setLoadHint(text) {
  const el = document.getElementById('load-hint');
  if (el) el.textContent = text;
}
function hideAppLoading() {
  const el = document.getElementById('app-loading');
  if (!el) return;
  el.style.opacity = '0';
  setTimeout(() => el.remove(), 300);
}

const _showJsError = (function() {
  let _errEl = null, _errTimer = null, _lastMsg = '';
  return function(text) {
    if (text === _lastMsg) return;
    _lastMsg = text;
    if (!_errEl) {
      _errEl = document.createElement('div');
      _errEl.style.cssText = 'position:fixed;bottom:0;left:0;right:0;background:#fee;border-top:2px solid #c00;' +
        'padding:10px 14px;font:12px monospace;color:#900;z-index:99999;white-space:pre-wrap;transition:opacity .3s';
      document.body.appendChild(_errEl);
    }
    _errEl.textContent = text;
    _errEl.style.opacity = '1';
    _errEl.style.display = '';
    clearTimeout(_errTimer);
    _errTimer = setTimeout(() => { _errEl.style.opacity = '0'; _lastMsg = ''; }, 8000);
  };
})();
window.onerror = function(msg, url, line, col, err) {
  _showJsError('JS Error [L' + line + ']: ' + msg);
};
// window.onerror 捕获不到 Promise 拒绝，这里兜底显示
window.addEventListener('unhandledrejection', (e) => {
  const r = e.reason;
  _showJsError('Promise Error: ' + ((r && (r.message || String(r))) || String(r)));
});

/* ════════════════════════════════════════════════════════════
   主题切换
   ════════════════════════════════════════════════════════════ */
let _themeMode = '';
function applyTheme(mode) {
  _themeMode = mode;
  const html = document.documentElement;
  html.classList.remove('dark', 'light');
  if (mode === 'dark') html.classList.add('dark');
  else if (mode === 'light') html.classList.add('light');
  syncTitlebar(mode);
}
function syncTitlebar(mode) {
  let dark;
  if (mode === 'dark') dark = true;
  else if (mode === 'light') dark = false;
  else dark = window.matchMedia('(prefers-color-scheme:dark)').matches;
  apiCall('set_titlebar_dark', dark).catch(() => {});
}
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.classList.contains('dark') ||
    (!html.classList.contains('light') && window.matchMedia('(prefers-color-scheme:dark)').matches);
  const next = isDark ? 'light' : 'dark';
  applyTheme(next);
  apiCall('save_config', { theme: next }).catch(() => {});
}
function applyForceAnimation(on) {
  document.documentElement.classList.toggle('force-anim', !!on);
}

/* ════════════════════════════════════════════════════════════
   全局状态
   ════════════════════════════════════════════════════════════ */
const state = {
  pending: [], processed: [], failed: [],
  roots: [], adhoc_files: [],
  view: 'pending',
  hasAutoSwitched: false,
  sourceFilter: null,
  selected: new Set(),
  selAnchor: null,
  primaryId: null,
  processing: false,
  search: '',
  searchMode: 'all',
  sortBy: 'default',
  aiConnected: false,
  llamaSynced: false,
  llamaEnabled: false,
  llamaIntegration: false,
  llamaRunning: false,
  llamaStarting: false,
  llamaPendingLaunch: false,
  pillMode: 'conn',
  settings_tab: 'config',
  active_preset_id: 'default',
  ptItems: [],
  ptSelected: new Set(),
  ptSelAnchor: null,
  ptLoadedEnabled: false,
  settings_scroll_to: null,
  thumbCache: new Map(),
  pixaiTaggerEnabled: false,
  whisperEnabled: false,
  gpuBusy: false,
  installing: false,
  pixaiTaggedIds: new Set(),
  pixaiRealIds: new Set(),
  pixaiAnimeIds: new Set(),
  pixaiUncertainIds: new Set(),
  whisperTranscribedIds: new Set(),
  hfDownloading: false,
  hfDlPct: 0,
  dlKind: null,
  settingsOpen: new Set(),
  settingsScroll: 0,
  llamaXargs: [],
  xargSelected: new Set(),
  xargSelAnchor: null,
};
const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));

function initDropdown(el, onChange, noValueActive) {
  const btn = el.querySelector('.dd-btn');
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const wasOpen = el.classList.contains('open');
    $$('.dd.open').forEach(d => d.classList.remove('open'));
    if (!wasOpen) {
      _positionPanel(el);
      el.classList.add('open');
    }
  });
  el.querySelectorAll('.dd-opt').forEach(opt => {
    opt.addEventListener('click', (e) => {
      e.stopPropagation();
      if (opt.classList.contains('disabled')) return;
      if (!(noValueActive && opt.dataset.value)) setDropdownValue(el, opt.dataset.value);
      el.classList.remove('open');
      if (onChange) onChange(opt.dataset.value, opt);
    });
  });
}
function _positionPanel(el) {
  const btn = el.querySelector('.dd-btn');
  const panel = el.querySelector('.dd-panel');
  const r = btn.getBoundingClientRect();
  panel.style.position = 'fixed';
  panel.style.top = (r.bottom + 6) + 'px';
  const pw = panel.offsetWidth || 150;
  panel.style.left = Math.min(r.left, window.innerWidth - pw - 8) + 'px';
  panel.style.minWidth = r.width + 'px';
  panel.style.maxHeight = Math.max(120, window.innerHeight - r.bottom - 14) + 'px';
}
function setDropdownValue(el, val) {
  el.querySelectorAll('.dd-opt').forEach(o => o.classList.toggle('active', o.dataset.value === val));
  const active = el.querySelector('.dd-opt.active');
  const label = el.querySelector('.dd-label');
  if (active && label) label.textContent = active.textContent;
}
function setDropdownDisabled(el, val, disabled) {
  const opt = el.querySelector(`.dd-opt[data-value="${val}"]`);
  if (opt) opt.classList.toggle('disabled', disabled);
}
function getDropdownValue(el) {
  const active = el.querySelector('.dd-opt.active');
  return active ? active.dataset.value : '';
}
document.addEventListener('click', () => {
  $$('.dd.open').forEach(d => d.classList.remove('open'));
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') $$('.dd.open').forEach(d => d.classList.remove('open'));
});

let _tipBox = null;
function _placeTip(target) {
  const text = target.getAttribute('data-tip');
  if (!text) return;
  if (target.hasAttribute('data-tip-trunc') && target.scrollWidth - target.clientWidth <= 1) return;
  if (!_tipBox) {
    _tipBox = document.createElement('div');
    _tipBox.className = 'hover-tip';
    document.body.appendChild(_tipBox);
  }
  _tipBox.classList.toggle('light', !!(target.classList && target.classList.contains('selected-info')));
  _tipBox.textContent = text;
  _tipBox.classList.add('show');
  const r = target.getBoundingClientRect();
  if (_tipBox.classList.contains('light')) {
    _tipBox.style.maxWidth = (window.innerWidth - Math.max(8, r.left) - 15) + 'px';
  } else {
    _tipBox.style.maxWidth = '';
  }
  const tw = _tipBox.offsetWidth, th = _tipBox.offsetHeight;
  const left = Math.min(Math.max(8, r.left), window.innerWidth - tw - 8);
  let top = r.top - th - 8;
  const below = top < 8;
  if (below) top = r.bottom + 8;
  _tipBox.classList.toggle('below', below);
  _tipBox.style.left = left + 'px';
  _tipBox.style.top = top + 'px';
  _tipBox.style.setProperty('--ax', Math.min(Math.max(6, r.left - left + 8), tw - 16) + 'px');
}
function _hideTip() { if (_tipBox) _tipBox.classList.remove('show'); }
document.addEventListener('mouseover', (e) => {
  const t = e.target.closest ? e.target.closest('[data-tip]') : null;
  if (t) _placeTip(t); else _hideTip();
});
document.addEventListener('mouseout', (e) => { if (!e.relatedTarget) _hideTip(); });
document.addEventListener('scroll', _hideTip, true);
document.addEventListener('click', _hideTip, true);

function icon(name, size) {
  let s = size || '1em';
  return '<svg class="ic" style="width:' + s + ';height:' + s + '" aria-hidden="true"><use href="#ic-' + name + '"></use></svg>';
}

function ddArrow(cls) {
  return '<svg class="' + (cls || 'dd-arrow') + '" viewBox="0 0 10 6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M1 1l4 4 4-4"/></svg>';
}

function toast(msg, kind, replace) {
  const wrap = $('#toast-wrap');
  if (replace && wrap.lastChild) wrap.lastChild.remove();
  if (wrap.lastChild && wrap.lastChild.textContent === msg) wrap.lastChild.remove();
  const el = document.createElement('div');
  el.className = 'toast-item';
  el.textContent = msg;
  wrap.appendChild(el);
  while (wrap.children.length > 4) wrap.firstChild.remove();
  requestAnimationFrame(() => el.classList.add('show'));
  const dur = kind === 'err' ? 4000 : 2600;
  setTimeout(() => {
    el.classList.remove('show');
    el.classList.add('hide');
    el.addEventListener('transitionend', () => el.remove(), { once: true });
    setTimeout(() => el.remove(), 400);
  }, dur);
}
function showConfirm(msg, opts) {
  opts = opts || {};
  const bg = $('#confirmBg');
  $('#confirmTitle').textContent = opts.title || '确认';
  $('#confirmMsg').textContent = msg;
  $('#confirmOk').textContent = opts.okText || '确定';
  bg.classList.add('show');
  $('#confirmOk').focus();
  return new Promise(resolve => {
    function close(val) {
      bg.classList.remove('show');
      $('#confirmOk').removeEventListener('click', onOk);
      $('#confirmCancel').removeEventListener('click', onCancel);
      document.removeEventListener('keydown', onKey);
      resolve(val);
    }
    function onOk() { close(true); }
    function onCancel() { close(false); }
    function onKey(e) { if (e.key === 'Escape') { e.stopPropagation(); close(false); } }
    $('#confirmOk').addEventListener('click', onOk);
    $('#confirmCancel').addEventListener('click', onCancel);
    document.addEventListener('keydown', onKey);
  });
}
function showPrompt(msg, opts) {
  opts = opts || {};
  const bg = $('#promptBg');
  $('#promptTitle').textContent = opts.title || '输入';
  const msgEl = $('#promptMsg');
  msgEl.textContent = msg || '';
  msgEl.style.display = msg ? '' : 'none';
  const input = $('#promptInput');
  input.value = opts.defaultValue || '';
  input.placeholder = opts.placeholder || '';
  $('#promptOk').textContent = opts.okText || '确定';
  bg.classList.add('show');
  input.focus();
  input.select();
  return new Promise(resolve => {
    function close(val) {
      bg.classList.remove('show');
      $('#promptOk').removeEventListener('click', onOk);
      $('#promptCancel').removeEventListener('click', onCancel);
      input.removeEventListener('keydown', onKey);
      resolve(val);
    }
    function onOk() {
      const v = input.value.trim();
      if (!v) {
        input.classList.remove('shake');
        void input.offsetWidth;
        input.classList.add('shake');
        setTimeout(() => input.classList.remove('shake'), 400);
        input.focus();
        return;
      }
      close(v);
    }
    function onCancel() { close(null); }
    function onKey(e) {
      if (e.key === 'Enter') { e.preventDefault(); onOk(); }
      else if (e.key === 'Escape') { e.stopPropagation(); close(null); }
    }
    $('#promptOk').addEventListener('click', onOk);
    $('#promptCancel').addEventListener('click', onCancel);
    input.addEventListener('keydown', onKey);
  });
}
const _escEl = document.createElement('div');
function esc(s) {
  _escEl.textContent = s == null ? '' : String(s);
  return _escEl.innerHTML.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function fmtDur(sec) {
  if (!sec) return '';
  sec = Math.round(sec);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (h > 0) return h + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
  return m + ':' + String(s).padStart(2, '0');
}

/* ════════════════════════════════════════════════════════════
   pywebview 桥
   ════════════════════════════════════════════════════════════ */
let _pywebviewApi = null, _bridgeWait = null;
function pywebviewReady(timeout) {
  if (_pywebviewApi) return Promise.resolve(_pywebviewApi);
  if (window.pywebview && window.pywebview.api && typeof window.pywebview.api.scan === 'function') {
    _pywebviewApi = window.pywebview.api;
    return Promise.resolve(_pywebviewApi);
  }
  if (!_bridgeWait) {
    _bridgeWait = new Promise((res) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        _pywebviewApi = window.pywebview.api;
        res(_pywebviewApi);
      };
      window.addEventListener('pywebviewready', finish, { once: true });
      let waited = 0;
      const t = setInterval(() => {
        const api = window.pywebview && window.pywebview.api;
        if (api && typeof api.scan === 'function') {
          clearInterval(t);
          finish();
          return;
        }
        waited += 40;
        if (waited % 10000 < 40) toast('后端就绪较慢，正在等待连接…', 'err');
      }, 40);
    });
  }
  if (timeout && timeout > 0) {
    return Promise.race([
      _bridgeWait,
      new Promise((_, rej) => setTimeout(
        () => rej(new Error('后端启动超时（' + Math.round(timeout / 1000) + 's 未就绪）')),
        timeout))
    ]);
  }
  return _bridgeWait;
}
async function apiCall(method, ...args) {
  const api = await pywebviewReady();
  return await api[method](...args);
}
async function callApi(method, ...args) {
  try {
    return await apiCall(method, ...args);
  } catch (e) {
    toast('操作失败: ' + ((e && e.message) || e), 'err');
    return undefined;
  }
}

function createLimiter(max) {
  const queue = [];
  let active = 0;
  const pump = () => {
    while (active < max && queue.length) {
      const run = queue.shift();
      active++;
      run().finally(() => { active--; pump(); });
    }
  };
  return fn => new Promise((resolve, reject) => {
    queue.push(() => fn().then(resolve, reject));
    pump();
  });
}

function pickSelection(sel, anchor, id, e, rangeIds) {
  e = e || {};
  if (e.ctrlKey || e.metaKey) {
    if (sel.has(id)) sel.delete(id);
    else sel.add(id);
    return id;
  }
  if (e.shiftKey && anchor != null) {
    if (rangeIds) {
      const a = rangeIds.indexOf(anchor), b = rangeIds.indexOf(id);
      if (a >= 0 && b >= 0) {
        for (let i = Math.min(a, b); i <= Math.max(a, b); i++) sel.add(rangeIds[i]);
      } else {
        sel.add(id);
      }
      return anchor;
    }
    for (let i = Math.min(anchor, id); i <= Math.max(anchor, id); i++) sel.add(i);
    return anchor;
  }
  if (sel.size === 1 && sel.has(id)) {
    sel.clear();
    return null;
  }
  sel.clear();
  sel.add(id);
  return id;
}

/* ════════════════════════════════════════════════════════════
   后端 → 前端 推送回调
   ════════════════════════════════════════════════════════════ */
window.__ui = {
  appendLog(line, level) {
    const log = $('#log');
    const le = $('#logEmpty'); if (le) le.remove();
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    const d = document.createElement('div');
    d.className = 'ln' + (level ? ' ' + level.toLowerCase() : '');
    d.textContent = line;
    log.appendChild(d);
    const excess = log.childElementCount - 1000;
    if (excess > 0) {
      const range = document.createRange();
      range.setStartBefore(log.firstChild);
      range.setEndAfter(log.children[excess - 1]);
      range.deleteContents();
    }
    if (atBottom) log.scrollTop = log.scrollHeight;
  },
  llamaStateChanged() {
    apiCall('get_llama_status').then(st => {
      if (!st) return;
      syncLlamaState(st);
      if (st.starting && state.llamaIntegration) startLlamaPillPolling();
    }).catch(() => {});
  },
  setProgress(cur, tot) {
    const pct = tot > 0 ? Math.round(cur / tot * 100) : 0;
    $('#prog-bar').style.width = pct + '%';
    $('#prog-num').textContent = tot > 0 ? `${cur}/${tot} (${pct}%)` : '';
  },
  hfDownloadProgress(ev) {
    if (typeof window.__onHfDownloadProgress === 'function') window.__onHfDownloadProgress(ev);
  },
  hfDownloadDone(ev) {
    if (typeof window.__onHfDownloadDone === 'function') window.__onHfDownloadDone(ev);
  },
  whisperModelProgress(ev) {
    if (typeof window.__onWhisperModelProgress === 'function') window.__onWhisperModelProgress(ev);
  },
  whisperModelDone(ev) {
    if (typeof window.__onWhisperModelDone === 'function') window.__onWhisperModelDone(ev);
  },
  onFileDone(entry) {
    const idx = state.pending.findIndex(v => v.path === entry.original_path);
    const src = idx >= 0 ? state.pending[idx] : null;
    if (idx >= 0) state.pending.splice(idx, 1);
    const e = {
      id: entry.id,
      path: entry.new_path,
      name: entry.new_name,
      size: (entry.info && entry.info.size) || 0,
      mtime: (src && src.mtime) || 0,
      duration: (entry.info && entry.info.duration) || 0,
      resolution: (entry.info && entry.info.resolution) || '',
      status: entry.status === 'ok' || entry.status === 'skipped' ? 'processed' : 'failed',
      title: entry.title,
      plot: entry.plot,
      tags: entry.tags || [],
      original_name: entry.original_name,
      processed_at: entry.processed_at,
    };
    if (e.status === 'processed') state.processed.unshift(e);
    else state.failed.unshift(e);
    updateStats();
    if (state.view === 'pending') removeFromGrid(entry.id);
    else if (state.view === e.status) refreshGridData();
    updateSelectedInfo();
  },
  onProcessDone(summary, scanResult) {
    state.processing = false;
    state.gpuBusy = false;
    $('#btn-stop').classList.remove('active');
    $('#btn-stop').classList.add('done');
    $('#prog-bar').className = 'done';
    updateMiniProg();
    if (scanResult && !scanResult.error) {
      loadFromResult(scanResult, true);
    } else {
      updateStats();
    }
    const ok = summary && summary.ok;
    const okN = summary && summary.ok_count != null ? summary.ok_count : 0;
    const errN = summary && summary.error_count != null ? summary.error_count : 0;
    const cancelN = summary && summary.cancelled_count != null ? summary.cancelled_count : 0;
    let msg = ok ? '处理完成' : '处理结束（有错误）';
    if (errN > 0) msg += `：成功 ${okN}，失败 ${errN}`;
    if (cancelN > 0) msg += `，${cancelN} 个未处理`;
    toast(msg, ok && errN === 0 ? 'ok' : 'err');
  },
};

/* ════════════════════════════════════════════════════════════
   连接检测 / 本地推理启动状态胶囊
   ════════════════════════════════════════════════════════════ */
function withTimeout(promise, ms) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('timeout')), ms);
    promise.then(v => { clearTimeout(t); resolve(v); },
                  e => { clearTimeout(t); reject(e); });
  });
}

function updateLlamaPill(st) {
  const dot = $('#conn-dot'), label = $('#conn-label'), pill = $('#conn-pill');
  if (!dot || !st) return;
  const hint = '；点击进入本地推理配置';
  let cls, text, title;
  if (st.starting) {
    cls = 'yellow'; text = '正在启动';
    title = '本地推理服务正在启动…' + hint;
  } else if (st.running) {
    cls = 'green'; text = '运行中';
    title = '本地推理服务运行中'
      + (st.model ? '；模型: ' + String(st.model).split(/[\\/]/).pop() : '')
      + (st.port ? '；端口: ' + st.port : '') + hint;
  } else if (st.launch_failed) {
    cls = 'red'; text = '启动失败';
    title = '启动失败: ' + String(st.launch_failed) + hint;
  } else {
    cls = 'gray'; text = '未启动';
    title = '本地推理服务未启动（点击「开始处理」会自动启动）' + hint;
  }
  dot.className = 'conn-dot ' + cls;
  label.textContent = text;
  pill.dataset.tip = title;
}

let _llamaPollTimer = null;
let _llamaPollToken = 0;
function startLlamaPillPolling() {
  if (_llamaPollTimer) { clearTimeout(_llamaPollTimer); _llamaPollTimer = null; }
  const token = ++_llamaPollToken;
  const tick = async () => {
    try {
      const st = await apiCall('get_llama_status');
      if (token !== _llamaPollToken) return;
      if (st) syncLlamaState(st);
      if (!state.llamaIntegration || (!st.starting && !state.llamaPendingLaunch)) {
        stopLlamaPillPolling();
        return;
      }
      _llamaPollTimer = setTimeout(tick, 2000);
    } catch (e) {
      if (token !== _llamaPollToken) return;
      _llamaPollTimer = setTimeout(tick, 5000);
    }
  };
  tick();
}
function stopLlamaPillPolling() {
  _llamaPollToken++;
  if (_llamaPollTimer) { clearTimeout(_llamaPollTimer); _llamaPollTimer = null; }
}

function updateConnectionUI(r) {
  const dot = $('#conn-dot'), label = $('#conn-label'), pill = $('#conn-pill');
  if (!dot) return;
  if (state.llamaIntegration) return;
  state.aiConnected = !!r.ok;
  dot.className = 'conn-dot ' + (r.ok ? 'green' : 'red');
  label.textContent = r.ok ? 'AI 已连接' : '连接失败';
  const addr = r.base_url ? '；服务地址: ' + r.base_url : '';
  pill.dataset.tip = (r.ok ? '已连接' : (r.message || '连接失败')) + addr + '；点击配置';
  updateStartBtn();
}

function updateStartBtn() {
  const btn = $('#btn-start');
  if (!btn) return;
  const ready = state.llamaIntegration || state.aiConnected;
  btn.disabled = !ready || state.pending.length === 0 || state.processing || state.gpuBusy || state.llamaPendingLaunch;
  btn.dataset.tip = !ready ? '请等待 AI 服务连接'
    : state.pending.length === 0 ? '请先添加视频'
    : state.llamaPendingLaunch ? '本地推理服务启动中…'
    : state.llamaIntegration && !state.aiConnected ? '本地推理服务未就绪，点击自动启动'
    : (state.processing || state.gpuBusy) ? '正在处理中…' : '开始处理';
  if (typeof updateSelToolbar === 'function') updateSelToolbar();
}

let _connInflight = null;
async function checkConnection() {
  if (state.llamaIntegration) return state.llamaRunning;
  if (_connInflight) return _connInflight;
  _connInflight = (async () => {
    try {
      const r = await withTimeout(apiCall('check_connection'), 20000);
      updateConnectionUI(r);
      return r.ok;
    } catch (e) {
      updateConnectionUI({ ok: false });
      return false;
    } finally {
      _connInflight = null;
    }
  })();
  return _connInflight;
}

async function initConnection() {
  try {
    const st = await apiCall('get_llama_status');
    if (st) syncLlamaState(st);
  } catch (e) {  }
  if (state.llamaIntegration) {
    if (state.llamaStarting) startLlamaPillPolling();
    return;
  }
  try {
    const r = await withTimeout(apiCall('check_connection'), 20000);
    updateConnectionUI(r);
  } catch (e) {
    updateConnectionUI({ ok: false });
    await new Promise(res => setTimeout(res, 3000));
    await checkConnection();
  }
}
