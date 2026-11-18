/* ════════════════════════════════════════════════════════════
   手动重命名 Modal
   ════════════════════════════════════════════════════════════ */

let _renameItems = [];
let _renameTimer = null;
let _renameSeq = 0;
let _renameRegex = false;

function _basename(p) { return String(p || '').split(/[\\/]/).pop(); }

function _syncRegexUI() {
  $$('#dd-rename-find .dd-btn, #dd-rename-repl .dd-btn').forEach(b =>
    b.classList.toggle('on', _renameRegex));
  $$('#dd-rename-find .dd-opt[data-mode], #dd-rename-repl .dd-opt[data-mode]').forEach(o =>
    o.classList.toggle('active', o.dataset.mode === (_renameRegex ? 'regex' : 'literal')));
}

function openRenameDialog() {
  _renameItems = [...state.selected].map(id => {
    const v = findVideoById(id);
    return v ? { id: v.id, path: v.path } : null;
  }).filter(Boolean);
  if (!_renameItems.length) return;
  if ($('#modal').classList.contains('show')) closeSettings();
  $('#rename-find').value = '';
  $('#rename-repl').value = '';
  setCaseMode('');
  setMatchMode(true);
  _renameRegex = false;
  _syncRegexUI();
  const subBtn = $('#btn-subtoggle');
  const dirs = new Set(_renameItems.map(v => _renameDir(v.path)));
  subBtn.style.display = dirs.size === 1 ? '' : 'none';
  subBtn.classList.remove('on');
  $('#renameModal').classList.add('show');
  $('#rename-find').focus();
  refreshRenamePreview();
}

function _renameDir(p) { return String(p || '').split(/[\\/]/).slice(0, -1).join('/').toLowerCase(); }

function setCaseMode(val) {
  $$('#rename-case input').forEach(r => r.checked = (r.value === val));
  _syncPills();
}
function setMatchMode(all) {
  const b = $('#btn-matchtoggle');
  b.classList.toggle('on', !all);
  b.textContent = all ? '匹配全部' : '匹配首个';
}
function _syncPills() {
  $$('.rename-case input').forEach(r => {
    r.closest('.rp-opt').classList.toggle('on', r.checked);
  });
}

function closeRenameDialog() {
  clearTimeout(_renameTimer);
  $('#renameModal').classList.remove('show');
}

function _currentRenameParams() {
  const caseRadio = $('#rename-case input:checked');
  return { mode: 'replace',
           text: $('#rename-find').value,
           text2: $('#rename-repl').value,
           useRegex: _renameRegex,
           matchAll: !$('#btn-matchtoggle').classList.contains('on'),
           matchSubtitles: $('#btn-subtoggle').classList.contains('on'),
           caseMode: caseRadio ? caseRadio.value : '' };
}

function scheduleRenamePreview() {
  clearTimeout(_renameTimer);
  _renameTimer = setTimeout(refreshRenamePreview, 200);
}

async function refreshRenamePreview() {
  const { mode, text, text2, useRegex, matchAll, matchSubtitles, caseMode } = _currentRenameParams();
  const seq = ++_renameSeq;
  if (!text && !caseMode) {
    if (matchSubtitles) {
      const r = await callApi('manual_rename', _renameItems, mode, text, text2,
                              true, useRegex, matchAll, caseMode, matchSubtitles);
      if (seq !== _renameSeq) return;
      if (!r) {
        $('#rename-preview').innerHTML = '';
        $('#rename-foot-note').textContent = '预览失败：后端桥接未就绪，请稍候再输入重试';
        $('#btn-confirmrename').disabled = true;
        return;
      }
      renderRenamePreview(r.items || [], false);
      return;
    }
    renderRenamePreview(_renameItems.map(it => {
      const n = _basename(it.path);
      return { name: n, new_name: n, changed: false, note: '' };
    }), false);
    return;
  }
  const r = await callApi('manual_rename', _renameItems, mode, text, text2, true, useRegex, matchAll, caseMode, matchSubtitles);
  if (seq !== _renameSeq) return;
  if (!r) {
    $('#rename-preview').innerHTML = '';
    $('#rename-foot-note').textContent = '预览失败：后端桥接未就绪，请稍候再输入重试';
    $('#btn-confirmrename').disabled = true;
    return;
  }
  renderRenamePreview(r.items || [], true);
  if (r.errors && r.errors.length) {
    $('#rename-foot-note').textContent += `；${r.errors.length} 个文件不可用`;
  }
}

function renderRenamePreview(items, hasText) {
  const el = $('#rename-preview');
  const note = $('#rename-foot-note');
  const btn = $('#btn-confirmrename');
  const changed = items.filter(i => i.changed);
  btn.disabled = !changed.length;
  if (!items.length) {
    el.innerHTML = '';
    note.textContent = `已选 ${_renameItems.length} 个视频`;
    return;
  }
  if (!changed.length) {
    const vCount = items.filter(i => i.kind !== 'sub').length;
    note.textContent = hasText ? '没有文件名会变化'
                               : `已选 ${vCount} 个视频`;
  } else {
    const blocked = items.filter(i => i.note && !i.changed).length;
    const unchanged = items.length - changed.length - blocked;
    note.textContent = `共 ${changed.length} 个文件将重命名` +
      (blocked ? `，${blocked} 个不可用` : '') +
      (unchanged ? `，${unchanged} 个无变化` : '');
  }
  let html = '';
  for (const i of items) {
    html += '<div class="rename-pv-row' + (i.changed ? '' : ' rp-dim') +
      (i.kind === 'sub' ? ' rp-sub' : '') + '">' +
      (i.kind === 'sub' ? '<span class="rp-tag">字幕</span>' : '') +
      '<span class="rp-old">' + esc(i.name) +
      '</span><span class="rp-arrow">→</span><span class="rp-new">' + esc(i.new_name) +
      (i.note ? '</span><span class="rp-note">' + esc(i.note) + '</span>' : '</span>') +
      '</div>';
  }
  el.innerHTML = html;
}

async function confirmRename() {
  const { mode, text, text2, useRegex, matchAll, matchSubtitles, caseMode } = _currentRenameParams();
  if (!text && !caseMode) { toast('请输入查找文本', 'err'); return; }
  const btn = $('#btn-confirmrename');
  btn.disabled = true;
  const r = await callApi('manual_rename', _renameItems, mode, text, text2, false, useRegex, matchAll, caseMode, matchSubtitles);
  if (!r) { btn.disabled = false; return; }
  closeRenameDialog();
  if (r.scan && !r.scan.error) loadFromResult(r.scan, true);
  else updateStats();
  const parts = [];
  if (r.renamed > 0) parts.push(`成功 ${r.renamed}`);
  if (r.sub_renamed > 0) parts.push(`字幕 ${r.sub_renamed}`);
  if (r.skipped > 0) parts.push(`${r.skipped} 个无变化`);
  if (r.errors && r.errors.length) parts.push(`失败 ${r.errors.length}：${r.errors[0]}`);
  toast('重命名' + (parts.length ? '：' + parts.join('，') : '完成'),
        r.errors && r.errors.length ? 'err' : 'ok');
}

initDropdown($('#dd-rename-find'), (val, opt) => {
  if (opt && opt.dataset.mode) {
    _renameRegex = opt.dataset.mode === 'regex';
  } else {
    $('#rename-find').value = val;
    _renameRegex = true;
  }
  _syncRegexUI();
  scheduleRenamePreview();
}, true);
initDropdown($('#dd-rename-repl'), (val, opt) => {
  if (opt && opt.dataset.mode) {
    _renameRegex = opt.dataset.mode === 'regex';
  } else {
    $('#rename-repl').value = val;
  }
  _syncRegexUI();
  scheduleRenamePreview();
}, true);

$('#btn-closerename').addEventListener('click', closeRenameDialog);
$('#btn-cancelrename').addEventListener('click', closeRenameDialog);
$('#btn-confirmrename').addEventListener('click', confirmRename);
$('#btn-matchtoggle').addEventListener('click', () => {
  const on = $('#btn-matchtoggle').classList.toggle('on');
  $('#btn-matchtoggle').textContent = on ? '匹配首个' : '匹配全部';
  scheduleRenamePreview();
});
$('#btn-subtoggle').addEventListener('click', async () => {
  const on = $('#btn-subtoggle').classList.toggle('on');
  scheduleRenamePreview();
  if (on && !$('#rename-find').value && !$('#rename-case input:checked').value) {
    const r = await callApi('manual_rename', _renameItems, 'replace', '', '',
                            true, true, true, '', true);
    if (r) toast(`匹配到 ${r.sub_total || 0} 个字幕`, 'ok');
  }
});
$('#renameModal').addEventListener('input', scheduleRenamePreview);
$('#renameModal').addEventListener('change', e => {
  if (e.target.matches('.rename-case input')) _syncPills();
  scheduleRenamePreview();
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && $('#renameModal').classList.contains('show')) closeRenameDialog();
});
