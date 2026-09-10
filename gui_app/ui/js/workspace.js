/* ════════════════════════════════════════════════════════════
   设置 Modal - 缓存清理页（环形总览 + 分类条 + 分类卡片）
   ════════════════════════════════════════════════════════════ */

const CW_CACHES = [
  { key: 'thumb',   title: '缩略图',   unit: '个', desc: '删除后查看视频时重新生成' },
  { key: 'nfo',     title: 'NFO',      unit: '个', desc: '详情页数据源，删除后需重新生成' },
  { key: 'probe',   title: '探针缓存', unit: '个', desc: '时长 / 分辨率 / 编码探测结果，自动重建' },
  { key: 'similar', title: '去重指纹', unit: '条', desc: '「移除重复视频」的比对依据，自动重建' },
];

const CW_RECORDS = [
  { key: 'missing', title: '失效记录',   unit: '条', size: false,
    desc: '对应视频已不在磁盘，一并清理孤立缓存' },
  { key: 'history', title: '已处理记录', unit: '条', size: false,
    desc: '重置后已处理视频重新进入待处理', warn: '影响处理状态' },
];

function cwFmtSize(bytes) {
  if (!bytes || bytes <= 0) return '0 MB';
  const mb = bytes / 1048576;
  if (mb >= 1024) return (mb / 1024).toFixed(2) + ' GB';
  if (mb >= 100) return mb.toFixed(0) + ' MB';
  if (mb >= 1) return mb.toFixed(1) + ' MB';
  return Math.max(mb, 0.01).toFixed(2) + ' MB';
}

function cwFmtCount(n) {
  return n < 0 ? '—' : Number(n || 0).toLocaleString('en-US');
}

function renderWorkspaceTab(stats) {
  const body = $('#modal-body');
  const caches = CW_CACHES.map(c => ({ ...c, count: stats[c.key]?.count ?? 0, bytes: stats[c.key]?.size ?? 0 }));
  const totalBytes = caches.reduce((s, c) => s + c.bytes, 0);
  const historyCount = stats.history_count || 0;
  const missingCount = stats.missing_count || 0;
  const totalItems = historyCount + caches.reduce((s, c) => s + Math.max(c.count, 0), 0);

  const bars = caches.map(c => `
      <div class="cw-bar-row">
        <span class="cw-bar-label">${c.title}</span>
        <span class="cw-bar-track"><i class="cw-bar-fill"
          data-share="${totalBytes ? (c.bytes / totalBytes * 100) : 0}"></i></span>
        <span class="cw-bar-size">${cwFmtSize(c.bytes)}</span>
      </div>`).join('') + `
      <button class="btn primary cw-clear-all" id="btn-clear-all" ${totalItems <= 0 ? 'disabled' : ''}>
        清理全部缓存并重置已处理记录
      </button>`;

  const cards = [...CW_RECORDS.map(r => ({
    ...r,
    count: r.key === 'missing' ? missingCount : historyCount,
    foot: `${cwFmtCount(r.key === 'missing' ? missingCount : historyCount)} ${r.unit}，—`,
  })), ...caches.map(c => ({
    key: c.key, title: c.title, desc: c.desc, unit: c.unit,
    count: c.count,
    foot: `${cwFmtCount(c.count)} ${c.unit} · ${cwFmtSize(c.bytes)}`,
  }))].map(c => `
      <div class="cw-card" data-key="${c.key}">
        <div class="cw-card-head">
          <span class="cw-card-title">${c.title}</span>
          <button class="ws-btn cw-clear" data-key="${c.key}" ${c.count <= 0 ? 'disabled' : ''}>清理</button>
        </div>
        <div class="cw-card-desc">${c.desc}${c.warn ? ` <span class="cw-warn">「${c.warn}」</span>` : ''}</div>
        <div class="cw-card-foot">${c.foot}</div>
      </div>`).join('');

  body.innerHTML = `
    <div class="cw-page">
      <div class="cw-grid">${cards}</div>
      <div class="cw-hero">
        <div class="cw-bars">${bars}</div>
      </div>
    </div>`;

  requestAnimationFrame(() => requestAnimationFrame(() => {
    body.querySelectorAll('.cw-bar-fill').forEach(el => { el.style.width = el.dataset.share + '%'; });
  }));

  async function workspaceAction(apiMethod, args, confirmMsg, successToast, failMsg, afterFn) {
    if (confirmMsg && !await showConfirm(confirmMsg)) return;
    const r = await callApi(apiMethod, ...args);
    if (!r) return;
    if (!r.ok) { toast(failMsg || '操作失败', 'err'); return; }
    toast(typeof successToast === 'function' ? successToast(r) : successToast, 'ok');
    const stats2 = await callApi('get_workspace_stats');
    if (stats2) renderWorkspaceTab(stats2);
    if (afterFn) await afterFn();
  }

  const clearHandler = {
    missing: () => workspaceAction('prune_history', [], null,
      r => `已清理 ${r.removed} 条失效记录、${r.thumbs || 0} 个缩略图、${r.nfos || 0} 个 NFO`),
    history: () => workspaceAction('clear_workspace', [true, false, false, false, false, false],
      `确定重置全部已处理记录吗？\n\n${historyCount} 条已处理记录将被移除，对应视频重新进入待处理列表。\n不会删除视频文件与任何缓存。`,
      '已重置已处理记录'),
    thumb: () => workspaceAction('clear_workspace', [false, true, false, false, false, false],
      `确定清理 ${cwFmtCount(caches[0].count)} 个缩略图缓存吗？\n\n下次查看对应视频时会重新生成缩略图。`,
      r => `已清理 ${r.cleared.thumbs} 个缩略图`),
    nfo: () => workspaceAction('clear_workspace', [false, false, true, false, false, false],
      `确定清理 ${cwFmtCount(caches[1].count)} 个 NFO 缓存吗？\n\n「自动输出 NFO 至目录」关闭时，详情页依赖此缓存。`,
      r => `已清理 ${r.cleared.nfo} 个 NFO`),
    probe: () => workspaceAction('clear_workspace', [false, false, false, false, true, false],
      '确定清理探针缓存吗？\n\n视频的时长 / 分辨率 / 编码等信息下次使用时会自动重新探测。',
      '已清理探针缓存'),
    similar: () => workspaceAction('clear_workspace', [false, false, false, false, false, true],
      '确定清理去重指纹缓存吗？\n\n下次使用「移除重复视频」时会自动重建指纹。',
      '已清理去重指纹缓存'),
  };
  body.querySelectorAll('.cw-clear').forEach(btn => {
    btn.onclick = () => (clearHandler[btn.dataset.key] || (() => {}))();
  });

  $('#btn-clear-all').onclick = () => workspaceAction('clear_workspace', [true, true, true, false, true, true],
    '确定清除全部缓存吗？\n\n将删除：\n• 所有已处理记录（已处理视频会重新进入待处理）\n• 所有缩略图缓存\n• 所有 NFO 缓存\n• 视频探针缓存与去重指纹缓存（下次使用时自动重建）\n\n不会删除任何视频文件，也不会清空已添加的源。',
    r => `已清除：${r.cleared.history ? '已处理记录、' : ''}${r.cleared.thumbs} 个缩略图、${r.cleared.nfo} 个 NFO${r.cleared.probe ? '、探针缓存' : ''}${r.cleared.similar ? '、去重缓存' : ''}`,
    '清除失败',
    async () => { const sr = await callApi('scan'); if (sr) loadFromResult(sr); });
}
