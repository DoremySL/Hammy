/* ════════════════════════════════════════════════════════════
   设置 Modal - 工作区页
   ════════════════════════════════════════════════════════════ */
function renderWorkspaceTab(stats) {
  const body = $('#modal-body');
  const totalCacheMb = (stats.thumb_size_mb || 0) + (stats.nfo_size_mb || 0);
  const missingHint = stats.reconcile.missing > 0
    ? `（其中 ${stats.reconcile.missing} 条记录的视频已不在磁盘上）`
    : '';
  body.innerHTML = `
    <div class="ws-page">
      <div class="group" style="text-align:center">
        <div class="stat-cards">
          <div class="stat-card"><div class="num">${stats.history_count}</div><div class="lbl">已处理记录</div></div>
          <div class="stat-card"><div class="num">${totalCacheMb.toFixed(1)} MB</div><div class="lbl">缓存总占用</div></div>
          <div class="stat-card"><div class="num">${stats.thumb_count}</div><div class="lbl">缩略图（${stats.thumb_size_mb} MB）</div></div>
          <div class="stat-card"><div class="num">${stats.nfo_count}</div><div class="lbl">NFO（${stats.nfo_size_mb} MB）</div></div>
        </div>
        ${missingHint ? `<div class="help" style="font-size:11px">${icon('warning')} ${stats.reconcile.missing} 条记录对应的视频已不在磁盘上</div>` : ''}
      </div>
      <div class="group ws-group">
        <div class="ws-center">
          <div class="ws-actions">
            <button class="ws-btn" id="btn-prune-history">清理失效记录 (${stats.reconcile.missing})</button>
            <button class="ws-btn" id="btn-clear-thumbs">清理缩略图 (${stats.thumb_count})</button>
            <button class="ws-btn" id="btn-clear-nfo">清理 NFO (${stats.nfo_count})</button>
            <button class="ws-btn danger" id="btn-clear-cache">清除全部</button>
          </div>
          <div class="help" style="margin-top:10px;font-size:11px;text-align:center">
            仅删除缓存数据，<strong>不影响视频文件</strong>。「清除全部」会重置已处理状态。
          </div>
        </div>
      </div>
    </div>`;
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

  $('#btn-prune-history').onclick = () => workspaceAction('prune_history', [], null,
    r => `已清理 ${r.removed} 条失效记录、${r.thumbs || 0} 个缩略图、${r.nfos || 0} 个 NFO`);
  $('#btn-clear-thumbs').onclick = () => workspaceAction('clear_workspace', [false, true, false, false],
    `确定清理 ${stats.thumb_count} 个缩略图缓存吗？\n\n下次查看对应视频时会重新生成缩略图。`,
    r => `已清理 ${r.cleared.thumbs} 个缩略图`);
  $('#btn-clear-nfo').onclick = () => workspaceAction('clear_workspace', [false, false, true, false],
    `确定清理 ${stats.nfo_count} 个 NFO 缓存吗？\n\n「自动输出 NFO 至视频目录」关闭时，详情页依赖此缓存。`,
    r => `已清理 ${r.cleared.nfo} 个 NFO`);
  $('#btn-clear-cache').onclick = () => workspaceAction('clear_workspace', [true, true, true, false],
    '确定清除全部缓存吗？\n\n将删除：\n• 所有已处理记录（已处理视频会重新进入待处理）\n• 所有缩略图缓存\n• 所有 NFO 缓存\n\n不会删除任何视频文件，也不会清空已添加的源。',
    r => `已清除：${r.cleared.history ? '已处理记录、' : ''}${r.cleared.thumbs} 个缩略图、${r.cleared.nfo} 个 NFO`,
    '清除失败',
    async () => { const sr = await callApi('scan'); if (sr) loadFromResult(sr); });
}
