/* ════════════════════════════════════════════════════════════
   主题切换
   ════════════════════════════════════════════════════════════ */
(function() {
  const el = $('#themeToggle');
  if (el) el.addEventListener('click', toggleTheme);
  const mq = window.matchMedia('(prefers-color-scheme:dark)');
  mq.addEventListener('change', () => {
    if (_themeMode === '') syncTitlebar('');
  });
})();

/* ════════════════════════════════════════════════════════════
   禁止拖拽导航
   ════════════════════════════════════════════════════════════ */
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
  window.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); }, false);
});

$('#conn-pill').addEventListener('click', () => openSettings('ai'));
$$('.stats .pill[data-view]').forEach(p => p.addEventListener('click', () => switchView(p.dataset.view)));
$('#searchMode .dd-btn').insertAdjacentHTML('beforeend', ddArrow());

/* ════════════════════════════════════════════════════════════
   启动
   ════════════════════════════════════════════════════════════ */
renderGrid();
updateStats();

pywebviewReady(30000).then(async () => {
  setLoadHint('正在加载配置…');
  try {
    const cfg = await apiCall('get_config');
    if (cfg && cfg.theme) applyTheme(cfg.theme);
    if (cfg) applyForceAnimation(cfg.force_animation !== false);
    const exp = (cfg && cfg.experimental) || {};
    state.thumbOptimize = Number(((cfg || {}).video || {}).frame_time_tags) === 2;
    // 各 id 集合互相独立，并行拉取避免逐个串行等待
    const idFetches = [];
    if (exp.pixai_tagger_enabled) {
      state.pixaiTaggerEnabled = true;
      idFetches.push(
        apiCall('get_pixai_tagged_ids').then(ids => { if (ids) state.pixaiTaggedIds = new Set(ids); }).catch(() => {}),
        apiCall('get_pixai_real_ids').then(ids => { if (ids) state.pixaiRealIds = new Set(ids); }).catch(() => {}),
        apiCall('get_pixai_anime_ids').then(ids => { if (ids) state.pixaiAnimeIds = new Set(ids); }).catch(() => {}),
        apiCall('get_pixai_uncertain_ids').then(ids => { if (ids) state.pixaiUncertainIds = new Set(ids); }).catch(() => {}),
      );
    }
    if (exp.whisper_enabled) {
      state.whisperEnabled = true;
      idFetches.push(
        apiCall('get_whisper_transcribed_ids').then(ids => { if (ids) state.whisperTranscribedIds = new Set(ids); }).catch(() => {}),
      );
    }
    if (idFetches.length) await Promise.all(idFetches);
  } catch (e) { console.warn('[启动] 配置加载失败，使用默认主题', e); }

  setLoadHint('正在检测服务…');
  initConnection().catch(() => {});

  setLoadHint('正在扫描源…');
  try {
    const r = await apiCall('scan');
    if (r && r.error) {
      toast('扫描出错: ' + r.error, 'err');
    } else if (r) {
      loadFromResult(r, true);
    }
  } catch (e) {
    toast('扫描失败: ' + ((e && e.message) || e), 'err');
  }
  hideAppLoading();
}).catch(e => {
  hideAppLoading();
  toast('启动失败: ' + ((e && e.message) || e), 'err');
});
