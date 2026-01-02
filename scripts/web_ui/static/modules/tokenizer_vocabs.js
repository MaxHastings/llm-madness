import { api, fetchJson } from './api.js';
import { els } from './dom.js';
import { isSectionActive, scheduleAutoRefresh } from './auto_refresh.js';
import { emitEvent, onEvent } from './events.js';

function formatDate(value) {
  if (!value) return '-';
  return value.replace('T', ' ');
}

function formatBytes(bytes) {
  if (bytes == null) return '-';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let idx = 0;
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }
  return `${value.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`;
}

function formatDatasetLabel(path) {
  if (!path) return 'dataset -';
  const parts = path.split('/').filter(Boolean);
  const idx = parts.lastIndexOf('datasets');
  if (idx !== -1 && parts[idx + 1]) {
    return `dataset ${parts[idx + 1]}`;
  }
  const file = parts[parts.length - 1] || path;
  return `dataset ${file}`;
}

function formatDatasetTitle(item) {
  if (!item) return null;
  const name = item.dataset_name;
  const id = item.dataset_id;
  if (name && id) return `${name} (${id})`;
  if (name) return name;
  if (id) return id;
  if (item.dataset_manifest) {
    return formatDatasetLabel(item.dataset_manifest).replace(/^dataset\s+/i, '');
  }
  return null;
}

function formatRunLabel(runDir) {
  if (!runDir) return 'run -';
  const parts = runDir.split('/').filter(Boolean);
  return `run ${parts[parts.length - 1] || runDir}`;
}

function formatApiError(err, fallback) {
  if (!err) return fallback;
  const message = err.message || String(err);
  try {
    const parsed = JSON.parse(message);
    if (parsed && typeof parsed === 'object') {
      return parsed.error || parsed.message || fallback || message;
    }
  } catch (parseError) {
    // Not JSON, use the raw message.
  }
  return message || fallback;
}

let logStream = null;
let selectedRunDir = null;
let activeVocabTab = 'report';
let tokenSearchTimer = null;
let progressLines = [];
let progressTimer = null;
let progressStart = null;

function setVocabDetails(payload, label) {
  if (payload && Object.prototype.hasOwnProperty.call(payload, 'raw')) {
    els.tokenizerVocabDetails.value = payload.raw || '';
    els.tokenizerVocabConfig.value = '';
    els.tokenizerVocabDetailsMeta.textContent = label || '';
    els.tokenizerVocabConfigMeta.textContent = '';
    return;
  }
  const report = payload?.report ?? payload ?? {};
  const manifest = payload?.manifest ?? null;
  const view = manifest ? { report, manifest } : report;
  const config = payload?.config ?? {};
  els.tokenizerVocabDetails.value = JSON.stringify(view, null, 2);
  els.tokenizerVocabConfig.value = JSON.stringify(config, null, 2);
  els.tokenizerVocabDetailsMeta.textContent = label || '';
  els.tokenizerVocabConfigMeta.textContent = Object.keys(config || {}).length ? 'config loaded' : '';
}

function setVocabTab(tab) {
  const previewRoot = els.tokenizerVocabTabSelect?.closest('.vocab-preview');
  const panels = previewRoot ? previewRoot.querySelectorAll('.preview-panel') : [];
  activeVocabTab = tab;
  if (els.tokenizerVocabTabSelect) {
    els.tokenizerVocabTabSelect.value = tab;
  }
  panels.forEach((panel) => {
    panel.classList.toggle('is-hidden', panel.dataset.preview !== tab);
  });
  if (tab === 'tokens') {
    loadVocabTokens();
  }
}

function updateProgressLine(line) {
  if (!els.tokenizerVocabProgressLine) return;
  if (line) {
    progressLines.push(line);
    if (progressLines.length > 6) {
      progressLines = progressLines.slice(-6);
    }
  }
  els.tokenizerVocabProgressLine.textContent = progressLines.join('\n');
  if (line && line.includes('[tokenizer] run complete')) {
    setProgressMeta('generation complete');
    stopLogStream();
  }
}

function formatElapsed(startMs) {
  if (!startMs) return '';
  const delta = Math.max(0, Math.floor((Date.now() - startMs) / 1000));
  const minutes = Math.floor(delta / 60);
  const seconds = delta % 60;
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function setProgressMeta(label) {
  if (!els.tokenizerVocabProgressMeta) return;
  els.tokenizerVocabProgressMeta.textContent = label || '';
}

function stopLogStream() {
  if (logStream) {
    logStream.close();
    logStream = null;
  }
  if (progressTimer) {
    window.clearInterval(progressTimer);
    progressTimer = null;
  }
}

function startLogStream(runDir) {
  if (!runDir) return;
  stopLogStream();
  progressLines = [];
  progressStart = Date.now();
  setProgressMeta(`generating vocab • ${runDir}`);
  updateProgressLine('[tokenizer] waiting for output...');
  progressTimer = window.setInterval(() => {
    setProgressMeta(`generating vocab • ${runDir} • ${formatElapsed(progressStart)}`);
  }, 1000);
  logStream = new EventSource(`/api/run/stream?run_dir=${encodeURIComponent(runDir)}&kind=process`);
  logStream.onmessage = (event) => {
    if (event.data) updateProgressLine(event.data);
  };
  logStream.onerror = () => {
    setProgressMeta('generation stream disconnected');
    stopLogStream();
  };
}

async function viewVocab(runDir) {
  if (!runDir) return;
  try {
    const data = await fetchJson(`/api/tokenizer_vocabs/report?run_dir=${encodeURIComponent(runDir)}`);
    const payload = {
      run_dir: data.run_dir,
      manifest: data.manifest,
      config: data.config,
      report: data.report,
    };
    const datasetManifest = data?.manifest?.inputs?.dataset_manifest;
    const label = datasetManifest ? `loaded ${runDir} • dataset ${datasetManifest}` : `loaded ${runDir}`;
    setVocabDetails(payload, label);
    selectedRunDir = runDir;
    renderSelectedRun();
    if (activeVocabTab === 'tokens') {
      loadVocabTokens();
    }
  } catch (err) {
    setVocabDetails({ raw: '' }, `failed to load vocab: ${err.message}`);
  }
}

async function deleteVocab(runDir) {
  if (!runDir) return;
  const ok = window.confirm(`Delete tokenizer vocab ${runDir}? This cannot be undone.`);
  if (!ok) return;
  await api('/api/run/delete', { run_dir: runDir });
  setVocabDetails({ raw: '' }, 'vocab deleted');
  if (selectedRunDir === runDir) {
    selectedRunDir = null;
    renderSelectedRun();
  }
  await loadVocabList();
  emitEvent('tokenizer_vocabs:changed');
}

function renderSelectedRun() {
  const rows = els.tokenizerVocabList.querySelectorAll('.vocab-card');
  rows.forEach((row) => {
    row.classList.toggle('selected', row.dataset.runDir === selectedRunDir);
  });
}

function renderVocabTokens(tokens) {
  els.tokenizerVocabTokens.innerHTML = '';
  if (!tokens || !tokens.length) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'No tokens to show.';
    els.tokenizerVocabTokens.appendChild(empty);
    return;
  }
  tokens.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'token-row';
    const id = document.createElement('span');
    id.className = 'token-id';
    id.textContent = item.id;
    const token = document.createElement('span');
    token.className = 'token-value';
    token.textContent = item.token;
    row.appendChild(id);
    row.appendChild(token);
    els.tokenizerVocabTokens.appendChild(row);
  });
}

async function loadVocabTokens() {
  if (!selectedRunDir) {
    els.tokenizerVocabTokenMeta.textContent = 'Select a vocab run to view tokens.';
    els.tokenizerVocabTokens.innerHTML = '';
    return;
  }
  const query = (els.tokenizerVocabTokenSearch.value || '').trim();
  const limitRaw = parseInt(els.tokenizerVocabTokenLimit.value, 10);
  const limit = Number.isFinite(limitRaw) ? Math.min(Math.max(limitRaw, 10), 2000) : 200;
  els.tokenizerVocabTokenLimit.value = limit;
  try {
    const data = await fetchJson(
      `/api/tokenizer_vocabs/vocab?run_dir=${encodeURIComponent(selectedRunDir)}&q=${encodeURIComponent(query)}&limit=${limit}`
    );
    renderVocabTokens(data.tokens || []);
    const total = data.total ?? 0;
    const shown = data.shown ?? (data.tokens || []).length;
    els.tokenizerVocabTokenMeta.textContent = query
      ? `showing ${shown} of ${total} tokens matching "${query}"`
      : `showing ${shown} of ${total} tokens`;
  } catch (err) {
    els.tokenizerVocabTokenMeta.textContent = `failed to load tokens: ${err.message}`;
  }
}

async function loadVocabList({ preserveSelection = false } = {}) {
  const priorSelection = selectedRunDir;
  const data = await fetchJson('/api/tokenizer_vocabs');
  const items = data.vocabs || [];
  els.tokenizerVocabList.innerHTML = '';
  if (!preserveSelection) {
    setVocabDetails({ raw: '' }, '');
    selectedRunDir = null;
  }
  if (!items.length) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'No vocabularies yet.';
    els.tokenizerVocabList.appendChild(empty);
    if (preserveSelection && priorSelection) {
      setVocabDetails({ raw: '' }, 'selected vocab no longer available');
      selectedRunDir = null;
    }
    return;
  }
  let selectionStillExists = false;
  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'artifact-card selectable-card vocab-card';
    row.dataset.runDir = item.run_dir;
    row.addEventListener('click', () => viewVocab(item.run_dir));
    const title = document.createElement('div');
    title.className = 'artifact-title';
    title.textContent = item.name ? `${item.name} v${item.version ?? '-'}` : item.run_id;
    const datasetTitle = document.createElement('div');
    datasetTitle.className = 'meta';
    const datasetLabel = formatDatasetTitle(item);
    datasetTitle.textContent = datasetLabel ? `dataset ${datasetLabel}` : formatDatasetLabel(item.dataset_manifest);
    const summary = document.createElement('div');
    summary.className = 'vocab-summary';
    const stats = [
      `vocab ${item.vocab_size ?? '-'}`,
      `tokens ${item.token_count ?? '-'}`,
      item.input_bytes != null ? `input ${formatBytes(item.input_bytes)}` : 'input -',
      formatDate(item.created_at),
    ];
    summary.textContent = stats.join(' • ');
    const runMeta = document.createElement('div');
    runMeta.className = 'meta';
    runMeta.textContent = formatRunLabel(item.run_dir);
    const actions = document.createElement('div');
    actions.className = 'artifact-actions';
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = 'Delete';
    deleteBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      deleteVocab(item.run_dir);
    });
    actions.appendChild(deleteBtn);
    row.appendChild(title);
    row.appendChild(datasetTitle);
    row.appendChild(summary);
    row.appendChild(runMeta);
    row.appendChild(actions);
    els.tokenizerVocabList.appendChild(row);
    if (priorSelection && item.run_dir === priorSelection) {
      selectionStillExists = true;
    }
  });
  if (preserveSelection) {
    if (selectionStillExists) {
      selectedRunDir = priorSelection;
    } else if (priorSelection) {
      setVocabDetails({ raw: '' }, 'selected vocab no longer available');
      selectedRunDir = null;
    }
  }
  renderSelectedRun();
}

async function loadConfigs() {
  const priorSelection = els.tokenizerVocabConfigSelect.value;
  const data = await api('/api/configs/meta', { scope: 'tokenizer' });
  const items = data.configs || [];
  els.tokenizerVocabConfigSelect.innerHTML = '';
  const available = new Set();
  items.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.path;
    const name = item.name || item.path;
    const version = item.version != null ? `v${item.version}` : 'v?';
    const vocab = item.vocab_size != null ? `vocab ${item.vocab_size}` : '';
    opt.textContent = [name, version, vocab].filter(Boolean).join(' • ');
    els.tokenizerVocabConfigSelect.appendChild(opt);
    available.add(opt.value);
  });
  if (!items.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No tokenizer configs found';
    els.tokenizerVocabConfigSelect.appendChild(opt);
  } else if (priorSelection && available.has(priorSelection)) {
    els.tokenizerVocabConfigSelect.value = priorSelection;
  }
}

async function loadDatasets() {
  const priorSelection = els.tokenizerVocabDatasetSelect.value;
  const data = await fetchJson('/api/datasets');
  const items = data.datasets || [];
  els.tokenizerVocabDatasetSelect.innerHTML = '';
  const available = new Set();
  items.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.manifest_path;
    const name = item.name ? `${item.name} (${item.id})` : item.id;
    opt.textContent = `${name} • files ${item.file_count ?? '-'}`;
    els.tokenizerVocabDatasetSelect.appendChild(opt);
    available.add(opt.value);
  });
  if (!items.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No dataset manifests found';
    els.tokenizerVocabDatasetSelect.appendChild(opt);
  } else if (priorSelection && available.has(priorSelection)) {
    els.tokenizerVocabDatasetSelect.value = priorSelection;
  }
}

async function createVocab() {
  const config = els.tokenizerVocabConfigSelect.value;
  const datasetManifest = els.tokenizerVocabDatasetSelect.value;
  const runName = (els.tokenizerVocabName.value || '').trim();
  if (!config) {
    els.tokenizerVocabMeta.textContent = 'Select a tokenizer config.';
    return;
  }
  if (!datasetManifest) {
    els.tokenizerVocabMeta.textContent = 'Select a dataset manifest.';
    return;
  }
  try {
    const res = await api('/api/run', {
      stage: 'tokenizer',
      config,
      dataset_manifest: datasetManifest,
      run_name: runName || null,
    });
    if (res.run_id) {
      els.tokenizerVocabMeta.textContent = `started tokenizer run ${res.run_id}`;
      startLogStream(res.run_dir);
    } else {
      els.tokenizerVocabMeta.textContent = res.error || 'run failed';
    }
    await loadVocabList();
    emitEvent('tokenizer_vocabs:changed');
  } catch (err) {
    els.tokenizerVocabMeta.textContent = formatApiError(err, 'run failed');
  }
}

export function initTokenizerVocabs() {
  els.tokenizerVocabRefreshBtn.addEventListener('click', loadVocabList);
  if (els.tokenizerVocabRefreshListBtn) {
    els.tokenizerVocabRefreshListBtn.addEventListener('click', loadVocabList);
  }
  els.tokenizerVocabRefreshConfigsBtn.addEventListener('click', loadConfigs);
  els.tokenizerVocabRefreshDatasetsBtn.addEventListener('click', loadDatasets);
  els.tokenizerVocabCreateBtn.addEventListener('click', createVocab);
  if (els.tokenizerVocabTabSelect) {
    els.tokenizerVocabTabSelect.addEventListener('change', (event) => {
      setVocabTab(event.target.value);
    });
  }
  if (els.tokenizerVocabTokenRefreshBtn) {
    els.tokenizerVocabTokenRefreshBtn.addEventListener('click', loadVocabTokens);
  }
  if (els.tokenizerVocabTokenSearch) {
    els.tokenizerVocabTokenSearch.addEventListener('input', () => {
      if (tokenSearchTimer) window.clearTimeout(tokenSearchTimer);
      tokenSearchTimer = window.setTimeout(loadVocabTokens, 250);
    });
  }
  loadVocabList();
  loadConfigs();
  loadDatasets();
  setVocabTab('report');
  onEvent('datasets:changed', loadDatasets);
  onEvent('tokenizer_configs:changed', loadConfigs);
  scheduleAutoRefresh({
    intervalMs: 30000,
    isEnabled: () => isSectionActive('tokenizer-vocabs'),
    task: () => loadVocabList({ preserveSelection: true }),
  });
  scheduleAutoRefresh({
    intervalMs: 30000,
    isEnabled: () => isSectionActive('tokenizer-vocabs'),
    task: loadConfigs,
  });
  scheduleAutoRefresh({
    intervalMs: 30000,
    isEnabled: () => isSectionActive('tokenizer-vocabs'),
    task: loadDatasets,
  });
}
