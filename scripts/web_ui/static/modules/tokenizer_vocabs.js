import { api, fetchJson } from './api.js';
import { els } from './dom.js';

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

let logStream = null;

function setVocabDetails(payload, label) {
  const raw = payload?.raw || JSON.stringify(payload ?? {}, null, 2);
  els.tokenizerVocabDetails.value = raw || '';
  els.tokenizerVocabDetailsMeta.textContent = label || '';
}

function appendLogLine(line) {
  const next = `${els.tokenizerVocabLogs.value}${els.tokenizerVocabLogs.value ? '\n' : ''}${line}`;
  const maxChars = 20000;
  els.tokenizerVocabLogs.value = next.length > maxChars ? next.slice(-maxChars) : next;
  els.tokenizerVocabLogs.scrollTop = els.tokenizerVocabLogs.scrollHeight;
}

function stopLogStream() {
  if (logStream) {
    logStream.close();
    logStream = null;
  }
}

function startLogStream(runDir) {
  if (!runDir) return;
  stopLogStream();
  els.tokenizerVocabLogs.value = '';
  els.tokenizerVocabLogsMeta.textContent = `streaming logs for ${runDir}`;
  logStream = new EventSource(`/api/run/stream?run_dir=${encodeURIComponent(runDir)}&kind=process`);
  logStream.onmessage = (event) => {
    if (event.data) appendLogLine(event.data);
  };
  logStream.onerror = () => {
    els.tokenizerVocabLogsMeta.textContent = 'log stream disconnected';
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
  await loadVocabList();
}

async function loadVocabList() {
  const data = await fetchJson('/api/tokenizer_vocabs');
  const items = data.vocabs || [];
  els.tokenizerVocabList.innerHTML = '';
  setVocabDetails({ raw: '' }, '');
  if (!items.length) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'No vocabularies yet.';
    els.tokenizerVocabList.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'artifact-card';
    const title = document.createElement('div');
    title.className = 'artifact-title';
    title.textContent = item.name ? `${item.name} v${item.version ?? '-'}` : item.run_id;
    const datasetTitle = document.createElement('div');
    datasetTitle.className = 'meta';
    const datasetLabel = formatDatasetTitle(item);
    datasetTitle.textContent = datasetLabel ? `dataset ${datasetLabel}` : formatDatasetLabel(item.dataset_manifest);
    const vocabMeta = document.createElement('div');
    vocabMeta.className = 'meta';
    vocabMeta.textContent = `vocab ${item.vocab_size ?? '-'}`;
    const tokenMeta = document.createElement('div');
    tokenMeta.className = 'meta';
    tokenMeta.textContent = `tokens ${item.token_count ?? '-'}`;
    const dateMeta = document.createElement('div');
    dateMeta.className = 'meta';
    dateMeta.textContent = formatDate(item.created_at);
    const inputMeta = document.createElement('div');
    inputMeta.className = 'meta';
    inputMeta.textContent = item.input_bytes != null ? `input ${formatBytes(item.input_bytes)}` : 'input -';
    const runMeta = document.createElement('div');
    runMeta.className = 'meta';
    runMeta.textContent = formatRunLabel(item.run_dir);
    const actions = document.createElement('div');
    actions.className = 'artifact-actions';
    const viewBtn = document.createElement('button');
    viewBtn.textContent = 'View';
    viewBtn.addEventListener('click', () => viewVocab(item.run_dir));
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = 'Delete';
    deleteBtn.addEventListener('click', () => deleteVocab(item.run_dir));
    actions.appendChild(viewBtn);
    actions.appendChild(deleteBtn);
    row.appendChild(title);
    row.appendChild(datasetTitle);
    row.appendChild(vocabMeta);
    row.appendChild(tokenMeta);
    row.appendChild(dateMeta);
    row.appendChild(inputMeta);
    row.appendChild(runMeta);
    row.appendChild(actions);
    els.tokenizerVocabList.appendChild(row);
  });
}

async function loadConfigs() {
  const data = await api('/api/configs/meta', { scope: 'tokenizer' });
  const items = data.configs || [];
  els.tokenizerVocabConfigSelect.innerHTML = '';
  items.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.path;
    const name = item.name || item.path;
    const version = item.version != null ? `v${item.version}` : 'v?';
    const vocab = item.vocab_size != null ? `vocab ${item.vocab_size}` : '';
    opt.textContent = [name, version, vocab].filter(Boolean).join(' • ');
    els.tokenizerVocabConfigSelect.appendChild(opt);
  });
  if (!items.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No tokenizer configs found';
    els.tokenizerVocabConfigSelect.appendChild(opt);
  }
}

async function loadDatasets() {
  const data = await fetchJson('/api/datasets');
  const items = data.datasets || [];
  els.tokenizerVocabDatasetSelect.innerHTML = '';
  items.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.manifest_path;
    const name = item.name ? `${item.name} (${item.id})` : item.id;
    opt.textContent = `${name} • files ${item.file_count ?? '-'}`;
    els.tokenizerVocabDatasetSelect.appendChild(opt);
  });
  if (!items.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No dataset manifests found';
    els.tokenizerVocabDatasetSelect.appendChild(opt);
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
}

export function initTokenizerVocabs() {
  els.tokenizerVocabRefreshBtn.addEventListener('click', loadVocabList);
  els.tokenizerVocabRefreshConfigsBtn.addEventListener('click', loadConfigs);
  els.tokenizerVocabRefreshDatasetsBtn.addEventListener('click', loadDatasets);
  els.tokenizerVocabCreateBtn.addEventListener('click', createVocab);
  loadVocabList();
  loadConfigs();
  loadDatasets();
}
