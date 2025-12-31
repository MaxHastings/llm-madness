import { api, fetchJson } from './api.js';
import { els } from './dom.js';
import { loadTokenizerVocabs, setTokenizerVocabSelection } from './pipeline.js';

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

async function loadVocabList() {
  const data = await fetchJson('/api/tokenizer_vocabs');
  const items = data.vocabs || [];
  els.tokenizerVocabList.innerHTML = '';
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
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = `vocab ${item.vocab_size ?? '-'} • tokens ${item.token_count ?? '-'} • ${formatDate(item.created_at)}`;
    const details = document.createElement('div');
    details.className = 'meta';
    const sizeLabel = item.input_bytes != null ? `input ${formatBytes(item.input_bytes)}` : 'input -';
    details.textContent = `${sizeLabel} • ${item.run_dir}`;
    const actions = document.createElement('div');
    actions.className = 'artifact-actions';
    const useBtn = document.createElement('button');
    useBtn.textContent = 'Use in Run';
    useBtn.addEventListener('click', async () => {
      if (!item.tokenizer_path) return;
      await loadTokenizerVocabs();
      setTokenizerVocabSelection(item.tokenizer_path);
      const navBtn = document.querySelector('[data-section="runs"]');
      if (navBtn) navBtn.click();
    });
    actions.appendChild(useBtn);
    row.appendChild(title);
    row.appendChild(meta);
    row.appendChild(details);
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
  });
  if (res.run_id) {
    els.tokenizerVocabMeta.textContent = `started tokenizer run ${res.run_id}`;
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
