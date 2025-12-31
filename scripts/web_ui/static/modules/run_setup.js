import { api, fetchJson } from './api.js';
import { els } from './dom.js';
import { refreshRunList } from './runs.js';

function formatDatasetLabel(item) {
  if (!item) return '';
  const name = item.name ? `${item.name} (${item.id})` : item.id;
  const count = item.file_count != null ? `files ${item.file_count}` : 'files ?';
  return `${name} — ${count}`;
}

function formatTrainingSummary(item) {
  if (!item) return '';
  const parts = [];
  if (item.n_layer != null) parts.push(`${item.n_layer}L`);
  if (item.n_head != null) parts.push(`${item.n_head}H`);
  if (item.n_embd != null) parts.push(`${item.n_embd}D`);
  if (item.block_size != null) parts.push(`B${item.block_size}`);
  return parts.length ? parts.join(' ') : '';
}

export async function loadDatasets() {
  const data = await fetchJson('/api/datasets');
  const items = data.datasets || [];
  els.datasetSelect.innerHTML = '';
  const none = document.createElement('option');
  none.value = '';
  none.textContent = 'Select dataset';
  els.datasetSelect.appendChild(none);
  items.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.manifest_path;
    opt.textContent = formatDatasetLabel(item);
    els.datasetSelect.appendChild(opt);
  });
}

export function setDatasetSelection(path) {
  if (!path) return;
  const exists = Array.from(els.datasetSelect.options).some((opt) => opt.value === path);
  if (!exists) {
    const opt = document.createElement('option');
    opt.value = path;
    opt.textContent = path.split('/').slice(-2).join('/');
    els.datasetSelect.appendChild(opt);
  }
  els.datasetSelect.value = path;
}

export async function loadTokenizerVocabs() {
  const data = await fetchJson('/api/tokenizer_vocabs');
  const items = data.vocabs || [];
  els.runTokenizerVocabSelect.innerHTML = '';
  const none = document.createElement('option');
  none.value = '';
  none.textContent = 'Select tokenizer vocab';
  els.runTokenizerVocabSelect.appendChild(none);
  items.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.tokenizer_path || '';
    const name = item.name ? `${item.name} v${item.version ?? '-'}` : item.run_id;
    const vocab = item.vocab_size != null ? `vocab ${item.vocab_size}` : '';
    opt.textContent = [name, vocab].filter(Boolean).join(' • ');
    els.runTokenizerVocabSelect.appendChild(opt);
  });
}

export function setTokenizerVocabSelection(path) {
  if (!path) return;
  const exists = Array.from(els.runTokenizerVocabSelect.options).some((opt) => opt.value === path);
  if (!exists) {
    const opt = document.createElement('option');
    opt.value = path;
    opt.textContent = path.split('/').slice(-2).join('/');
    els.runTokenizerVocabSelect.appendChild(opt);
  }
  els.runTokenizerVocabSelect.value = path;
}

export async function loadTrainingConfigs() {
  const data = await api('/api/configs/meta', { scope: 'training' });
  const items = data.configs || [];
  els.trainingConfigSelect.innerHTML = '';
  const none = document.createElement('option');
  none.value = '';
  none.textContent = 'Select training config';
  els.trainingConfigSelect.appendChild(none);
  items.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.path;
    const name = item.name || item.path;
    const version = item.version != null ? `v${item.version}` : 'v?';
    const summary = formatTrainingSummary(item);
    opt.textContent = [name, version, summary].filter(Boolean).join(' • ');
    els.trainingConfigSelect.appendChild(opt);
  });
}

export function setTrainingConfigSelection(path) {
  if (!path) return;
  const exists = Array.from(els.trainingConfigSelect.options).some((opt) => opt.value === path);
  if (!exists) {
    const opt = document.createElement('option');
    opt.value = path;
    opt.textContent = path.split('/').slice(-2).join('/');
    els.trainingConfigSelect.appendChild(opt);
  }
  els.trainingConfigSelect.value = path;
}

async function createTrainRun() {
  const datasetManifest = els.datasetSelect.value;
  const tokenizerPath = els.runTokenizerVocabSelect.value;
  const trainingConfig = els.trainingConfigSelect.value;
  if (!datasetManifest) {
    els.runSetupMeta.textContent = 'Select a dataset.';
    return;
  }
  if (!tokenizerPath) {
    els.runSetupMeta.textContent = 'Select a tokenizer vocab.';
    return;
  }
  if (!trainingConfig) {
    els.runSetupMeta.textContent = 'Select a training config.';
    return;
  }
  const payload = {
    stage: 'train',
    config: trainingConfig,
    dataset_manifest: datasetManifest,
    tokenizer_path: tokenizerPath,
  };
  const data = await api('/api/run', payload);
  els.runSetupMeta.textContent = data.run_id ? `started train (${data.run_id})` : data.error || 'run failed';
  await refreshRunList();
}

export function initRunSetup() {
  els.runCreateBtn.addEventListener('click', createTrainRun);
  els.refreshDatasetsBtn.addEventListener('click', loadDatasets);
  els.refreshTokenizerVocabsBtn.addEventListener('click', loadTokenizerVocabs);
  els.refreshTrainingConfigsBtn.addEventListener('click', loadTrainingConfigs);
  loadDatasets();
  loadTokenizerVocabs();
  loadTrainingConfigs();
}
