import { api, fetchJson } from './api.js';
import { els } from './dom.js';
import { refreshRunList } from './runs.js';

function formatDatasetLabel(item) {
  if (!item) return '';
  const name = item.name ? `${item.name} (${item.id})` : item.id;
  const count = item.file_count != null ? `files ${item.file_count}` : 'files ?';
  return `${name} — ${count}`;
}

async function loadDatasets() {
  const data = await fetchJson('/api/datasets');
  const items = data.datasets || [];
  els.datasetSelect.innerHTML = '';
  const none = document.createElement('option');
  none.value = '';
  none.textContent = 'No dataset (use defaults)';
  els.datasetSelect.appendChild(none);
  items.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.manifest_path;
    opt.textContent = formatDatasetLabel(item);
    els.datasetSelect.appendChild(opt);
  });
}

export async function loadTokenizerVocabs() {
  const data = await fetchJson('/api/tokenizer_vocabs');
  const items = data.vocabs || [];
  els.runTokenizerVocabSelect.innerHTML = '';
  const none = document.createElement('option');
  none.value = '';
  none.textContent = 'No tokenizer vocab';
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

async function loadConfigs() {
  const data = await api('/api/configs', { scope: 'pipeline' });
  els.configSelect.innerHTML = '';
  data.configs.forEach((name) => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    els.configSelect.appendChild(opt);
  });
  if (data.configs.length) {
    const preferred = data.configs.find((name) => name === 'pipeline.json') || data.configs[0];
    els.configSelect.value = preferred;
  }
}

async function loadSelectedConfig() {
  const name = els.configSelect.value;
  if (!name) return;
  const data = await fetchJson(`/api/configs/${encodeURIComponent(name)}`);
  if (data.raw) {
    els.configEditor.value = data.raw;
    els.configMeta.textContent = `loaded ${name}`;
  } else {
    els.configMeta.textContent = data.error || 'failed to load config';
  }
}

function parseConfigEditor() {
  const raw = els.configEditor.value;
  try {
    JSON.parse(raw);
    return { ok: true, raw };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

async function saveSelectedConfig() {
  const name = els.configSelect.value;
  const parsed = parseConfigEditor();
  if (!parsed.ok) {
    els.configMeta.textContent = `invalid json: ${parsed.error}`;
    return false;
  }
  const data = await api('/api/configs/save', { name, raw: parsed.raw });
  els.configMeta.textContent = data.status ? `saved ${name}` : data.error || 'save failed';
  return true;
}

async function startRun(stage) {
  const name = els.configSelect.value;
  if (!name) return;
  const ok = await saveSelectedConfig();
  if (!ok) return;
  const payload = { stage, config: name };
  if (stage === 'tokenizer' || stage === 'train') {
    const datasetManifest = els.datasetSelect.value;
    if (datasetManifest) payload.dataset_manifest = datasetManifest;
  }
  if (stage === 'train') {
    const tokenizerPath = els.runTokenizerVocabSelect.value;
    if (tokenizerPath) payload.tokenizer_path = tokenizerPath;
  }
  const data = await api('/api/run', payload);
  els.configMeta.textContent = data.run_id ? `started ${stage} (${data.run_id})` : data.error || 'run failed';
  await refreshRunList();
}

export function initPipeline() {
  els.loadConfigBtn.addEventListener('click', loadSelectedConfig);
  els.saveConfigBtn.addEventListener('click', saveSelectedConfig);
  els.configSelect.addEventListener('change', loadSelectedConfig);
  els.runPipelineBtn.addEventListener('click', () => startRun('pipeline'));
  els.runTokenizerBtn.addEventListener('click', () => startRun('tokenizer'));
  els.runTrainBtn.addEventListener('click', () => startRun('train'));
  els.refreshDatasetsBtn.addEventListener('click', loadDatasets);
  els.refreshTokenizerVocabsBtn.addEventListener('click', loadTokenizerVocabs);
  loadConfigs().then(loadSelectedConfig);
  loadDatasets();
  loadTokenizerVocabs();
}
