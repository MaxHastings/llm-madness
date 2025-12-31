import { api, fetchJson } from './api.js';
import { els } from './dom.js';
import { refreshRunList } from './runs.js';

async function loadConfigs() {
  const data = await api('/api/configs');
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
  const data = await api('/api/run', { stage, config: name });
  els.configMeta.textContent = data.run_id ? `started ${stage} (${data.run_id})` : data.error || 'run failed';
  await refreshRunList();
}

export function initPipeline() {
  els.loadConfigBtn.addEventListener('click', loadSelectedConfig);
  els.saveConfigBtn.addEventListener('click', saveSelectedConfig);
  els.configSelect.addEventListener('change', loadSelectedConfig);
  els.runPipelineBtn.addEventListener('click', () => startRun('pipeline'));
  els.runGenerateBtn.addEventListener('click', () => startRun('generate'));
  els.runTokenizerBtn.addEventListener('click', () => startRun('tokenizer'));
  els.runTrainBtn.addEventListener('click', () => startRun('train'));
  loadConfigs().then(loadSelectedConfig);
}
