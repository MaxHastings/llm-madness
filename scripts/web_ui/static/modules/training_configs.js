import { api, fetchJson } from './api.js';
import { els } from './dom.js';
import { setPipelineTrainingConfig } from './pipeline.js';

let configs = [];
let selectedPath = null;

function formatDate(value) {
  if (!value) return '-';
  return value.replace('T', ' ');
}

function slugify(raw) {
  return raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48) || 'training';
}

function padVersion(value) {
  return `${value}`.padStart(3, '0');
}

function nextVersion(name) {
  const target = name.toLowerCase();
  const versions = configs
    .filter((item) => (item.name || '').toLowerCase() === target && Number.isInteger(item.version))
    .map((item) => item.version);
  if (!versions.length) return 1;
  return Math.max(...versions) + 1;
}

function parseEditor() {
  const raw = els.trainingConfigEditor.value;
  try {
    const parsed = JSON.parse(raw);
    return { ok: true, raw, parsed };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

function ensureMeta(parsed, name, parentId) {
  const updated = { ...parsed };
  const meta = { ...(updated.meta || {}) };
  const safeName = name || meta.name || 'default';
  meta.name = safeName;
  meta.version = nextVersion(safeName);
  meta.created_at = new Date().toISOString().slice(0, 19);
  meta.id = meta.id || `${slugify(safeName)}_${meta.version}`;
  meta.parent_id = parentId || meta.parent_id || null;
  updated.meta = meta;
  return updated;
}

function formatModelSummary(item) {
  if (!item) return '-';
  const parts = [];
  if (item.n_layer != null) parts.push(`${item.n_layer}L`);
  if (item.n_head != null) parts.push(`${item.n_head}H`);
  if (item.n_embd != null) parts.push(`${item.n_embd}D`);
  if (item.block_size != null) parts.push(`B${item.block_size}`);
  return parts.length ? parts.join(' ') : '-';
}

function renderList() {
  els.trainingConfigList.innerHTML = '';
  if (!configs.length) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'No training configs yet.';
    els.trainingConfigList.appendChild(empty);
    return;
  }
  configs.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'config-row';
    row.dataset.path = item.path;
    row.classList.toggle('active', item.path === selectedPath);
    row.innerHTML = `
      <div class="config-row-main">
        <span class="config-title">${item.name || item.path}</span>
        <span class="config-meta">v${item.version ?? '-'} • ${formatModelSummary(item)}</span>
      </div>
      <div class="config-row-sub">${formatDate(item.created_at)}</div>
    `;
    row.addEventListener('click', () => loadConfig(item.path));
    const actions = document.createElement('div');
    actions.className = 'artifact-actions';
    const useBtn = document.createElement('button');
    useBtn.textContent = 'Use in Run';
    useBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      const ok = setPipelineTrainingConfig(item.path);
      if (ok) {
        const navBtn = document.querySelector('[data-section="runs"]');
        if (navBtn) navBtn.click();
      }
    });
    actions.appendChild(useBtn);
    row.appendChild(actions);
    els.trainingConfigList.appendChild(row);
  });
}

async function loadList() {
  const data = await api('/api/configs/meta', { scope: 'training' });
  configs = data.configs || [];
  renderList();
}

async function loadConfig(path) {
  const data = await fetchJson(`/api/configs/${encodeURIComponent(path)}`);
  if (!data.raw) {
    els.trainingConfigMeta.textContent = data.error || 'failed to load config';
    return;
  }
  selectedPath = path;
  els.trainingConfigEditor.value = data.raw;
  els.trainingConfigMeta.textContent = `loaded ${path}`;
  renderList();
}

function newConfigTemplate() {
  return {
    meta: {
      id: null,
      name: 'new-training',
      version: 1,
      created_at: new Date().toISOString().slice(0, 19),
      parent_id: null,
    },
    seed: 1337,
    model: {
      block_size: 128,
      n_layer: 4,
      n_head: 4,
      n_embd: 256,
      dropout: 0.1,
    },
    training: {
      device: 'auto',
      batch_size: 32,
      max_iters: 2000,
      learning_rate: 0.0003,
      weight_decay: 0.1,
      log_interval: 50,
      eval_interval: 200,
      eval_iters: 50,
      val_split: 0.1,
      save_interval: 500,
      grad_clip: 1.0,
      warmup_iters: 200,
      sample_prompt: '1 + 1 =',
      sample_length: 32,
      sample_temperature: 1.0,
      sample_top_k: null,
    },
  };
}

function startNew() {
  selectedPath = null;
  els.trainingConfigEditor.value = JSON.stringify(newConfigTemplate(), null, 2);
  els.trainingConfigMeta.textContent = 'new training config';
  renderList();
}

async function saveNewVersion() {
  const parsed = parseEditor();
  if (!parsed.ok) {
    els.trainingConfigMeta.textContent = `invalid json: ${parsed.error}`;
    return;
  }
  const baseName = parsed.parsed.meta?.name || 'training';
  const parentId = parsed.parsed.meta?.id || null;
  const updated = ensureMeta(parsed.parsed, baseName, parentId);
  const version = updated.meta.version;
  const fileName = `${slugify(updated.meta.name)}__v${padVersion(version)}.json`;
  const path = `training/${fileName}`;
  const raw = JSON.stringify(updated, null, 2);
  const result = await api('/api/configs/save', { name: path, raw });
  if (result.status !== 'saved') {
    els.trainingConfigMeta.textContent = result.error || 'save failed';
    return;
  }
  selectedPath = path;
  els.trainingConfigEditor.value = raw;
  els.trainingConfigMeta.textContent = `saved ${path}`;
  await loadList();
  renderList();
}

async function duplicateSelected() {
  if (!selectedPath) {
    els.trainingConfigMeta.textContent = 'select a config first';
    return;
  }
  const parsed = parseEditor();
  if (!parsed.ok) {
    els.trainingConfigMeta.textContent = `invalid json: ${parsed.error}`;
    return;
  }
  parsed.parsed.meta = parsed.parsed.meta || {};
  parsed.parsed.meta.id = null;
  parsed.parsed.meta.parent_id = null;
  parsed.parsed.meta.name = `${parsed.parsed.meta.name || 'training'}-copy`;
  els.trainingConfigEditor.value = JSON.stringify(parsed.parsed, null, 2);
  selectedPath = null;
  els.trainingConfigMeta.textContent = 'duplicated; save as new version';
  renderList();
}

async function deleteSelected() {
  if (!selectedPath) {
    els.trainingConfigMeta.textContent = 'select a config first';
    return;
  }
  const ok = window.confirm(`Delete ${selectedPath}? This cannot be undone.`);
  if (!ok) return;
  await api('/api/configs/delete', { name: selectedPath });
  selectedPath = null;
  els.trainingConfigEditor.value = '';
  els.trainingConfigMeta.textContent = 'deleted';
  await loadList();
}

export function initTrainingConfigs() {
  els.trainingConfigNewBtn.addEventListener('click', startNew);
  els.trainingConfigSaveBtn.addEventListener('click', saveNewVersion);
  els.trainingConfigDuplicateBtn.addEventListener('click', duplicateSelected);
  els.trainingConfigDeleteBtn.addEventListener('click', deleteSelected);
  loadList().then(() => {
    if (configs.length) {
      loadConfig(configs[0].path);
    } else {
      startNew();
    }
  });
}
