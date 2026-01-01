import { api, fetchJson } from './api.js';
import { els } from './dom.js';

let configs = [];
let selectedPath = null;
let configTab = 'editor';
let editorTimer = null;

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

function safeText(value) {
  if (value === null || value === undefined) return '-';
  if (Array.isArray(value)) return value.length ? value.join(', ') : '-';
  return `${value}`;
}

function formatSortName(item) {
  return (item.name || item.path || '').toLowerCase();
}

function formatSortDate(item) {
  const ts = item.created_at ? Date.parse(item.created_at) : 0;
  return Number.isFinite(ts) ? ts : 0;
}

function renderKeyValues(container, entries) {
  container.innerHTML = '';
  if (!entries.length) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'No details to show.';
    container.appendChild(empty);
    return;
  }
  entries.forEach(([label, value]) => {
    const row = document.createElement('div');
    row.className = 'summary-row';
    const key = document.createElement('span');
    key.className = 'summary-key';
    key.textContent = label;
    const val = document.createElement('span');
    val.className = 'summary-value';
    val.textContent = safeText(value);
    row.appendChild(key);
    row.appendChild(val);
    container.appendChild(row);
  });
}

function renderSummary(parsed) {
  if (!parsed) {
    renderKeyValues(els.trainingConfigSummary, []);
    renderKeyValues(els.trainingConfigHyperparams, []);
    return;
  }
  const meta = parsed.meta || {};
  const training = parsed.training || {};
  const model = parsed.model || {};
  renderKeyValues(els.trainingConfigSummary, [
    ['Name', meta.name],
    ['Version', meta.version],
    ['ID', meta.id],
    ['Parent', meta.parent_id],
    ['Created', meta.created_at],
    ['Model', formatModelSummary(model)],
    ['Seed', parsed.seed],
    ['Device', training.device],
    ['Batch size', training.batch_size],
    ['Max iters', training.max_iters],
  ]);
  renderKeyValues(els.trainingConfigHyperparams, [
    ['Learning rate', training.learning_rate],
    ['Weight decay', training.weight_decay],
    ['Warmup iters', training.warmup_iters],
    ['Grad clip', training.grad_clip],
    ['Eval interval', training.eval_interval],
    ['Eval iters', training.eval_iters],
    ['Log interval', training.log_interval],
    ['Save interval', training.save_interval],
    ['Val split', training.val_split],
    ['Sample length', training.sample_length],
    ['Sample temperature', training.sample_temperature],
    ['Sample top-k', training.sample_top_k],
    ['Dropout', model.dropout],
  ]);
}

function setConfigTab(tab) {
  const previewRoot = els.trainingConfigTabSelect?.closest('.config-editor-panel');
  const panels = previewRoot ? previewRoot.querySelectorAll('.preview-panel') : [];
  configTab = tab;
  if (els.trainingConfigTabSelect) {
    els.trainingConfigTabSelect.value = tab;
  }
  panels.forEach((panel) => {
    panel.classList.toggle('is-hidden', panel.dataset.preview !== tab);
  });
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
  const query = (els.trainingConfigSearch?.value || '').trim().toLowerCase();
  const sortBy = els.trainingConfigSort?.value || 'recent';
  let items = configs.slice();
  if (query) {
    items = items.filter((item) => {
      const hay = `${item.name || ''} ${item.path || ''}`.toLowerCase();
      return hay.includes(query);
    });
  }
  if (sortBy === 'name') {
    items.sort((a, b) => formatSortName(a).localeCompare(formatSortName(b)));
  } else if (sortBy === 'version') {
    items.sort((a, b) => {
      const nameCmp = formatSortName(a).localeCompare(formatSortName(b));
      if (nameCmp !== 0) return nameCmp;
      return (b.version || 0) - (a.version || 0);
    });
  } else {
    items.sort((a, b) => formatSortDate(b) - formatSortDate(a));
  }
  if (!items.length) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'No training configs yet.';
    els.trainingConfigList.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'artifact-card selectable-card config-card';
    row.dataset.path = item.path;
    row.classList.toggle('selected', item.path === selectedPath);
    const title = document.createElement('div');
    title.className = 'artifact-title';
    title.textContent = item.name || item.path;
    const summary = document.createElement('div');
    summary.className = 'config-summary';
    summary.textContent = `v${item.version ?? '-'} • ${formatModelSummary(item)}`;
    const date = document.createElement('div');
    date.className = 'meta';
    date.textContent = formatDate(item.created_at);
    row.addEventListener('click', () => loadConfig(item.path));
    const actions = document.createElement('div');
    actions.className = 'artifact-actions';
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = 'Delete';
    deleteBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      deleteConfig(item.path);
    });
    actions.appendChild(deleteBtn);
    row.appendChild(title);
    row.appendChild(summary);
    row.appendChild(date);
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
  let parsed = data.config;
  if (!parsed) {
    try {
      parsed = JSON.parse(data.raw);
    } catch (err) {
      parsed = null;
    }
  }
  renderSummary(parsed);
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
  const template = newConfigTemplate();
  els.trainingConfigEditor.value = JSON.stringify(template, null, 2);
  els.trainingConfigMeta.textContent = 'new training config';
  renderSummary(template);
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
  renderSummary(updated);
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
  renderSummary(parsed.parsed);
  renderList();
}

async function deleteConfig(path) {
  if (!path) return;
  const ok = window.confirm(`Delete ${path}? This cannot be undone.`);
  if (!ok) return;
  await api('/api/configs/delete', { name: path });
  if (selectedPath === path) {
    selectedPath = null;
    els.trainingConfigEditor.value = '';
    els.trainingConfigMeta.textContent = 'deleted';
    renderSummary(null);
  }
  await loadList();
}

export function initTrainingConfigs() {
  els.trainingConfigNewBtn.addEventListener('click', startNew);
  els.trainingConfigSaveBtn.addEventListener('click', saveNewVersion);
  els.trainingConfigDuplicateBtn.addEventListener('click', duplicateSelected);
  if (els.trainingConfigSearch) {
    els.trainingConfigSearch.addEventListener('input', renderList);
  }
  if (els.trainingConfigSort) {
    els.trainingConfigSort.addEventListener('change', renderList);
  }
  if (els.trainingConfigTabSelect) {
    els.trainingConfigTabSelect.addEventListener('change', (event) => {
      setConfigTab(event.target.value);
    });
  }
  els.trainingConfigEditor.addEventListener('input', () => {
    if (editorTimer) window.clearTimeout(editorTimer);
    editorTimer = window.setTimeout(() => {
      const parsed = parseEditor();
      if (parsed.ok) {
        renderSummary(parsed.parsed);
      }
    }, 200);
  });
  loadList().then(() => {
    if (configs.length) {
      loadConfig(configs[0].path);
    } else {
      startNew();
    }
  });
  setConfigTab('editor');
}
