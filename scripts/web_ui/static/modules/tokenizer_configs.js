import { api, fetchJson } from './api.js';
import { els } from './dom.js';
import { isSectionActive, scheduleAutoRefresh } from './auto_refresh.js';
import { emitEvent } from './events.js';

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
    .slice(0, 48) || 'tokenizer';
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
    renderKeyValues(els.tokenizerConfigSummary, []);
    renderKeyValues(els.tokenizerConfigDefaults, []);
    return;
  }
  const meta = parsed.meta || {};
  renderKeyValues(els.tokenizerConfigSummary, [
    ['Name', meta.name],
    ['Version', meta.version],
    ['ID', meta.id],
    ['Parent', meta.parent_id],
    ['Created', meta.created_at],
    ['Algorithm', parsed.algorithm],
    ['Vocab size', parsed.vocab_size],
    ['Min frequency', parsed.min_frequency],
  ]);
  renderKeyValues(els.tokenizerConfigDefaults, [
    ['Special tokens', parsed.special_tokens],
    ['Discover regex', parsed.discover_special_token_regex],
    ['Add prefix space', parsed.add_prefix_space],
    ['Byte level', parsed.byte_level],
    ['Split digits', parsed.split_digits],
  ]);
}

function setConfigTab(tab) {
  const previewRoot = els.tokenizerConfigTabSelect?.closest('.config-editor-panel');
  const panels = previewRoot ? previewRoot.querySelectorAll('.preview-panel') : [];
  configTab = tab;
  if (els.tokenizerConfigTabSelect) {
    els.tokenizerConfigTabSelect.value = tab;
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
  const raw = els.tokenizerConfigEditor.value;
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

function renderList() {
  els.tokenizerConfigList.innerHTML = '';
  const query = (els.tokenizerConfigSearch?.value || '').trim().toLowerCase();
  const sortBy = els.tokenizerConfigSort?.value || 'recent';
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
    empty.textContent = 'No tokenizer configs yet.';
    els.tokenizerConfigList.appendChild(empty);
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
    summary.textContent = `v${item.version ?? '-'} • vocab ${item.vocab_size ?? '-'}`;
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
    els.tokenizerConfigList.appendChild(row);
  });
}

async function loadList() {
  const data = await api('/api/configs/meta', { scope: 'tokenizer' });
  configs = data.configs || [];
  renderList();
}

async function loadConfig(path) {
  const data = await fetchJson(`/api/configs/${encodeURIComponent(path)}`);
  if (!data.raw) {
    els.tokenizerConfigMeta.textContent = data.error || 'failed to load config';
    return;
  }
  selectedPath = path;
  els.tokenizerConfigEditor.value = data.raw;
  els.tokenizerConfigMeta.textContent = `loaded ${path}`;
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
      name: 'new-tokenizer',
      version: 1,
      created_at: new Date().toISOString().slice(0, 19),
      parent_id: null,
    },
    algorithm: 'bpe',
    vocab_size: 4096,
    min_frequency: 2,
    special_tokens: ['<|unk|>', '<|begin|>', '<|end|>', '<|endoftext|>'],
    discover_special_token_regex: '<\\|speaker=[^|]+\\|>',
    add_prefix_space: false,
    byte_level: true,
    split_digits: true,
  };
}

function startNew() {
  selectedPath = null;
  const template = newConfigTemplate();
  els.tokenizerConfigEditor.value = JSON.stringify(template, null, 2);
  els.tokenizerConfigMeta.textContent = 'new tokenizer config';
  renderSummary(template);
  renderList();
}

async function saveNewVersion() {
  const parsed = parseEditor();
  if (!parsed.ok) {
    els.tokenizerConfigMeta.textContent = `invalid json: ${parsed.error}`;
    return;
  }
  const baseName = parsed.parsed.meta?.name || 'tokenizer';
  const parentId = parsed.parsed.meta?.id || null;
  const updated = ensureMeta(parsed.parsed, baseName, parentId);
  const version = updated.meta.version;
  const fileName = `${slugify(updated.meta.name)}__v${padVersion(version)}.json`;
  const path = `tokenizer/${fileName}`;
  const raw = JSON.stringify(updated, null, 2);
  const result = await api('/api/configs/save', { name: path, raw });
  if (result.status !== 'saved') {
    els.tokenizerConfigMeta.textContent = result.error || 'save failed';
    return;
  }
  selectedPath = path;
  els.tokenizerConfigEditor.value = raw;
  els.tokenizerConfigMeta.textContent = `saved ${path}`;
  renderSummary(updated);
  await loadList();
  renderList();
  emitEvent('tokenizer_configs:changed');
}

async function duplicateSelected() {
  if (!selectedPath) {
    els.tokenizerConfigMeta.textContent = 'select a config first';
    return;
  }
  const parsed = parseEditor();
  if (!parsed.ok) {
    els.tokenizerConfigMeta.textContent = `invalid json: ${parsed.error}`;
    return;
  }
  parsed.parsed.meta = parsed.parsed.meta || {};
  parsed.parsed.meta.id = null;
  parsed.parsed.meta.parent_id = null;
  parsed.parsed.meta.name = `${parsed.parsed.meta.name || 'tokenizer'}-copy`;
  els.tokenizerConfigEditor.value = JSON.stringify(parsed.parsed, null, 2);
  selectedPath = null;
  els.tokenizerConfigMeta.textContent = 'duplicated; save as new version';
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
    els.tokenizerConfigEditor.value = '';
    els.tokenizerConfigMeta.textContent = 'deleted';
    renderSummary(null);
  }
  await loadList();
  emitEvent('tokenizer_configs:changed');
}

export function initTokenizerConfigs() {
  els.tokenizerConfigNewBtn.addEventListener('click', startNew);
  els.tokenizerConfigSaveBtn.addEventListener('click', saveNewVersion);
  els.tokenizerConfigDuplicateBtn.addEventListener('click', duplicateSelected);
  if (els.tokenizerConfigSearch) {
    els.tokenizerConfigSearch.addEventListener('input', renderList);
  }
  if (els.tokenizerConfigSort) {
    els.tokenizerConfigSort.addEventListener('change', renderList);
  }
  if (els.tokenizerConfigTabSelect) {
    els.tokenizerConfigTabSelect.addEventListener('change', (event) => {
      setConfigTab(event.target.value);
    });
  }
  els.tokenizerConfigEditor.addEventListener('input', () => {
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
  scheduleAutoRefresh({
    intervalMs: 30000,
    isEnabled: () => isSectionActive('tokenizer-configs'),
    task: loadList,
  });
}
