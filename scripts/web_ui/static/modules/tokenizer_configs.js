import { api, fetchJson } from './api.js';
import { els } from './dom.js';

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
    .slice(0, 48) || 'tokenizer';
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
  if (!configs.length) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'No tokenizer configs yet.';
    els.tokenizerConfigList.appendChild(empty);
    return;
  }
  configs.forEach((item) => {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'config-row';
    row.dataset.path = item.path;
    row.classList.toggle('active', item.path === selectedPath);
    row.innerHTML = `
      <div class="config-row-main">
        <span class="config-title">${item.name || item.path}</span>
        <span class="config-meta">v${item.version ?? '-'} • vocab ${item.vocab_size ?? '-'}</span>
      </div>
      <div class="config-row-sub">${formatDate(item.created_at)}</div>
    `;
    row.addEventListener('click', () => loadConfig(item.path));
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
  els.tokenizerConfigEditor.value = JSON.stringify(newConfigTemplate(), null, 2);
  els.tokenizerConfigMeta.textContent = 'new tokenizer config';
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
  await loadList();
  renderList();
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
  renderList();
}

async function deleteSelected() {
  if (!selectedPath) {
    els.tokenizerConfigMeta.textContent = 'select a config first';
    return;
  }
  const ok = window.confirm(`Delete ${selectedPath}? This cannot be undone.`);
  if (!ok) return;
  await api('/api/configs/delete', { name: selectedPath });
  selectedPath = null;
  els.tokenizerConfigEditor.value = '';
  els.tokenizerConfigMeta.textContent = 'deleted';
  await loadList();
}

export function initTokenizerConfigs() {
  els.tokenizerConfigNewBtn.addEventListener('click', startNew);
  els.tokenizerConfigSaveBtn.addEventListener('click', saveNewVersion);
  els.tokenizerConfigDuplicateBtn.addEventListener('click', duplicateSelected);
  els.tokenizerConfigDeleteBtn.addEventListener('click', deleteSelected);
  loadList().then(() => {
    if (configs.length) {
      loadConfig(configs[0].path);
    } else {
      startNew();
    }
  });
}
