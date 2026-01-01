import { api, fetchJson } from './api.js';
import { els } from './dom.js';

let currentPath = '';
const selections = new Set();

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

function renderSelections() {
  const items = Array.from(selections).sort();
  els.datasetSelections.value = items.join('\n');
  els.datasetMeta.textContent = items.length ? `${items.length} selection(s)` : 'Select files and folders under data/.';
}

function setManifestPreview(payload, label) {
  const manifest = payload?.manifest ?? null;
  const raw = payload?.raw || (manifest ? JSON.stringify(manifest, null, 2) : '');
  els.datasetManifestPreview.value = raw || '';
  els.datasetManifestMeta.textContent = label || '';
  renderManifestFiles(manifest?.files);
}

function renderManifestFiles(files) {
  els.datasetManifestFiles.innerHTML = '';
  if (!files || !files.length) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'No files to show.';
    els.datasetManifestFiles.appendChild(empty);
    return;
  }
  files.forEach((file) => {
    const row = document.createElement('div');
    row.className = 'manifest-file';
    const name = document.createElement('span');
    name.textContent = file.path || '-';
    const size = document.createElement('span');
    size.className = 'file-size';
    size.textContent = formatBytes(file.size_bytes);
    row.appendChild(name);
    row.appendChild(size);
    els.datasetManifestFiles.appendChild(row);
  });
}

function setPreviewTab(tab) {
  const tabs = els.datasetPreviewTabs?.querySelectorAll('.tab') || [];
  const previewRoot = els.datasetPreviewTabs?.closest('.dataset-preview');
  const panels = previewRoot ? previewRoot.querySelectorAll('.preview-panel') : [];
  tabs.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.tab === tab);
  });
  panels.forEach((panel) => {
    panel.classList.toggle('is-hidden', panel.dataset.preview !== tab);
  });
}

async function viewManifest(path) {
  if (!path) return;
  try {
    const data = await fetchJson(`/api/datasets/manifest?path=${encodeURIComponent(path)}`);
    setManifestPreview(data, `loaded ${data.path || path}`);
  } catch (err) {
    setManifestPreview({ raw: '' }, `failed to load manifest: ${err.message}`);
  }
}

async function deleteManifest(runDir) {
  if (!runDir) return;
  const ok = window.confirm(`Delete dataset manifest ${runDir}? This cannot be undone.`);
  if (!ok) return;
  await api('/api/run/delete', { run_dir: runDir });
  setManifestPreview({ raw: '' }, 'manifest deleted');
  await refreshDatasetManifests();
}

function renderEntries(entries, parent) {
  els.datasetEntries.innerHTML = '';

  if (parent != null) {
    const row = document.createElement('div');
    row.className = 'file-row';
    const spacer = document.createElement('span');
    spacer.className = 'file-spacer';
    const btn = document.createElement('button');
    btn.textContent = 'Up';
    btn.addEventListener('click', () => loadPath(parent));
    row.appendChild(spacer);
    row.appendChild(btn);
    els.datasetEntries.appendChild(row);
  }

  entries.forEach((entry) => {
    const row = document.createElement('div');
    row.className = 'file-row';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = selections.has(entry.rel_path);
    checkbox.addEventListener('change', () => {
      if (checkbox.checked) {
        selections.add(entry.rel_path);
      } else {
        selections.delete(entry.rel_path);
      }
      renderSelections();
    });

    const label = document.createElement('span');
    label.className = `file-name file-${entry.type}`;
    label.textContent = entry.type === 'dir' ? `${entry.name}/` : entry.name;
    if (entry.type === 'dir') {
      label.addEventListener('click', () => loadPath(entry.rel_path));
      label.title = 'Click to open folder';
    }

    const meta = document.createElement('span');
    meta.className = 'file-meta';
    if (entry.type === 'file') {
      meta.textContent = formatBytes(entry.size_bytes);
    } else {
      meta.textContent = 'folder';
    }

    row.appendChild(checkbox);
    row.appendChild(label);
    row.appendChild(meta);
    els.datasetEntries.appendChild(row);
  });

  if (!entries.length) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'No .txt files or folders here.';
    els.datasetEntries.appendChild(empty);
  }
}

async function loadPath(path) {
  const data = await fetchJson(`/api/data/list?path=${encodeURIComponent(path || '')}`);
  currentPath = data.path || '';
  els.datasetPath.textContent = currentPath ? `data/${currentPath}` : 'data/';
  renderEntries(data.entries || [], data.parent);
}

async function refreshDatasetManifests() {
  const data = await fetchJson('/api/datasets');
  const items = data.datasets || [];
  els.datasetManifestList.innerHTML = '';
  setManifestPreview({ raw: '' }, '');
  if (!items.length) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'No dataset manifests yet.';
    els.datasetManifestList.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'dataset-manifest';
    const title = document.createElement('div');
    title.textContent = item.name ? `${item.name} (${item.id})` : item.id;
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = `files ${item.file_count ?? '-'} | size ${formatBytes(item.total_bytes)} | ${item.created_at ?? ''}`;
    const path = document.createElement('div');
    path.className = 'meta';
    path.textContent = item.manifest_path;
    const actions = document.createElement('div');
    actions.className = 'artifact-actions';
    const viewBtn = document.createElement('button');
    viewBtn.textContent = 'View';
    viewBtn.addEventListener('click', () => viewManifest(item.manifest_path));
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = 'Delete';
    deleteBtn.addEventListener('click', () => deleteManifest(item.run_dir));
    actions.appendChild(viewBtn);
    actions.appendChild(deleteBtn);
    row.appendChild(title);
    row.appendChild(meta);
    row.appendChild(path);
    row.appendChild(actions);
    els.datasetManifestList.appendChild(row);
  });
}

async function createManifest() {
  const items = Array.from(selections).sort();
  if (!items.length) {
    els.datasetCreateMeta.textContent = 'Select at least one file or folder.';
    return;
  }
  const name = els.datasetName.value.trim() || null;
  const enableHashes = els.datasetEnableHashes.checked;
  const res = await api('/api/datasets/create', {
    name,
    selections: items,
    enable_content_hashes: enableHashes,
  });
  els.datasetCreateMeta.textContent = `created ${res.dataset_id} (files ${res.file_count}, ${formatBytes(res.total_bytes)})`;
  selections.clear();
  renderSelections();
  await refreshDatasetManifests();
}

export function initDatasets() {
  els.datasetCreateBtn.addEventListener('click', createManifest);
  els.datasetRefreshBtn.addEventListener('click', () => loadPath(currentPath));
  els.datasetRefreshManifestsBtn.addEventListener('click', refreshDatasetManifests);
  if (els.datasetPreviewTabs) {
    els.datasetPreviewTabs.addEventListener('click', (event) => {
      const btn = event.target.closest('.tab');
      if (!btn) return;
      setPreviewTab(btn.dataset.tab);
    });
  }
  renderSelections();
  loadPath(currentPath);
  refreshDatasetManifests();
  setPreviewTab('manifest');
}
