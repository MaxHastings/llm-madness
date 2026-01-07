import { api, fetchJson } from './api.js';
import { els } from './dom.js';
import { isSectionActive, scheduleAutoRefresh } from './auto_refresh.js';
import { emitEvent } from './events.js';

let currentPath = '';
const selections = new Set();
const selectionSizes = new Map();
let selectedManifestPath = null;
let progressStream = null;
let activeRunDir = null;

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

function estimateTokens(bytes) {
  if (bytes == null) return null;
  return Math.max(0, Math.round(bytes / 4));
}

function formatTokens(tokens) {
  if (tokens == null) return '-';
  const units = ['', 'K', 'M', 'B'];
  let value = tokens;
  let idx = 0;
  while (value >= 1000 && idx < units.length - 1) {
    value /= 1000;
    idx += 1;
  }
  const precision = idx === 0 ? 0 : 1;
  return `${value.toFixed(precision)}${units[idx]} tokens`;
}

function formatElapsed(seconds) {
  if (!Number.isFinite(seconds)) return null;
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
}

function extractRunId(runDir) {
  if (!runDir) return '';
  const normalized = runDir.replace(/\\/g, '/');
  const parts = normalized.split('/').filter(Boolean);
  return parts[parts.length - 1] || runDir;
}

function stopProgressStream() {
  if (progressStream) {
    progressStream.close();
    progressStream = null;
  }
}

function renderProgress(payload) {
  if (!els.datasetProgressMeta || !els.datasetProgressLine) return;
  if (!payload || typeof payload !== 'object') {
    els.datasetProgressMeta.textContent = '';
    els.datasetProgressLine.textContent = '';
    return;
  }
  const stage = payload.stage || 'dataset';
  const status = payload.status || 'running';
  const updatedAt = payload.updated_at || null;
  const metaParts = [stage, status];
  if (updatedAt) metaParts.push(`updated ${updatedAt}`);
  els.datasetProgressMeta.textContent = metaParts.join(' | ');

  const lines = [];
  if (payload.message) lines.push(payload.message);
  const processed = Number.isFinite(payload.processed_files) ? payload.processed_files : null;
  const total = Number.isFinite(payload.total_files) ? payload.total_files : null;
  if (processed != null && total != null) {
    const pct = total ? ` (${Math.min(100, (processed / total) * 100).toFixed(1)}%)` : '';
    lines.push(`files: ${processed}/${total}${pct}`);
  }
  const elapsed = formatElapsed(payload.elapsed_seconds);
  if (elapsed) lines.push(`elapsed: ${elapsed}`);
  els.datasetProgressLine.textContent = lines.join('\n');
}

function startProgressStream(runDir) {
  if (!runDir) return;
  stopProgressStream();
  activeRunDir = runDir;
  renderProgress({ stage: 'dataset', status: 'running', message: `creating dataset | ${runDir}` });
  const params = new URLSearchParams({
    run_dir: runDir,
    kind: 'progress',
  });
  progressStream = new EventSource(`/api/run/stream?${params.toString()}`);
  progressStream.onmessage = (event) => {
    if (!event.data) return;
    try {
      const payload = JSON.parse(event.data);
      renderProgress(payload);
      if (payload.status === 'complete') {
        stopProgressStream();
        els.datasetCreateMeta.textContent = `created ${extractRunId(activeRunDir)}`;
        refreshDatasetManifests();
        emitEvent('datasets:changed');
      } else if (payload.status === 'failed') {
        stopProgressStream();
        els.datasetCreateMeta.textContent = `dataset failed: ${payload.message || 'unknown error'}`;
      }
    } catch (err) {
      renderProgress({ message: event.data });
    }
  };
  progressStream.onerror = () => {
    if (els.datasetProgressMeta) {
      els.datasetProgressMeta.textContent = 'Progress stream disconnected. Retrying...';
    }
  };
}

function renderSelections() {
  const items = Array.from(selections).sort();
  els.datasetSelections.value = items.join('\n');
  if (!items.length) {
    els.datasetMeta.textContent = 'Select files and folders under data/.';
    return;
  }
  let total = 0;
  let unknown = false;
  selections.forEach((path) => {
    if (!selectionSizes.has(path)) {
      unknown = true;
      return;
    }
    total += selectionSizes.get(path);
  });
  const sizeLabel = unknown ? 'size unknown' : `${formatBytes(total)} total`;
  const tokensLabel = unknown ? null : `~${formatTokens(estimateTokens(total))}`;
  const metaParts = [`${items.length} selection(s)`, sizeLabel];
  if (tokensLabel) {
    metaParts.push(tokensLabel);
  }
  els.datasetMeta.textContent = metaParts.join(' | ');
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
    const tokenEstimate = estimateTokens(file.size_bytes);
    const tokenLabel = tokenEstimate == null ? '-' : `~${formatTokens(tokenEstimate)}`;
    size.textContent = `${formatBytes(file.size_bytes)} | ${tokenLabel}`;
    row.appendChild(name);
    row.appendChild(size);
    els.datasetManifestFiles.appendChild(row);
  });
}

function setPreviewTab(tab) {
  const previewRoot = els.datasetPreviewSelect?.closest('.dataset-preview');
  const panels = previewRoot ? previewRoot.querySelectorAll('.preview-panel') : [];
  if (els.datasetPreviewSelect) {
    els.datasetPreviewSelect.value = tab;
  }
  panels.forEach((panel) => {
    panel.classList.toggle('is-hidden', panel.dataset.preview !== tab);
  });
}

async function viewManifest(path) {
  if (!path) return;
  try {
    const data = await fetchJson(`/api/datasets/manifest?path=${encodeURIComponent(path)}`);
    setManifestPreview(data, `loaded ${data.path || path}`);
    selectedManifestPath = path;
    renderSelectedManifest();
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
  selectedManifestPath = null;
  renderSelectedManifest();
  await refreshDatasetManifests();
  emitEvent('datasets:changed');
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
        if (entry.size_bytes != null) {
          selectionSizes.set(entry.rel_path, entry.size_bytes);
        }
      } else {
        selections.delete(entry.rel_path);
        selectionSizes.delete(entry.rel_path);
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
    if (entry.size_bytes != null) {
      const tokenEstimate = estimateTokens(entry.size_bytes);
      const tokenLabel = tokenEstimate == null ? '-' : `~${formatTokens(tokenEstimate)}`;
      meta.textContent = `${formatBytes(entry.size_bytes)} | ${tokenLabel}`;
    } else if (entry.type === 'dir') {
      meta.textContent = 'folder';
    } else {
      meta.textContent = '-';
    }

    row.appendChild(checkbox);
    row.appendChild(label);
    row.appendChild(meta);
    els.datasetEntries.appendChild(row);

    if (checkbox.checked && entry.size_bytes != null && !selectionSizes.has(entry.rel_path)) {
      selectionSizes.set(entry.rel_path, entry.size_bytes);
    }
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
    row.className = 'artifact-card selectable-card dataset-manifest-card';
    row.dataset.manifestPath = item.manifest_path;
    row.addEventListener('click', () => {
      selectedManifestPath = item.manifest_path;
      renderSelectedManifest();
      viewManifest(item.manifest_path);
    });
    const title = document.createElement('div');
    title.className = 'artifact-title';
    title.textContent = item.name ? `${item.name} (${item.id})` : item.id;
    const meta = document.createElement('div');
    meta.className = 'meta';
    const tokenEstimate = estimateTokens(item.total_bytes);
    const tokenLabel = tokenEstimate == null ? '-' : `~${formatTokens(tokenEstimate)}`;
    meta.textContent = `files ${item.file_count ?? '-'} | size ${formatBytes(item.total_bytes)} | ${tokenLabel} | ${item.created_at ?? ''}`;
    const path = document.createElement('div');
    path.className = 'meta';
    path.textContent = item.manifest_path;
    const actions = document.createElement('div');
    actions.className = 'artifact-actions';
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = 'Delete';
    deleteBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      deleteManifest(item.run_dir);
    });
    actions.appendChild(deleteBtn);
    row.appendChild(title);
    row.appendChild(meta);
    row.appendChild(path);
    row.appendChild(actions);
    els.datasetManifestList.appendChild(row);
  });
  renderSelectedManifest();
}

function renderSelectedManifest() {
  const rows = els.datasetManifestList.querySelectorAll('.dataset-manifest-card');
  rows.forEach((row) => {
    row.classList.toggle('selected', row.dataset.manifestPath === selectedManifestPath);
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
    async: true,
  });
  if (res.status === 'started' && res.run_dir) {
    els.datasetCreateMeta.textContent = `creating ${res.dataset_id}`;
    startProgressStream(res.run_dir);
  } else if (res.status === 'created') {
    els.datasetCreateMeta.textContent = `created ${res.dataset_id} (files ${res.file_count}, ${formatBytes(res.total_bytes)})`;
  } else {
    els.datasetCreateMeta.textContent = res.error || 'dataset create failed';
  }
  selections.clear();
  selectionSizes.clear();
  renderSelections();
  await refreshDatasetManifests();
  emitEvent('datasets:changed');
}

export function initDatasets() {
  els.datasetCreateBtn.addEventListener('click', createManifest);
  els.datasetRefreshBtn.addEventListener('click', () => loadPath(currentPath));
  els.datasetRefreshManifestsBtn.addEventListener('click', refreshDatasetManifests);
  if (els.datasetPreviewSelect) {
    els.datasetPreviewSelect.addEventListener('change', (event) => {
      setPreviewTab(event.target.value);
    });
  }
  renderSelections();
  loadPath(currentPath);
  refreshDatasetManifests();
  setPreviewTab('manifest');
  scheduleAutoRefresh({
    intervalMs: 30000,
    isEnabled: () => isSectionActive('datasets'),
    task: () => loadPath(currentPath),
  });
  scheduleAutoRefresh({
    intervalMs: 30000,
    isEnabled: () => isSectionActive('datasets'),
    task: refreshDatasetManifests,
  });
}
