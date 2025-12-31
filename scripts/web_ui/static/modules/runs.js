import { api, fetchJson } from './api.js';
import { els } from './dom.js';
import { loadRunIntoInspector } from './session.js';

let runsCache = [];
let selectedRun = null;
let selectedRunDetails = null;
let currentLogTab = 'logs';
let autoRefreshEnabled = true;

function formatTime(ts) {
  if (!ts) return '-';
  return ts.replace('T', ' ');
}

function truncateLine(text, max = 120) {
  if (!text) return '-';
  if (text.length <= max) return text;
  return `${text.slice(0, max - 3)}...`;
}

function formatManifestValue(value) {
  if (value == null) return '-';
  if (Array.isArray(value)) {
    return value.length ? value.join(', ') : '-';
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value);
    if (!entries.length) return '-';
    return entries.map(([key, val]) => `${key}: ${formatManifestValue(val)}`).join(' | ');
  }
  return `${value}`;
}

function updateFilterOptions(items) {
  const currentStage = els.runsFilterStage.value || 'all';
  const currentStatus = els.runsFilterStatus.value || 'all';
  const stages = Array.from(new Set(items.map((row) => row.stage).filter(Boolean))).sort();
  const statuses = Array.from(new Set(items.map((row) => row.status).filter(Boolean))).sort();

  els.runsFilterStage.innerHTML = '';
  const stageAll = document.createElement('option');
  stageAll.value = 'all';
  stageAll.textContent = 'All stages';
  els.runsFilterStage.appendChild(stageAll);
  stages.forEach((stage) => {
    const opt = document.createElement('option');
    opt.value = stage;
    opt.textContent = stage;
    els.runsFilterStage.appendChild(opt);
  });
  els.runsFilterStage.value = stages.includes(currentStage) ? currentStage : 'all';

  els.runsFilterStatus.innerHTML = '';
  const statusAll = document.createElement('option');
  statusAll.value = 'all';
  statusAll.textContent = 'All status';
  els.runsFilterStatus.appendChild(statusAll);
  statuses.forEach((status) => {
    const opt = document.createElement('option');
    opt.value = status;
    opt.textContent = status;
    els.runsFilterStatus.appendChild(opt);
  });
  els.runsFilterStatus.value = statuses.includes(currentStatus) ? currentStatus : 'all';

  if (!els.runsSort.options.length) {
    const newest = document.createElement('option');
    newest.value = 'newest';
    newest.textContent = 'Newest first';
    const oldest = document.createElement('option');
    oldest.value = 'oldest';
    oldest.textContent = 'Oldest first';
    els.runsSort.appendChild(newest);
    els.runsSort.appendChild(oldest);
    els.runsSort.value = 'newest';
  }
}

function applyRunFilters(items) {
  const stageFilter = els.runsFilterStage.value;
  const statusFilter = els.runsFilterStatus.value;
  let filtered = items.slice();
  if (stageFilter && stageFilter !== 'all') {
    filtered = filtered.filter((row) => row.stage === stageFilter);
  }
  if (statusFilter && statusFilter !== 'all') {
    filtered = filtered.filter((row) => row.status === statusFilter);
  }
  const sortOrder = els.runsSort.value || 'newest';
  filtered.sort((a, b) => {
    const left = a.start_time || '';
    const right = b.start_time || '';
    if (left === right) return 0;
    return sortOrder === 'newest' ? (left < right ? 1 : -1) : (left > right ? 1 : -1);
  });
  return filtered;
}

function renderRunRow(item) {
  const row = document.createElement('div');
  row.className = 'run-item';

  const main = document.createElement('div');
  main.className = 'run-main';

  const top = document.createElement('div');
  top.className = 'run-top';

  const stage = document.createElement('span');
  stage.className = 'run-stage';
  stage.textContent = item.stage || 'unknown';

  const runId = document.createElement('span');
  runId.className = 'run-id';
  runId.textContent = item.run_id || 'unknown';

  const time = document.createElement('span');
  time.className = 'run-time';
  time.textContent = formatTime(item.start_time);

  top.appendChild(stage);
  top.appendChild(runId);
  top.appendChild(time);

  const bottom = document.createElement('div');
  bottom.className = 'run-bottom';

  const status = document.createElement('span');
  const statusClass = `status-${item.status || 'unknown'}`;
  status.className = `status-chip ${statusClass}`;
  status.textContent = item.status || 'unknown';

  const duration = document.createElement('span');
  duration.className = 'run-duration';
  duration.textContent = item.duration || '-';

  const lastLog = document.createElement('span');
  lastLog.className = 'run-last';
  lastLog.textContent = truncateLine(item.last_log || '-');

  bottom.appendChild(status);
  bottom.appendChild(duration);
  bottom.appendChild(lastLog);

  main.appendChild(top);
  main.appendChild(bottom);

  const actions = document.createElement('div');
  actions.className = 'run-actions';

  const viewBtn = document.createElement('button');
  viewBtn.textContent = 'View';
  viewBtn.addEventListener('click', () => showRunDetails(item.run_dir));
  actions.appendChild(viewBtn);

  if (item.stage === 'train') {
    const loadBtn = document.createElement('button');
    loadBtn.textContent = 'Load';
    loadBtn.addEventListener('click', () => loadRunIntoInspector(item));
    actions.appendChild(loadBtn);
  }

  if (item.is_active) {
    const stopBtn = document.createElement('button');
    stopBtn.textContent = 'Stop';
    stopBtn.addEventListener('click', async () => {
      await api(`/api/stop/${encodeURIComponent(item.run_id)}`);
      await refreshRunList();
    });
    actions.appendChild(stopBtn);
  }

  const deleteBtn = document.createElement('button');
  deleteBtn.textContent = 'Delete';
  deleteBtn.addEventListener('click', async () => {
    const confirmDelete = window.confirm(`Delete run ${item.run_id}? This cannot be undone.`);
    if (!confirmDelete) return;
    await api('/api/run/delete', { run_dir: item.run_dir });
    if (selectedRun && selectedRun.run_dir === item.run_dir) {
      clearRunDetail();
    }
    await refreshRunList();
  });
  actions.appendChild(deleteBtn);

  row.appendChild(main);
  row.appendChild(actions);
  return row;
}

function renderRunSection(container, items) {
  container.innerHTML = '';
  if (!items.length) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'No runs to show.';
    container.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    container.appendChild(renderRunRow(item));
  });
}

function renderSummary(summary) {
  els.runSummary.innerHTML = '';
  if (!summary) return;
  const rows = [
    ['Stage', summary.stage],
    ['Status', summary.status],
    ['Run ID', summary.run_id],
    ['Started', formatTime(summary.start_time)],
    ['Ended', formatTime(summary.end_time)],
    ['Duration', summary.duration || '-'],
    ['Run Dir', summary.run_dir],
  ];
  rows.forEach(([label, value]) => {
    const row = document.createElement('div');
    const key = document.createElement('span');
    key.textContent = label;
    const val = document.createElement('span');
    val.textContent = value || '-';
    row.appendChild(key);
    row.appendChild(val);
    els.runSummary.appendChild(row);
  });
}

function renderManifest(manifest) {
  els.runManifest.innerHTML = '';
  if (!manifest) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'No manifest found for this run.';
    els.runManifest.appendChild(empty);
    return;
  }
  const fields = [
    ['Git SHA', manifest.git_sha],
    ['Notes', manifest.notes],
    ['Error', manifest.error],
    ['Inputs', manifest.inputs],
    ['Outputs', manifest.outputs],
    ['Config', manifest.config],
  ];
  fields.forEach(([label, value]) => {
    if (value == null || value === '' || (typeof value === 'object' && !Object.keys(value || {}).length)) {
      return;
    }
    const row = document.createElement('div');
    const key = document.createElement('span');
    key.textContent = label;
    const val = document.createElement('span');
    val.textContent = formatManifestValue(value);
    row.appendChild(key);
    row.appendChild(val);
    els.runManifest.appendChild(row);
  });
  if (!els.runManifest.childElementCount) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'No extra manifest fields to show.';
    els.runManifest.appendChild(empty);
  }
}

function setLogTab(tab) {
  currentLogTab = tab;
  document.querySelectorAll('.run-tabs .tab').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.tab === tab);
  });
  if (!selectedRunDetails) {
    els.runLogs.textContent = '';
    return;
  }
  const lines = tab === 'process' ? selectedRunDetails.process_log : selectedRunDetails.logs;
  els.runLogs.textContent = (lines || []).join('\n') || 'No logs available.';
}

async function showRunDetails(runDir) {
  const data = await fetchJson(`/api/run/${encodeURIComponent(runDir)}`);
  selectedRun = data.summary || null;
  selectedRunDetails = {
    logs: data.logs || [],
    process_log: data.process_log || [],
  };
  autoRefreshEnabled = false;
  els.runDetailMeta.textContent = selectedRun ? `${selectedRun.stage} | ${selectedRun.run_id}` : 'Run detail';
  els.runLoadStatus.textContent = 'Auto-refresh paused while viewing this run.';
  els.loadRunFromDrawer.disabled = !(selectedRun && selectedRun.stage === 'train');
  renderSummary(data.summary);
  renderManifest(data.manifest);
  setLogTab(currentLogTab);
  renderRunsFromCache();
}

function clearRunDetail() {
  selectedRun = null;
  selectedRunDetails = null;
  els.runDetailMeta.textContent = 'Select a run to inspect.';
  els.runLoadStatus.textContent = '';
  els.runSummary.innerHTML = '';
  els.runManifest.innerHTML = '';
  els.runLogs.textContent = '';
  els.loadRunFromDrawer.disabled = true;
  autoRefreshEnabled = true;
}

function renderRunsFromCache() {
  const filtered = applyRunFilters(runsCache);
  const activeRuns = filtered.filter((row) => row.is_active || row.status === 'running' || row.status === 'queued');
  const historyRuns = filtered.filter((row) => !activeRuns.includes(row));
  renderRunSection(els.runsActiveList, activeRuns);
  renderRunSection(els.runsHistoryList, historyRuns);
  const activeCount = runsCache.filter((row) => row.is_active || row.status === 'running' || row.status === 'queued').length;
  const pauseNote = autoRefreshEnabled ? '' : ' | auto-refresh paused';
  els.runsMeta.textContent = `${runsCache.length} runs | ${activeCount} active${pauseNote}`;
}

export async function refreshRunList() {
  const data = await api('/api/runs', { scope: 'all' });
  runsCache = data.runs || [];
  updateFilterOptions(runsCache);
  renderRunsFromCache();
}

export function initRuns() {
  els.refreshRunsBtn.addEventListener('click', refreshRunList);
  els.runsFilterStage.addEventListener('change', renderRunsFromCache);
  els.runsFilterStatus.addEventListener('change', renderRunsFromCache);
  els.runsSort.addEventListener('change', renderRunsFromCache);
  els.loadRunFromDrawer.addEventListener('click', async () => {
    if (!selectedRun) return;
    await loadRunIntoInspector(selectedRun);
  });
  els.clearRunDetailBtn.addEventListener('click', () => {
    clearRunDetail();
    renderRunsFromCache();
  });
  document.querySelectorAll('.run-tabs .tab').forEach((btn) => {
    btn.addEventListener('click', () => setLogTab(btn.dataset.tab));
  });
  refreshRunList();
  setInterval(() => {
    if (autoRefreshEnabled) {
      refreshRunList();
    }
  }, 8000);
}
