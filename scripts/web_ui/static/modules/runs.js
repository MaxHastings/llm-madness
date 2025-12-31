import { api, fetchJson } from './api.js';
import { els } from './dom.js';
import { renderLossChart } from './loss_chart.js';
import { loadRunIntoInspector } from './session.js';

let runsCache = [];
let selectedRun = null;
let selectedRunDetails = null;
let currentLogTab = 'logs';
let autoRefreshEnabled = true;
let liveLogsEnabled = true;
let logStream = null;
let lossStream = null;
let runLossLogs = [];
let datasetsCache = null;
let datasetsCacheAt = 0;
let tokenizerVocabsCache = null;
let tokenizerVocabsCacheAt = 0;

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

function extractRunId(path, segment) {
  if (!path) return null;
  const normalized = path.replace(/\\/g, '/');
  const parts = normalized.split('/');
  const idx = parts.lastIndexOf(segment);
  if (idx === -1 || idx + 1 >= parts.length) return null;
  return parts[idx + 1];
}

function formatHighlight(value, fallback = '-') {
  if (!value) return fallback;
  return value;
}

async function loadDatasetsCache() {
  const now = Date.now();
  if (datasetsCache && now - datasetsCacheAt < 30000) {
    return datasetsCache;
  }
  const data = await fetchJson('/api/datasets');
  datasetsCache = data.datasets || [];
  datasetsCacheAt = now;
  return datasetsCache;
}

async function loadTokenizerVocabsCache() {
  const now = Date.now();
  if (tokenizerVocabsCache && now - tokenizerVocabsCacheAt < 30000) {
    return tokenizerVocabsCache;
  }
  const data = await fetchJson('/api/tokenizer_vocabs');
  tokenizerVocabsCache = data.vocabs || [];
  tokenizerVocabsCacheAt = now;
  return tokenizerVocabsCache;
}

async function updateRunHighlights(manifest, runDir) {
  if (!els.runDatasetName || !els.runTokenizerName || !els.runTrainingConfigName) return;
  els.runDatasetName.textContent = '-';
  els.runTokenizerName.textContent = '-';
  els.runTrainingConfigName.textContent = '-';
  if (!manifest) return;
  const guardRunDir = runDir;

  const configMeta = manifest.config?.meta;
  const trainingName = configMeta?.name || configMeta?.id || configMeta?.parent_id;
  els.runTrainingConfigName.textContent = formatHighlight(trainingName);

  const dataPath = manifest.inputs?.data;
  const tokenizerPath = manifest.inputs?.tokenizer;
  const datasetId = extractRunId(dataPath, 'datasets');
  const tokenizerRunId = extractRunId(tokenizerPath, 'tokenizer');

  if (dataPath) {
    const datasets = await loadDatasetsCache();
    if (guardRunDir && (!selectedRun || selectedRun.run_dir !== guardRunDir)) return;
    const match = datasets.find((row) => row.snapshot_path === dataPath || row.id === datasetId);
    if (match) {
      const label = match.name || match.id;
      els.runDatasetName.textContent = formatHighlight(label, `dataset ${datasetId || '-'}`);
    } else if (datasetId) {
      els.runDatasetName.textContent = `dataset ${datasetId}`;
    } else {
      els.runDatasetName.textContent = dataPath;
    }
  }

  if (tokenizerPath) {
    const vocabs = await loadTokenizerVocabsCache();
    if (guardRunDir && (!selectedRun || selectedRun.run_dir !== guardRunDir)) return;
    const match = vocabs.find((row) => row.tokenizer_path === tokenizerPath || row.run_id === tokenizerRunId);
    if (match) {
      const label = match.name || match.config_id || match.run_id;
      els.runTokenizerName.textContent = formatHighlight(label, `tokenizer ${tokenizerRunId || '-'}`);
    } else if (tokenizerRunId) {
      els.runTokenizerName.textContent = `tokenizer ${tokenizerRunId}`;
    } else {
      els.runTokenizerName.textContent = tokenizerPath;
    }
  }
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
  const searchTerm = (els.runsSearch.value || '').trim().toLowerCase();
  let filtered = items.slice();
  if (stageFilter && stageFilter !== 'all') {
    filtered = filtered.filter((row) => row.stage === stageFilter);
  }
  if (statusFilter && statusFilter !== 'all') {
    filtered = filtered.filter((row) => row.status === statusFilter);
  }
  if (searchTerm) {
    filtered = filtered.filter((row) => {
      const haystack = [
        row.run_id,
        row.run_dir,
        row.stage,
        row.status,
        row.last_log,
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      return haystack.includes(searchTerm);
    });
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
  if (selectedRun && selectedRun.run_dir === item.run_dir) {
    row.classList.add('selected');
  }
  row.dataset.runDir = item.run_dir;

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
  row.addEventListener('click', (event) => {
    if (event.target.closest('button')) return;
    showRunDetails(item.run_dir);
  });
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
  if (liveLogsEnabled) {
    startLogStream(tab);
  }
  ensureLossStream();
}

function parseLossLine(line) {
  if (!line || line[0] !== '{') return null;
  try {
    const parsed = JSON.parse(line);
    if (!Number.isFinite(parsed.iter)) return null;
    if (parsed.train_loss == null && parsed.val_loss == null) return null;
    return parsed;
  } catch (err) {
    return null;
  }
}

function renderRunLossChart() {
  if (!els.runLossChart) return;
  if (!selectedRun || selectedRun.stage !== 'train') {
    renderLossChart(els.runLossChart, [], { emptyMessage: 'No loss data yet.' });
    if (els.runLossMeta) {
      els.runLossMeta.textContent = '';
    }
    return;
  }
  renderLossChart(els.runLossChart, runLossLogs, { emptyMessage: 'No loss data yet.' });
  if (!els.runLossMeta) return;
  const last = runLossLogs[runLossLogs.length - 1];
  if (last) {
    const train = last.train_loss != null ? last.train_loss.toFixed(4) : '-';
    const val = last.val_loss != null ? last.val_loss.toFixed(4) : '-';
    els.runLossMeta.textContent = `iter ${last.iter} train ${train} val ${val}`;
  } else {
    els.runLossMeta.textContent = 'no loss data yet';
  }
}

function seedRunLossLogs(lines) {
  runLossLogs = [];
  (lines || []).forEach((line) => {
    const entry = parseLossLine(line);
    if (!entry) return;
    runLossLogs.push(entry);
  });
  if (runLossLogs.length > 1000) {
    runLossLogs = runLossLogs.slice(-1000);
  }
  renderRunLossChart();
}

async function showRunDetails(runDir) {
  const data = await fetchJson(`/api/run/${encodeURIComponent(runDir)}`);
  selectedRun = data.summary || null;
  selectedRunDetails = {
    logs: data.logs || [],
    process_log: data.process_log || [],
  };
  autoRefreshEnabled = false;
  if (els.runsAutoRefresh) {
    els.runsAutoRefresh.checked = false;
  }
  liveLogsEnabled = !!els.liveRunLogs && els.liveRunLogs.checked;
  els.runDetailMeta.textContent = selectedRun ? `${selectedRun.stage} | ${selectedRun.run_id}` : 'Run detail';
  els.runLoadStatus.textContent = 'Auto-refresh paused while viewing this run.';
  els.loadRunFromDrawer.disabled = !(selectedRun && selectedRun.stage === 'train');
  renderSummary(data.summary);
  renderManifest(data.manifest);
  updateRunHighlights(data.manifest, data.summary?.run_dir);
  if (selectedRun && selectedRun.stage === 'train') {
    seedRunLossLogs(data.logs || []);
  } else {
    runLossLogs = [];
    renderRunLossChart();
  }
  setLogTab(currentLogTab);
  renderRunsFromCache();
}

function stopLogStream() {
  if (logStream) {
    logStream.close();
    logStream = null;
  }
}

function stopLossStream() {
  if (lossStream) {
    lossStream.close();
    lossStream = null;
  }
}

function ensureLossStream() {
  if (!selectedRun || selectedRun.stage !== 'train') {
    stopLossStream();
    return;
  }
  if (!liveLogsEnabled || currentLogTab === 'logs') {
    stopLossStream();
    return;
  }
  startLossStream();
}

function startLossStream() {
  if (!selectedRun || selectedRun.stage !== 'train') return;
  stopLossStream();
  const params = new URLSearchParams({
    run_dir: selectedRun.run_dir,
    kind: 'logs',
  });
  lossStream = new EventSource(`/api/run/stream?${params.toString()}`);
  lossStream.onmessage = (event) => {
    if (!event.data) return;
    const entry = parseLossLine(event.data);
    if (!entry) return;
    runLossLogs.push(entry);
    if (runLossLogs.length > 1000) {
      runLossLogs = runLossLogs.slice(-1000);
    }
    renderRunLossChart();
  };
}

function startLogStream(kind) {
  if (!selectedRun) return;
  stopLogStream();
  const params = new URLSearchParams({
    run_dir: selectedRun.run_dir,
    kind,
  });
  logStream = new EventSource(`/api/run/stream?${params.toString()}`);
  if (kind === 'logs') {
    selectedRunDetails.logs = [];
  } else {
    selectedRunDetails.process_log = [];
  }
  els.runLogs.textContent = '';
  logStream.onmessage = (event) => {
    if (!event.data) return;
    if (kind === 'logs') {
      selectedRunDetails.logs.push(event.data);
      if (selectedRunDetails.logs.length > 500) {
        selectedRunDetails.logs = selectedRunDetails.logs.slice(-500);
      }
      if (selectedRun && selectedRun.stage === 'train') {
        const entry = parseLossLine(event.data);
        if (entry) {
          runLossLogs.push(entry);
          if (runLossLogs.length > 1000) {
            runLossLogs = runLossLogs.slice(-1000);
          }
          renderRunLossChart();
        }
      }
    } else {
      selectedRunDetails.process_log.push(event.data);
      if (selectedRunDetails.process_log.length > 500) {
        selectedRunDetails.process_log = selectedRunDetails.process_log.slice(-500);
      }
    }
    if (currentLogTab === kind) {
      els.runLogs.textContent += `${event.data}\n`;
      els.runLogs.scrollTop = els.runLogs.scrollHeight;
    }
    const now = new Date();
    const stamp = now.toTimeString().slice(0, 8);
    els.runLoadStatus.textContent = `Live stream connected | ${stamp}`;
  };
  logStream.onerror = () => {
    els.runLoadStatus.textContent = 'Live stream disconnected. Retrying...';
  };
}

async function refreshSelectedRunDetails() {
  if (!selectedRun) return;
  const data = await fetchJson(`/api/run/${encodeURIComponent(selectedRun.run_dir)}`);
  selectedRun = data.summary || selectedRun;
  if (!liveLogsEnabled) {
    selectedRunDetails = {
      logs: data.logs || [],
      process_log: data.process_log || [],
    };
  }
  renderSummary(data.summary);
  renderManifest(data.manifest);
  updateRunHighlights(data.manifest, data.summary?.run_dir || selectedRun?.run_dir);
  if (!liveLogsEnabled && selectedRun.stage === 'train') {
    seedRunLossLogs(data.logs || []);
  } else if (!liveLogsEnabled) {
    runLossLogs = [];
    renderRunLossChart();
  }
  if (!liveLogsEnabled) {
    setLogTab(currentLogTab);
  }
  const now = new Date();
  const stamp = now.toTimeString().slice(0, 8);
  els.runLoadStatus.textContent = `Live logs refreshed at ${stamp}`;
}

function clearRunDetail() {
  selectedRun = null;
  selectedRunDetails = null;
  els.runDetailMeta.textContent = 'Select a run to inspect.';
  els.runLoadStatus.textContent = '';
  els.runSummary.innerHTML = '';
  els.runManifest.innerHTML = '';
  els.runLogs.textContent = '';
  if (els.runDatasetName) els.runDatasetName.textContent = '-';
  if (els.runTokenizerName) els.runTokenizerName.textContent = '-';
  if (els.runTrainingConfigName) els.runTrainingConfigName.textContent = '-';
  runLossLogs = [];
  renderRunLossChart();
  els.loadRunFromDrawer.disabled = true;
  autoRefreshEnabled = true;
  if (els.runsAutoRefresh) {
    els.runsAutoRefresh.checked = true;
  }
  liveLogsEnabled = !!els.liveRunLogs && els.liveRunLogs.checked;
  stopLogStream();
  stopLossStream();
}

function renderRunsFromCache() {
  const filtered = applyRunFilters(runsCache);
  const activeRuns = filtered.filter((row) => row.is_active || row.status === 'running' || row.status === 'queued');
  const historyRuns = filtered.filter((row) => !activeRuns.includes(row));
  renderRunSection(els.runsActiveList, activeRuns);
  renderRunSection(els.runsHistoryList, historyRuns);
  const activeCount = activeRuns.length;
  const filteredCount = filtered.length;
  const totalCount = runsCache.length;
  const countLabel = filteredCount === totalCount ? `${totalCount} runs` : `${filteredCount} of ${totalCount} runs`;
  const pauseNote = autoRefreshEnabled ? '' : ' | auto-refresh paused';
  els.runsMeta.textContent = `${countLabel} | ${activeCount} active${pauseNote}`;
}

export async function refreshRunList() {
  const data = await api('/api/runs', { scope: 'train' });
  runsCache = data.runs || [];
  updateFilterOptions(runsCache);
  renderRunsFromCache();
}

export function initRuns() {
  els.refreshRunsBtn.addEventListener('click', refreshRunList);
  els.runsSearch.addEventListener('input', renderRunsFromCache);
  els.runsFilterStage.addEventListener('change', renderRunsFromCache);
  els.runsFilterStatus.addEventListener('change', renderRunsFromCache);
  els.runsSort.addEventListener('change', renderRunsFromCache);
  els.runsAutoRefresh.addEventListener('change', () => {
    autoRefreshEnabled = els.runsAutoRefresh.checked;
    renderRunsFromCache();
  });
  els.liveRunLogs.addEventListener('change', () => {
    liveLogsEnabled = els.liveRunLogs.checked;
    if (liveLogsEnabled) {
      refreshSelectedRunDetails();
      startLogStream(currentLogTab);
      ensureLossStream();
    } else {
      stopLogStream();
      stopLossStream();
    }
  });
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
  liveLogsEnabled = !!els.liveRunLogs && els.liveRunLogs.checked;
  refreshRunList();
  setInterval(() => {
    if (autoRefreshEnabled) {
      refreshRunList();
    }
    if (!selectedRun) return;
    if (liveLogsEnabled || (els.liveRunLogs && els.liveRunLogs.checked)) {
      refreshSelectedRunDetails();
    }
  }, 8000);
}
