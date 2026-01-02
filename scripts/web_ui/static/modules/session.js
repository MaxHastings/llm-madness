import { api } from './api.js';
import { els } from './dom.js';
import { resetTokens } from './prompt.js';
import { isSectionActive, scheduleAutoRefresh } from './auto_refresh.js';

export async function refreshCheckpoints() {
  try {
    const data = await api('/api/checkpoints');
    els.checkpointSelect.innerHTML = '';
    data.checkpoints.forEach((ckpt) => {
      const opt = document.createElement('option');
      opt.value = ckpt;
      opt.textContent = ckpt;
      if (ckpt === data.current) opt.selected = true;
      els.checkpointSelect.appendChild(opt);
    });
    els.runMeta.textContent = data.run_dir;
    els.checkpointMeta.textContent = data.checkpoints.length ? '' : 'no checkpoints found';
  } catch (err) {
    els.checkpointSelect.innerHTML = '';
    els.checkpointMeta.textContent = 'load a training run to see checkpoints';
  }
}

async function loadCheckpoint() {
  const picked = els.checkpointSelect.value;
  const data = await api('/api/load_checkpoint', { checkpoint: picked });
  els.checkpointMeta.textContent = data.status;
}

export async function loadRunIntoInspector(item) {
  if (!item) return;
  const data = await api('/api/load_run', { run_dir: item.run_dir });
  els.runMeta.textContent = data.run_dir || els.runMeta.textContent;
  els.checkpointMeta.textContent = data.status || data.error || 'failed to load run';
  resetTokens();
  await refreshCheckpoints();
}

async function refreshSessionRuns() {
  const priorSelection = els.sessionRunSelect.value;
  const data = await api('/api/runs', { scope: 'all' });
  const runs = (data.runs || []).filter((run) => run.stage === 'train');
  runs.sort((a, b) => (a.start_time || '') < (b.start_time || '') ? 1 : -1);
  els.sessionRunSelect.innerHTML = '';
  if (!runs.length) {
    els.sessionRunMeta.textContent = 'no training runs found';
    return;
  }
  let selectionStillExists = false;
  const baseName = (path) => {
    if (!path) return 'unknown';
    const parts = path.split('/');
    return parts[parts.length - 1] || path;
  };
  runs.forEach((run) => {
    const opt = document.createElement('option');
    opt.value = run.run_dir;
    const label = run.run_id || baseName(run.run_dir) || 'unknown';
    opt.textContent = `${label} (${run.status || 'unknown'})`;
    els.sessionRunSelect.appendChild(opt);
    if (priorSelection && priorSelection === run.run_dir) {
      selectionStillExists = true;
    }
  });
  if (selectionStillExists) {
    els.sessionRunSelect.value = priorSelection;
  }
  els.sessionRunMeta.textContent = `${runs.length} training runs`;
}

export function initSession() {
  els.loadCheckpointBtn.addEventListener('click', loadCheckpoint);
  els.loadSessionRunBtn.addEventListener('click', async () => {
    const runDir = els.sessionRunSelect.value;
    if (!runDir) return;
    await loadRunIntoInspector({ run_dir: runDir });
  });
  refreshCheckpoints();
  refreshSessionRuns();
  scheduleAutoRefresh({
    intervalMs: 15000,
    isEnabled: () => isSectionActive('inspector'),
    task: refreshCheckpoints,
  });
  scheduleAutoRefresh({
    intervalMs: 15000,
    isEnabled: () => isSectionActive('inspector'),
    task: refreshSessionRuns,
  });
}
