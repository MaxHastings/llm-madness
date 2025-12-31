import { api } from './api.js';
import { els } from './dom.js';
import { resetTokens } from './prompt.js';
import { refreshTokenizerReport, refreshTrainingLogs } from './training.js';

export async function refreshCheckpoints() {
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
  await refreshTokenizerReport();
  await refreshTrainingLogs();
}

export function initSession() {
  els.loadCheckpointBtn.addEventListener('click', loadCheckpoint);
  refreshCheckpoints();
}
