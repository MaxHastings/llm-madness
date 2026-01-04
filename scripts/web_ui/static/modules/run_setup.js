import { api, fetchJson } from './api.js';
import { els } from './dom.js';
import { refreshRunList } from './runs.js';
import { isSectionActive, scheduleAutoRefresh } from './auto_refresh.js';
import { onEvent } from './events.js';

function formatDatasetLabel(item) {
  if (!item) return '';
  const name = item.name ? `${item.name} (${item.id})` : item.id;
  const count = item.file_count != null ? `files ${item.file_count}` : 'files ?';
  return `${name} — ${count}`;
}

function formatTrainingSummary(item) {
  if (!item) return '';
  const parts = [];
  if (item.n_layer != null) parts.push(`${item.n_layer}L`);
  if (item.n_head != null) parts.push(`${item.n_head}H`);
  if (item.n_embd != null) parts.push(`${item.n_embd}D`);
  if (item.block_size != null) parts.push(`B${item.block_size}`);
  return parts.length ? parts.join(' ') : '';
}

export async function loadDatasets() {
  const priorSelection = els.datasetSelect.value;
  const data = await fetchJson('/api/datasets');
  const items = data.datasets || [];
  els.datasetSelect.innerHTML = '';
  const none = document.createElement('option');
  none.value = '';
  none.textContent = 'Select dataset';
  els.datasetSelect.appendChild(none);
  const available = new Set();
  items.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.manifest_path;
    opt.textContent = formatDatasetLabel(item);
    els.datasetSelect.appendChild(opt);
    available.add(opt.value);
  });
  if (priorSelection && available.has(priorSelection)) {
    els.datasetSelect.value = priorSelection;
  }
}

export function setDatasetSelection(path) {
  if (!path) return;
  const exists = Array.from(els.datasetSelect.options).some((opt) => opt.value === path);
  if (!exists) {
    const opt = document.createElement('option');
    opt.value = path;
    opt.textContent = path.split('/').slice(-2).join('/');
    els.datasetSelect.appendChild(opt);
  }
  els.datasetSelect.value = path;
}

export async function loadTokenizerVocabs() {
  const priorSelection = els.runTokenizerVocabSelect.value;
  const data = await fetchJson('/api/tokenizer_vocabs');
  const items = data.vocabs || [];
  els.runTokenizerVocabSelect.innerHTML = '';
  const none = document.createElement('option');
  none.value = '';
  none.textContent = 'Select tokenizer vocab';
  els.runTokenizerVocabSelect.appendChild(none);
  const available = new Set();
  items.forEach((item) => {
    const opt = document.createElement('option');
    const exists = Boolean(item.tokenizer_exists);
    opt.value = exists ? item.tokenizer_path || '' : '';
    opt.disabled = !exists;
    const name = item.name ? `${item.name} v${item.version ?? '-'}` : item.run_id;
    const vocab = item.vocab_size != null ? `vocab ${item.vocab_size}` : '';
    const status = item.status && item.status !== 'completed' ? item.status : '';
    opt.textContent = [name, vocab, status].filter(Boolean).join(' • ');
    els.runTokenizerVocabSelect.appendChild(opt);
    if (opt.value) {
      available.add(opt.value);
    }
  });
  if (priorSelection && available.has(priorSelection)) {
    els.runTokenizerVocabSelect.value = priorSelection;
  }
}

export function setTokenizerVocabSelection(path) {
  if (!path) return;
  const exists = Array.from(els.runTokenizerVocabSelect.options).some((opt) => opt.value === path);
  if (!exists) {
    const opt = document.createElement('option');
    opt.value = path;
    opt.textContent = path.split('/').slice(-2).join('/');
    els.runTokenizerVocabSelect.appendChild(opt);
  }
  els.runTokenizerVocabSelect.value = path;
}

export async function loadTrainingConfigs() {
  const priorSelection = els.trainingConfigSelect.value;
  const data = await api('/api/configs/meta', { scope: 'training' });
  const items = data.configs || [];
  els.trainingConfigSelect.innerHTML = '';
  const none = document.createElement('option');
  none.value = '';
  none.textContent = 'Select training config';
  els.trainingConfigSelect.appendChild(none);
  const available = new Set();
  items.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.path;
    const name = item.name || item.path;
    const version = item.version != null ? `v${item.version}` : 'v?';
    const summary = formatTrainingSummary(item);
    opt.textContent = [name, version, summary].filter(Boolean).join(' • ');
    els.trainingConfigSelect.appendChild(opt);
    available.add(opt.value);
  });
  if (priorSelection && available.has(priorSelection)) {
    els.trainingConfigSelect.value = priorSelection;
  }
}

export function setTrainingConfigSelection(path) {
  if (!path) return;
  const exists = Array.from(els.trainingConfigSelect.options).some((opt) => opt.value === path);
  if (!exists) {
    const opt = document.createElement('option');
    opt.value = path;
    opt.textContent = path.split('/').slice(-2).join('/');
    els.trainingConfigSelect.appendChild(opt);
  }
  els.trainingConfigSelect.value = path;
}

export async function loadInitCheckpoints() {
  const priorSelection = els.runInitCheckpointSelect.value;
  const data = await fetchJson('/api/checkpoints');
  const items = data.checkpoints || [];
  els.runInitCheckpointSelect.innerHTML = '';
  const none = document.createElement('option');
  none.value = '';
  none.textContent = 'Start fresh (no checkpoint)';
  els.runInitCheckpointSelect.appendChild(none);
  const available = new Set();
  items.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.checkpoint_path;
    const name = item.run_name ? `${item.run_name} (${item.run_id})` : item.run_id;
    const meta = item.iter != null ? `iter ${item.iter}` : item.checkpoint;
    opt.textContent = `${name} • ${meta}`;
    els.runInitCheckpointSelect.appendChild(opt);
    available.add(opt.value);
  });
  if (priorSelection && available.has(priorSelection)) {
    els.runInitCheckpointSelect.value = priorSelection;
  }
  checkCheckpointCompatibility();
}

async function checkCheckpointCompatibility() {
  const checkpointPath = els.runInitCheckpointSelect.value;
  if (!checkpointPath) {
    if (els.runInitCheckpointMeta) {
      els.runInitCheckpointMeta.textContent = '';
    }
    return;
  }
  const trainingConfig = els.trainingConfigSelect.value;
  const tokenizerPath = els.runTokenizerVocabSelect.value;
  if (!trainingConfig || !tokenizerPath) {
    if (els.runInitCheckpointMeta) {
      els.runInitCheckpointMeta.textContent = 'Select a tokenizer vocab and training config to validate.';
    }
    return;
  }
  try {
    const data = await api('/api/checkpoint/compat', {
      checkpoint_path: checkpointPath,
      config: trainingConfig,
      tokenizer_path: tokenizerPath,
    });
    if (!els.runInitCheckpointMeta) return;
    if (data.ok) {
      els.runInitCheckpointMeta.textContent = 'Checkpoint compatible.';
    } else {
      const errors = (data.errors || []).join(' | ') || 'Checkpoint incompatible.';
      els.runInitCheckpointMeta.textContent = errors;
    }
  } catch (err) {
    if (els.runInitCheckpointMeta) {
      els.runInitCheckpointMeta.textContent = `Compatibility check failed: ${err.message}`;
    }
  }
}

async function createTrainRun() {
  const datasetManifest = els.datasetSelect.value;
  const tokenizerPath = els.runTokenizerVocabSelect.value;
  const trainingConfig = els.trainingConfigSelect.value;
  const runName = (els.runName?.value || '').trim();
  let initCheckpoint = els.runInitCheckpointSelect.value;
  let initMode = (els.runInitModeSelect.value || '').trim().toLowerCase();
  const overrides = {};
  if (!datasetManifest) {
    els.runSetupMeta.textContent = 'Select a dataset.';
    return;
  }
  if (!tokenizerPath) {
    els.runSetupMeta.textContent = 'Select a tokenizer vocab.';
    return;
  }
  if (!trainingConfig) {
    els.runSetupMeta.textContent = 'Select a training config.';
    return;
  }
  if (!initCheckpoint) {
    initCheckpoint = null;
    initMode = 'fresh';
  } else if (!initMode || initMode === 'fresh') {
    initMode = 'fork';
  }
  const lr = (els.runOverrideLearningRate?.value || '').trim();
  if (lr) {
    const parsed = Number(lr);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      els.runSetupMeta.textContent = 'Invalid learning rate override.';
      return;
    }
    overrides['training.learning_rate'] = String(parsed);
  }
  const maxIters = (els.runOverrideMaxIters?.value || '').trim();
  if (maxIters) {
    const parsed = Number(maxIters);
    if (!Number.isInteger(parsed) || parsed <= 0) {
      els.runSetupMeta.textContent = 'Invalid max iters override.';
      return;
    }
    overrides['training.max_iters'] = String(parsed);
  }
  const batchSize = (els.runOverrideBatchSize?.value || '').trim();
  if (batchSize) {
    const parsed = Number(batchSize);
    if (!Number.isInteger(parsed) || parsed <= 0) {
      els.runSetupMeta.textContent = 'Invalid batch size override.';
      return;
    }
    overrides['training.batch_size'] = String(parsed);
  }
  const evalInterval = (els.runOverrideEvalInterval?.value || '').trim();
  if (evalInterval) {
    const parsed = Number(evalInterval);
    if (!Number.isInteger(parsed) || parsed <= 0) {
      els.runSetupMeta.textContent = 'Invalid eval interval override.';
      return;
    }
    overrides['training.eval_interval'] = String(parsed);
  }
  const warmupIters = (els.runOverrideWarmupIters?.value || '').trim();
  if (warmupIters) {
    const parsed = Number(warmupIters);
    if (!Number.isInteger(parsed) || parsed < 0) {
      els.runSetupMeta.textContent = 'Invalid warmup iters override.';
      return;
    }
    overrides['training.warmup_iters'] = String(parsed);
  }
  const seed = (els.runOverrideSeed?.value || '').trim();
  if (seed) {
    const parsed = Number(seed);
    if (!Number.isInteger(parsed) || parsed <= 0) {
      els.runSetupMeta.textContent = 'Invalid seed override.';
      return;
    }
    overrides.seed = String(parsed);
  }
  const device = (els.runOverrideDevice?.value || '').trim();
  if (device) {
    overrides['training.device'] = device;
  }
  const payload = {
    stage: 'train',
    config: trainingConfig,
    dataset_manifest: datasetManifest,
    tokenizer_path: tokenizerPath,
    run_name: runName || null,
    init_checkpoint: initCheckpoint,
    init_mode: initCheckpoint ? initMode : null,
    overrides,
  };
  const data = await api('/api/run', payload);
  els.runSetupMeta.textContent = data.run_id ? `started train (${data.run_id})` : data.error || 'run failed';
  await refreshRunList();
}

export function initRunSetup() {
  els.runCreateBtn.addEventListener('click', createTrainRun);
  els.refreshDatasetsBtn.addEventListener('click', loadDatasets);
  els.refreshTokenizerVocabsBtn.addEventListener('click', loadTokenizerVocabs);
  els.refreshTrainingConfigsBtn.addEventListener('click', loadTrainingConfigs);
  els.refreshInitCheckpointsBtn.addEventListener('click', loadInitCheckpoints);
  els.runInitCheckpointSelect.addEventListener('change', checkCheckpointCompatibility);
  els.trainingConfigSelect.addEventListener('change', checkCheckpointCompatibility);
  els.runTokenizerVocabSelect.addEventListener('change', checkCheckpointCompatibility);
  loadDatasets();
  loadTokenizerVocabs();
  loadTrainingConfigs();
  loadInitCheckpoints();
  onEvent('datasets:changed', loadDatasets);
  onEvent('tokenizer_vocabs:changed', loadTokenizerVocabs);
  onEvent('training_configs:changed', loadTrainingConfigs);
  scheduleAutoRefresh({
    intervalMs: 30000,
    isEnabled: () => isSectionActive('runs'),
    task: loadDatasets,
  });
  scheduleAutoRefresh({
    intervalMs: 30000,
    isEnabled: () => isSectionActive('runs'),
    task: loadTokenizerVocabs,
  });
  scheduleAutoRefresh({
    intervalMs: 30000,
    isEnabled: () => isSectionActive('runs'),
    task: loadTrainingConfigs,
  });
  scheduleAutoRefresh({
    intervalMs: 30000,
    isEnabled: () => isSectionActive('runs'),
    task: loadInitCheckpoints,
  });
}
