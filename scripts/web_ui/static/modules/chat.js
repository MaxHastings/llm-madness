import { api } from './api.js';
import { els } from './dom.js';
import { isSectionActive, scheduleAutoRefresh } from './auto_refresh.js';

const state = {
  messages: [],
  endIds: [],
  busy: false,
  activeAssistantIndex: null,
};

function endsWithSequence(values, seq) {
  if (!seq.length || values.length < seq.length) return false;
  const offset = values.length - seq.length;
  for (let i = 0; i < seq.length; i += 1) {
    if (values[offset + i] !== seq[i]) return false;
  }
  return true;
}

function buildRawTranscript() {
  let raw = '';
  state.messages.forEach((msg) => {
    if (msg.role === 'user') {
      raw += `<|user|>\n${msg.text}\n`;
      return;
    }
    raw += `<|assistant|>\n${msg.text}\n`;
    if (msg.hasEnd) {
      raw += '<|end|>\n';
    }
  });
  return raw;
}

function buildPromptWithAssistantCue() {
  return `${buildRawTranscript()}<|assistant|>\n`;
}

function renderMessages() {
  els.chatMessages.innerHTML = '';
  if (!state.messages.length) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'No messages yet.';
    els.chatMessages.appendChild(empty);
    return;
  }
  state.messages.forEach((msg) => {
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${msg.role}`;
    const role = document.createElement('div');
    role.className = 'chat-role';
    role.textContent = msg.role;
    const text = document.createElement('div');
    text.className = 'chat-text';
    text.textContent = msg.text || (msg.isStreaming ? '…' : '');
    bubble.appendChild(role);
    bubble.appendChild(text);
    els.chatMessages.appendChild(bubble);
  });
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function renderTranscript() {
  if (els.chatTranscript) {
    els.chatTranscript.value = buildRawTranscript();
  }
}

function updateDebugView() {
  const showDebug = !!els.chatDebugToggle?.checked;
  const rawWrap = els.chatTranscript?.closest('.chat-raw-wrap');
  if (rawWrap) {
    rawWrap.classList.toggle('is-hidden', !showDebug);
  }
  if (els.chatMessages) {
    els.chatMessages.classList.toggle('is-hidden', showDebug);
  }
  if (els.chatTranscriptMeta) {
    els.chatTranscriptMeta.textContent = showDebug
      ? 'Debug view shows special tokens.'
      : 'Special tokens are hidden.';
  }
}

function setStatus(text) {
  if (els.chatStatus) {
    els.chatStatus.textContent = text || '';
  }
}

function addMessage(role, text) {
  state.messages.push({ role, text, hasEnd: false, isStreaming: false });
  renderMessages();
  renderTranscript();
}

function startAssistantMessage() {
  const msg = { role: 'assistant', text: '', hasEnd: false, isStreaming: true };
  state.messages.push(msg);
  state.activeAssistantIndex = state.messages.length - 1;
  renderMessages();
  renderTranscript();
}

function updateAssistantMessage(text, { hasEnd = false, done = false } = {}) {
  const idx = state.activeAssistantIndex;
  if (idx == null || idx < 0 || idx >= state.messages.length) return;
  const msg = state.messages[idx];
  msg.text = text;
  if (hasEnd) msg.hasEnd = true;
  msg.isStreaming = !done;
  if (done) {
    state.activeAssistantIndex = null;
  }
  renderMessages();
  renderTranscript();
}

async function loadEndTokenIds() {
  try {
    const data = await api('/api/tokenize', { text: '<|end|>' });
    if (Array.isArray(data.ids) && data.ids.length) {
      state.endIds = data.ids;
      return;
    }
  } catch (err) {
    // Ignore; fallback below.
  }
  state.endIds = [];
}

function clampNumber(value, min, max, fallback) {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(min, Math.min(value, max));
}

async function generateAssistant() {
  if (state.busy) return;
  const last = state.messages[state.messages.length - 1];
  if (!last || last.role !== 'user') {
    setStatus('add a user message first');
    return;
  }
  state.busy = true;
  els.chatSendBtn.disabled = true;
  setStatus('generating...');
  const prompt = buildPromptWithAssistantCue();
  startAssistantMessage();
  let latestText = '';
  try {
    const tokenized = await api('/api/tokenize', { text: prompt });
    const ids = Array.isArray(tokenized.ids) ? tokenized.ids.slice() : [];
    if (!ids.length) {
      state.messages.pop();
      state.activeAssistantIndex = null;
      renderMessages();
      renderTranscript();
      setStatus('prompt tokenization returned no ids');
      return;
    }
    const maxTokens = clampNumber(parseInt(els.chatMaxTokens.value || '128', 10), 1, 512, 128);
    const temperature = clampNumber(parseFloat(els.chatTemperature.value || '1.0'), 0, 2, 1.0);
    const topP = clampNumber(parseFloat(els.chatTopP.value || '0.9'), 0, 1, 1.0);
    const topK = clampNumber(parseInt(els.chatTopK.value || '0', 10), 0, 200, 0);
    const repetitionPenalty = clampNumber(parseFloat(els.chatRepetitionPenalty.value || '1.0'), 1, 3, 1.0);
    els.chatMaxTokens.value = maxTokens;
    els.chatTemperature.value = temperature;
    els.chatTopP.value = topP;
    els.chatTopK.value = topK;
    els.chatRepetitionPenalty.value = repetitionPenalty;

    let generated = [];
    let hitEnd = false;
    let lastDecodedLength = 0;
    let lastDecodeAt = performance.now();
    const decodeBatch = async (force = false) => {
      if (!generated.length) return;
      const now = performance.now();
      const delta = generated.length - lastDecodedLength;
      if (!force && delta < 4 && now - lastDecodeAt < 160) return;
      const decoded = await api('/api/decode', { ids: generated });
      latestText = decoded.text || '';
      lastDecodedLength = generated.length;
      lastDecodeAt = now;
      updateAssistantMessage(latestText);
    };
    for (let i = 0; i < maxTokens; i += 1) {
      const data = await api('/api/sample', {
        ids,
        temperature,
        top_p: topP,
        top_k: topK,
        repetition_penalty: repetitionPenalty,
      });
      if (!data || data.error) {
        setStatus(data?.error || 'generation failed');
        break;
      }
      if (data.id == null) break;
      ids.push(data.id);
      generated.push(data.id);
      if (state.endIds.length && endsWithSequence(ids, state.endIds)) {
        hitEnd = true;
        if (endsWithSequence(generated, state.endIds)) {
          generated = generated.slice(0, generated.length - state.endIds.length);
        }
        break;
      }
      await decodeBatch();
    }

    await decodeBatch(true);
    updateAssistantMessage(latestText, { hasEnd: hitEnd, done: true });
    const stopLabel = hitEnd ? 'hit <|end|>' : `max ${maxTokens}`;
    setStatus(`generated ${generated.length} tokens (${stopLabel})`);
  } catch (err) {
    updateAssistantMessage(latestText, { done: true });
    setStatus(err?.message || 'generation failed');
  } finally {
    state.busy = false;
    els.chatSendBtn.disabled = false;
  }
}

async function handleSend() {
  const text = els.chatInput.value.trim();
  if (text) {
    addMessage('user', text);
    els.chatInput.value = '';
  }
  await generateAssistant();
}

function clearChat() {
  state.messages = [];
  state.activeAssistantIndex = null;
  renderMessages();
  renderTranscript();
  setStatus('');
  if (els.chatInput) {
    els.chatInput.value = '';
  }
}

async function refreshChatRuns() {
  const priorSelection = els.chatSessionRunSelect.value;
  const data = await api('/api/runs', { scope: 'all' });
  const runs = (data.runs || []).filter((run) => run.stage === 'train');
  runs.sort((a, b) => (a.start_time || '') < (b.start_time || '') ? 1 : -1);
  els.chatSessionRunSelect.innerHTML = '';
  if (!runs.length) {
    els.chatSessionRunMeta.textContent = 'no training runs found';
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
    const label = run.run_name ? `${run.run_name} - ${run.run_id || baseName(run.run_dir)}` : (run.run_id || baseName(run.run_dir) || 'unknown');
    opt.textContent = `${label} (${run.status || 'unknown'})`;
    els.chatSessionRunSelect.appendChild(opt);
    if (priorSelection && priorSelection === run.run_dir) {
      selectionStillExists = true;
    }
  });
  if (selectionStillExists) {
    els.chatSessionRunSelect.value = priorSelection;
  }
  els.chatSessionRunMeta.textContent = `${runs.length} training runs`;
}

async function refreshChatCheckpoints() {
  try {
    const data = await api('/api/checkpoints');
    els.chatCheckpointSelect.innerHTML = '';
    data.checkpoints.forEach((ckpt) => {
      const opt = document.createElement('option');
      opt.value = ckpt;
      opt.textContent = ckpt;
      if (ckpt === data.current) opt.selected = true;
      els.chatCheckpointSelect.appendChild(opt);
    });
    const runLabel = data.run_name ? `${data.run_name} - ${data.run_dir}` : data.run_dir;
    els.chatRunMeta.textContent = runLabel;
    els.chatCheckpointMeta.textContent = data.checkpoints.length ? '' : 'no checkpoints found';
  } catch (err) {
    els.chatCheckpointSelect.innerHTML = '';
    els.chatCheckpointMeta.textContent = 'load a training run to see checkpoints';
  }
}

async function loadChatRun() {
  const runDir = els.chatSessionRunSelect.value;
  if (!runDir) return;
  const data = await api('/api/load_run', { run_dir: runDir });
  if (data.run_name) {
    els.chatRunMeta.textContent = `${data.run_name} - ${data.run_dir || runDir}`;
  } else {
    els.chatRunMeta.textContent = data.run_dir || runDir || els.chatRunMeta.textContent;
  }
  els.chatCheckpointMeta.textContent = data.status || data.error || 'failed to load run';
  await refreshChatCheckpoints();
  await loadEndTokenIds();
}

async function loadChatCheckpoint() {
  const picked = els.chatCheckpointSelect.value;
  if (!picked) return;
  const data = await api('/api/load_checkpoint', { checkpoint: picked });
  els.chatCheckpointMeta.textContent = data.status || data.error || 'failed to load checkpoint';
}

export function initChat() {
  if (!els.chatSendBtn) return;
  els.chatSendBtn.addEventListener('click', handleSend);
  els.chatClearBtn.addEventListener('click', clearChat);
  els.chatInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  });
  els.chatDebugToggle.addEventListener('change', updateDebugView);
  els.chatLoadSessionRunBtn.addEventListener('click', loadChatRun);
  els.chatLoadCheckpointBtn.addEventListener('click', loadChatCheckpoint);
  renderMessages();
  renderTranscript();
  updateDebugView();
  refreshChatCheckpoints();
  refreshChatRuns();
  loadEndTokenIds();
  scheduleAutoRefresh({
    intervalMs: 15000,
    isEnabled: () => isSectionActive('chat'),
    task: refreshChatCheckpoints,
  });
  scheduleAutoRefresh({
    intervalMs: 15000,
    isEnabled: () => isSectionActive('chat'),
    task: refreshChatRuns,
  });
}
