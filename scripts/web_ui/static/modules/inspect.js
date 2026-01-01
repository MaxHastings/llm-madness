import { api } from './api.js';
import { els } from './dom.js';
import { state } from './state.js';
import { appendToken } from './prompt.js';

function renderLayerTopk(layers) {
  els.layerTopkTableBody.innerHTML = '';
  layers.forEach((layer) => {
    const tr = document.createElement('tr');
    const layerTd = document.createElement('td');
    layerTd.textContent = `${layer.layer}`;
    const tokensTd = document.createElement('td');
    layer.topk.forEach((row) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'token-btn';
      btn.textContent = `${row.token} (${row.prob.toFixed(3)})`;
      btn.addEventListener('click', () => appendToken(row.id));
      tokensTd.appendChild(btn);
    });
    tr.appendChild(layerTd);
    tr.appendChild(tokensTd);
    els.layerTopkTableBody.appendChild(tr);
  });
}

function appendRolloutToken(token, index) {
  const el = document.createElement('span');
  el.className = 'token';
  el.textContent = `${index}:${token}`;
  el.title = `step ${index}`;
  els.inspectRollout.appendChild(el);
}

function setInspectorTab(tab) {
  const previewRoot = els.inspectorTabSelect?.closest('.inspector-preview');
  const panels = previewRoot ? previewRoot.querySelectorAll('.preview-panel') : [];
  if (els.inspectorTabSelect) {
    els.inspectorTabSelect.value = tab;
  }
  panels.forEach((panel) => {
    panel.classList.toggle('is-hidden', panel.dataset.preview !== tab);
  });
}

async function runGreedyRollout() {
  if (!state.ids.length) return;
  const rawCount = parseInt(els.inspectRolloutCount.value || '32', 10);
  if (!Number.isFinite(rawCount) || rawCount < 1) return;
  const count = Math.min(rawCount, 256);
  els.inspectRollout.innerHTML = '';
  els.inspectRolloutMeta.textContent = 'generating...';
  els.inspectRolloutBtn.disabled = true;
  let generated = 0;
  try {
    for (let i = 0; i < count; i += 1) {
      const data = await api('/api/next', { ids: state.ids, top_k: 1 });
      if (!data.topk || !data.topk.length) break;
      const choice = data.topk[0];
      await appendToken(choice.id);
      appendRolloutToken(choice.token, i);
      generated += 1;
    }
  } finally {
    els.inspectRolloutBtn.disabled = false;
    els.inspectRolloutMeta.textContent = generated ? `greedy rollout: ${generated} tokens` : '';
  }
}

export function initInspect() {
  if (els.inspectorTabSelect) {
    els.inspectorTabSelect.addEventListener('change', (event) => {
      setInspectorTab(event.target.value);
    });
    setInspectorTab('tokens');
  }
  els.inspectBtn.addEventListener('click', async () => {
    if (!state.ids.length) return;
    const topK = parseInt(els.inspectTopK.value || '8', 10);
    const payload = { ids: state.ids, top_k: topK };
    if (state.activeToken && Number.isFinite(state.activeToken.index)) {
      payload.index = state.activeToken.index;
    }
    const data = await api('/api/inspect', payload);
    if (data.error) {
      els.inspectMeta.textContent = data.error;
      renderLayerTopk([]);
      return;
    }
    els.inspectMeta.textContent = data.meta || '';
    renderLayerTopk(data.layers || []);
  });

  els.inspectRolloutBtn.addEventListener('click', runGreedyRollout);
}
