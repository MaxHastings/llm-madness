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

async function runSampling() {
  if (!state.ids.length) return;
  const rawSteps = parseInt(els.inspectSampleSteps.value || '32', 10);
  if (!Number.isFinite(rawSteps) || rawSteps < 1) return;
  const steps = Math.min(rawSteps, 256);
  const temperature = parseFloat(els.inspectTemperature.value || '1.0');
  const topP = parseFloat(els.inspectTopP.value || '0.9');
  const topK = parseInt(els.inspectTopKSample.value || '0', 10);
  const repetitionPenalty = parseFloat(els.inspectRepetitionPenalty.value || '1.0');
  const safeTemp = Number.isFinite(temperature) ? Math.max(0, Math.min(temperature, 2)) : 1.0;
  const safeTopP = Number.isFinite(topP) ? Math.max(0, Math.min(topP, 1)) : 1.0;
  const safeTopK = Number.isFinite(topK) ? Math.max(0, Math.min(topK, 200)) : 0;
  const safePenalty = Number.isFinite(repetitionPenalty) ? Math.max(1, Math.min(repetitionPenalty, 3)) : 1.0;
  els.inspectTemperature.value = safeTemp;
  els.inspectTopP.value = safeTopP;
  els.inspectTopKSample.value = safeTopK;
  els.inspectRepetitionPenalty.value = safePenalty;
  if (els.promptRolloutMeta) {
    els.promptRolloutMeta.textContent = 'generating...';
  }
  els.inspectGenerateBtn.disabled = true;
  let generated = 0;
  try {
    for (let i = 0; i < steps; i += 1) {
      const data = await api('/api/sample', {
        ids: state.ids,
        temperature: safeTemp,
        top_p: safeTopP,
        top_k: safeTopK,
        repetition_penalty: safePenalty,
      });
      if (!data || data.error) {
        if (els.promptRolloutMeta) {
          els.promptRolloutMeta.textContent = data?.error || 'generation failed';
        }
        break;
      }
      if (data.id == null) break;
      await appendToken(data.id);
      generated += 1;
    }
  } finally {
    els.inspectGenerateBtn.disabled = false;
    if (els.promptRolloutMeta) {
      els.promptRolloutMeta.textContent = generated
        ? `generated ${generated} tokens @ temp ${safeTemp} top-p ${safeTopP} top-k ${safeTopK || 'off'} repeat ${safePenalty}`
        : '';
    }
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

  els.inspectGenerateBtn.addEventListener('click', runSampling);
}
