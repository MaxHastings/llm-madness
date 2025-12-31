import { api } from './api.js';
import { els } from './dom.js';
import { state } from './state.js';
import { appendToken } from './prompt.js';

function renderInspectTokens(tokens) {
  els.inspectTokens.innerHTML = '';
  tokens.forEach((tok, i) => {
    const el = document.createElement('span');
    el.className = 'token';
    el.textContent = `${i}:${tok}`;
    el.title = `index ${i}`;
    els.inspectTokens.appendChild(el);
  });
}

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

function renderHeatmap(matrix) {
  const ctx = els.inspectHeatmap.getContext('2d');
  const width = els.inspectHeatmap.clientWidth;
  const height = els.inspectHeatmap.clientHeight;
  els.inspectHeatmap.width = width;
  els.inspectHeatmap.height = height;
  ctx.clearRect(0, 0, width, height);
  if (!matrix || !matrix.length) {
    ctx.fillStyle = '#6e665d';
    ctx.fillText('No attention data.', 10, 20);
    return;
  }
  const rows = matrix.length;
  const cols = matrix[0].length;
  let min = Infinity;
  let max = -Infinity;
  matrix.forEach((row) => row.forEach((val) => {
    min = Math.min(min, val);
    max = Math.max(max, val);
  }));
  const cellW = width / cols;
  const cellH = height / rows;
  matrix.forEach((row, r) => {
    row.forEach((val, c) => {
      const norm = max === min ? 0.5 : (val - min) / (max - min);
      const alpha = 0.2 + norm * 0.8;
      ctx.fillStyle = `rgba(255, 107, 53, ${alpha})`;
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
    });
  });
}

export function initInspect() {
  els.inspectBtn.addEventListener('click', async () => {
    if (!state.ids.length) return;
    const layer = parseInt(els.inspectLayer.value || '0', 10);
    const head = parseInt(els.inspectHead.value || '0', 10);
    const mode = els.inspectMode.value;
    const data = await api('/api/inspect', { ids: state.ids, layer, head, mode, top_k: 10 });
    els.inspectMeta.textContent = data.meta || '';
    renderInspectTokens(data.tokens || []);

    if (mode === 'attention') {
      renderHeatmap(data.attention);
      const minVal = data.min_val != null ? data.min_val.toFixed(4) : '-';
      const maxVal = data.max_val != null ? data.max_val.toFixed(4) : '-';
      els.inspectLegend.textContent = `rows=query tokens, cols=key tokens | min ${minVal} max ${maxVal}`;
      els.inspectTableBody.innerHTML = '';
      els.layerTopkWrap.style.display = 'none';
      els.inspectTableBody.parentElement.style.display = 'table';
    } else if (mode === 'mlp') {
      renderHeatmap(null);
      els.inspectLegend.textContent = 'top activations for last token';
      els.inspectTableBody.innerHTML = '';
      els.layerTopkWrap.style.display = 'none';
      els.inspectTableBody.parentElement.style.display = 'table';
      (data.activations || []).forEach((row) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${row.index}</td><td>${row.value.toFixed(4)}</td>`;
        els.inspectTableBody.appendChild(tr);
      });
    } else if (mode === 'layer_topk') {
      renderHeatmap(null);
      els.inspectLegend.textContent = 'layer-wise next-token probabilities (last token only)';
      els.inspectTableBody.innerHTML = '';
      els.inspectTableBody.parentElement.style.display = 'none';
      els.layerTopkWrap.style.display = 'block';
      renderLayerTopk(data.layers || []);
    }
  });
}
