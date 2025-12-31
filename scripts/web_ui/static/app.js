const promptInput = document.getElementById('promptInput');
const promptMeta = document.getElementById('promptMeta');
const tokenList = document.getElementById('tokenList');
const nextMeta = document.getElementById('nextMeta');
const nextTableBody = document.getElementById('nextTable').querySelector('tbody');
const checkpointSelect = document.getElementById('checkpointSelect');
const checkpointMeta = document.getElementById('checkpointMeta');
const runMeta = document.getElementById('runMeta');
const runSelect = document.getElementById('runSelect');
const runStatus = document.getElementById('runStatus');
const tokenReportMeta = document.getElementById('tokenReportMeta');
const tokenReportTable = document.getElementById('tokenReportTable').querySelector('tbody');
const lossChart = document.getElementById('lossChart');
const lossMeta = document.getElementById('lossMeta');
const sampleList = document.getElementById('sampleList');
const inspectMeta = document.getElementById('inspectMeta');
const inspectLegend = document.getElementById('inspectLegend');
const inspectTable = document.getElementById('inspectTable').querySelector('tbody');
const inspectHeatmap = document.getElementById('inspectHeatmap');
const inspectTokens = document.getElementById('inspectTokens');
const layerTopkWrap = document.getElementById('layerTopkWrap');
const layerTopkTable = document.getElementById('layerTopkTable').querySelector('tbody');
const configSelect = document.getElementById('configSelect');
const loadConfigBtn = document.getElementById('loadConfig');
const saveConfigBtn = document.getElementById('saveConfig');
const configEditor = document.getElementById('configEditor');
const configMeta = document.getElementById('configMeta');
const runsList = document.getElementById('runsList');
const refreshRunsBtn = document.getElementById('refreshRuns');
const runsMeta = document.getElementById('runsMeta');
const runDetail = document.getElementById('runDetail');
const runDetailMeta = document.getElementById('runDetailMeta');
const runPipelineBtn = document.getElementById('runPipeline');
const runGenerateBtn = document.getElementById('runGenerate');
const runTokenizerBtn = document.getElementById('runTokenizer');
const runTrainBtn = document.getElementById('runTrain');

const state = { ids: [], tokens: [] };

async function api(path, payload) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg);
  }
  return res.json();
}

async function loadConfigs() {
  const data = await api('/api/configs');
  configSelect.innerHTML = '';
  data.configs.forEach((name) => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    configSelect.appendChild(opt);
  });
  if (data.configs.length) {
    const preferred = data.configs.find((name) => name === 'pipeline.json') || data.configs[0];
    configSelect.value = preferred;
  }
}

async function loadSelectedConfig() {
  const name = configSelect.value;
  if (!name) return;
  const res = await fetch(`/api/configs/${encodeURIComponent(name)}`);
  const data = await res.json();
  if (data.raw) {
    configEditor.value = data.raw;
    configMeta.textContent = `loaded ${name}`;
  } else {
    configMeta.textContent = data.error || 'failed to load config';
  }
}

function parseConfigEditor() {
  const raw = configEditor.value;
  try {
    JSON.parse(raw);
    return { ok: true, raw };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

async function saveSelectedConfig() {
  const name = configSelect.value;
  const parsed = parseConfigEditor();
  if (!parsed.ok) {
    configMeta.textContent = `invalid json: ${parsed.error}`;
    return;
  }
  const data = await api('/api/configs/save', { name, raw: parsed.raw });
  configMeta.textContent = data.status ? `saved ${name}` : data.error || 'save failed';
}

async function startRun(stage) {
  const name = configSelect.value;
  if (!name) return;
  const parsed = parseConfigEditor();
  if (!parsed.ok) {
    configMeta.textContent = `invalid json: ${parsed.error}`;
    return;
  }
  await saveSelectedConfig();
  const data = await api('/api/run', { stage, config: name });
  runsMeta.textContent = data.run_id ? `started ${stage} (${data.run_id})` : data.error || 'run failed';
  await refreshRunList();
}

function renderRunList(items) {
  runsList.innerHTML = '';
  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'run-item';
    const info = document.createElement('div');
    const stage = document.createElement('div');
    stage.className = 'run-stage';
    stage.textContent = item.stage || 'unknown';
    const status = document.createElement('div');
    status.className = 'run-status';
    status.textContent = item.status || 'unknown';
    info.appendChild(stage);
    info.appendChild(status);
    const actions = document.createElement('div');
    const viewBtn = document.createElement('button');
    viewBtn.textContent = 'View';
    viewBtn.addEventListener('click', () => showRunDetails(item.run_dir));
    const stopBtn = document.createElement('button');
    stopBtn.textContent = 'Stop';
    stopBtn.addEventListener('click', async () => {
      const runId = item.run_dir.split('/').pop();
      await api(`/api/stop/${encodeURIComponent(runId)}`);
      await refreshRunList();
    });
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = 'Delete';
    deleteBtn.addEventListener('click', async () => {
      await api('/api/run/delete', { run_dir: item.run_dir });
      runDetail.textContent = '';
      runDetailMeta.textContent = '';
      await refreshRunList();
    });
    actions.appendChild(viewBtn);
    actions.appendChild(stopBtn);
    actions.appendChild(deleteBtn);
    row.appendChild(info);
    row.appendChild(actions);
    runsList.appendChild(row);
  });
}

async function showRunDetails(runDir) {
  const res = await fetch(`/api/run/${encodeURIComponent(runDir)}`);
  const data = await res.json();
  runDetailMeta.textContent = data.run_dir || 'run detail';
  const payload = {
    manifest: data.manifest || null,
    logs: data.logs || [],
    process_log: data.process_log || [],
  };
  runDetail.textContent = JSON.stringify(payload, null, 2);
}

async function refreshRunList() {
  const data = await api('/api/runs', { scope: 'all' });
  renderRunList(data.runs || []);
  runsMeta.textContent = `${(data.runs || []).length} runs`;
}

function renderTokens(tokens, ids) {
  tokenList.innerHTML = '';
  tokens.forEach((tok, i) => {
    const el = document.createElement('span');
    el.className = 'token';
    el.textContent = `${i}: ${tok} (${ids[i]})`;
    el.title = `index ${i} id ${ids[i]}`;
    tokenList.appendChild(el);
  });
}

function renderInspectTokens(tokens) {
  inspectTokens.innerHTML = '';
  tokens.forEach((tok, i) => {
    const el = document.createElement('span');
    el.className = 'token';
    el.textContent = `${i}:${tok}`;
    el.title = `index ${i}`;
    inspectTokens.appendChild(el);
  });
}

function renderLayerTopk(layers) {
  layerTopkTable.innerHTML = '';
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
    layerTopkTable.appendChild(tr);
  });
}

function renderTopK(rows) {
  nextTableBody.innerHTML = '';
  rows.forEach((row, idx) => {
    const tr = document.createElement('tr');
    const rank = document.createElement('td');
    rank.textContent = `${idx + 1}`;

    const tokenTd = document.createElement('td');
    const tokenBtn = document.createElement('button');
    tokenBtn.className = 'token-btn';
    tokenBtn.type = 'button';
    tokenBtn.textContent = row.token;
    tokenBtn.addEventListener('click', () => appendToken(row.id));
    tokenTd.appendChild(tokenBtn);

    const idTd = document.createElement('td');
    idTd.textContent = `${row.id}`;

    const probTd = document.createElement('td');
    probTd.textContent = row.prob.toFixed(4);

    tr.appendChild(rank);
    tr.appendChild(tokenTd);
    tr.appendChild(idTd);
    tr.appendChild(probTd);
    nextTableBody.appendChild(tr);
  });
}

function renderLossChart(logs) {
  lossChart.innerHTML = '';
  if (!logs.length) {
    lossChart.textContent = 'No logs yet.';
    return;
  }

  const margin = { top: 12, right: 16, bottom: 28, left: 40 };
  const width = lossChart.clientWidth;
  const height = lossChart.clientHeight;
  const plotW = Math.max(10, width - margin.left - margin.right);
  const plotH = Math.max(10, height - margin.top - margin.bottom);

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);

  const losses = logs.filter((row) => row.train_loss != null);
  const vals = logs.filter((row) => row.val_loss != null);
  const all = losses.concat(vals);
  const minLoss = Math.min(...all.map((row) => row.train_loss ?? row.val_loss));
  const maxLoss = Math.max(...all.map((row) => row.train_loss ?? row.val_loss));
  const minIter = Math.min(...logs.map((row) => row.iter));
  const maxIter = Math.max(...logs.map((row) => row.iter));

  function scaleX(iter) {
    if (maxIter === minIter) return margin.left;
    return margin.left + ((iter - minIter) / (maxIter - minIter)) * plotW;
  }

  function scaleY(loss) {
    if (maxLoss === minLoss) return margin.top + plotH / 2;
    return margin.top + (1 - (loss - minLoss) / (maxLoss - minLoss)) * plotH;
  }

  function drawLine(rows, color, key) {
    if (!rows.length) return;
    const path = rows
      .map((row, idx) => {
        const x = scaleX(row.iter);
        const y = scaleY(row[key]);
        return `${idx === 0 ? 'M' : 'L'}${x} ${y}`;
      })
      .join(' ');
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', path);
    p.setAttribute('fill', 'none');
    p.setAttribute('stroke', color);
    p.setAttribute('stroke-width', '2');
    svg.appendChild(p);
  }

  function drawAxis() {
    const axisColor = '#cbbfb1';
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', margin.left);
    xAxis.setAttribute('y1', margin.top + plotH);
    xAxis.setAttribute('x2', margin.left + plotW);
    xAxis.setAttribute('y2', margin.top + plotH);
    xAxis.setAttribute('stroke', axisColor);
    svg.appendChild(xAxis);

    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', margin.left);
    yAxis.setAttribute('y1', margin.top);
    yAxis.setAttribute('x2', margin.left);
    yAxis.setAttribute('y2', margin.top + plotH);
    yAxis.setAttribute('stroke', axisColor);
    svg.appendChild(yAxis);

    const ticks = 4;
    for (let i = 0; i <= ticks; i += 1) {
      const t = i / ticks;
      const iter = minIter + t * (maxIter - minIter);
      const x = scaleX(iter);
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', x);
      label.setAttribute('y', margin.top + plotH + 18);
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('font-size', '10');
      label.setAttribute('fill', '#6e665d');
      label.textContent = Math.round(iter);
      svg.appendChild(label);
    }

    for (let i = 0; i <= ticks; i += 1) {
      const t = i / ticks;
      const loss = minLoss + t * (maxLoss - minLoss);
      const y = scaleY(loss);
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', margin.left - 6);
      label.setAttribute('y', y + 3);
      label.setAttribute('text-anchor', 'end');
      label.setAttribute('font-size', '10');
      label.setAttribute('fill', '#6e665d');
      label.textContent = loss.toFixed(2);
      svg.appendChild(label);
    }

    const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xLabel.setAttribute('x', margin.left + plotW / 2);
    xLabel.setAttribute('y', height - 4);
    xLabel.setAttribute('text-anchor', 'middle');
    xLabel.setAttribute('font-size', '10');
    xLabel.setAttribute('fill', '#6e665d');
    xLabel.textContent = 'iteration';
    svg.appendChild(xLabel);

    const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yLabel.setAttribute('x', 10);
    yLabel.setAttribute('y', margin.top + plotH / 2);
    yLabel.setAttribute('text-anchor', 'middle');
    yLabel.setAttribute('font-size', '10');
    yLabel.setAttribute('fill', '#6e665d');
    yLabel.setAttribute('transform', `rotate(-90 10 ${margin.top + plotH / 2})`);
    yLabel.textContent = 'loss';
    svg.appendChild(yLabel);
  }

  function drawLegend() {
    const legend = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    legend.setAttribute('x', margin.left + 6);
    legend.setAttribute('y', margin.top + 12);
    legend.setAttribute('font-size', '10');
    legend.setAttribute('fill', '#6e665d');
    legend.textContent = 'train (orange) / val (blue)';
    svg.appendChild(legend);
  }

  drawAxis();
  drawLine(losses, '#ff6b35', 'train_loss');
  drawLine(vals, '#1f7a8c', 'val_loss');
  drawLegend();
  lossChart.appendChild(svg);
}

function renderHeatmap(matrix) {
  const ctx = inspectHeatmap.getContext('2d');
  const width = inspectHeatmap.clientWidth;
  const height = inspectHeatmap.clientHeight;
  inspectHeatmap.width = width;
  inspectHeatmap.height = height;
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

async function refreshCheckpoints() {
  const data = await api('/api/checkpoints');
  checkpointSelect.innerHTML = '';
  data.checkpoints.forEach((ckpt) => {
    const opt = document.createElement('option');
    opt.value = ckpt;
    opt.textContent = ckpt;
    if (ckpt === data.current) opt.selected = true;
    checkpointSelect.appendChild(opt);
  });
  runMeta.textContent = data.run_dir;
}

async function refreshRuns() {
  const data = await api('/api/runs');
  runSelect.innerHTML = '';
  data.runs.forEach((run) => {
    const opt = document.createElement('option');
    opt.value = run;
    opt.textContent = run;
    if (run === data.current) opt.selected = true;
    runSelect.appendChild(opt);
  });
}

async function refreshTokenizerReport() {
  const data = await api('/api/tokenizer_report');
  if (data.error) {
    tokenReportMeta.textContent = data.error;
    return;
  }
  tokenReportMeta.textContent = `tokens: ${data.total_tokens} unique: ${data.unique_tokens} vocab: ${data.vocab_size} coverage: ${(data.coverage * 100).toFixed(2)}% unk_rate: ${(data.unk_rate * 100).toFixed(2)}%`;
  tokenReportTable.innerHTML = '';
  data.top_tokens.forEach((row) => {
    const tr = document.createElement('tr');
    const tokenTd = document.createElement('td');
    tokenTd.textContent = row.token;
    const idTd = document.createElement('td');
    idTd.textContent = `${row.id}`;
    const countTd = document.createElement('td');
    countTd.textContent = `${row.count}`;
    tr.appendChild(tokenTd);
    tr.appendChild(idTd);
    tr.appendChild(countTd);
    tokenReportTable.appendChild(tr);
  });
}

async function refreshTrainingLogs() {
  const data = await api('/api/training_logs');
  renderLossChart(data.logs || []);
  if (data.logs && data.logs.length) {
    const last = data.logs[data.logs.length - 1];
    const trainPpl = last.train_ppl != null ? last.train_ppl.toFixed(2) : '-';
    const valPpl = last.val_ppl != null ? last.val_ppl.toFixed(2) : '-';
    lossMeta.textContent = `latest iter ${last.iter} train ${last.train_loss ?? '-'} val ${last.val_loss ?? '-'} ppl ${trainPpl}/${valPpl}`;
  } else {
    lossMeta.textContent = 'no logs found';
  }
  sampleList.textContent = '';
  (data.samples || []).forEach((row) => {
    sampleList.textContent += `[${row.iter}] ${row.sample}\n\n`;
  });
}

async function updateFromIds() {
  if (!state.ids.length) {
    state.tokens = [];
    renderTokens([], []);
    promptMeta.textContent = 'no tokens';
    return;
  }
  const tokensData = await api('/api/ids_to_tokens', { ids: state.ids });
  state.tokens = tokensData.tokens;
  renderTokens(state.tokens, state.ids);
  const decoded = await api('/api/decode', { ids: state.ids });
  promptInput.value = decoded.text;
  promptMeta.textContent = `chars: ${decoded.text.length} tokens: ${state.ids.length}`;
}

async function probeNext() {
  if (!state.ids.length) return;
  const topK = parseInt(document.getElementById('topK').value || '8', 10);
  const data = await api('/api/next', { ids: state.ids, top_k: topK });
  renderTopK(data.topk);
  nextMeta.textContent = `loaded from ${data.checkpoint}`;
}

async function appendToken(tokenId) {
  state.ids.push(tokenId);
  await updateFromIds();
  await probeNext();
}

loadConfigBtn.addEventListener('click', loadSelectedConfig);
saveConfigBtn.addEventListener('click', saveSelectedConfig);
refreshRunsBtn.addEventListener('click', refreshRunList);
runPipelineBtn.addEventListener('click', () => startRun('pipeline'));
runGenerateBtn.addEventListener('click', () => startRun('generate'));
runTokenizerBtn.addEventListener('click', () => startRun('tokenizer'));
runTrainBtn.addEventListener('click', () => startRun('train'));
configSelect.addEventListener('change', loadSelectedConfig);

document.getElementById('tokenizeBtn').addEventListener('click', async () => {
  const data = await api('/api/tokenize', { text: promptInput.value });
  state.ids = data.ids;
  state.tokens = data.tokens;
  renderTokens(data.tokens, data.ids);
  promptMeta.textContent = `chars: ${promptInput.value.length} tokens: ${data.ids.length}`;
});

document.getElementById('decodeBtn').addEventListener('click', async () => {
  if (!state.ids.length) return;
  const data = await api('/api/decode', { ids: state.ids });
  promptInput.value = data.text;
});

document.getElementById('nextBtn').addEventListener('click', probeNext);

document.getElementById('loadCheckpoint').addEventListener('click', async () => {
  const picked = checkpointSelect.value;
  const data = await api('/api/load_checkpoint', { checkpoint: picked });
  checkpointMeta.textContent = data.status;
});

document.getElementById('loadRun').addEventListener('click', async () => {
  const picked = runSelect.value;
  const data = await api('/api/load_run', { run_dir: picked });
  runStatus.textContent = data.status;
  state.ids = [];
  state.tokens = [];
  renderTokens([], []);
  promptMeta.textContent = 'no tokens';
  runMeta.textContent = data.run_dir || runMeta.textContent;
  await refreshCheckpoints();
  await refreshTokenizerReport();
  await refreshTrainingLogs();
});

document.getElementById('tokenReportBtn').addEventListener('click', refreshTokenizerReport);
document.getElementById('refreshLogs').addEventListener('click', refreshTrainingLogs);

document.getElementById('inspectBtn').addEventListener('click', async () => {
  if (!state.ids.length) return;
  const layer = parseInt(document.getElementById('inspectLayer').value || '0', 10);
  const head = parseInt(document.getElementById('inspectHead').value || '0', 10);
  const mode = document.getElementById('inspectMode').value;
  const data = await api('/api/inspect', { ids: state.ids, layer, head, mode, top_k: 10 });
  inspectMeta.textContent = data.meta || '';
  renderInspectTokens(data.tokens || []);

  if (mode === 'attention') {
    renderHeatmap(data.attention);
    const minVal = data.min_val != null ? data.min_val.toFixed(4) : '-';
    const maxVal = data.max_val != null ? data.max_val.toFixed(4) : '-';
    inspectLegend.textContent = `rows=query tokens, cols=key tokens | min ${minVal} max ${maxVal}`;
    inspectTable.innerHTML = '';
    layerTopkWrap.style.display = 'none';
    inspectTable.style.display = 'table';
  } else {
    if (mode === 'mlp') {
      renderHeatmap(null);
      inspectLegend.textContent = 'top activations for last token';
      inspectTable.innerHTML = '';
      layerTopkWrap.style.display = 'none';
      inspectTable.style.display = 'table';
      (data.activations || []).forEach((row) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${row.index}</td><td>${row.value.toFixed(4)}</td>`;
        inspectTable.appendChild(tr);
      });
    } else if (mode === 'layer_topk') {
      renderHeatmap(null);
      inspectLegend.textContent = 'layer-wise next-token probabilities (last token only)';
      inspectTable.innerHTML = '';
      inspectTable.style.display = 'none';
      layerTopkWrap.style.display = 'block';
      renderLayerTopk(data.layers || []);
    }
  }
});

refreshCheckpoints();
refreshRuns();
refreshTokenizerReport();
refreshTrainingLogs();
loadConfigs().then(() => loadSelectedConfig());
refreshRunList();
