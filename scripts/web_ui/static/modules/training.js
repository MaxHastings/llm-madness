import { api } from './api.js';
import { els } from './dom.js';

function renderLossChart(logs) {
  els.lossChart.innerHTML = '';
  if (!logs.length) {
    els.lossChart.textContent = 'No logs yet.';
    return;
  }

  const margin = { top: 12, right: 16, bottom: 28, left: 40 };
  const width = els.lossChart.clientWidth;
  const height = els.lossChart.clientHeight;
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
  els.lossChart.appendChild(svg);
}

export async function refreshTokenizerReport() {
  const data = await api('/api/tokenizer_report');
  if (data.error) {
    els.tokenReportMeta.textContent = data.error;
    return;
  }
  els.tokenReportMeta.textContent = `tokens: ${data.total_tokens} unique: ${data.unique_tokens} vocab: ${data.vocab_size} coverage: ${(data.coverage * 100).toFixed(2)}% unk_rate: ${(data.unk_rate * 100).toFixed(2)}%`;
  els.tokenReportTableBody.innerHTML = '';
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
    els.tokenReportTableBody.appendChild(tr);
  });
}

export async function refreshTrainingLogs() {
  const data = await api('/api/training_logs');
  renderLossChart(data.logs || []);
  if (data.logs && data.logs.length) {
    const last = data.logs[data.logs.length - 1];
    const trainPpl = last.train_ppl != null ? last.train_ppl.toFixed(2) : '-';
    const valPpl = last.val_ppl != null ? last.val_ppl.toFixed(2) : '-';
    els.lossMeta.textContent = `latest iter ${last.iter} train ${last.train_loss ?? '-'} val ${last.val_loss ?? '-'} ppl ${trainPpl}/${valPpl}`;
  } else {
    els.lossMeta.textContent = 'no logs found';
  }
  els.sampleList.textContent = '';
  (data.samples || []).forEach((row) => {
    els.sampleList.textContent += `[${row.iter}] ${row.sample}\n\n`;
  });
}

export function initTraining() {
  els.tokenReportBtn.addEventListener('click', refreshTokenizerReport);
  els.refreshLogsBtn.addEventListener('click', refreshTrainingLogs);
  refreshTokenizerReport();
  refreshTrainingLogs();
}
