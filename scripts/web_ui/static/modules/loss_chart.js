export function renderLossChart(container, logs, options = {}) {
  container.innerHTML = '';
  const emptyMessage = options.emptyMessage || 'No logs yet.';
  if (!logs || !logs.length) {
    container.textContent = emptyMessage;
    return;
  }

  const trainRows = logs.filter((row) => row.train_loss != null);
  const valRows = logs.filter((row) => row.val_loss != null);
  const iterRows = logs.filter((row) => Number.isFinite(row.iter));
  if (!trainRows.length && !valRows.length) {
    container.textContent = emptyMessage;
    return;
  }

  const width = Math.max(container.clientWidth || 0, 280);
  const height = Math.max(container.clientHeight || 0, 160);
  const margin = { top: 12, right: 36, bottom: 28, left: 40 };
  const plotW = Math.max(10, width - margin.left - margin.right);
  const plotH = Math.max(10, height - margin.top - margin.bottom);
  const styles = getComputedStyle(document.documentElement);
  const axisColor = styles.getPropertyValue('--chart-axis').trim() || '#cbbfb1';
  const labelColor = styles.getPropertyValue('--chart-label').trim() || '#6e665d';
  const trainColor = styles.getPropertyValue('--chart-train').trim() || '#ff6b35';
  const valColor = styles.getPropertyValue('--chart-val').trim() || '#1f7a8c';

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);

  const minIter = Math.min(...iterRows.map((row) => row.iter));
  const maxIter = Math.max(...iterRows.map((row) => row.iter));

  function rangeFor(rows, key) {
    if (!rows.length) return null;
    const values = rows.map((row) => row[key]);
    return [Math.min(...values), Math.max(...values)];
  }

  const trainRange = rangeFor(trainRows, 'train_loss');
  const valRange = rangeFor(valRows, 'val_loss');
  const [minTrain, maxTrain] = trainRange || valRange || [0, 1];
  const [minVal, maxVal] = valRange || trainRange || [0, 1];

  function scaleX(iter) {
    if (maxIter === minIter) return margin.left;
    return margin.left + ((iter - minIter) / (maxIter - minIter)) * plotW;
  }

  function scaleY(value, minValLocal, maxValLocal) {
    if (maxValLocal === minValLocal) return margin.top + plotH / 2;
    return margin.top + (1 - (value - minValLocal) / (maxValLocal - minValLocal)) * plotH;
  }

  function drawLine(rows, color, key, minValLocal, maxValLocal) {
    if (!rows.length) return;
    const path = rows
      .map((row, idx) => {
        const x = scaleX(row.iter);
        const y = scaleY(row[key], minValLocal, maxValLocal);
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
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', margin.left);
    xAxis.setAttribute('y1', margin.top + plotH);
    xAxis.setAttribute('x2', margin.left + plotW);
    xAxis.setAttribute('y2', margin.top + plotH);
    xAxis.setAttribute('stroke', axisColor);
    svg.appendChild(xAxis);

    const yAxisLeft = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxisLeft.setAttribute('x1', margin.left);
    yAxisLeft.setAttribute('y1', margin.top);
    yAxisLeft.setAttribute('x2', margin.left);
    yAxisLeft.setAttribute('y2', margin.top + plotH);
    yAxisLeft.setAttribute('stroke', axisColor);
    svg.appendChild(yAxisLeft);

    const yAxisRight = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxisRight.setAttribute('x1', margin.left + plotW);
    yAxisRight.setAttribute('y1', margin.top);
    yAxisRight.setAttribute('x2', margin.left + plotW);
    yAxisRight.setAttribute('y2', margin.top + plotH);
    yAxisRight.setAttribute('stroke', axisColor);
    svg.appendChild(yAxisRight);

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
      label.setAttribute('fill', labelColor);
      label.textContent = Math.round(iter);
      svg.appendChild(label);
    }

    for (let i = 0; i <= ticks; i += 1) {
      const t = i / ticks;
      const trainVal = minTrain + t * (maxTrain - minTrain);
      const yLeft = scaleY(trainVal, minTrain, maxTrain);
      const leftLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      leftLabel.setAttribute('x', margin.left - 6);
      leftLabel.setAttribute('y', yLeft + 3);
      leftLabel.setAttribute('text-anchor', 'end');
      leftLabel.setAttribute('font-size', '10');
      leftLabel.setAttribute('fill', trainColor);
      leftLabel.textContent = trainVal.toFixed(2);
      svg.appendChild(leftLabel);

      const valVal = minVal + t * (maxVal - minVal);
      const yRight = scaleY(valVal, minVal, maxVal);
      const rightLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      rightLabel.setAttribute('x', margin.left + plotW + 6);
      rightLabel.setAttribute('y', yRight + 3);
      rightLabel.setAttribute('text-anchor', 'start');
      rightLabel.setAttribute('font-size', '10');
      rightLabel.setAttribute('fill', valColor);
      rightLabel.textContent = valVal.toFixed(2);
      svg.appendChild(rightLabel);
    }

    const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xLabel.setAttribute('x', margin.left + plotW / 2);
    xLabel.setAttribute('y', height - 4);
    xLabel.setAttribute('text-anchor', 'middle');
    xLabel.setAttribute('font-size', '10');
    xLabel.setAttribute('fill', labelColor);
    xLabel.textContent = 'iteration';
    svg.appendChild(xLabel);

    const leftLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    leftLabel.setAttribute('x', 10);
    leftLabel.setAttribute('y', margin.top + plotH / 2);
    leftLabel.setAttribute('text-anchor', 'middle');
    leftLabel.setAttribute('font-size', '10');
    leftLabel.setAttribute('fill', trainColor);
    leftLabel.setAttribute('transform', `rotate(-90 10 ${margin.top + plotH / 2})`);
    leftLabel.textContent = 'train loss';
    svg.appendChild(leftLabel);

    const rightLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    rightLabel.setAttribute('x', width - 10);
    rightLabel.setAttribute('y', margin.top + plotH / 2);
    rightLabel.setAttribute('text-anchor', 'middle');
    rightLabel.setAttribute('font-size', '10');
    rightLabel.setAttribute('fill', valColor);
    rightLabel.setAttribute('transform', `rotate(-90 ${width - 10} ${margin.top + plotH / 2})`);
    rightLabel.textContent = 'val loss';
    svg.appendChild(rightLabel);
  }

  function drawLegend() {
    const legend = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    legend.setAttribute('x', margin.left + 6);
    legend.setAttribute('y', margin.top + 12);
    legend.setAttribute('font-size', '10');
    legend.setAttribute('fill', labelColor);
    legend.textContent = 'train (orange, left) / val (blue, right)';
    svg.appendChild(legend);
  }

  drawAxis();
  drawLine(trainRows, trainColor, 'train_loss', minTrain, maxTrain);
  drawLine(valRows, valColor, 'val_loss', minVal, maxVal);
  drawLegend();
  container.appendChild(svg);
}
