import { api } from './api.js';
import { els } from './dom.js';
import { renderLossChart } from './loss_chart.js';

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
  renderLossChart(els.lossChart, data.logs || []);
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
