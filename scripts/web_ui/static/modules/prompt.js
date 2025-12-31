import { api } from './api.js';
import { els } from './dom.js';
import { state } from './state.js';

function renderTokens(tokens, ids) {
  els.tokenList.innerHTML = '';
  tokens.forEach((tok, i) => {
    const el = document.createElement('span');
    el.className = 'token';
    el.textContent = `${i}: ${tok} (${ids[i]})`;
    el.title = `index ${i} id ${ids[i]}`;
    els.tokenList.appendChild(el);
  });
}

function renderTopK(rows) {
  els.nextTableBody.innerHTML = '';
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
    els.nextTableBody.appendChild(tr);
  });
}

async function updateFromIds() {
  if (!state.ids.length) {
    state.tokens = [];
    renderTokens([], []);
    els.promptMeta.textContent = 'no tokens';
    return;
  }
  const tokensData = await api('/api/ids_to_tokens', { ids: state.ids });
  state.tokens = tokensData.tokens;
  renderTokens(state.tokens, state.ids);
  const decoded = await api('/api/decode', { ids: state.ids });
  els.promptInput.value = decoded.text;
  els.promptMeta.textContent = `chars: ${decoded.text.length} tokens: ${state.ids.length}`;
}

async function probeNext() {
  if (!state.ids.length) return;
  const topK = parseInt(els.topK.value || '8', 10);
  const data = await api('/api/next', { ids: state.ids, top_k: topK });
  renderTopK(data.topk);
  els.nextMeta.textContent = `loaded from ${data.checkpoint}`;
}

export async function appendToken(tokenId) {
  state.ids.push(tokenId);
  await updateFromIds();
  await probeNext();
}

export function resetTokens() {
  state.ids = [];
  state.tokens = [];
  renderTokens([], []);
  els.promptMeta.textContent = 'no tokens';
}

export function initPrompt() {
  els.tokenizeBtn.addEventListener('click', async () => {
    const data = await api('/api/tokenize', { text: els.promptInput.value });
    state.ids = data.ids;
    state.tokens = data.tokens;
    renderTokens(data.tokens, data.ids);
    els.promptMeta.textContent = `chars: ${els.promptInput.value.length} tokens: ${data.ids.length}`;
  });

  els.decodeBtn.addEventListener('click', async () => {
    if (!state.ids.length) return;
    const data = await api('/api/decode', { ids: state.ids });
    els.promptInput.value = data.text;
  });

  els.nextBtn.addEventListener('click', probeNext);
}
