import { api } from './api.js';
import { els } from './dom.js';
import { state } from './state.js';

function formatActiveTokenLabel(activeToken) {
  if (!activeToken) return 'Active token: -';
  const tokenLabel = activeToken.token ?? '-';
  return `Active token: ${tokenLabel} (id ${activeToken.id})`;
}

function updateActiveTokenUI() {
  const label = formatActiveTokenLabel(state.activeToken);
  if (els.inspectorActiveToken) {
    els.inspectorActiveToken.textContent = label;
  }
  if (els.inspectActiveToken) {
    els.inspectActiveToken.textContent = '';
  }
}

function setActiveToken(index, tokens, ids) {
  if (!tokens.length || index == null || index < 0 || index >= tokens.length) {
    state.activeToken = null;
    updateActiveTokenUI();
    return;
  }
  state.activeToken = { index, token: tokens[index], id: ids[index] };
  updateActiveTokenUI();
}

function normalizeActiveToken(tokens, ids) {
  if (!tokens.length) {
    state.activeToken = null;
    updateActiveTokenUI();
    return;
  }
  if (!state.activeToken || state.activeToken.index == null || state.activeToken.index >= tokens.length) {
    setActiveToken(tokens.length - 1, tokens, ids);
  }
}

function renderTokens(tokens, ids) {
  normalizeActiveToken(tokens, ids);
  els.tokenList.innerHTML = '';
  tokens.forEach((tok, i) => {
    const el = document.createElement('span');
    el.className = 'token';
    if (state.activeToken && state.activeToken.index === i) {
      el.classList.add('active');
    }
    el.textContent = `${i}: ${tok} (${ids[i]})`;
    el.title = `index ${i} id ${ids[i]}`;
    el.addEventListener('click', () => {
      setActiveToken(i, tokens, ids);
      renderTokens(tokens, ids);
    });
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
  if (state.tokens.length) {
    setActiveToken(state.tokens.length - 1, state.tokens, state.ids);
    renderTokens(state.tokens, state.ids);
  }
  await probeNext();
}

export function resetTokens() {
  state.ids = [];
  state.tokens = [];
  state.activeToken = null;
  updateActiveTokenUI();
  renderTokens([], []);
  els.promptMeta.textContent = 'no tokens';
}

export function initPrompt() {
  updateActiveTokenUI();
  els.tokenizeBtn.addEventListener('click', async () => {
    const data = await api('/api/tokenize', { text: els.promptInput.value });
    state.ids = data.ids;
    state.tokens = data.tokens;
    setActiveToken(state.tokens.length - 1, state.tokens, state.ids);
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
