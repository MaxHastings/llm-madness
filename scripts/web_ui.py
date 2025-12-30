#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from llm_madness.model import GPT, GPTConfig
from llm_madness.tokenizer import load_tokenizer
from llm_madness.utils import find_latest_run


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLM Madness - Inspector</title>
  <style>
    :root {
      --bg: #f4f0e9;
      --panel: #ffffff;
      --ink: #1d1c1a;
      --muted: #6d6a65;
      --accent: #ff6b35;
      --border: #e2dcd2;
      --mono: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      background: linear-gradient(140deg, #f7f2ea 0%, #f1e9de 100%);
      color: var(--ink);
    }
    header {
      padding: 18px 22px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    header h1 {
      margin: 0;
      font-size: 16px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    main {
      padding: 20px 22px 32px;
      display: grid;
      gap: 18px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      padding: 14px;
      border-radius: 10px;
      box-shadow: 0 6px 20px rgba(50, 40, 30, 0.08);
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }
    textarea {
      width: 100%;
      min-height: 140px;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 8px;
      font-family: var(--mono);
      font-size: 13px;
      background: #fbfaf8;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }
    button, select, input {
      font-family: var(--sans);
      font-size: 12px;
      padding: 7px 10px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #f3efe8;
    }
    button.primary {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    .token-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      font-family: var(--mono);
      font-size: 12px;
    }
    .token {
      padding: 4px 6px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fdfaf5;
    }
    .meta {
      font-size: 12px;
      color: var(--muted);
      font-family: var(--mono);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-family: var(--mono);
      font-size: 12px;
    }
    th, td {
      text-align: left;
      padding: 6px 4px;
      border-bottom: 1px solid var(--border);
    }
    th {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 10px;
    }
  </style>
</head>
<body>
  <header>
    <h1>LLM Madness Inspector</h1>
    <div class="meta" id="runMeta">Loading...</div>
  </header>
  <main>
    <section class="panel">
      <h2>Checkpoint</h2>
      <div class="controls">
        <select id="checkpointSelect"></select>
        <button id="loadCheckpoint" class="primary">Load</button>
        <span class="meta" id="checkpointMeta"></span>
      </div>
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Prompt</h2>
        <div class="controls">
          <button id="tokenizeBtn" class="primary">Tokenize</button>
          <button id="decodeBtn">Decode</button>
        </div>
        <textarea id="promptInput" placeholder="Type a prompt..." spellcheck="false"></textarea>
        <div class="meta" id="promptMeta"></div>
      </div>
      <div class="panel">
        <h2>Tokens</h2>
        <div id="tokenList" class="token-list"></div>
      </div>
    </section>
    <section class="panel">
      <h2>Next Token (Top-k)</h2>
      <div class="controls">
        <label>Top-k <input id="topK" type="number" value="8" min="1" max="50" /></label>
        <button id="nextBtn" class="primary">Probe</button>
      </div>
      <div id="nextMeta" class="meta"></div>
      <table id="nextTable">
        <thead>
          <tr><th>Rank</th><th>Token</th><th>ID</th><th>Prob</th></tr>
        </thead>
        <tbody></tbody>
      </table>
    </section>
  </main>

  <script>
    const promptInput = document.getElementById('promptInput');
    const promptMeta = document.getElementById('promptMeta');
    const tokenList = document.getElementById('tokenList');
    const nextMeta = document.getElementById('nextMeta');
    const nextTable = document.getElementById('nextTable').querySelector('tbody');
    const checkpointSelect = document.getElementById('checkpointSelect');
    const checkpointMeta = document.getElementById('checkpointMeta');
    const runMeta = document.getElementById('runMeta');

    let state = { ids: [] };

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

    function renderTokens(tokens, ids) {
      tokenList.innerHTML = '';
      tokens.forEach((tok, i) => {
        const el = document.createElement('span');
        el.className = 'token';
        el.textContent = `${tok} (${ids[i]})`;
        tokenList.appendChild(el);
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

    document.getElementById('tokenizeBtn').addEventListener('click', async () => {
      const data = await api('/api/tokenize', { text: promptInput.value });
      state.ids = data.ids;
      renderTokens(data.tokens, data.ids);
      promptMeta.textContent = `chars: ${promptInput.value.length} tokens: ${data.ids.length}`;
    });

    document.getElementById('decodeBtn').addEventListener('click', async () => {
      if (!state.ids.length) return;
      const data = await api('/api/decode', { ids: state.ids });
      promptInput.value = data.text;
    });

    document.getElementById('nextBtn').addEventListener('click', async () => {
      if (!state.ids.length) return;
      const topK = parseInt(document.getElementById('topK').value || '8', 10);
      const data = await api('/api/next', { ids: state.ids, top_k: topK });
      nextTable.innerHTML = '';
      data.topk.forEach((row, idx) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${idx + 1}</td><td>${row.token}</td><td>${row.id}</td><td>${row.prob.toFixed(4)}</td>`;
        nextTable.appendChild(tr);
      });
      nextMeta.textContent = `loaded from ${data.checkpoint}`;
    });

    document.getElementById('loadCheckpoint').addEventListener('click', async () => {
      const picked = checkpointSelect.value;
      const data = await api('/api/load_checkpoint', { checkpoint: picked });
      checkpointMeta.textContent = data.status;
    });

    refreshCheckpoints();
  </script>
</body>
</html>
"""


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps" and (not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available()):
        return torch.device("cpu")
    return torch.device(name)


class ServerState:
    def __init__(self, run_dir: Path, checkpoint: str | None, device_override: str):
        self.run_dir = run_dir
        self.run_config = json.loads((run_dir / "run.json").read_text())
        self.tokenizer = load_tokenizer(run_dir / "tokenizer.json")
        device_name = device_override if device_override != "auto" else self.run_config.get("device", "auto")
        self.device = select_device(device_name)
        self.model = self._build_model()
        self.current_checkpoint = None
        self.load_checkpoint(checkpoint)

    def _build_model(self) -> GPT:
        model_cfg = self.run_config.get("config", {}).get("model", {})
        gpt_config = GPTConfig(
            vocab_size=self.tokenizer.get_vocab_size(),
            block_size=int(model_cfg.get("block_size", 128)),
            n_layer=int(model_cfg.get("n_layer", 4)),
            n_head=int(model_cfg.get("n_head", 4)),
            n_embd=int(model_cfg.get("n_embd", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        return GPT(gpt_config).to(self.device)

    def list_checkpoints(self) -> list[str]:
        ckpt_dir = self.run_dir / "checkpoints"
        checkpoints = []
        if ckpt_dir.exists():
            checkpoints.extend(sorted(p.name for p in ckpt_dir.glob("checkpoint_*.pt")))
        if (self.run_dir / "latest.pt").exists():
            checkpoints.append("latest.pt")
        return checkpoints

    def load_checkpoint(self, checkpoint: str | None) -> None:
        if checkpoint is None:
            checkpoint = "latest.pt"
        if checkpoint == "latest.pt":
            ckpt_path = self.run_dir / "latest.pt"
        else:
            ckpt_path = self.run_dir / "checkpoints" / checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.current_checkpoint = ckpt_path.name


STATE: ServerState | None = None


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/":
            self.send_response(404)
            self.end_headers()
            return
        data = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        global STATE
        if STATE is None:
            self._send_json({"error": "server not ready"}, status=500)
            return
        try:
            if self.path == "/api/tokenize":
                payload = self._read_json()
                text = payload.get("text", "")
                encoding = STATE.tokenizer.encode(text)
                tokens = [STATE.tokenizer.id_to_token(idx) for idx in encoding.ids]
                self._send_json({"ids": encoding.ids, "tokens": tokens})
                return
            if self.path == "/api/decode":
                payload = self._read_json()
                ids = payload.get("ids", [])
                text = STATE.tokenizer.decode(ids)
                self._send_json({"text": text})
                return
            if self.path == "/api/next":
                payload = self._read_json()
                ids = payload.get("ids", [])
                top_k = int(payload.get("top_k", 8))
                if not ids:
                    self._send_json({"topk": [], "checkpoint": STATE.current_checkpoint})
                    return
                idx = torch.tensor([ids], dtype=torch.long, device=STATE.device)
                idx = idx[:, -STATE.model.config.block_size :]
                with torch.no_grad():
                    logits, _ = STATE.model(idx)
                    logits = logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    topk = torch.topk(probs, k=min(top_k, probs.size(-1)))
                results = []
                for prob, token_id in zip(topk.values[0], topk.indices[0]):
                    tid = int(token_id.item())
                    results.append(
                        {
                            "id": tid,
                            "token": STATE.tokenizer.id_to_token(tid),
                            "prob": float(prob.item()),
                        }
                    )
                self._send_json({"topk": results, "checkpoint": STATE.current_checkpoint})
                return
            if self.path == "/api/checkpoints":
                checkpoints = STATE.list_checkpoints()
                self._send_json(
                    {
                        "checkpoints": checkpoints,
                        "current": STATE.current_checkpoint,
                        "run_dir": str(STATE.run_dir),
                    }
                )
                return
            if self.path == "/api/load_checkpoint":
                payload = self._read_json()
                ckpt = payload.get("checkpoint")
                STATE.load_checkpoint(ckpt)
                self._send_json({"status": f"loaded {STATE.current_checkpoint}"})
                return
            self._send_json({"error": "unknown endpoint"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Web UI for inspecting a trained model.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        latest = find_latest_run(Path("runs/train"))
        if latest is None:
            raise SystemExit("no training runs found; pass --run-dir")
        run_dir = latest

    global STATE
    STATE = ServerState(run_dir, args.checkpoint, args.device)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"server running on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
