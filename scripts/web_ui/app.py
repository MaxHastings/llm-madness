#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import torch

from llm_madness.utils import find_latest_run, timestamp

from .state import ServerState

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"
STATIC_DIR = BASE_DIR / "static"


STATE: ServerState | None = None
DEVICE_OVERRIDE = "auto"
RUNS_DIR = Path("runs/train")
CONFIGS_DIR = Path("configs")
RUNS_ROOT = Path("runs")
DATA_ROOT = Path("data")
RUN_PROCS: dict[str, dict] = {}


def guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".css":
        return "text/css"
    if suffix == ".js":
        return "application/javascript"
    if suffix == ".svg":
        return "image/svg+xml"
    if suffix == ".png":
        return "image/png"
    return "text/plain"


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

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, data: str, content_type: str = "text/html") -> None:
        encoded = data.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/configs/"):
            rel = unquote(self.path[len("/api/configs/"):])
            target = (CONFIGS_DIR / rel).resolve()
            if CONFIGS_DIR.resolve() not in target.parents:
                self._send_json({"error": "invalid config path"}, status=400)
                return
            if not target.exists():
                self._send_json({"error": "config not found"}, status=404)
                return
            raw = target.read_text()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            self._send_json({"name": rel, "raw": raw, "config": parsed})
            return
        if self.path.startswith("/api/run/"):
            rel = unquote(self.path[len("/api/run/"):])
            run_path = Path(rel).resolve()
            if run_path.exists() and run_path.is_dir():
                if RUNS_ROOT.resolve() not in run_path.parents and DATA_ROOT.resolve() not in run_path.parents:
                    self._send_json({"error": "invalid run path"}, status=400)
                    return
                manifest_path = run_path / "run.json"
                payload = {"run_dir": str(run_path)}
                if manifest_path.exists():
                    payload["manifest"] = json.loads(manifest_path.read_text())
                logs_path = run_path / "logs.jsonl"
                if logs_path.exists():
                    payload["logs"] = logs_path.read_text().splitlines()[-200:]
                self._send_json(payload)
                return
            self._send_json({"error": "run not found"}, status=404)
            return
        if self.path in ("/", "/index.html"):
            self._send_text(TEMPLATE_PATH.read_text())
            return
        if self.path.startswith("/static/"):
            rel = unquote(self.path[len("/static/"):])
            target = (STATIC_DIR / rel).resolve()
            if STATIC_DIR not in target.parents and target != STATIC_DIR:
                self.send_response(403)
                self.end_headers()
                return
            if not target.exists() or not target.is_file():
                self.send_response(404)
                self.end_headers()
                return
            self._send_file(target, guess_mime(target))
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        global STATE
        try:
            state_required = {
                "/api/tokenize",
                "/api/decode",
                "/api/ids_to_tokens",
                "/api/next",
                "/api/checkpoints",
                "/api/load_checkpoint",
                "/api/tokenizer_report",
                "/api/training_logs",
                "/api/inspect",
            }
            if self.path in state_required and STATE is None:
                self._send_json({"error": "server not ready"}, status=500)
                return
            if self.path == "/api/configs":
                configs = []
                if CONFIGS_DIR.exists():
                    for path in sorted(CONFIGS_DIR.rglob("*.json")):
                        rel = path.relative_to(CONFIGS_DIR)
                        configs.append(str(rel))
                self._send_json({"configs": configs})
                return
            if self.path == "/api/configs/save":
                payload = self._read_json()
                name = payload.get("name")
                raw = payload.get("raw")
                if not name or raw is None:
                    self._send_json({"error": "name and raw are required"}, status=400)
                    return
                target = (CONFIGS_DIR / name).resolve()
                if CONFIGS_DIR.resolve() not in target.parents:
                    self._send_json({"error": "invalid config path"}, status=400)
                    return
                try:
                    json.loads(raw)
                except json.JSONDecodeError as exc:
                    self._send_json({"error": f"invalid json: {exc}"}, status=400)
                    return
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(raw)
                self._send_json({"status": "saved", "name": name})
                return
            if self.path == "/api/run":
                payload = self._read_json()
                stage = payload.get("stage", "pipeline")
                config_name = payload.get("config", "pipeline.json")
                if config_name.startswith("configs/"):
                    config_name = config_name[len("configs/"):]
                config_path = (CONFIGS_DIR / config_name).resolve()
                if CONFIGS_DIR.resolve() not in config_path.parents:
                    self._send_json({"error": "invalid config path"}, status=400)
                    return
                if not config_path.exists():
                    self._send_json({"error": "config not found"}, status=404)
                    return
                run_id = timestamp()
                cmd = [sys.executable, "-m"]
                if stage == "pipeline":
                    cmd += ["scripts.pipeline", "--config", str(config_path), "--set", f"run.id={run_id}"]
                    run_dir = RUNS_ROOT / "pipeline" / run_id
                elif stage == "tokenizer":
                    cmd += ["scripts.train_tokenizer", "--config", str(config_path), "--set", f"run.id={run_id}"]
                    run_dir = Path("runs/tokenizer") / run_id
                elif stage == "train":
                    cmd += ["scripts.train_model", "--config", str(config_path), "--set", f"run.id={run_id}"]
                    run_dir = Path("runs/train") / run_id
                elif stage == "generate":
                    cmd += ["scripts.generate", "--config", str(config_path), "--set", f"run.id={run_id}"]
                    run_dir = Path("data/generated") / run_id
                elif stage == "combine":
                    cmd += ["scripts.data_combine", "--config", str(config_path), "--set", f"run.id={run_id}"]
                    run_dir = Path("data/combined") / run_id
                else:
                    self._send_json({"error": f"unknown stage '{stage}'"}, status=400)
                    return
                run_dir.mkdir(parents=True, exist_ok=True)
                log_path = run_dir / "process.log"
                log_file = log_path.open("a")
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
                RUN_PROCS[run_id] = {"process": proc, "stage": stage, "run_dir": str(run_dir)}
                self._send_json({"run_id": run_id, "run_dir": str(run_dir), "stage": stage})
                return
            if self.path.startswith("/api/stop/"):
                run_id = unquote(self.path[len("/api/stop/"):])
                info = RUN_PROCS.get(run_id)
                if not info:
                    self._send_json({"error": "run not found"}, status=404)
                    return
                proc = info["process"]
                proc.terminate()
                self._send_json({"status": "stopping", "run_id": run_id})
                return
            if self.path == "/api/runs":
                payload = self._read_json()
                scope = payload.get("scope", "train") if isinstance(payload, dict) else "train"
                if scope == "all":
                    manifests = []
                    for root in (RUNS_ROOT, DATA_ROOT):
                        if not root.exists():
                            continue
                        for path in root.rglob("run.json"):
                            run_dir = path.parent
                            try:
                                manifest = json.loads(path.read_text())
                            except json.JSONDecodeError:
                                continue
                            manifests.append(
                                {
                                    "run_dir": str(run_dir),
                                    "stage": manifest.get("stage"),
                                    "status": manifest.get("status"),
                                    "start_time": manifest.get("start_time"),
                                }
                            )
                    manifests.sort(key=lambda item: item.get("start_time") or "", reverse=True)
                    self._send_json({"runs": manifests})
                    return
                runs = []
                if RUNS_DIR.exists():
                    for entry in sorted(RUNS_DIR.iterdir()):
                        if entry.is_dir():
                            runs.append(str(entry))
                current = str(STATE.run_dir) if STATE else None
                self._send_json({"runs": runs, "current": current})
                return
            if self.path == "/api/tokenize":
                payload = self._read_json()
                text = payload.get("text", "")
                encoding = STATE.tokenizer.encode(text)
                tokens = STATE.ids_to_tokens(encoding.ids)
                self._send_json({"ids": encoding.ids, "tokens": tokens})
                return
            if self.path == "/api/decode":
                payload = self._read_json()
                ids = payload.get("ids", [])
                text = STATE.tokenizer.decode(ids)
                self._send_json({"text": text})
                return
            if self.path == "/api/ids_to_tokens":
                payload = self._read_json()
                ids = payload.get("ids", [])
                tokens = STATE.ids_to_tokens(ids)
                self._send_json({"tokens": tokens})
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
            if self.path == "/api/load_run":
                payload = self._read_json()
                run_dir = payload.get("run_dir", "")
                run_path = Path(run_dir)
                if not run_path.exists() or not run_path.is_dir():
                    self._send_json({"error": "run dir not found"}, status=404)
                    return
                resolved = run_path.resolve()
                if RUNS_DIR.resolve() not in resolved.parents:
                    self._send_json({"error": "invalid run dir"}, status=400)
                    return
                STATE = ServerState(resolved, None, DEVICE_OVERRIDE)
                self._send_json({"status": f"loaded {STATE.run_dir}", "run_dir": str(STATE.run_dir)})
                return
            if self.path == "/api/tokenizer_report":
                payload = self._read_json()
                top_n = int(payload.get("top_n", 25))
                report = STATE.tokenizer_report(top_n=top_n)
                self._send_json(report)
                return
            if self.path == "/api/training_logs":
                self._send_json(STATE.training_logs())
                return
            if self.path == "/api/inspect":
                payload = self._read_json()
                ids = payload.get("ids", [])
                layer = int(payload.get("layer", 0))
                head = int(payload.get("head", 0))
                mode = payload.get("mode", "attention")
                top_k = int(payload.get("top_k", 10))
                result = STATE.inspect(ids, layer, head, mode, top_k)
                self._send_json(result)
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
    global DEVICE_OVERRIDE
    DEVICE_OVERRIDE = args.device
    STATE = ServerState(run_dir, args.checkpoint, DEVICE_OVERRIDE)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"server running on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
