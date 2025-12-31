#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

os.environ.setdefault("KMP_SHM_DISABLE", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import torch

from llm_madness.datasets.manifest import create_dataset_manifest
from llm_madness.utils import find_latest_run, timestamp

from .state import ServerState

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"
STATIC_DIR = BASE_DIR / "static"
REPO_ROOT = BASE_DIR.parents[1]


STATE: ServerState | None = None
DEVICE_OVERRIDE = "auto"
RUNS_DIR = Path("runs/train")
CONFIGS_DIR = Path("configs")
RUNS_ROOT = Path("runs")
DATA_ROOT = REPO_ROOT / "data"
DATASETS_DIR = RUNS_ROOT / "datasets"
TOKENIZER_RUNS_DIR = RUNS_ROOT / "tokenizer"
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


def parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def format_duration(seconds: float | None) -> str | None:
    if seconds is None or seconds < 0:
        return None
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def read_last_line(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        if size == 0:
            return None
        read_size = min(size, 4096)
        handle.seek(-read_size, 2)
        chunk = handle.read(read_size)
    lines = chunk.splitlines()
    if not lines:
        return None
    last = lines[-1].decode("utf-8", errors="ignore").strip()
    return last or None


def normalize_status(status: str | None) -> str:
    if not status:
        return "unknown"
    lowered = status.lower()
    if lowered in {"complete", "completed", "success", "succeeded", "finished"}:
        return "completed"
    if lowered in {"running", "active", "in_progress"}:
        return "running"
    if lowered in {"queued", "pending"}:
        return "queued"
    if lowered in {"stopped", "cancelled", "canceled"}:
        return "stopped"
    if lowered in {"failed", "error"}:
        return "failed"
    return lowered


def is_any_run_active() -> bool:
    return any(info["process"].poll() is None for info in RUN_PROCS.values())


def list_config_paths(scope: str) -> list[Path]:
    scope = (scope or "pipeline").lower()
    if scope == "pipeline":
        return sorted(
            [
                path
                for path in CONFIGS_DIR.iterdir()
                if path.is_file() and path.suffix == ".json" and path.name.startswith("pipeline")
            ]
        )
    if scope == "tokenizer":
        base = CONFIGS_DIR / "tokenizer"
        if not base.exists():
            return []
        return sorted([path for path in base.iterdir() if path.is_file() and path.suffix == ".json"])
    if scope == "training":
        base = CONFIGS_DIR / "training"
        if not base.exists():
            return []
        return sorted([path for path in base.iterdir() if path.is_file() and path.suffix == ".json"])
    if scope == "all":
        return sorted(CONFIGS_DIR.rglob("*.json"))
    raise ValueError(f"unknown config scope: {scope}")


def describe_config(path: Path) -> dict:
    rel = str(path.relative_to(CONFIGS_DIR))
    payload: dict = {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        payload = {}
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    model = payload.get("model", {}) if isinstance(payload, dict) else {}
    return {
        "name": meta.get("name") or path.stem,
        "version": meta.get("version"),
        "created_at": meta.get("created_at"),
        "id": meta.get("id"),
        "path": rel,
        "vocab_size": payload.get("vocab_size") if isinstance(payload, dict) else None,
        "algorithm": payload.get("algorithm") if isinstance(payload, dict) else None,
        "block_size": model.get("block_size"),
        "n_layer": model.get("n_layer"),
        "n_head": model.get("n_head"),
        "n_embd": model.get("n_embd"),
    }


def infer_stage(run_dir: Path, manifest: dict | None) -> str:
    if manifest and manifest.get("stage"):
        return manifest["stage"]
    stage = run_dir.parent.name
    return stage or "unknown"


def run_is_active(run_id: str) -> bool:
    info = RUN_PROCS.get(run_id)
    if not info:
        return False
    return info["process"].poll() is None


def build_run_summary(run_dir: Path, manifest: dict | None = None) -> dict:
    run_id = run_dir.name
    manifest_path = run_dir / "run.json"
    has_manifest = manifest is not None or manifest_path.exists()
    if manifest is None and has_manifest:
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = None
    stage = infer_stage(run_dir, manifest)
    is_active = run_is_active(run_id)
    status = normalize_status(manifest.get("status") if manifest else None)
    if is_active:
        status = "running"
    start_time = manifest.get("start_time") if manifest else None
    end_time = manifest.get("end_time") if manifest else None
    start_dt = parse_iso(start_time)
    end_dt = parse_iso(end_time)
    if start_dt:
        if is_active:
            duration_seconds = (datetime.now() - start_dt).total_seconds()
        elif end_dt:
            duration_seconds = (end_dt - start_dt).total_seconds()
        else:
            duration_seconds = None
    else:
        duration_seconds = None
    duration = format_duration(duration_seconds)
    last_log = read_last_line(run_dir / "logs.jsonl") or read_last_line(run_dir / "process.log")
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stage": stage,
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "last_log": last_log,
        "has_manifest": has_manifest,
        "is_active": is_active,
    }


class Handler(BaseHTTPRequestHandler):
    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except ConnectionResetError:
            return

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
        if self.path.startswith("/api/data/list"):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            rel = params.get("path", [""])[0]
            rel = rel.strip().lstrip("/")
            target = (DATA_ROOT / rel).resolve()
            if DATA_ROOT.resolve() not in target.parents and target != DATA_ROOT.resolve():
                self._send_json({"error": "invalid data path"}, status=400)
                return
            if not target.exists() or not target.is_dir():
                self._send_json({"error": "path not found"}, status=404)
                return
            entries = []
            for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if child.name.startswith("."):
                    continue
                if child.is_dir():
                    entries.append(
                        {
                            "name": child.name,
                            "type": "dir",
                            "rel_path": str(child.relative_to(DATA_ROOT)),
                        }
                    )
                elif child.is_file() and child.suffix.lower() == ".txt":
                    entries.append(
                        {
                            "name": child.name,
                            "type": "file",
                            "rel_path": str(child.relative_to(DATA_ROOT)),
                            "size_bytes": child.stat().st_size,
                        }
                    )
            parent = None
            if target != DATA_ROOT.resolve():
                parent = str(target.parent.relative_to(DATA_ROOT))
            self._send_json({"path": rel, "parent": parent, "entries": entries})
            return
        if self.path == "/api/datasets":
            datasets = []
            if DATASETS_DIR.exists():
                for run_dir in sorted(DATASETS_DIR.iterdir(), reverse=True):
                    if not run_dir.is_dir():
                        continue
                    manifest_path = run_dir / "dataset_manifest.json"
                    if not manifest_path.exists():
                        continue
                    try:
                        manifest = json.loads(manifest_path.read_text())
                    except json.JSONDecodeError:
                        continue
                    datasets.append(
                        {
                            "id": manifest.get("id") or run_dir.name,
                            "name": manifest.get("name"),
                            "created_at": manifest.get("created_at"),
                            "file_count": manifest.get("file_count"),
                            "total_bytes": manifest.get("total_bytes"),
                            "manifest_path": str(manifest_path),
                            "snapshot_path": manifest.get("snapshot_path"),
                            "run_dir": str(run_dir),
                        }
                    )
            self._send_json({"datasets": datasets})
            return
        if self.path.startswith("/api/datasets/manifest"):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            path = params.get("path", [""])[0]
            if not path:
                self._send_json({"error": "path required"}, status=400)
                return
            target = Path(unquote(path)).resolve()
            if RUNS_ROOT.resolve() not in target.parents:
                self._send_json({"error": "invalid manifest path"}, status=400)
                return
            if not target.exists() or not target.is_file():
                self._send_json({"error": "manifest not found"}, status=404)
                return
            raw = target.read_text()
            try:
                parsed_manifest = json.loads(raw)
            except json.JSONDecodeError:
                parsed_manifest = None
            self._send_json({"path": str(target), "raw": raw, "manifest": parsed_manifest})
            return
        if self.path == "/api/tokenizer_vocabs":
            vocabs = []
            if TOKENIZER_RUNS_DIR.exists():
                for run_dir in sorted(TOKENIZER_RUNS_DIR.iterdir(), reverse=True):
                    if not run_dir.is_dir():
                        continue
                    manifest_path = run_dir / "run.json"
                    if not manifest_path.exists():
                        continue
                    try:
                        manifest = json.loads(manifest_path.read_text())
                    except json.JSONDecodeError:
                        continue
                    report_path = run_dir / "report.json"
                    report = {}
                    if report_path.exists():
                        try:
                            report = json.loads(report_path.read_text())
                        except json.JSONDecodeError:
                            report = {}
                    config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
                    meta = config.get("meta", {}) if isinstance(config, dict) else {}
                    input_path = report.get("input_path") or manifest.get("inputs", {}).get("input")
                    dataset_manifest = manifest.get("inputs", {}).get("dataset_manifest")
                    input_bytes = None
                    if input_path:
                        try:
                            input_bytes = Path(input_path).stat().st_size
                        except FileNotFoundError:
                            input_bytes = None
                    vocabs.append(
                        {
                            "run_id": run_dir.name,
                            "run_dir": str(run_dir),
                            "status": manifest.get("status"),
                            "created_at": manifest.get("start_time"),
                            "name": meta.get("name"),
                            "version": meta.get("version"),
                            "config_id": meta.get("id"),
                            "vocab_size": report.get("vocab_size") or config.get("vocab_size"),
                            "token_count": report.get("token_count"),
                            "input_path": input_path,
                            "dataset_manifest": dataset_manifest,
                            "input_bytes": input_bytes,
                            "tokenizer_path": report.get("output_path") or str(run_dir / "tokenizer.json"),
                        }
                    )
            self._send_json({"vocabs": vocabs})
            return
        if self.path.startswith("/api/tokenizer_vocabs/report"):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            run_dir = params.get("run_dir", [""])[0]
            if not run_dir:
                self._send_json({"error": "run_dir required"}, status=400)
                return
            run_path = Path(unquote(run_dir)).resolve()
            if RUNS_ROOT.resolve() not in run_path.parents:
                self._send_json({"error": "invalid run path"}, status=400)
                return
            if not run_path.exists() or not run_path.is_dir():
                self._send_json({"error": "run not found"}, status=404)
                return
            manifest = None
            report = None
            config = None
            manifest_path = run_path / "run.json"
            report_path = run_path / "report.json"
            config_path = run_path / "tokenizer_config.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                except json.JSONDecodeError:
                    manifest = None
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text())
                except json.JSONDecodeError:
                    report = None
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text())
                except json.JSONDecodeError:
                    config = None
            self._send_json(
                {
                    "run_dir": str(run_path),
                    "manifest": manifest,
                    "report": report,
                    "config": config,
                }
            )
            return
        if self.path.startswith("/api/run/stream"):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            run_dir = params.get("run_dir", [""])[0]
            kind = params.get("kind", ["logs"])[0]
            if not run_dir:
                self._send_json({"error": "run_dir required"}, status=400)
                return
            run_path = Path(unquote(run_dir)).resolve()
            if run_path.exists() and run_path.is_dir():
                if RUNS_ROOT.resolve() not in run_path.parents and DATA_ROOT.resolve() not in run_path.parents:
                    self._send_json({"error": "invalid run path"}, status=400)
                    return
                filename = "logs.jsonl" if kind == "logs" else "process.log"
                target = run_path / filename
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                def send_line(line: str) -> None:
                    payload = f"data: {line.rstrip()}\n\n".encode("utf-8")
                    self.wfile.write(payload)
                    self.wfile.flush()

                last_ping = time.time()
                try:
                    last_size = -1
                    while True:
                        if not target.exists():
                            time.sleep(1)
                            continue
                        size = target.stat().st_size
                        if size < last_size:
                            last_size = -1
                        if last_size == -1:
                            with target.open("r", encoding="utf-8", errors="ignore") as handle:
                                lines = handle.read().splitlines()[-200:]
                                for line in lines:
                                    send_line(line)
                            last_size = target.stat().st_size

                        with target.open("r", encoding="utf-8", errors="ignore") as handle:
                            handle.seek(last_size)
                            while True:
                                line = handle.readline()
                                if line:
                                    send_line(line)
                                else:
                                    break
                            last_size = handle.tell()

                        if time.time() - last_ping > 10:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            last_ping = time.time()
                        time.sleep(1)
                except (BrokenPipeError, ConnectionResetError):
                    return
                return
            self._send_json({"error": "run not found"}, status=404)
            return
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
                manifest = None
                if manifest_path.exists():
                    try:
                        manifest = json.loads(manifest_path.read_text())
                    except json.JSONDecodeError:
                        manifest = None
                    payload["manifest"] = manifest
                logs_path = run_path / "logs.jsonl"
                if logs_path.exists():
                    payload["logs"] = logs_path.read_text().splitlines()[-200:]
                proc_log = run_path / "process.log"
                if proc_log.exists():
                    payload["process_log"] = proc_log.read_text().splitlines()[-200:]
                payload["summary"] = build_run_summary(run_path, manifest)
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
                payload = self._read_json()
                scope = payload.get("scope", "pipeline") if isinstance(payload, dict) else "pipeline"
                if CONFIGS_DIR.exists():
                    for path in list_config_paths(scope):
                        configs.append(str(path.relative_to(CONFIGS_DIR)))
                self._send_json({"configs": configs})
                return
            if self.path == "/api/configs/meta":
                payload = self._read_json()
                scope = payload.get("scope", "pipeline") if isinstance(payload, dict) else "pipeline"
                configs = []
                if CONFIGS_DIR.exists():
                    for path in list_config_paths(scope):
                        configs.append(describe_config(path))
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
            if self.path == "/api/configs/delete":
                payload = self._read_json()
                name = payload.get("name")
                if not name:
                    self._send_json({"error": "name required"}, status=400)
                    return
                target = (CONFIGS_DIR / name).resolve()
                if CONFIGS_DIR.resolve() not in target.parents:
                    self._send_json({"error": "invalid config path"}, status=400)
                    return
                if not target.exists():
                    self._send_json({"error": "config not found"}, status=404)
                    return
                target.unlink()
                self._send_json({"status": "deleted", "name": name})
                return
            if self.path == "/api/datasets/create":
                payload = self._read_json()
                selections = payload.get("selections", [])
                name = payload.get("name") or None
                enable_hashes = bool(payload.get("enable_content_hashes", False))
                if not isinstance(selections, list) or not selections:
                    self._send_json({"error": "selections required"}, status=400)
                    return
                result = create_dataset_manifest(
                    name=name,
                    selections=[str(item) for item in selections],
                    data_root=DATA_ROOT,
                    runs_root=RUNS_ROOT,
                    enable_content_hashes=enable_hashes,
                    repo_root=REPO_ROOT,
                )
                self._send_json(
                    {
                        "status": "created",
                        "dataset_id": result["run_dir"].name,
                        "manifest_path": str(result["manifest_path"]),
                        "snapshot_path": str(result["snapshot_path"]),
                        "file_count": result["manifest"]["file_count"],
                        "total_bytes": result["manifest"]["total_bytes"],
                    }
                )
                return
            if self.path == "/api/run":
                if is_any_run_active():
                    self._send_json({"error": "another run is active; stop it first"}, status=400)
                    return
                payload = self._read_json()
                stage = payload.get("stage", "train")
                config_name = payload.get("config", "training/default__v001.json")
                dataset_manifest = payload.get("dataset_manifest")
                tokenizer_path = payload.get("tokenizer_path")
                if stage not in {"tokenizer", "train"}:
                    self._send_json({"error": "unsupported stage"}, status=400)
                    return
                if config_name.startswith("configs/"):
                    config_name = config_name[len("configs/"):]
                config_path = (CONFIGS_DIR / config_name).resolve()
                if CONFIGS_DIR.resolve() not in config_path.parents:
                    self._send_json({"error": "invalid config path"}, status=400)
                    return
                if not config_path.exists():
                    self._send_json({"error": "config not found"}, status=404)
                    return
                dataset_manifest_path = None
                if dataset_manifest:
                    candidate = Path(str(dataset_manifest)).resolve()
                    if RUNS_ROOT.resolve() not in candidate.parents:
                        self._send_json({"error": "invalid dataset manifest path"}, status=400)
                        return
                    if not candidate.exists():
                        self._send_json({"error": "dataset manifest not found"}, status=404)
                        return
                    dataset_manifest_path = candidate
                tokenizer_path_resolved = None
                if tokenizer_path:
                    candidate = Path(str(tokenizer_path)).resolve()
                    if RUNS_ROOT.resolve() not in candidate.parents:
                        self._send_json({"error": "invalid tokenizer path"}, status=400)
                        return
                    if not candidate.exists():
                        self._send_json({"error": "tokenizer path not found"}, status=404)
                        return
                    tokenizer_path_resolved = candidate
                run_id = timestamp()
                cmd = [sys.executable, "-m"]
                if stage == "tokenizer":
                    if dataset_manifest_path is None:
                        self._send_json({"error": "dataset manifest required for tokenizer run"}, status=400)
                        return
                    cmd += ["scripts.train_tokenizer", "--config", str(config_path), "--set", f"run.id={run_id}"]
                    cmd += ["--dataset-manifest", str(dataset_manifest_path)]
                    run_dir = RUNS_ROOT / "tokenizer" / run_id
                elif stage == "train":
                    if dataset_manifest_path is None:
                        self._send_json({"error": "dataset manifest required for training run"}, status=400)
                        return
                    if tokenizer_path_resolved is None:
                        self._send_json({"error": "tokenizer vocab required for training run"}, status=400)
                        return
                    cmd += ["scripts.train_model", "--config", str(config_path), "--set", f"run.id={run_id}"]
                    cmd += ["--dataset-manifest", str(dataset_manifest_path)]
                    cmd += ["--tokenizer", str(tokenizer_path_resolved)]
                    run_dir = RUNS_ROOT / "train" / run_id
                else:
                    self._send_json({"error": f"unknown stage '{stage}'"}, status=400)
                    return
                run_dir.mkdir(parents=True, exist_ok=True)
                log_path = run_dir / "process.log"
                log_file = log_path.open("a")
                env = os.environ.copy()
                env.setdefault("TOKENIZERS_PARALLELISM", "false")
                env.setdefault("KMP_SHM_DISABLE", "1")
                env.setdefault("KMP_USE_SHM", "0")
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
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
            if self.path == "/api/run/delete":
                payload = self._read_json()
                run_dir = payload.get("run_dir")
                if not run_dir:
                    self._send_json({"error": "run_dir required"}, status=400)
                    return
                run_path = Path(run_dir).resolve()
                if RUNS_ROOT.resolve() not in run_path.parents and DATA_ROOT.resolve() not in run_path.parents:
                    self._send_json({"error": "invalid run path"}, status=400)
                    return
                run_id = run_path.name
                if run_id in RUN_PROCS and RUN_PROCS[run_id]["process"].poll() is None:
                    self._send_json({"error": "run is still active; stop it first"}, status=400)
                    return
                RUN_PROCS.pop(run_id, None)
                if not run_path.exists():
                    self._send_json({"error": "run not found"}, status=404)
                    return
                for child in sorted(run_path.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                run_path.rmdir()
                self._send_json({"status": "deleted", "run_dir": str(run_path)})
                return
            if self.path == "/api/runs":
                payload = self._read_json()
                scope = payload.get("scope", "train") if isinstance(payload, dict) else "train"
                if scope == "train":
                    summaries = []
                    run_dirs: set[Path] = set()
                    train_root = RUNS_ROOT / "train"
                    if train_root.exists():
                        for run_dir in sorted(train_root.iterdir()):
                            if run_dir.is_dir():
                                run_dirs.add(run_dir)
                    for info in RUN_PROCS.values():
                        if info.get("stage") == "train":
                            run_dirs.add(Path(info["run_dir"]))
                    for run_dir in sorted(run_dirs):
                        summaries.append(build_run_summary(run_dir))
                    summaries.sort(key=lambda item: item.get("start_time") or item.get("run_id") or "", reverse=True)
                    self._send_json({"runs": summaries})
                    return
                if scope == "all":
                    summaries = []
                    run_dirs: set[Path] = set()
                    for root in (RUNS_ROOT, DATA_ROOT):
                        if not root.exists():
                            continue
                        for path in root.rglob("run.json"):
                            run_dirs.add(path.parent)
                    for info in RUN_PROCS.values():
                        run_dirs.add(Path(info["run_dir"]))
                    for run_dir in sorted(run_dirs):
                        summaries.append(build_run_summary(run_dir))
                    summaries.sort(key=lambda item: item.get("start_time") or item.get("run_id") or "", reverse=True)
                    self._send_json({"runs": summaries})
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
                top_k = int(payload.get("top_k", 10))
                result = STATE.inspect(ids, top_k)
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

    def find_latest_with_manifest(root: Path) -> Path | None:
        if not root.exists():
            return None
        candidates = []
        for entry in root.iterdir():
            if entry.is_dir() and (entry / "run.json").exists():
                candidates.append(entry)
        return sorted(candidates, key=lambda p: p.name)[-1] if candidates else None

    run_dir = args.run_dir
    if run_dir is None:
        latest = find_latest_with_manifest(Path("runs/train"))
        if latest is None:
            print("warning: no training runs with run.json found; starting UI without a loaded run")
            run_dir = None
        else:
            run_dir = latest

    global STATE
    global DEVICE_OVERRIDE
    DEVICE_OVERRIDE = args.device
    if run_dir is not None:
        if not (run_dir / "run.json").exists():
            print(f"warning: run.json missing in {run_dir}; starting UI without a loaded run")
        else:
            STATE = ServerState(run_dir, args.checkpoint, DEVICE_OVERRIDE)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"server running on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
