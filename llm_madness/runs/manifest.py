from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from llm_madness.utils import git_sha, write_json


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_manifest(path: Path, payload: dict) -> None:
    write_json(path, payload)


def start_manifest(
    stage: str,
    run_dir: Path,
    config: dict,
    inputs: dict | None = None,
    outputs: dict | None = None,
    notes: str | None = None,
    repo_root: Path | None = None,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": stage,
        "status": "running",
        "start_time": iso_now(),
        "end_time": None,
        "config": config,
        "inputs": inputs or {},
        "outputs": outputs or {},
        "git_sha": git_sha(repo_root or run_dir),
        "notes": notes,
    }
    write_manifest(run_dir / "run.json", manifest)
    return manifest


def finish_manifest(
    run_dir: Path,
    status: str,
    outputs: dict | None = None,
    error: str | None = None,
) -> dict:
    path = run_dir / "run.json"
    payload: dict[str, Any] = {}
    if path.exists():
        payload = Path(path).read_text()
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)
    payload["status"] = status
    payload["end_time"] = iso_now()
    if outputs:
        payload["outputs"] = outputs
    if error:
        payload["error"] = error
    write_manifest(path, payload)
    return payload
