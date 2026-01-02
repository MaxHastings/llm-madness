from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: Path | str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def write_json(path: Path | str, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def git_sha(root: Path | str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def list_text_files(paths: Iterable[Path]) -> list[Path]:
    results: list[Path] = []
    for path in paths:
        if path.is_dir():
            results.extend(sorted(path.rglob("*.txt")))
        elif path.is_file():
            results.append(path)
    return results


def find_latest_run(base_dir: Path, filename: str | None = None) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    latest = sorted(candidates, key=lambda p: p.name)[-1]
    if filename:
        candidate_file = latest / filename
        if candidate_file.exists():
            return candidate_file
    return latest
