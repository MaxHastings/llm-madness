from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from llm_madness.runs import finish_manifest, start_manifest
from llm_madness.utils import ensure_dir, timestamp, write_json


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass(frozen=True)
class DatasetFile:
    rel_path: str
    size_bytes: int
    mtime: str


def _resolve_under(root: Path, rel: str) -> Path:
    candidate = (root / rel).resolve()
    if candidate == root.resolve() or root.resolve() not in candidate.parents:
        raise ValueError(f"invalid dataset path: {rel}")
    return candidate


def expand_txt_paths(data_root: Path, selections: list[str]) -> list[Path]:
    seen: set[Path] = set()
    results: list[Path] = []
    for rel in selections:
        rel = rel.strip().lstrip("/")
        if not rel:
            continue
        target = _resolve_under(data_root, rel)
        if target.is_dir():
            for path in sorted(target.rglob("*.txt")):
                if path.is_file() and path not in seen:
                    seen.add(path)
                    results.append(path)
            continue
        if target.is_file() and target.suffix.lower() == ".txt":
            if target not in seen:
                seen.add(target)
                results.append(target)
            continue
        raise ValueError(f"unsupported selection (txt only): {rel}")
    return sorted(results, key=lambda p: str(p))


def build_snapshot(paths: list[Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        text = path.read_text(errors="ignore")
        if text and not text.endswith("\n"):
            text += "\n"
        chunks.append(text)
    return "".join(chunks)


def create_dataset_manifest(
    *,
    name: str | None,
    selections: list[str],
    data_root: Path,
    runs_root: Path,
    enable_content_hashes: bool = False,
    repo_root: Path | None = None,
    run_id: str | None = None,
) -> dict:
    run_dir = ensure_dir(runs_root / "datasets" / (run_id or timestamp()))
    manifest_path = run_dir / "dataset_manifest.json"
    snapshot_path = run_dir / "snapshot.txt"

    start_manifest(
        "dataset_manifest",
        run_dir,
        {"name": name, "selections": selections, "enable_content_hashes": enable_content_hashes},
        inputs={"data_root": str(data_root)},
        outputs={"dataset_manifest": str(manifest_path), "snapshot": str(snapshot_path)},
        repo_root=repo_root,
    )

    try:
        expanded = expand_txt_paths(data_root, selections)
        files: list[DatasetFile] = []
        total_bytes = 0
        for path in expanded:
            stat = path.stat()
            rel = str(path.relative_to(data_root))
            mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
            files.append(DatasetFile(rel_path=rel, size_bytes=int(stat.st_size), mtime=mtime))
            total_bytes += int(stat.st_size)

        snapshot = build_snapshot(expanded)
        snapshot_path.write_text(snapshot)

        payload = {
            "kind": "DatasetManifest",
            "version": 1,
            "id": run_dir.name,
            "name": name,
            "created_at": _iso_now(),
            "data_root": str(data_root),
            "selections": selections,
            "txt_only": True,
            "recursive": True,
            "hashing": {"enabled": bool(enable_content_hashes), "algorithm": "sha256"},
            "file_count": len(files),
            "total_bytes": total_bytes,
            "files": [
                {"path": f.rel_path, "size_bytes": f.size_bytes, "mtime": f.mtime}
                for f in files
            ],
            "snapshot_path": str(snapshot_path),
        }
        write_json(manifest_path, payload)
        finish_manifest(run_dir, "complete", outputs={"dataset_manifest": str(manifest_path), "snapshot": str(snapshot_path)})
        return {"run_dir": run_dir, "manifest_path": manifest_path, "snapshot_path": snapshot_path, "manifest": payload}
    except Exception as exc:
        finish_manifest(run_dir, "failed", error=str(exc))
        raise


def load_dataset_manifest(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("kind") != "DatasetManifest":
        raise ValueError(f"not a DatasetManifest: {path}")
    return raw

