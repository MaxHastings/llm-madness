from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

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
    snapshot, _ = build_snapshots(paths)
    return snapshot


def build_snapshot_jsonl(paths: list[Path]) -> str:
    _, snapshot_jsonl = build_snapshots(paths)
    return snapshot_jsonl


def build_snapshots(paths: list[Path], progress_cb=None) -> tuple[str, str]:
    chunks: list[str] = []
    jsonl_lines: list[str] = []
    total = len(paths)
    for idx, path in enumerate(paths, start=1):
        text = path.read_text(encoding="utf-8", errors="replace")
        jsonl_lines.append(json.dumps({"text": text}, ensure_ascii=True))
        # Force UTF-8 so the snapshot is always valid for tokenizers.
        if text and not text.endswith("\n"):
            text += "\n"
        chunks.append(text)
        if progress_cb:
            progress_cb(idx, total, path)
    snapshot = "".join(chunks)
    snapshot_jsonl = "" if not jsonl_lines else "\n".join(jsonl_lines) + "\n"
    return snapshot, snapshot_jsonl


def _write_progress(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def create_dataset_manifest(
    *,
    name: str | None,
    selections: list[str],
    data_root: Path,
    runs_root: Path,
    enable_content_hashes: bool = False,
    repo_root: Path | None = None,
    run_id: str | None = None,
    progress_path: Path | None = None,
) -> dict:
    run_dir = ensure_dir(runs_root / "datasets" / (run_id or timestamp()))
    manifest_path = run_dir / "dataset_manifest.json"
    snapshot_path = run_dir / "snapshot.txt"
    snapshot_jsonl_path = run_dir / "snapshot.jsonl"
    started = time.time()

    def push_progress(status: str, message: str, processed: int | None = None, total: int | None = None) -> None:
        if progress_path is None:
            return
        payload = {
            "kind": "RunProgress",
            "version": 1,
            "stage": "dataset",
            "status": status,
            "message": message,
            "processed_files": processed,
            "total_files": total,
            "elapsed_seconds": time.time() - started,
        }
        _write_progress(progress_path, payload)

    start_manifest(
        "dataset_manifest",
        run_dir,
        {"name": name, "selections": selections, "enable_content_hashes": enable_content_hashes},
        inputs={"data_root": str(data_root)},
        outputs={
            "dataset_manifest": str(manifest_path),
            "snapshot": str(snapshot_path),
            "snapshot_jsonl": str(snapshot_jsonl_path),
        },
        repo_root=repo_root,
    )

    try:
        push_progress("running", "expanding selections")
        expanded = expand_txt_paths(data_root, selections)
        files: list[DatasetFile] = []
        total_bytes = 0
        total_files = len(expanded)
        for idx, path in enumerate(expanded, start=1):
            stat = path.stat()
            rel = str(path.relative_to(data_root))
            mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
            files.append(DatasetFile(rel_path=rel, size_bytes=int(stat.st_size), mtime=mtime))
            total_bytes += int(stat.st_size)
            push_progress("running", "collecting file stats", idx, total_files)

        push_progress("running", "building snapshots", 0, total_files)
        def on_progress(idx: int, total: int, path: Path) -> None:
            push_progress("running", f"processing {path.name}", idx, total)

        snapshot, snapshot_jsonl = build_snapshots(expanded, progress_cb=on_progress)
        snapshot_path.write_text(snapshot, encoding="utf-8")
        snapshot_jsonl_path.write_text(snapshot_jsonl, encoding="utf-8")

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
            "snapshot_jsonl_path": str(snapshot_jsonl_path),
        }
        write_json(manifest_path, payload)
        push_progress("complete", "dataset ready", total_files, total_files)
        finish_manifest(
            run_dir,
            "complete",
            outputs={
                "dataset_manifest": str(manifest_path),
                "snapshot": str(snapshot_path),
                "snapshot_jsonl": str(snapshot_jsonl_path),
            },
        )
        return {
            "run_dir": run_dir,
            "manifest_path": manifest_path,
            "snapshot_path": snapshot_path,
            "snapshot_jsonl_path": snapshot_jsonl_path,
            "manifest": payload,
        }
    except Exception as exc:
        push_progress("failed", str(exc))
        finish_manifest(run_dir, "failed", error=str(exc))
        raise


def load_dataset_manifest(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("kind") != "DatasetManifest":
        raise ValueError(f"not a DatasetManifest: {path}")
    return raw

