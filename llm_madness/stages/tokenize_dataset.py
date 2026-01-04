from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llm_madness.runs import finish_manifest, start_manifest
from llm_madness.tokenizer import load_tokenizer
from llm_madness.utils import ensure_dir, sha256_file, timestamp


@dataclass(frozen=True)
class TokenizeSplit:
    split_idx: int
    line_count: int


def _compute_split(path: Path, val_split: float) -> TokenizeSplit:
    if val_split <= 0.0:
        return TokenizeSplit(split_idx=0, line_count=0)
    if val_split >= 1.0:
        raise ValueError("val_split must be in (0, 1)")
    line_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _ in handle:
            line_count += 1
    if line_count < 2:
        return TokenizeSplit(split_idx=line_count, line_count=line_count)
    split_idx = int(line_count * (1.0 - val_split))
    split_idx = max(1, min(line_count - 1, split_idx))
    return TokenizeSplit(split_idx=split_idx, line_count=line_count)


def _dtype_for_vocab(vocab_size: int) -> np.dtype:
    if vocab_size < 0:
        raise ValueError("vocab_size must be >= 0")
    if vocab_size < 2**16:
        return np.dtype(np.uint16)
    if vocab_size < 2**32:
        return np.dtype(np.uint32)
    raise ValueError("vocab_size too large for uint32")


def _write_ids(handle, ids: list[int], dtype: np.dtype) -> int:
    if not ids:
        return 0
    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(handle)
    return int(arr.size)


def _write_progress(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_tokenize_dataset(
    config: dict,
    snapshot_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    repo_root: Path,
    dataset_manifest: Path | None = None,
) -> dict:
    tok_cfg = config.get("tokenize", {})
    val_split = float(tok_cfg.get("val_split", 0.0))
    chunk_size_chars = int(tok_cfg.get("chunk_size_chars", 2_000_000))
    batch_size = int(tok_cfg.get("batch_size", 8))
    use_content_hash = bool(tok_cfg.get("use_content_hash", True))

    run_id = config.get("run", {}).get("id") if isinstance(config.get("run"), dict) else None
    run_dir = ensure_dir(output_dir / (run_id or timestamp()))
    train_path = run_dir / "train.bin"
    val_path = run_dir / "val.bin"
    meta_path = run_dir / "meta.json"
    progress_path = run_dir / "progress.json"

    start_manifest(
        "tokenize_dataset",
        run_dir,
        config,
        inputs={
            "snapshot": str(snapshot_path),
            "tokenizer": str(tokenizer_path),
            "dataset_manifest": str(dataset_manifest) if dataset_manifest else None,
        },
        outputs={"run_dir": str(run_dir), "meta": str(meta_path), "train": str(train_path), "val": str(val_path)},
        repo_root=repo_root,
    )

    try:
        tokenizer = load_tokenizer(tokenizer_path)
        vocab_size = int(tokenizer.get_vocab_size())
        dtype = _dtype_for_vocab(vocab_size)

        split = _compute_split(snapshot_path, val_split)
        has_val = val_split > 0.0 and split.line_count >= 2

        t0 = time.perf_counter()
        train_tokens = 0
        val_tokens = 0
        last_progress = 0.0

        _write_progress(
            progress_path,
            {
                "kind": "TokenizeProgress",
                "version": 1,
                "stage": "tokenize_dataset",
                "status": "running",
                "run_id": run_dir.name,
                "snapshot_path": str(snapshot_path),
                "tokenizer_path": str(tokenizer_path),
                "total_lines": split.line_count,
                "processed_lines": 0,
                "train_tokens": 0,
                "val_tokens": 0,
                "message": "tokenizing dataset",
            },
        )

        buffer = ""
        current_target = "train"
        batch_chunks: list[str] = []
        batch_targets: list[str] = []

        def flush_batch(force: bool = False) -> None:
            nonlocal batch_chunks, batch_targets, train_tokens, val_tokens
            if not batch_chunks:
                return
            if not force and len(batch_chunks) < max(1, batch_size):
                return
            encodings = tokenizer.encode_batch(batch_chunks)
            for encoding, target in zip(encodings, batch_targets):
                ids = encoding.ids
                if target == "train":
                    train_tokens += _write_ids(train_handle, ids, dtype)
                else:
                    val_tokens += _write_ids(val_handle, ids, dtype)
            batch_chunks = []
            batch_targets = []

        def push_buffer(target: str) -> None:
            nonlocal buffer
            if not buffer:
                return
            batch_chunks.append(buffer)
            batch_targets.append(target)
            buffer = ""
            flush_batch()

        with train_path.open("ab") as train_handle, val_path.open("ab") as val_handle:
            with snapshot_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_idx, line in enumerate(handle):
                    target = "val" if (has_val and line_idx >= split.split_idx) else "train"
                    if target != current_target:
                        push_buffer(current_target)
                        current_target = target
                    buffer += line
                    if len(buffer) >= chunk_size_chars:
                        push_buffer(current_target)
                    if time.perf_counter() - last_progress >= 1.0:
                        _write_progress(
                            progress_path,
                            {
                                "kind": "TokenizeProgress",
                                "version": 1,
                                "stage": "tokenize_dataset",
                                "status": "running",
                                "run_id": run_dir.name,
                                "snapshot_path": str(snapshot_path),
                                "tokenizer_path": str(tokenizer_path),
                                "total_lines": split.line_count,
                                "processed_lines": line_idx + 1,
                                "train_tokens": train_tokens,
                                "val_tokens": val_tokens,
                                "elapsed_seconds": time.perf_counter() - t0,
                                "message": "tokenizing dataset",
                            },
                        )
                        last_progress = time.perf_counter()
                push_buffer(current_target)
                flush_batch(force=True)

        # Ensure val.bin exists even if empty so training can memmap it reliably.
        val_path.touch(exist_ok=True)

        elapsed = time.perf_counter() - t0
        snapshot_fingerprint = None
        tokenizer_fingerprint = None
        if use_content_hash:
            snapshot_fingerprint = sha256_file(snapshot_path)
            tokenizer_fingerprint = sha256_file(tokenizer_path)
        else:
            ss = snapshot_path.stat()
            ts = tokenizer_path.stat()
            snapshot_fingerprint = f"stat:{ss.st_size}:{int(ss.st_mtime)}"
            tokenizer_fingerprint = f"stat:{ts.st_size}:{int(ts.st_mtime)}"

        meta = {
            "kind": "TokenDataset",
            "version": 1,
            "id": run_dir.name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "snapshot_path": str(snapshot_path),
            "snapshot_fingerprint": snapshot_fingerprint,
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "val_split": val_split,
            "split": {"line_count": split.line_count, "split_idx": split.split_idx},
            "vocab_size": vocab_size,
            "dtype": str(dtype),
            "paths": {"train": str(train_path), "val": str(val_path)},
            "counts": {"train_tokens": train_tokens, "val_tokens": val_tokens},
            "timing": {"seconds": elapsed},
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "tokenizer.json").write_text(
            tokenizer_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        if dataset_manifest:
            (run_dir / "dataset_manifest.json").write_text(
                dataset_manifest.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        _write_progress(
            progress_path,
            {
                "kind": "TokenizeProgress",
                "version": 1,
                "stage": "tokenize_dataset",
                "status": "complete",
                "run_id": run_dir.name,
                "snapshot_path": str(snapshot_path),
                "tokenizer_path": str(tokenizer_path),
                "total_lines": split.line_count,
                "processed_lines": split.line_count,
                "train_tokens": train_tokens,
                "val_tokens": val_tokens,
                "elapsed_seconds": elapsed,
                "message": "tokenization complete",
                "meta_path": str(meta_path),
            },
        )
        finish_manifest(run_dir, "complete", outputs={"run_dir": str(run_dir), "meta": str(meta_path)})
        return {"run_dir": run_dir, "meta_path": meta_path, "train_path": train_path, "val_path": val_path}
    except Exception as exc:
        try:
            _write_progress(
                progress_path,
                {
                    "kind": "TokenizeProgress",
                    "version": 1,
                    "stage": "tokenize_dataset",
                    "status": "failed",
                    "run_id": run_dir.name,
                    "snapshot_path": str(snapshot_path),
                    "tokenizer_path": str(tokenizer_path),
                    "message": str(exc),
                },
            )
        except Exception:
            pass
        finish_manifest(run_dir, "failed", error=str(exc))
        raise

