from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from llm_madness.utils import git_sha, sha256_text


CHECKPOINT_KIND = "Checkpoint"
CHECKPOINT_VERSION = 2


@dataclass(frozen=True)
class CheckpointBundle:
    payload: dict[str, Any]
    path: Path


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def is_checkpoint_v2(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    return payload.get("kind") == CHECKPOINT_KIND and int(payload.get("version", 0)) >= CHECKPOINT_VERSION


def load_checkpoint(path: Path, map_location: str | torch.device | None = None) -> CheckpointBundle:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint payload: {path}")
    return CheckpointBundle(payload=payload, path=path)


def checkpoint_supports_resume(path: Path) -> tuple[bool, str | None]:
    bundle = load_checkpoint(path, map_location="cpu")
    if not is_checkpoint_v2(bundle.payload):
        return False, "checkpoint is not v2"
    if bundle.payload.get("optimizer_state") is None:
        return False, "resume requires optimizer_state in checkpoint"
    return True, None


def validate_checkpoint(payload: dict, model_config: dict, tokenizer_path: Path | None) -> list[str]:
    errors: list[str] = []
    if not is_checkpoint_v2(payload):
        errors.append("checkpoint is not v2")
        return errors
    ckpt_model = payload.get("model_config") or {}
    for key in ("block_size", "n_layer", "n_head", "n_embd", "vocab_size"):
        if key in model_config and key in ckpt_model and model_config[key] != ckpt_model[key]:
            errors.append(f"model {key} mismatch: {model_config[key]} != {ckpt_model[key]}")
    if tokenizer_path is not None:
        ckpt_tokenizer = payload.get("tokenizer_path")
        if ckpt_tokenizer:
            try:
                ckpt_path = Path(str(ckpt_tokenizer)).resolve()
            except OSError:
                ckpt_path = Path(str(ckpt_tokenizer))
            try:
                target_path = tokenizer_path.resolve()
            except OSError:
                target_path = tokenizer_path
            if ckpt_path != target_path:
                ckpt_sha = payload.get("tokenizer_sha256")
                try:
                    target_sha = sha256_text(tokenizer_path.read_text())
                except OSError:
                    target_sha = None
                if not (ckpt_sha and target_sha and ckpt_sha == target_sha):
                    errors.append("tokenizer path mismatch")
    return errors


def build_checkpoint_payload(
    *,
    run_dir: Path,
    repo_root: Path | None,
    model_state: dict,
    optimizer_state: dict | None,
    rng_state: dict | None,
    model_config: dict,
    training_config: dict,
    tokenizer_path: Path | None,
    iter_num: int,
) -> dict[str, Any]:
    tokenizer_sha = None
    if tokenizer_path and tokenizer_path.exists():
        tokenizer_sha = sha256_text(tokenizer_path.read_text())
    payload: dict[str, Any] = {
        "kind": CHECKPOINT_KIND,
        "version": CHECKPOINT_VERSION,
        "created_at": iso_now(),
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "git_sha": git_sha(repo_root or run_dir),
        "iter": int(iter_num),
        "model_state": model_state,
        "model_config": model_config,
        "training_config": training_config,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        "tokenizer_sha256": tokenizer_sha,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if rng_state is not None:
        payload["rng_state"] = rng_state
    return payload
