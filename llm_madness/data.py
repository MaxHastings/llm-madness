from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from tokenizers import Tokenizer


@dataclass
class DataSplit:
    train_text: str
    val_text: str | None


@dataclass(frozen=True)
class TokenDataset:
    train: np.memmap
    val: np.memmap | None
    dtype: np.dtype
    vocab_size: int


def split_text_by_lines(text: str, val_split: float) -> DataSplit:
    if val_split <= 0.0:
        return DataSplit(train_text=text, val_text=None)
    if val_split >= 1.0:
        raise ValueError("val_split must be in (0, 1)")
    lines = text.splitlines(keepends=True)
    if len(lines) < 2:
        return DataSplit(train_text=text, val_text=None)
    split_idx = int(len(lines) * (1.0 - val_split))
    split_idx = max(1, min(len(lines) - 1, split_idx))
    return DataSplit(train_text="".join(lines[:split_idx]), val_text="".join(lines[split_idx:]))


def encode_text(tokenizer: Tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer.encode(text).ids
    return torch.tensor(ids, dtype=torch.long)


def decode_ids(tokenizer: Tokenizer, ids: Iterable[int]) -> str:
    return tokenizer.decode(list(ids))


def get_batch(tokens: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    if tokens.numel() < block_size + 1:
        raise ValueError("not enough tokens for the requested block size")
    max_start = tokens.numel() - block_size - 1
    idx = torch.randint(0, max_start + 1, (batch_size,), device=tokens.device)
    offsets = torch.arange(block_size, device=tokens.device)
    x_idx = idx[:, None] + offsets[None, :]
    x = tokens[x_idx]
    y = tokens[x_idx + 1]
    return x.to(device), y.to(device)


def load_token_dataset(run_dir: str, *, mode: str = "r") -> TokenDataset:
    import json
    from pathlib import Path

    base = Path(run_dir)
    meta_path = base / "meta.json"
    raw = json.loads(meta_path.read_text())
    if not isinstance(raw, dict) or raw.get("kind") != "TokenDataset":
        raise ValueError(f"not a TokenDataset: {meta_path}")
    dtype = np.dtype(str(raw.get("dtype")))
    vocab_size = int(raw.get("vocab_size", 0))
    paths = raw.get("paths") or {}
    train_path = Path(str(paths.get("train")))
    val_path = Path(str(paths.get("val")))
    train = np.memmap(train_path, dtype=dtype, mode=mode)
    val = None
    if val_path.exists() and val_path.stat().st_size > 0:
        val = np.memmap(val_path, dtype=dtype, mode=mode)
    return TokenDataset(train=train, val=val, dtype=dtype, vocab_size=vocab_size)


def get_batch_memmap(tokens: np.memmap, block_size: int, batch_size: int, device: torch.device):
    if tokens.size < block_size + 1:
        raise ValueError("not enough tokens for the requested block size")
    max_start = int(tokens.size) - block_size - 1
    idx = torch.randint(0, max_start + 1, (batch_size,), device="cpu")
    offsets = torch.arange(block_size, device="cpu")
    x_idx = (idx[:, None] + offsets[None, :]).numpy()
    x_np = tokens[x_idx]
    y_np = tokens[x_idx + 1]
    x = torch.from_numpy(x_np.astype(np.int64, copy=False)).to(device)
    y = torch.from_numpy(y_np.astype(np.int64, copy=False)).to(device)
    return x, y
