from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from tokenizers import Tokenizer


@dataclass
class DataSplit:
    train_text: str
    val_text: str | None


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
    idx = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in idx])
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in idx])
    return x.to(device), y.to(device)
