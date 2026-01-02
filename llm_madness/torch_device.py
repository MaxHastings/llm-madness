from __future__ import annotations

import torch


def _mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def select_device(name: str, *, strict: bool = False) -> torch.device:
    """
    Select a torch.device from a user-provided string.

    - "auto": cuda -> mps -> cpu
    - explicit "cuda"/"cuda:0": requires CUDA available when strict=True
    - "mps": requires MPS available when strict=True
    """

    name = (name or "auto").strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if name.startswith("cuda"):
        if strict and not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device(name)

    if name == "mps":
        if strict and not _mps_available():
            raise ValueError("MPS requested but torch.backends.mps.is_available() is False")
        return torch.device("mps")

    return torch.device(name)

