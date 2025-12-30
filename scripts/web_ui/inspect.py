from __future__ import annotations

import torch


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def attention_from_trace(trace: dict, layer: int, head: int) -> tuple[int, int, list[list[float]], float, float]:
    attn = trace.get("attn", [])
    if not attn:
        return 0, 0, [], 0.0, 0.0
    layer_idx = clamp(layer, 0, len(attn) - 1)
    layer_attn = attn[layer_idx]
    head_idx = clamp(head, 0, layer_attn.size(1) - 1)
    matrix = layer_attn[0, head_idx].tolist()
    min_val = float(min(min(row) for row in matrix)) if matrix else 0.0
    max_val = float(max(max(row) for row in matrix)) if matrix else 0.0
    return layer_idx, head_idx, matrix, min_val, max_val


def mlp_from_trace(trace: dict, layer: int, top_k: int) -> tuple[int, list[dict]]:
    mlp = trace.get("mlp", [])
    if not mlp:
        return 0, []
    layer_idx = clamp(layer, 0, len(mlp) - 1)
    hidden = mlp[layer_idx][0, -1]
    values, indices = torch.topk(hidden, k=min(top_k, hidden.numel()))
    activations = [
        {"index": int(idx.item()), "value": float(val.item())}
        for idx, val in zip(indices, values)
    ]
    return layer_idx, activations
