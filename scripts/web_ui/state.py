from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

from llm_madness.model import GPT, GPTConfig
from llm_madness.tokenizer import load_tokenizer

from .inspect import attention_from_trace, mlp_from_trace


class ServerState:
    def __init__(self, run_dir: Path, checkpoint: str | None, device_override: str):
        self.run_dir = run_dir
        self.run_config = json.loads((run_dir / "run.json").read_text())
        self.tokenizer = load_tokenizer(run_dir / "tokenizer.json")
        data_path = self.run_config.get("data_path")
        self.data_path = Path(data_path) if data_path else None
        self._tokenizer_report: dict | None = None
        device_name = device_override if device_override != "auto" else self.run_config.get("device", "auto")
        self.device = select_device(device_name)
        self.model = self._build_model()
        self.current_checkpoint: str | None = None
        self.load_checkpoint(checkpoint)

    def _build_model(self) -> GPT:
        model_cfg = self.run_config.get("config", {}).get("model", {})
        gpt_config = GPTConfig(
            vocab_size=self.tokenizer.get_vocab_size(),
            block_size=int(model_cfg.get("block_size", 128)),
            n_layer=int(model_cfg.get("n_layer", 4)),
            n_head=int(model_cfg.get("n_head", 4)),
            n_embd=int(model_cfg.get("n_embd", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        return GPT(gpt_config).to(self.device)

    def list_checkpoints(self) -> list[str]:
        ckpt_dir = self.run_dir / "checkpoints"
        checkpoints = []
        if ckpt_dir.exists():
            checkpoints.extend(sorted(p.name for p in ckpt_dir.glob("checkpoint_*.pt")))
        if (self.run_dir / "latest.pt").exists():
            checkpoints.append("latest.pt")
        return checkpoints

    def tokenizer_report(self, top_n: int = 25, max_chars: int = 200_000) -> dict:
        if self._tokenizer_report is not None:
            return self._tokenizer_report
        if self.data_path is None or not self.data_path.exists():
            return {"error": "data_path missing; rerun training with a dataset"}
        text = self.data_path.read_text(errors="ignore")
        if len(text) > max_chars:
            text = text[:max_chars]

        ids = self.tokenizer.encode(text).ids
        total_tokens = len(ids)
        vocab_size = self.tokenizer.get_vocab_size()
        unique_tokens = len(set(ids))
        coverage = unique_tokens / max(1, vocab_size)

        vocab = self.tokenizer.get_vocab()
        unk_id = None
        for token in ("<|unk|>", "<unk>", "[UNK]"):
            if token in vocab:
                unk_id = vocab[token]
                break
        unk_count = ids.count(unk_id) if unk_id is not None else 0
        unk_rate = unk_count / max(1, total_tokens)

        freq = Counter(ids)
        top_tokens = []
        for token_id, count in freq.most_common(top_n):
            top_tokens.append(
                {
                    "id": token_id,
                    "token": self.tokenizer.id_to_token(token_id),
                    "count": count,
                }
            )
        report = {
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "vocab_size": vocab_size,
            "coverage": coverage,
            "unk_rate": unk_rate,
            "top_tokens": top_tokens,
        }
        self._tokenizer_report = report
        return report

    def training_logs(self) -> dict:
        logs_path = self.run_dir / "logs.jsonl"
        samples_path = self.run_dir / "samples.jsonl"
        logs = []
        if logs_path.exists():
            for line in logs_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        samples = []
        if samples_path.exists():
            for line in samples_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return {"logs": logs, "samples": samples}

    def ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.tokenizer.id_to_token(idx) for idx in ids]

    def inspect(self, ids: list[int], layer: int, head: int, mode: str, top_k: int) -> dict:
        if not ids:
            return {"error": "no ids provided"}
        max_len = min(len(ids), self.model.config.block_size, 64)
        ids = ids[-max_len:]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        self.model.eval()
        tokens = [self.tokenizer.id_to_token(i) for i in ids]
        if mode == "attention":
            with torch.no_grad():
                _, trace = self.model.forward_with_trace(idx)
            layer_idx, head_idx, matrix, min_val, max_val = attention_from_trace(trace, layer, head)
            return {
                "attention": matrix,
                "tokens": tokens,
                "meta": f"layer {layer_idx} head {head_idx} tokens {len(tokens)}",
                "min_val": min_val,
                "max_val": max_val,
            }
        if mode == "mlp":
            with torch.no_grad():
                _, trace = self.model.forward_with_trace(idx)
            layer_idx, activations = mlp_from_trace(trace, layer, top_k)
            return {
                "activations": activations,
                "tokens": tokens,
                "meta": f"layer {layer_idx} token {len(tokens) - 1}",
            }
        if mode == "layer_topk":
            with torch.no_grad():
                _, hidden_states = self.model.forward_with_hidden_states(idx)
            results = []
            for layer_idx, hidden in enumerate(hidden_states):
                layer_hidden = self.model.ln_f(hidden)
                logits = self.model.head(layer_hidden)[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                values, indices = torch.topk(probs, k=min(top_k, probs.size(-1)))
                topk = []
                for prob, token_id in zip(values[0], indices[0]):
                    tid = int(token_id.item())
                    topk.append(
                        {
                            "id": tid,
                            "token": self.tokenizer.id_to_token(tid),
                            "prob": float(prob.item()),
                        }
                    )
                results.append({"layer": layer_idx, "topk": topk})
            return {"layers": results, "tokens": tokens, "meta": f"layers {len(results)}"}
        return {"error": f"unknown mode: {mode}"}

    def load_checkpoint(self, checkpoint: str | None) -> None:
        if checkpoint is None:
            latest_path = self.run_dir / "latest.pt"
            if latest_path.exists():
                ckpt_path = latest_path
            else:
                candidates = sorted((self.run_dir / "checkpoints").glob("checkpoint_*.pt"))
                if not candidates:
                    raise FileNotFoundError("no checkpoints found for run")
                ckpt_path = candidates[-1]
        else:
            if checkpoint == "latest.pt":
                ckpt_path = self.run_dir / "latest.pt"
            else:
                ckpt_path = self.run_dir / "checkpoints" / checkpoint
            if not ckpt_path.exists():
                raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.current_checkpoint = ckpt_path.name


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps" and (not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available()):
        return torch.device("cpu")
    return torch.device(name)
