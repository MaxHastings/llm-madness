#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import torch

from llm_madness.data import encode_text, get_batch, split_text_by_lines
from llm_madness.model import GPT, GPTConfig
from llm_madness.tokenizer import load_tokenizer
from llm_madness.utils import ensure_dir, find_latest_run, git_sha, sha256_text, timestamp, write_json


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def estimate_loss(
    model: GPT,
    tokens: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            xb, yb = get_batch(tokens, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def generate_sample(
    model: GPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
) -> str:
    model.eval()
    with torch.no_grad():
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    model.train()
    return tokenizer.decode(out[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small GPT model.")
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/train"))
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    data_path = args.data
    if data_path is None:
        latest = find_latest_run(Path("data/combined"), filename="combined.txt")
        if latest is None:
            latest = find_latest_run(Path("data/curated"), filename="curated.txt")
        if latest is None:
            raise SystemExit("no dataset found; pass --data")
        data_path = latest

    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        latest_tok = find_latest_run(Path("runs/tokenizer"), filename="tokenizer.json")
        if latest_tok is None:
            raise SystemExit("no tokenizer found; pass --tokenizer")
        tokenizer_path = latest_tok

    run_dir = ensure_dir(args.output_dir / timestamp())
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 1337))
    random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = load_tokenizer(tokenizer_path)
    text = data_path.read_text()
    split = split_text_by_lines(text, float(train_cfg.get("val_split", 0.0)))

    train_tokens = encode_text(tokenizer, split.train_text)
    val_tokens = encode_text(tokenizer, split.val_text) if split.val_text else None

    device = select_device(str(train_cfg.get("device", "auto")))

    gpt_config = GPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=int(model_cfg.get("block_size", 128)),
        n_layer=int(model_cfg.get("n_layer", 4)),
        n_head=int(model_cfg.get("n_head", 4)),
        n_embd=int(model_cfg.get("n_embd", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    model = GPT(gpt_config).to(device)

    if train_tokens.numel() < 2:
        raise SystemExit("training data is too small; need at least 2 tokens")
    train_block_size = min(gpt_config.block_size, train_tokens.numel() - 1)

    val_block_size: int | None = None
    if val_tokens is not None:
        if val_tokens.numel() < 2:
            print("warning: validation split too small; skipping validation")
            val_tokens = None
        else:
            val_block_size = min(gpt_config.block_size, val_tokens.numel() - 1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
    )

    max_iters = int(train_cfg.get("max_iters", 2000))
    batch_size = int(train_cfg.get("batch_size", 32))
    log_interval = int(train_cfg.get("log_interval", 50))
    eval_interval = int(train_cfg.get("eval_interval", 200))
    eval_iters = int(train_cfg.get("eval_iters", 50))
    save_interval = int(train_cfg.get("save_interval", 500))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    warmup_iters = int(train_cfg.get("warmup_iters", 200))
    sample_prompt = str(train_cfg.get("sample_prompt", "1 + 1 ="))
    sample_length = int(train_cfg.get("sample_length", 32))
    sample_temperature = float(train_cfg.get("sample_temperature", 1.0))
    sample_top_k = train_cfg.get("sample_top_k", None)
    sample_top_k = int(sample_top_k) if sample_top_k is not None else None

    def get_lr(iter_num: int) -> float:
        base_lr = float(train_cfg.get("learning_rate", 3e-4))
        if iter_num < warmup_iters:
            return base_lr * iter_num / max(1, warmup_iters)
        progress = (iter_num - warmup_iters) / max(1, max_iters - warmup_iters)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    run_info = {
        "config": config,
        "data_path": str(data_path),
        "data_sha256": sha256_text(text),
        "tokenizer_path": str(tokenizer_path),
        "device": str(device),
        "git_sha": git_sha(Path(__file__).resolve().parents[1]),
    }
    write_json(run_dir / "run.json", run_info)
    (run_dir / "tokenizer.json").write_text(tokenizer_path.read_text())
    (run_dir / "training_config.json").write_text(args.config.read_text())

    log_path = run_dir / "logs.jsonl"
    for iter_num in range(1, max_iters + 1):
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        xb, yb = get_batch(train_tokens, train_block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if iter_num % log_interval == 0:
            train_ppl = math.exp(loss.item())
            log_entry = {
                "iter": iter_num,
                "train_loss": float(loss.item()),
                "train_ppl": float(train_ppl),
                "lr": lr,
            }
            with log_path.open("a") as log_file:
                log_file.write(json.dumps(log_entry) + "\n")
            print(f"iter {iter_num} train loss {loss.item():.4f} lr {lr:.6f}")

        if val_tokens is not None and val_block_size is not None and iter_num % eval_interval == 0:
            val_loss = estimate_loss(
                model,
                val_tokens,
                val_block_size,
                batch_size,
                eval_iters,
                device,
            )
            val_ppl = math.exp(val_loss)
            with log_path.open("a") as log_file:
                log_file.write(
                    json.dumps({"iter": iter_num, "val_loss": val_loss, "val_ppl": val_ppl}) + "\n"
                )
            print(f"iter {iter_num} val loss {val_loss:.4f}")

        if iter_num % save_interval == 0 or iter_num == max_iters:
            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "iter": iter_num,
                "config": config,
            }
            ckpt_path = run_dir / "checkpoints" / f"checkpoint_{iter_num:07d}.pt"
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, run_dir / "latest.pt")

            sample_text = generate_sample(
                model,
                tokenizer,
                sample_prompt,
                sample_length,
                sample_temperature,
                sample_top_k,
                device,
            )
            sample_entry = {
                "iter": iter_num,
                "checkpoint": ckpt_path.name,
                "prompt": sample_prompt,
                "sample": sample_text,
            }
            with (run_dir / "samples.jsonl").open("a") as sample_file:
                sample_file.write(json.dumps(sample_entry) + "\n")

    print(f"training complete. run saved to {run_dir}")


if __name__ == "__main__":
    main()
