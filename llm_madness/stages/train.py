from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from llm_madness.checkpoints import build_checkpoint_payload, is_checkpoint_v2, load_checkpoint, validate_checkpoint
from llm_madness.data import encode_text, get_batch, get_batch_memmap, load_token_dataset, split_text_by_lines
from llm_madness.model import GPT, GPTConfig
from llm_madness.runs import finish_manifest, start_manifest
from llm_madness.torch_device import select_device
from llm_madness.tokenizer import load_tokenizer
from llm_madness.utils import ensure_dir, find_latest_run, sha256_file, timestamp


def estimate_loss(
    model: GPT,
    tokens: torch.Tensor | np.memmap,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            if isinstance(tokens, torch.Tensor):
                xb, yb = get_batch(tokens, block_size, batch_size, device)
            else:
                xb, yb = get_batch_memmap(tokens, block_size, batch_size, device)
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


def run_train(
    config: dict,
    data_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    repo_root: Path,
    init_checkpoint: Path | None = None,
    init_mode: str | None = None,
) -> dict:
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    init_checkpoint_cfg = train_cfg.get("init_checkpoint")
    init_mode_cfg = train_cfg.get("init_mode")

    if init_checkpoint is None and init_checkpoint_cfg:
        init_checkpoint = Path(str(init_checkpoint_cfg))
    if init_mode is None and init_mode_cfg:
        init_mode = str(init_mode_cfg)

    run_id = config.get("run", {}).get("id") if isinstance(config.get("run"), dict) else None
    run_dir = ensure_dir(output_dir / (run_id or timestamp()))
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 1337))
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        device = select_device(str(train_cfg.get("device", "auto")), strict=True)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

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

    def resolve_latest(path: Path) -> Path:
        if "latest" not in path.parts:
            return path
        parts = list(path.parts)
        idx = parts.index("latest")
        base = Path(*parts[:idx])
        suffix = Path(*parts[idx + 1 :])
        latest = find_latest_run(base)
        if latest is None:
            return path
        return latest / suffix

    init_mode_value = (init_mode or "").strip().lower() if init_mode else None
    init_checkpoint_path: Path | None = None
    if init_checkpoint is not None:
        init_checkpoint_path = resolve_latest(Path(init_checkpoint))
        if not init_mode_value:
            init_mode_value = "fork"
    if init_checkpoint_path is None:
        init_mode_value = init_mode_value or "fresh"

    start_manifest(
        "train",
        run_dir,
        config,
        inputs={
            "data": str(data_path),
            "tokenizer": str(tokenizer_path),
            "init_checkpoint": str(init_checkpoint_path) if init_checkpoint_path else None,
            "init_mode": init_mode_value if init_checkpoint_path else None,
        },
        outputs={"run_dir": str(run_dir), "device": str(device)},
        repo_root=repo_root,
    )

    try:
        if init_checkpoint_path is None and init_mode_value != "fresh":
            raise SystemExit("init_mode requires init_checkpoint")
        if init_checkpoint_path is not None and init_mode_value == "fresh":
            raise SystemExit("init_checkpoint requires init_mode fork or resume")

        tokenizer = load_tokenizer(tokenizer_path)
        gpt_config = GPTConfig(
            vocab_size=tokenizer.get_vocab_size(),
            block_size=int(model_cfg.get("block_size", 128)),
            n_layer=int(model_cfg.get("n_layer", 4)),
            n_head=int(model_cfg.get("n_head", 4)),
            n_embd=int(model_cfg.get("n_embd", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_rmsnorm=bool(model_cfg.get("use_rmsnorm", False)),
            use_swiglu=bool(model_cfg.get("use_swiglu", False)),
            use_rope=bool(model_cfg.get("use_rope", False)),
            use_sdpa=bool(model_cfg.get("use_sdpa", False)),
            use_kv_cache=bool(model_cfg.get("use_kv_cache", False)),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
        )
        model = GPT(gpt_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("learning_rate", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        )

        start_iter = 1
        if init_checkpoint_path is not None:
            bundle = load_checkpoint(init_checkpoint_path, map_location="cpu")
            if not is_checkpoint_v2(bundle.payload):
                raise SystemExit("checkpoint must be v2; run scripts/migrate_checkpoints.py")
            errors = validate_checkpoint(bundle.payload, gpt_config.__dict__, tokenizer_path)
            if errors:
                raise SystemExit(f"checkpoint incompatible: {', '.join(errors)}")
            model.load_state_dict(bundle.payload["model_state"])
            if init_mode_value == "resume":
                optimizer_state = bundle.payload.get("optimizer_state")
                if optimizer_state is None:
                    raise SystemExit("resume requires optimizer_state in checkpoint")
                optimizer.load_state_dict(optimizer_state)
                rng_state = bundle.payload.get("rng_state") or {}
                if rng_state.get("python") is not None:
                    random.setstate(rng_state["python"])
                if rng_state.get("torch") is not None:
                    torch.set_rng_state(rng_state["torch"])
                if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng_state["cuda"])
                start_iter = int(bundle.payload.get("iter", 0)) + 1
            elif init_mode_value != "fork":
                raise SystemExit(f"unsupported init_mode: {init_mode_value}")

        if start_iter > max_iters:
            raise SystemExit("checkpoint iter exceeds max_iters")

        val_split = float(train_cfg.get("val_split", 0.0))

        train_tokens: torch.Tensor | np.memmap
        val_tokens: torch.Tensor | np.memmap | None

        if data_path.is_dir() and (data_path / "meta.json").exists():
            ds = load_token_dataset(str(data_path), mode="r")
            if ds.vocab_size and ds.vocab_size != tokenizer.get_vocab_size():
                raise SystemExit("token dataset vocab_size does not match tokenizer vocab_size")
            train_tokens = ds.train
            val_tokens = ds.val
            print(
                f"[train] using token dataset {data_path} (train={ds.train.size} tokens, val={ds.val.size if ds.val is not None else 0} tokens)",
                flush=True,
            )
        else:
            cache_cfg = train_cfg.get("token_cache", {})
            cache_enabled = bool(cache_cfg.get("enabled", True))
            cache_use_content_hash = bool(cache_cfg.get("use_content_hash", False))
            cache_dir = Path(cache_cfg.get("dir", "runs/cache/tokens"))
            cache_dir = ensure_dir(cache_dir) if cache_enabled else cache_dir

            tok_hash: str | None = None
            data_hash: str | None = None
            if cache_enabled:
                t0 = time.perf_counter()
                if cache_use_content_hash:
                    tok_hash = sha256_file(tokenizer_path)
                    data_hash = sha256_file(data_path)
                else:
                    tok_stat = tokenizer_path.stat()
                    data_stat = data_path.stat()
                    # Use a filesystem-safe fingerprint for cache filenames.
                    tok_hash = f"stat_{tok_stat.st_size}_{int(tok_stat.st_mtime)}"
                    data_hash = f"stat_{data_stat.st_size}_{int(data_stat.st_mtime)}"
                print(f"[train] computed cache keys in {time.perf_counter() - t0:.2f}s", flush=True)

            def _cache_key() -> str:
                assert tok_hash is not None
                assert data_hash is not None
                val_tag = f"{val_split:.6f}".replace(".", "p")
                return f"v1__tok_{tok_hash[:16]}__data_{data_hash[:16]}__val_{val_tag}"

            train_tokens_path: Path | None = None
            val_tokens_path: Path | None = None
            meta_path: Path | None = None
            if cache_enabled:
                key = _cache_key()
                train_tokens_path = cache_dir / f"{key}__train.pt"
                val_tokens_path = cache_dir / f"{key}__val.pt"
                meta_path = cache_dir / f"{key}.json"

            if (
                cache_enabled
                and train_tokens_path is not None
                and val_tokens_path is not None
                and train_tokens_path.exists()
                and (val_split <= 0.0 or val_tokens_path.exists())
            ):
                t0 = time.perf_counter()
                train_tokens = torch.load(train_tokens_path, map_location="cpu")
                if val_split > 0.0:
                    cached_val = torch.load(val_tokens_path, map_location="cpu")
                    val_tokens = None if cached_val.numel() == 0 else cached_val
                else:
                    val_tokens = None
                print(f"[train] loaded cached tokens in {time.perf_counter() - t0:.2f}s", flush=True)
            else:
                t0 = time.perf_counter()
                text = data_path.read_text(encoding="utf-8", errors="replace")
                split = split_text_by_lines(text, val_split)
                train_tokens = encode_text(tokenizer, split.train_text)
                val_tokens = encode_text(tokenizer, split.val_text) if split.val_text else None
                print(f"[train] tokenized dataset in {time.perf_counter() - t0:.2f}s", flush=True)

                if cache_enabled and train_tokens_path is not None and val_tokens_path is not None and meta_path is not None:
                    t1 = time.perf_counter()
                    torch.save(train_tokens, train_tokens_path)
                    torch.save(val_tokens if val_tokens is not None else torch.empty(0, dtype=torch.long), val_tokens_path)
                    meta = {
                        "kind": "TokenCache",
                        "version": 1,
                        "data_path": str(data_path),
                        "data_fingerprint": data_hash,
                        "tokenizer_path": str(tokenizer_path),
                        "tokenizer_fingerprint": tok_hash,
                        "val_split": val_split,
                        "use_content_hash": cache_use_content_hash,
                        "train_tokens": str(train_tokens_path),
                        "val_tokens": str(val_tokens_path),
                        "val_tokens_empty": val_tokens is None,
                    }
                    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
                    print(f"[train] wrote token cache in {time.perf_counter() - t1:.2f}s", flush=True)

        train_len = int(train_tokens.numel()) if isinstance(train_tokens, torch.Tensor) else int(train_tokens.size)
        if train_len < 2:
            raise SystemExit("training data is too small; need at least 2 tokens")
        train_block_size = min(gpt_config.block_size, train_len - 1)

        val_block_size: int | None = None
        if val_tokens is not None:
            val_len = int(val_tokens.numel()) if isinstance(val_tokens, torch.Tensor) else int(val_tokens.size)
            if val_len < 2:
                print("warning: validation split too small; skipping validation")
                val_tokens = None
            else:
                val_block_size = min(gpt_config.block_size, val_len - 1)

        tokenizer_text = tokenizer_path.read_text(encoding="utf-8", errors="replace")
        (run_dir / "tokenizer.json").write_text(tokenizer_text, encoding="utf-8")
        (run_dir / "training_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if init_checkpoint_path is not None:
            train_cfg["init_checkpoint"] = str(init_checkpoint_path)
            train_cfg["init_mode"] = init_mode_value

        log_path = run_dir / "logs.jsonl"
        samples_path = run_dir / "samples.jsonl"
        for iter_num in range(start_iter, max_iters + 1):
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if isinstance(train_tokens, torch.Tensor):
                xb, yb = get_batch(train_tokens, train_block_size, batch_size, device)
            else:
                xb, yb = get_batch_memmap(train_tokens, train_block_size, batch_size, device)
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
                entry = {"iter": iter_num, "val_loss": val_loss, "val_ppl": math.exp(val_loss)}
                with log_path.open("a") as log_file:
                    log_file.write(json.dumps(entry) + "\n")
                print(f"iter {iter_num} val loss {val_loss:.4f}")

            if iter_num % save_interval == 0 or iter_num == max_iters:
                ckpt_name = f"checkpoint_{iter_num}.pt"
                ckpt_path = run_dir / "checkpoints" / ckpt_name
                rng_state = {
                    "python": random.getstate(),
                    "torch": torch.get_rng_state(),
                }
                if torch.cuda.is_available():
                    rng_state["cuda"] = torch.cuda.get_rng_state_all()
                payload = build_checkpoint_payload(
                    run_dir=run_dir,
                    repo_root=repo_root,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    rng_state=rng_state,
                    model_config=gpt_config.__dict__,
                    training_config=config,
                    tokenizer_path=tokenizer_path,
                    iter_num=iter_num,
                )
                torch.save(payload, ckpt_path)
                (run_dir / "latest.pt").write_bytes(ckpt_path.read_bytes())
                print(f"saved checkpoint {ckpt_name}")

            if iter_num % eval_interval == 0:
                sample = generate_sample(
                    model,
                    tokenizer,
                    sample_prompt,
                    sample_length,
                    sample_temperature,
                    sample_top_k,
                    device,
                )
                entry = {"iter": iter_num, "prompt": sample_prompt, "sample": sample}
                with samples_path.open("a") as sample_file:
                    sample_file.write(json.dumps(entry) + "\n")

        finish_manifest(run_dir, "complete", outputs={"run_dir": str(run_dir)})
        return {"run_dir": run_dir}
    except SystemExit as exc:
        finish_manifest(run_dir, "failed", error=str(exc))
        raise
    except Exception as exc:
        finish_manifest(run_dir, "failed", error=str(exc))
        raise
