from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


def discover_special_tokens(text: str, pattern: str) -> list[str]:
    found = sorted(set(re.findall(pattern, text)))
    return found


def load_tokenizer(path: Path | str) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def train_bpe_tokenizer(
    input_path: Path | str,
    output_path: Path | str,
    vocab_size: int,
    min_frequency: int = 2,
    special_tokens: Iterable[str] | None = None,
    discover_regex: str | None = None,
    add_prefix_space: bool = False,
    byte_level: bool = True,
    split_digits: bool = False,
    progress_heartbeat_seconds: int | None = 30,
) -> dict:
    input_path = Path(input_path)
    print(f"[tokenizer] loading input {input_path}", flush=True)
    text = input_path.read_text()
    print(f"[tokenizer] input length {len(text)} chars", flush=True)

    discovered: list[str] = []
    if discover_regex:
        print("[tokenizer] discovering special tokens", flush=True)
        discovered = discover_special_tokens(text, discover_regex)

    special_tokens = list(special_tokens or [])
    for token in discovered:
        if token not in special_tokens:
            special_tokens.append(token)

    print(f"[tokenizer] training bpe vocab_size={vocab_size} min_freq={min_frequency}", flush=True)
    tokenizer = Tokenizer(BPE(unk_token=special_tokens[0] if special_tokens else "<|unk|>"))
    pre_tokenizers = []
    if split_digits:
        pre_tokenizers.append(Digits(individual_digits=True))
    if byte_level:
        pre_tokenizers.append(ByteLevel(add_prefix_space=add_prefix_space))

    if pre_tokenizers:
        tokenizer.pre_tokenizer = pre_tokenizers[0] if len(pre_tokenizers) == 1 else Sequence(pre_tokenizers)
    if byte_level:
        tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    started = time.monotonic()
    stop_event = threading.Event()

    def heartbeat() -> None:
        while not stop_event.wait(float(progress_heartbeat_seconds or 0)):
            elapsed = int(time.monotonic() - started)
            mins, secs = divmod(elapsed, 60)
            print(f"[tokenizer] training... {mins}m {secs}s elapsed", flush=True)

    hb_thread: threading.Thread | None = None
    if progress_heartbeat_seconds and progress_heartbeat_seconds > 0:
        hb_thread = threading.Thread(target=heartbeat, daemon=True)
        hb_thread.start()
    tokenizer.train([str(input_path)], trainer)
    stop_event.set()
    print("[tokenizer] training complete, saving tokenizer", flush=True)
    tokenizer.save(str(output_path))
    print(f"[tokenizer] saved {output_path}", flush=True)
    return {
        "input_path": str(input_path),
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": special_tokens,
        "discover_regex": discover_regex,
        "discovered_tokens": discovered,
        "split_digits": split_digits,
        "output_path": str(output_path),
        "progress_heartbeat_seconds": progress_heartbeat_seconds,
    }
