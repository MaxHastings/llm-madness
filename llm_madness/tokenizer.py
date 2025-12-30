from __future__ import annotations

import re
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
) -> dict:
    input_path = Path(input_path)
    text = input_path.read_text()

    discovered: list[str] = []
    if discover_regex:
        discovered = discover_special_tokens(text, discover_regex)

    special_tokens = list(special_tokens or [])
    for token in discovered:
        if token not in special_tokens:
            special_tokens.append(token)

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
    tokenizer.train([str(input_path)], trainer)
    tokenizer.save(str(output_path))
    return {
        "input_path": str(input_path),
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": special_tokens,
        "discover_regex": discover_regex,
        "discovered_tokens": discovered,
        "split_digits": split_digits,
        "output_path": str(output_path),
    }
