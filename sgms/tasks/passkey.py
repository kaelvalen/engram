"""Passkey retrieval (spec §5.1): a needle (key→value pair) hidden at a
random depth in a long filler stream; the model must recall the value when
the key is re-presented at the end.

Layout::

    [filler …] k v [filler …] MARK k PAD
                                    ↑ label here (= v)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PasskeyConfig:
    vocab_size: int = 8192
    seq_len: int = 2048

    def __post_init__(self):
        if self.seq_len < 5:
            raise ValueError("seq_len must be ≥ 5")
        if self.vocab_size < 8:
            raise ValueError("vocab_size too small")


def make_passkey_batch(
    cfg: PasskeyConfig,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids, labels); labels = -100 except the final position."""
    filler = cfg.vocab_size - 1
    marker = cfg.vocab_size - 2
    pool = cfg.vocab_size - 2  # keys/values sampled from [0, pool)

    ids = torch.full((batch_size, cfg.seq_len), filler, dtype=torch.long)
    labels = torch.full((batch_size, cfg.seq_len), -100, dtype=torch.long)
    for b in range(batch_size):
        key = int(torch.randint(0, pool, (1,), generator=generator))
        value = int(torch.randint(0, pool, (1,), generator=generator))
        depth = int(torch.randint(0, cfg.seq_len - 4, (1,), generator=generator))
        ids[b, depth] = key
        ids[b, depth + 1] = value
        ids[b, -3] = marker
        ids[b, -2] = key
        # ids[-1] stays filler: the model must *produce* the value after the key.
        labels[b, -1] = value
    return ids.to(device), labels.to(device)
