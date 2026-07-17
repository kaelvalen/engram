"""MQAR — multi-query associative recall (Arora-style; spec §5.1, spike task).

Sequence layout (length ``seq_len``)::

    k₁ v₁ k₂ v₂ … kₙ vₙ [filler …] q₁ q₂ … qₘ

with n = ``num_pairs`` presented key→value pairs and m = n queries (all keys,
reshuffled).  Filler is a reserved token id (``vocab_size - 1``).  The model
is scored only on the position immediately after each query token — the
correct continuation is the value bound to that key.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MQARConfig:
    vocab_size: int = 8192
    num_pairs: int = 64
    seq_len: int = 512

    def __post_init__(self):
        # +1: the final query still needs a trailing position for its target.
        needed = 3 * self.num_pairs + 1
        if self.seq_len < needed:
            raise ValueError(
                f"seq_len={self.seq_len} too short for {self.num_pairs} pairs "
                f"(need ≥ {needed})"
            )
        if self.vocab_size < 8:
            raise ValueError("vocab_size too small")


def make_mqar_batch(
    cfg: MQARConfig,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    return_classes: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (input_ids, labels) with labels = -100 except after queries.

    With ``return_classes=True`` also returns per-token class ids for the
    specialization analysis (§7.2): 0=filler, 1=key, 2=value, 3=query.
    """
    filler = cfg.vocab_size - 1
    pool = cfg.vocab_size - 1  # keys/values sampled from [0, pool)
    n = cfg.num_pairs
    gap = cfg.seq_len - 3 * n - 1  # one trailing filler after the last query

    ids = torch.full((batch_size, cfg.seq_len), filler, dtype=torch.long)
    labels = torch.full((batch_size, cfg.seq_len), -100, dtype=torch.long)
    classes = torch.zeros(batch_size, cfg.seq_len, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randperm(pool, generator=generator)[:n]
        values = torch.randint(0, pool, (n,), generator=generator)
        kv = torch.stack([keys, values], dim=1).flatten()  # k1 v1 k2 v2 …
        ids[b, : 2 * n] = kv
        classes[b, 0 : 2 * n : 2] = 1
        classes[b, 1 : 2 * n : 2] = 2

        order = torch.randperm(n, generator=generator)
        queries = keys[order]
        q_start = 2 * n + gap
        ids[b, q_start : q_start + n] = queries
        classes[b, q_start : q_start + n] = 3
        # label[p] is the target for the prediction made at p-1 (the query).
        labels[b, q_start + 1 : q_start + n + 1] = values[order]
    ids, labels, classes = ids.to(device), labels.to(device), classes.to(device)
    if return_classes:
        return ids, labels, classes
    return ids, labels
