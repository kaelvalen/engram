"""State tracking probe (spec §5.1, S4-style): composition of permutations.

A fixed set of G permutations of N elements is drawn once (``perm_seed``) and
shared by every batch.  Each sequence applies a random chain of them to a
random start element; the model must output the final element::

    [x₀] [g₁ g₂ … g_T] [MARK] [PAD]
                              ↑ label here (= g_T∘…∘g₁(x₀))

Token ids: elements are 0..N-1, generator g is N+g, MARK is N+G, PAD is N+G+1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StateTrackConfig:
    num_elements: int = 5  # N — track the image of one element (the hard core)
    num_generators: int = 4  # G — fixed permutation set
    seq_len: int = 1024  # total length incl. x0, MARK, PAD
    perm_seed: int = 12345  # fixed ⇒ identical task across batches/seeds

    def __post_init__(self):
        if self.seq_len < 4:
            raise ValueError("seq_len must be ≥ 4")
        if self.num_elements < 2 or self.num_generators < 1:
            raise ValueError("need ≥2 elements and ≥1 generator")

    @property
    def vocab_size(self) -> int:
        return self.num_elements + self.num_generators + 2


def generator_permutations(cfg: StateTrackConfig) -> torch.Tensor:
    """The fixed (G, N) permutation table (row g maps element e → perms[g, e])."""
    g = torch.Generator().manual_seed(cfg.perm_seed)
    return torch.stack(
        [torch.randperm(cfg.num_elements, generator=g) for _ in range(cfg.num_generators)]
    )


def make_state_track_batch(
    cfg: StateTrackConfig,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids, labels); labels = -100 except the final position."""
    N, G = cfg.num_elements, cfg.num_generators
    mark = N + G
    pad = N + G + 1
    T = cfg.seq_len - 3  # number of applied generators
    perms = generator_permutations(cfg)

    ids = torch.full((batch_size, cfg.seq_len), pad, dtype=torch.long)
    labels = torch.full((batch_size, cfg.seq_len), -100, dtype=torch.long)
    for b in range(batch_size):
        x0 = int(torch.randint(0, N, (1,), generator=generator))
        ops = torch.randint(0, G, (T,), generator=generator)
        state = x0
        for op in ops.tolist():
            state = int(perms[op, state])
        ids[b, 0] = x0
        ids[b, 1 : T + 1] = N + ops
        ids[b, T + 1] = mark
        # ids[T+2] stays PAD: the model must *produce* the final element.
        labels[b, T + 2] = state
    return ids.to(device), labels.to(device)
