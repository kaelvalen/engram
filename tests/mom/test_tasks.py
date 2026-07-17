"""Passkey + state-tracking task tests (spec §5.1)."""

from __future__ import annotations

import pytest
import torch
from mom.tasks.passkey import PasskeyConfig, make_passkey_batch
from mom.tasks.state_tracking import (
    StateTrackConfig,
    generator_permutations,
    make_state_track_batch,
)

# ---------------------------------------------------------------------------
# passkey
# ---------------------------------------------------------------------------


def test_passkey_shapes_and_single_scored_position():
    cfg = PasskeyConfig(vocab_size=64, seq_len=32)
    ids, labels = make_passkey_batch(cfg, 8, torch.Generator().manual_seed(0))
    assert ids.shape == labels.shape == (8, 32)
    assert (labels != -100).sum() == 8  # exactly one scored position per sample


def test_passkey_label_recalls_needle_value():
    cfg = PasskeyConfig(vocab_size=64, seq_len=32)
    ids, labels = make_passkey_batch(cfg, 16, torch.Generator().manual_seed(1))
    filler, marker = 63, 62
    for b in range(16):
        seq, lab = ids[b], labels[b]
        # locate the needle: key is the token right before the final marker…
        key = seq[-2]
        assert seq[-3] == marker
        # …and the pair (key, value) appears earlier in the stream
        hits = (seq[:-3] == key).nonzero().flatten()
        assert len(hits) == 1
        value = seq[hits[0] + 1]
        assert lab[-1] == value
        assert seq[-1] == filler  # no answer leak into the input stream


def test_passkey_determinism_and_depth_variation():
    cfg = PasskeyConfig(vocab_size=64, seq_len=48)
    a = make_passkey_batch(cfg, 4, torch.Generator().manual_seed(7))
    b = make_passkey_batch(cfg, 4, torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    c = make_passkey_batch(cfg, 64, torch.Generator().manual_seed(3))
    depths = (c[0][:, :-3] != 63).any(dim=0).nonzero().flatten()
    assert len(depths) > 1  # needle lands at varying depths


def test_passkey_vocab_bounds_and_validation():
    cfg = PasskeyConfig(vocab_size=32, seq_len=16)
    ids, labels = make_passkey_batch(cfg, 4, torch.Generator().manual_seed(0))
    assert ids.min() >= 0 and ids.max() < 32
    assert labels[labels != -100].max() < 32
    with pytest.raises(ValueError):
        PasskeyConfig(seq_len=3)


# ---------------------------------------------------------------------------
# state tracking
# ---------------------------------------------------------------------------


def _cfg(**kw):
    base = dict(num_elements=4, num_generators=3, seq_len=12, perm_seed=99)
    base.update(kw)
    return StateTrackConfig(**base)


def test_state_track_shapes_and_scoring():
    cfg = _cfg()
    ids, labels = make_state_track_batch(cfg, 6, torch.Generator().manual_seed(0))
    assert ids.shape == labels.shape == (6, 12)
    assert (labels != -100).sum() == 6
    assert ids.max() < cfg.vocab_size


def test_state_track_label_is_correct_composition():
    cfg = _cfg()
    perms = generator_permutations(cfg)
    ids, labels = make_state_track_batch(cfg, 16, torch.Generator().manual_seed(2))
    N, G = cfg.num_elements, cfg.num_generators
    for b in range(16):
        seq, lab = ids[b], labels[b]
        state = int(seq[0])
        for tok in seq[1:-2].tolist():
            assert N <= tok < N + G  # all middle tokens are generators
            state = int(perms[tok - N, state])
        assert seq[-2] == N + G  # MARK
        assert lab[-1] == state
        assert seq[-1] == N + G + 1  # PAD — no answer leak


def test_state_track_generator_set_is_fixed_across_batches():
    cfg = _cfg()
    p1 = generator_permutations(cfg)
    p2 = generator_permutations(cfg)
    assert torch.equal(p1, p2)
    assert p1.shape == (cfg.num_generators, cfg.num_elements)
    for row in p1:
        assert sorted(row.tolist()) == list(range(cfg.num_elements))


def test_state_track_determinism():
    cfg = _cfg()
    a = make_state_track_batch(cfg, 4, torch.Generator().manual_seed(5))
    b = make_state_track_batch(cfg, 4, torch.Generator().manual_seed(5))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_state_track_vocab_size_property():
    cfg = _cfg()
    assert cfg.vocab_size == cfg.num_elements + cfg.num_generators + 2
    with pytest.raises(ValueError):
        StateTrackConfig(num_elements=1)
    with pytest.raises(ValueError):
        StateTrackConfig(seq_len=3)
