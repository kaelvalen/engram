from __future__ import annotations

from engram.data.audio import get_audio_loaders


def test_synthetic_audio_loader_shapes():
    tr, va = get_audio_loaders(
        batch_size=8, synthetic=True, train_size=32, val_size=16, num_workers=0
    )
    x, y = next(iter(tr))
    assert x.ndim == 3 and y.ndim == 1
    assert x.shape[0] == 8
