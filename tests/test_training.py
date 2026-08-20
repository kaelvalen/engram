from __future__ import annotations

from pathlib import Path

import torch
from engram.config import ENGRAMConfig, ModalityConfig
from engram.model import ENGRAMForClassification
from engram.training.loops import evaluate_epoch, train_epoch
from engram.training.trainer import Trainer, TrainerConfig
from torch.utils.data import DataLoader, TensorDataset


def _tiny_cfg() -> ENGRAMConfig:
    return ENGRAMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        delta_every=2,
        modalities=[ModalityConfig(name="ecg", input_dim=12, num_classes=5)],
    )


def _tiny_loader(n: int = 16, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n, 32, 12)
    y = torch.randint(0, 5, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_train_epoch_returns_loss_and_acc():
    model = ENGRAMForClassification(_tiny_cfg())
    loader = _tiny_loader()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")
    loss, acc = train_epoch(model, loader, opt, device, modality="ecg")
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert loss > 0
    assert 0.0 <= acc <= 1.0


def test_evaluate_epoch_returns_loss_and_acc():
    model = ENGRAMForClassification(_tiny_cfg())
    loader = _tiny_loader()
    device = torch.device("cpu")
    loss, acc = evaluate_epoch(model, loader, device, modality="ecg")
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_trainer_fit_one_epoch(tmp_path: Path):
    cfg = _tiny_cfg()
    model = ENGRAMForClassification(cfg)
    loader = _tiny_loader(n=8, batch_size=4)
    tcfg = TrainerConfig(epochs=1, log_every_epoch=False)
    trainer = Trainer(model, cfg, device=torch.device("cpu"), tcfg=tcfg)
    result = trainer.fit(loader, loader, modality="ecg", output_dir=tmp_path)
    assert "best_val_acc" in result
    assert "history" in result
    assert len(result["history"]) == 1


def test_trainer_checkpoint_saved(tmp_path: Path):
    cfg = _tiny_cfg()
    model = ENGRAMForClassification(cfg)
    loader = _tiny_loader(n=8, batch_size=4)
    tcfg = TrainerConfig(epochs=1, log_every_epoch=False)
    trainer = Trainer(model, cfg, device=torch.device("cpu"), tcfg=tcfg)
    trainer.fit(loader, loader, modality="ecg", output_dir=tmp_path, best_filename="best_ecg.pt")
    ckpt_path = tmp_path / "best_ecg.pt"
    assert ckpt_path.exists()
    import torch as _torch

    ckpt = _torch.load(ckpt_path, weights_only=False)
    assert "model_state" in ckpt
    assert "cfg" in ckpt


def test_train_epoch_updates_parameters():
    model = ENGRAMForClassification(_tiny_cfg())
    before = {n: p.clone() for n, p in model.named_parameters()}
    loader = _tiny_loader()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    train_epoch(model, loader, opt, torch.device("cpu"), modality="ecg")
    changed = any(not torch.allclose(before[n], p) for n, p in model.named_parameters())
    assert changed, "Parameters should change after a training epoch"
