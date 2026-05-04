from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from prism.config import PRISMConfig
from prism.model import PRISMForClassification

from .checkpoint import cfg_to_dict, save_checkpoint
from .loops import evaluate_epoch, train_epoch


@dataclass
class TrainerConfig:
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    early_stopping_patience: int | None = None  # None = disabled; else val metric
    early_stopping_min_delta: float = 0.0
    log_every_epoch: bool = True
    tensorboard_dir: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None


class Trainer:
    """Shared training loop: cosine LR, best checkpoint, optional TB / W&B, early stopping."""

    def __init__(
        self,
        model: PRISMForClassification,
        cfg: PRISMConfig,
        *,
        device: torch.device,
        tcfg: TrainerConfig | None = None,
    ):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.tcfg = tcfg or TrainerConfig()
        self._writer = None
        if self.tcfg.tensorboard_dir:
            from torch.utils.tensorboard import SummaryWriter

            Path(self.tcfg.tensorboard_dir).mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=self.tcfg.tensorboard_dir)

        self._wandb = None
        if self.tcfg.wandb_project:
            try:
                import wandb
            except ImportError as e:
                raise ImportError(
                    "wandb is not installed. pip install wandb or disable --wandb-project."
                ) from e
            self._wandb = wandb.init(
                project=self.tcfg.wandb_project,
                name=self.tcfg.wandb_run_name,
                config=cfg_to_dict(cfg),
            )

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)
        if self._wandb is not None:
            self._wandb.log({tag: value}, step=step)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        *,
        modality: str,
        output_dir: str | Path,
        best_filename: str = "best.pt",
        epoch_callback: Callable[[int, dict[str, float]], None] | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        opt = AdamW(self.model.parameters(), lr=self.tcfg.lr, weight_decay=self.tcfg.weight_decay)
        sched = CosineAnnealingLR(opt, T_max=self.tcfg.epochs)

        best_val = float("-inf")
        patience_left = self.tcfg.early_stopping_patience
        history: list[dict[str, float]] = []

        for epoch in range(1, self.tcfg.epochs + 1):

            def step_log(tag: str, val: float, step: int) -> None:
                self._log_scalar(tag, val, epoch * 10_000 + step)

            train_loss, train_acc = train_epoch(
                self.model,
                train_loader,
                opt,
                self.device,
                modality,
                max_grad_norm=self.tcfg.max_grad_norm,
                loss_log_fn=step_log if self._writer or self._wandb else None,
            )
            val_loss, val_acc = evaluate_epoch(self.model, val_loader, self.device, modality)
            sched.step()

            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            history.append(metrics)

            if self.tcfg.log_every_epoch:
                self._log_scalar("epoch/train_loss", train_loss, epoch)
                self._log_scalar("epoch/train_acc", train_acc, epoch)
                self._log_scalar("epoch/val_loss", val_loss, epoch)
                self._log_scalar("epoch/val_acc", val_acc, epoch)

            if epoch_callback:
                epoch_callback(epoch, metrics)

            improved = val_acc > best_val + self.tcfg.early_stopping_min_delta
            if improved:
                best_val = val_acc
                save_checkpoint(
                    output_dir / best_filename,
                    epoch=epoch,
                    model_state=self.model.state_dict(),
                    cfg=self.cfg,
                    metrics={"val_acc": val_acc, "val_loss": val_loss},
                )
                if patience_left is not None:
                    patience_left = self.tcfg.early_stopping_patience

            if (
                self.tcfg.early_stopping_patience is not None
                and patience_left is not None
                and not improved
            ):
                patience_left -= 1
                if patience_left <= 0:
                    break

        if self._writer is not None:
            self._writer.close()
        if self._wandb is not None:
            self._wandb.finish()

        return {"best_val_acc": best_val, "history": history}
