from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from prism.config import PRISMConfig
from prism.model import PRISMForClassification

from .checkpoint import cfg_to_dict, load_checkpoint, save_checkpoint
from .loops import evaluate_epoch, train_epoch
from .utils import get_rng_state, set_rng_state

logger = logging.getLogger(__name__)

_AMP_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _resolve_amp(amp: str | None) -> torch.dtype | None:
    if amp in (None, "", "off", "none"):
        return None
    if amp not in _AMP_DTYPES:
        raise ValueError(
            f"Unsupported amp dtype: {amp!r}. Supported: 'bf16' (or 'off' to disable). "
            "fp16 not supported here — would need a GradScaler. Use bf16 on Ampere+ GPUs."
        )
    return _AMP_DTYPES[amp]


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
    amp: str | None = None  # None | "bf16"
    # Metric key used to pick the best checkpoint. "val_acc" (default) or e.g.
    # "val_macro_auc" for multi-label PTB-XL. The metric must be present in the
    # per-epoch metrics dict (an epoch_callback may add it before selection).
    select_metric: str = "val_acc"
    # Extra key/value pairs to merge into the wandb config (e.g. training HPs).
    extra_wandb_config: dict[str, Any] | None = None


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
        self._amp_dtype = _resolve_amp(self.tcfg.amp)
        # Optional torch.compile of the whole model (config-gated). Falls back
        # silently if the backend is unavailable (e.g. no Triton / CPU-only).
        if getattr(self.cfg, "compile", False):
            try:
                self.model = torch.compile(self.model)
            except Exception as exc:
                logger.warning(
                    "torch.compile failed (%s: %s); continuing without compilation.",
                    type(exc).__name__,
                    exc,
                )
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
            config = cfg_to_dict(cfg)
            if self.tcfg.extra_wandb_config:
                config.update(self.tcfg.extra_wandb_config)
            self._wandb = wandb.init(
                project=self.tcfg.wandb_project,
                name=self.tcfg.wandb_run_name,
                config=config,
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
        resume_from: str | Path | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        opt = AdamW(self.model.parameters(), lr=self.tcfg.lr, weight_decay=self.tcfg.weight_decay)
        sched = CosineAnnealingLR(opt, T_max=self.tcfg.epochs)

        start_epoch = 1
        global_step = 0
        best_val = float("-inf")
        patience_left = self.tcfg.early_stopping_patience
        history: list[dict[str, float]] = []

        if resume_from is not None:
            ckpt = load_checkpoint(resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                opt.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                sched.load_state_dict(ckpt["scheduler_state"])
            if "rng_state" in ckpt:
                set_rng_state(ckpt["rng_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("global_step", 0)
            best_val = ckpt.get("metrics", {}).get(self.tcfg.select_metric, float("-inf"))
            logger.info(
                "Resumed from %s at epoch %d (best %s=%.4f)",
                resume_from,
                start_epoch - 1,
                self.tcfg.select_metric,
                best_val if best_val != float("-inf") else 0.0,
            )

        for epoch in range(start_epoch, self.tcfg.epochs + 1):

            def step_log(tag: str, val: float, step: int) -> None:
                nonlocal global_step
                global_step += 1
                self._log_scalar(tag, val, global_step)

            train_loss, train_acc = train_epoch(
                self.model,
                train_loader,
                opt,
                self.device,
                modality,
                max_grad_norm=self.tcfg.max_grad_norm,
                loss_log_fn=step_log if self._writer or self._wandb else None,
                amp_dtype=self._amp_dtype,
            )
            val_loss, val_acc = evaluate_epoch(
                self.model, val_loader, self.device, modality, amp_dtype=self._amp_dtype
            )
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

            # Selection metric (callback may have added e.g. val_macro_auc above).
            current = metrics.get(self.tcfg.select_metric, val_acc)
            improved = current > best_val + self.tcfg.early_stopping_min_delta
            if improved:
                best_val = current
                save_checkpoint(
                    output_dir / best_filename,
                    epoch=epoch,
                    model_state=self.model.state_dict(),
                    cfg=self.cfg,
                    metrics=dict(metrics),
                    optimizer_state=opt.state_dict(),
                    scheduler_state=sched.state_dict(),
                    rng_state=get_rng_state(),
                    global_step=global_step,
                )
                if patience_left is not None:
                    patience_left = self.tcfg.early_stopping_patience

            # Always save a latest checkpoint for resumption / crash recovery.
            save_checkpoint(
                output_dir / "last.pt",
                epoch=epoch,
                model_state=self.model.state_dict(),
                cfg=self.cfg,
                metrics=dict(metrics),
                optimizer_state=opt.state_dict(),
                scheduler_state=sched.state_dict(),
                rng_state=get_rng_state(),
                global_step=global_step,
            )

            if (
                self.tcfg.early_stopping_patience is not None
                and patience_left is not None
                and not improved
            ):
                patience_left -= 1
                if patience_left <= 0:
                    # Restore best checkpoint before returning.
                    best_path = output_dir / best_filename
                    if best_path.exists():
                        logger.info("Early stopping; restoring best checkpoint from %s", best_path)
                        ckpt = load_checkpoint(best_path, map_location=self.device)
                        self.model.load_state_dict(ckpt["model_state"])
                    break

        if self._writer is not None:
            self._writer.close()
        if self._wandb is not None:
            self._wandb.finish()

        # Keep ``best_val_acc`` for backward compatibility with existing tests and
        # callers, while ``best_val`` is the generic name when select_metric is not
        # accuracy (e.g. macro AUROC).
        return {"best_val": best_val, "best_val_acc": best_val, "history": history}
