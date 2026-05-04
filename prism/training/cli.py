from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from prism.config import ModalityConfig, PRISMConfig
from prism.data.paths import resolve_ptbxl_root
from prism.logging import setup_logging
from prism.model import PRISMForClassification
from prism.training.checkpoint import save_checkpoint
from prism.training.loops import cycle_loader, evaluate_epoch
from prism.training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


def _build_loaders_single(
    args: argparse.Namespace,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, str, PRISMConfig]:
    modality = args.modality
    if modality == "image":
        from prism.data.image import get_cifar_loaders

        patch_size = args.patch_size
        input_dim = patch_size * patch_size * 3
        train_loader, val_loader = get_cifar_loaders(
            root=os.path.join(args.data_root, "cifar"),
            batch_size=args.batch_size,
            patch_size=patch_size,
            num_workers=args.num_workers,
        )
        modalities = [
            ModalityConfig(
                name="image",
                input_dim=input_dim,
                num_classes=10,
                patch_size=patch_size,
            )
        ]
    elif modality == "ecg":
        from prism.data.ecg import get_ecg_loaders

        ecg_root = resolve_ptbxl_root(args.data_root)
        train_loader, val_loader, _test = get_ecg_loaders(
            root=ecg_root,
            batch_size=args.batch_size,
            window_size=args.window_size,
            num_workers=args.num_workers,
        )
        modalities = [
            ModalityConfig(
                name="ecg",
                input_dim=12,
                num_classes=5,
                window_size=args.window_size,
            )
        ]
    else:
        from prism.data.audio import get_audio_loaders

        train_loader, val_loader = get_audio_loaders(
            root=os.path.join(args.data_root, "audio"),
            batch_size=args.batch_size,
            num_mel_bins=args.mel_bins,
            patch_frames=args.patch_frames,
            num_workers=args.num_workers,
            synthetic=args.audio_synthetic,
        )
        modalities = [
            ModalityConfig(
                name="audio",
                input_dim=args.mel_bins,
                num_classes=args.audio_num_classes,
            )
        ]

    force_block = None if args.block_pattern == "hybrid" else args.block_pattern
    cfg = PRISMConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        delta_every=args.delta_every,
        modalities=modalities,
        force_block_type=force_block,
    )
    return train_loader, val_loader, modality, cfg


def _build_loaders_joint(
    args: argparse.Namespace,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    PRISMConfig,
]:
    from prism.data.ecg import get_ecg_loaders
    from prism.data.image import get_cifar_loaders

    patch_size = args.patch_size
    input_dim = patch_size * patch_size * 3
    img_train, img_val = get_cifar_loaders(
        root=os.path.join(args.data_root, "cifar"),
        batch_size=args.batch_size,
        patch_size=patch_size,
        num_workers=args.num_workers,
    )
    ecg_root = resolve_ptbxl_root(args.data_root)
    ecg_train, ecg_val, _ = get_ecg_loaders(
        root=ecg_root,
        batch_size=args.batch_size,
        window_size=args.window_size,
        num_workers=args.num_workers,
    )
    modalities = [
        ModalityConfig(name="ecg", input_dim=12, num_classes=5, window_size=args.window_size),
        ModalityConfig(name="image", input_dim=input_dim, num_classes=10, patch_size=patch_size),
    ]
    force_block = None if args.block_pattern == "hybrid" else args.block_pattern
    cfg = PRISMConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        delta_every=args.delta_every,
        modalities=modalities,
        force_block_type=force_block,
    )
    return ecg_train, ecg_val, img_train, img_val, cfg


def _apply_yaml_defaults(args: argparse.Namespace, yaml_path: str | None) -> None:
    if not yaml_path:
        return
    from prism.training.yaml_config import load_yaml_config

    data = load_yaml_config(yaml_path)
    train_section = data.get("train", data)
    for key, val in train_section.items():
        if key == "modalities" or key == "model":
            continue
        attr = key.replace("-", "_")
        if hasattr(args, attr):
            setattr(args, attr, val)
    if "model" in data:
        m = data["model"]
        for key, val in m.items():
            attr = key.replace("-", "_")
            if hasattr(args, attr):
                setattr(args, attr, val)


def run_joint_training(args: argparse.Namespace, device: torch.device) -> None:
    ecg_tr, ecg_va, img_tr, img_va, cfg = _build_loaders_joint(args)
    model = PRISMForClassification(cfg).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tcfg = TrainerConfig(
        tensorboard_dir=os.path.join(args.output_dir, "tb_joint") if args.tensorboard else None,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name or "prism-joint",
    )
    trainer_helper = Trainer(model, cfg, device=device, tcfg=tcfg)
    writer = trainer_helper._writer
    wandb_run = trainer_helper._wandb

    def log_scalars(epoch: int, metrics: dict[str, float]) -> None:
        if writer:
            for k, v in metrics.items():
                writer.add_scalar(f"joint/{k}", v, epoch)
        if wandb_run:
            wandb_run.log(metrics, step=epoch)

    best_mean = 0.0
    it_e = cycle_loader(ecg_tr)
    it_i = cycle_loader(img_tr)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train(True)
        steps = max(len(ecg_tr), len(img_tr))
        total_loss = 0.0
        n_tokens = 0

        for _ in range(steps):
            xe, ye = next(it_e)
            xi, yi = next(it_i)
            xe, ye = xe.to(device), ye.to(device)
            xi, yi = xi.to(device), yi.to(device)
            opt.zero_grad()
            out_e = model(xe, modality="ecg", labels=ye)
            out_i = model(xi, modality="image", labels=yi)
            loss = out_e["loss"] + out_i["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total_loss += loss.item()
            n_tokens += 1

        sched.step()
        vl_e, acc_e = evaluate_epoch(model, ecg_va, device, "ecg")
        vl_i, acc_i = evaluate_epoch(model, img_va, device, "image")
        mean_acc = (acc_e + acc_i) / 2.0
        metrics = {
            "train_loss": total_loss / max(n_tokens, 1),
            "val_loss_ecg": vl_e,
            "val_loss_image": vl_i,
            "val_acc_ecg": acc_e,
            "val_acc_image": acc_i,
            "val_acc_mean": mean_acc,
        }
        log_scalars(epoch, metrics)
        dt = time.time() - t0
        logger.info(
            "[%03d/%d] joint train_loss=%.4f | val acc ecg=%.4f img=%.4f mean=%.4f | %.1fs",
            epoch,
            args.epochs,
            metrics["train_loss"],
            acc_e,
            acc_i,
            mean_acc,
            dt,
        )

        if mean_acc > best_mean:
            best_mean = mean_acc
            save_checkpoint(
                out / "best_joint.pt",
                epoch=epoch,
                model_state=model.state_dict(),
                cfg=cfg,
                metrics=metrics,
            )
            logger.info("  saved best_joint.pt (mean val acc=%.4f)", mean_acc)

    if writer:
        writer.close()
    if wandb_run:
        wandb_run.finish()
    logger.info("Joint training done. Best mean val acc: %.4f", best_mean)


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="PRISM training")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML file with train/model keys")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "joint"],
        help="single modality or joint ecg+image",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        choices=["image", "ecg", "audio"],
        help="Used when mode=single",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--delta-every", type=int, default=4)
    parser.add_argument(
        "--block-pattern",
        type=str,
        default="hybrid",
        choices=["hybrid", "s4", "delta"],
        help="hybrid = default interleave; s4/delta = ablation (all one type)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./datasets",
        help="Root for downloaded data: datasets/cifar, datasets/ptbxl, … (not prism/data/)",
    )
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--patch-size", type=int, default=4, help="CIFAR patch side (image / joint)"
    )
    parser.add_argument("--window-size", type=int, default=250, help="ECG timesteps (ecg / joint)")
    parser.add_argument("--mel-bins", type=int, default=64, help="audio: mel frequency bins")
    parser.add_argument("--patch-frames", type=int, default=4, help="audio: frames per patch token")
    parser.add_argument("--audio-num-classes", type=int, default=10)
    parser.add_argument(
        "--audio-synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="audio: use synthetic dataset (no files)",
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="log TensorBoard under output-dir/tb"
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--early-stopping", type=int, default=0, help="patience on val acc; 0=off")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    _apply_yaml_defaults(args, args.config)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "joint":
        run_joint_training(args, device)
        return

    train_loader, val_loader, modality, cfg = _build_loaders_single(args)
    model = PRISMForClassification(cfg).to(device)

    tcfg = TrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.grad_clip,
        early_stopping_patience=args.early_stopping or None,
        tensorboard_dir=os.path.join(args.output_dir, "tb", modality) if args.tensorboard else None,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name or f"prism-{modality}",
    )
    trainer = Trainer(model, cfg, device=device, tcfg=tcfg)

    def on_epoch(epoch: int, m: dict[str, float]) -> None:
        logger.info(
            "[%03d/%d] train loss: %.4f acc: %.4f | val loss: %.4f acc: %.4f",
            epoch,
            args.epochs,
            m["train_loss"],
            m["train_acc"],
            m["val_loss"],
            m["val_acc"],
        )

    logger.info(
        "PRISM — %s | params: %s | device: %s | block_pattern=%s",
        modality.upper(),
        f"{sum(p.numel() for p in model.parameters()):,}",
        device,
        args.block_pattern,
    )

    trainer.fit(
        train_loader,
        val_loader,
        modality=modality,
        output_dir=args.output_dir,
        best_filename=f"best_{modality}.pt",
        epoch_callback=on_epoch,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
