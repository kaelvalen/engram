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

from engram.config import ENGRAMConfig, ModalityConfig
from engram.data.paths import resolve_ptbxl_root
from engram.logging import setup_logging
from engram.model import ENGRAMForClassification
from engram.training.checkpoint import save_checkpoint
from engram.training.loops import _autocast, cycle_loader, evaluate_epoch
from engram.training.trainer import Trainer, TrainerConfig, _resolve_amp
from engram.training.utils import set_seed

logger = logging.getLogger(__name__)


def _cfg_kwargs(args: argparse.Namespace) -> dict:
    """Shared ENGRAMConfig kwargs derived from CLI args (architecture knobs).

    `--layer-pattern` (explicit comma/space list) takes precedence over the
    legacy `--block-pattern` force/interleave; when given it also overrides
    num_layers to match.
    """
    kwargs: dict = dict(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        delta_every=args.delta_every,
        dropout=args.dropout,
        ssm_kind=args.ssm_kind,
        s4d_init=args.s4d_init,
        delta_backend=args.delta_backend,
        scan_backend=args.scan_backend,
        swa_window=args.swa_window,
        compile=args.compile,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    layer_pattern = getattr(args, "layer_pattern", None)
    if layer_pattern:
        tokens = [t for t in layer_pattern.replace(",", " ").split() if t]
        kwargs["block_pattern"] = tokens
        kwargs["num_layers"] = len(tokens)
    else:
        kwargs["force_block_type"] = None if args.block_pattern == "hybrid" else args.block_pattern
    return kwargs


def _build_loaders_single(
    args: argparse.Namespace,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, str, ENGRAMConfig]:
    modality = args.modality
    if modality == "image":
        from engram.data.image import get_cifar_loaders

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
        from engram.data.ecg import get_ecg_loaders

        ecg_root = resolve_ptbxl_root(args.data_root)
        # Any task other than legacy super-diagnostic is inherently multi-label.
        ml = args.ecg_multilabel or args.ecg_task != "superdiag"
        train_loader, val_loader, _test = get_ecg_loaders(
            root=ecg_root,
            batch_size=args.batch_size,
            window_size=args.window_size,
            num_workers=args.num_workers,
            multilabel=ml,
            task=args.ecg_task,
            seed=args.seed,
        )
        # num_classes is task-dependent (5 for super-diag, more for diag/subdiag/…);
        # read it from the loaded vocabulary rather than hardcoding.
        num_classes = train_loader.dataset.num_classes
        modalities = [
            ModalityConfig(
                name="ecg",
                input_dim=12,
                num_classes=num_classes,
                window_size=args.window_size,
                multilabel=ml,
            )
        ]
    else:
        from engram.data.audio import get_audio_loaders

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

    cfg = ENGRAMConfig(modalities=modalities, **_cfg_kwargs(args))
    return train_loader, val_loader, modality, cfg


def _build_loaders_joint(
    args: argparse.Namespace,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    ENGRAMConfig,
]:
    from engram.data.ecg import get_ecg_loaders
    from engram.data.image import get_cifar_loaders

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
    cfg = ENGRAMConfig(modalities=modalities, **_cfg_kwargs(args))
    return ecg_train, ecg_val, img_train, img_val, cfg


def _yaml_defaults(yaml_path: str | None) -> dict[str, object]:
    """Return parser defaults loaded from YAML.

    YAML values are applied as argparse defaults before the final parse, so
    explicit CLI flags still override the config file. This matches the
    documented precedence: built-in defaults < YAML < CLI.
    """
    if not yaml_path:
        return {}

    from engram.training.yaml_config import load_yaml_config

    data = load_yaml_config(yaml_path)
    defaults: dict[str, object] = {}

    train_section = data.get("train", data)
    for key, val in train_section.items():
        if key in {"modalities", "model"}:
            continue
        defaults[key.replace("-", "_")] = val

    if "model" in data:
        for key, val in data["model"].items():
            defaults[key.replace("-", "_")] = val

    return defaults


def _apply_yaml_defaults(parser: argparse.ArgumentParser, yaml_path: str | None) -> None:
    defaults = {
        key: val
        for key, val in _yaml_defaults(yaml_path).items()
        if any(action.dest == key for action in parser._actions)
    }
    if defaults:
        parser.set_defaults(**defaults)


def run_joint_training(args: argparse.Namespace, device: torch.device) -> None:
    ecg_tr, ecg_va, img_tr, img_va, cfg = _build_loaders_joint(args)
    model = ENGRAMForClassification(cfg).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    amp_dtype = _resolve_amp(args.amp)

    tcfg = TrainerConfig(
        tensorboard_dir=os.path.join(args.output_dir, "tb_joint") if args.tensorboard else None,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name or "engram-joint",
        amp=args.amp,
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
            with _autocast(device, amp_dtype):
                out_e = model(xe, modality="ecg", labels=ye)
                out_i = model(xi, modality="image", labels=yi)
                loss = out_e["loss"] + out_i["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total_loss += loss.item()
            n_tokens += 1

        sched.step()
        vl_e, acc_e = evaluate_epoch(model, ecg_va, device, "ecg", amp_dtype=amp_dtype)
        vl_i, acc_i = evaluate_epoch(model, img_va, device, "image", amp_dtype=amp_dtype)
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
    parser = argparse.ArgumentParser(description="ENGRAM training")
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
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument(
        "--block-pattern",
        type=str,
        default="hybrid",
        choices=["hybrid", "s4", "delta"],
        help="hybrid = default interleave; s4/delta = ablation (all one type)",
    )
    parser.add_argument(
        "--layer-pattern",
        type=str,
        default=None,
        help="Explicit per-layer pattern, e.g. 's4,s4,s4,delta' or 's4 s4 s4 swa'. "
        "Tokens: s4|delta|swa. Overrides --block-pattern and sets num_layers.",
    )
    parser.add_argument(
        "--ssm-kind",
        type=str,
        default="ssd",
        choices=["ssd", "s4d_legacy"],
        help="SSM impl for 's4' slots: ssd (Mamba-2 selective) or s4d_legacy.",
    )
    parser.add_argument(
        "--s4d-init",
        type=str,
        default="lin",
        choices=["lin", "legacy"],
        help="S4D-legacy A init (only when --ssm-kind s4d_legacy).",
    )
    parser.add_argument(
        "--delta-backend",
        type=str,
        default="reference",
        choices=["reference", "fla"],
        help="Gated-delta backend: reference (pure PyTorch) or fla (Triton, GPU).",
    )
    parser.add_argument(
        "--scan-backend",
        type=str,
        default="auto",
        choices=["auto", "assoc", "reference"],
        help="Linear-recurrence scan backend for SSD/S4D.",
    )
    parser.add_argument("--swa-window", type=int, default=128, help="Sliding-window attn span.")
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the model in the trainer."
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Recompute layer activations during backward to save VRAM (slower epochs).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./datasets",
        help="Root for downloaded data: datasets/cifar, datasets/ptbxl, … (not engram/data/)",
    )
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic CUDA mode (slower, but more reproducible)",
    )
    parser.add_argument(
        "--patch-size", type=int, default=4, help="CIFAR patch side (image / joint)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="ECG timesteps. Default 1000 = full 10 s record at 100 Hz, matching the "
        "xresnet1d101 baseline; shorter windows discard signal and bias the comparison.",
    )
    parser.add_argument(
        "--ecg-multilabel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="PTB-XL multi-label superclass targets + BCE loss + macro AUROC (paper protocol).",
    )
    parser.add_argument(
        "--ecg-task",
        type=str,
        default="superdiag",
        choices=["superdiag", "subdiag", "diag", "form", "rhythm", "all"],
        help="PTB-XL task group. Non-superdiag tasks are always multi-label.",
    )
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
    parser.add_argument(
        "--early-stopping", type=int, default=0, help="patience on val metric; 0=off"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from (e.g. output/last.pt)",
    )
    parser.add_argument(
        "--amp",
        type=str,
        default="off",
        choices=["off", "bf16"],
        help="Mixed precision (bf16 autocast). Recommended on Ampere+ GPUs.",
    )
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    _apply_yaml_defaults(parser, config_args.config)

    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "joint":
        run_joint_training(args, device)
        return

    train_loader, val_loader, modality, cfg = _build_loaders_single(args)
    model = ENGRAMForClassification(cfg).to(device)

    # PTB-XL primary metric is macro one-vs-rest AUROC for all tasks (including the
    # legacy single-label super-diagnostic ablation). Non-ECG modalities use accuracy.
    use_auc = modality == "ecg"

    tcfg = TrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.grad_clip,
        early_stopping_patience=args.early_stopping or None,
        tensorboard_dir=os.path.join(args.output_dir, "tb", modality) if args.tensorboard else None,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name or f"engram-{modality}",
        amp=args.amp,
        select_metric="val_macro_auc" if use_auc else "val_acc",
        extra_wandb_config={
            "seed": args.seed,
            "batch_size": args.batch_size,
            "modality": modality,
            "ecg_task": args.ecg_task if modality == "ecg" else None,
        },
    )
    trainer = Trainer(model, cfg, device=device, tcfg=tcfg)

    def on_epoch(epoch: int, m: dict[str, float]) -> None:
        auc_msg = ""
        if use_auc:
            # Any task other than the legacy single-label super-diagnostic is
            # inherently multi-label; --ecg-multilabel can also force it.
            ml = args.ecg_multilabel or args.ecg_task != "superdiag"
            if ml:
                from engram.training.loops import evaluate_multilabel_auc

                auc = evaluate_multilabel_auc(
                    model,
                    val_loader,
                    device,
                    modality,
                    amp_dtype=_resolve_amp(args.amp),
                )
            else:
                from engram.training.loops import evaluate_macro_auc

                num_classes = train_loader.dataset.num_classes
                auc = evaluate_macro_auc(
                    model,
                    val_loader,
                    device,
                    modality,
                    num_classes,
                    amp_dtype=_resolve_amp(args.amp),
                )
            m["val_macro_auc"] = auc
            auc_msg = f" | val macro-AUC: {auc:.4f}"
        logger.info(
            "[%03d/%d] train loss: %.4f acc: %.4f | val loss: %.4f acc: %.4f%s",
            epoch,
            args.epochs,
            m["train_loss"],
            m["train_acc"],
            m["val_loss"],
            m["val_acc"],
            auc_msg,
        )

    logger.info(
        "ENGRAM — %s | params: %s | device: %s | block_pattern=%s",
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
        resume_from=args.resume,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
