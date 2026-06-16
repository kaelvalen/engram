"""Train lightweight baselines (1D ResNet or Transformer) on ECG or image patches."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from prism.baselines import ResNet1DClassifier, TransformerSequenceClassifier
from prism.data.paths import resolve_ptbxl_root
from prism.training.loops import accuracy, evaluate_macro_auc, evaluate_multilabel_auc
from prism.training.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def _loaders(args: argparse.Namespace):
    if args.task == "ecg":
        from prism.data.ecg import get_ecg_loaders

        root = resolve_ptbxl_root(args.data_root)
        # Any task other than the legacy single-label super-diagnostic is
        # inherently multi-label; --ecg-multilabel can also force it.
        ml = args.ecg_multilabel or args.ecg_task != "superdiag"
        train_loader, val_loader, _ = get_ecg_loaders(
            root=root,
            batch_size=args.batch_size,
            window_size=args.window_size,
            num_workers=args.num_workers,
            multilabel=ml,
            task=args.ecg_task,
        )
        return train_loader, val_loader, 12, train_loader.dataset.num_classes, ml
    from prism.data.image import get_cifar_loaders

    patch_size = args.patch_size
    train_loader, val_loader = get_cifar_loaders(
        root=os.path.join(args.data_root, "cifar"),
        batch_size=args.batch_size,
        patch_size=patch_size,
        num_workers=args.num_workers,
    )
    dim = patch_size * patch_size * 3
    return train_loader, val_loader, dim, 10, False


def train_epoch_baseline(model, loader, optimizer, device):
    model.train(True)
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(x, labels=labels)
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        b = x.size(0)
        total_loss += out["loss"].item() * b
        total_acc += accuracy(out["logits"], labels) * b
        n += b
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate_epoch_baseline(model, loader, device):
    model.train(False)
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        out = model(x, labels=labels)
        b = x.size(0)
        total_loss += out["loss"].item() * b
        total_acc += accuracy(out["logits"], labels) * b
        n += b
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate_baseline_auc(model, loader, device, multilabel: bool):
    if multilabel:
        return evaluate_multilabel_auc(model, loader, device, modality="ecg")
    return evaluate_macro_auc(
        model, loader, device, modality="ecg", num_classes=loader.dataset.num_classes
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet1d", "transformer"], default="resnet1d")
    parser.add_argument("--task", choices=["ecg", "image"], default="image")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--data-root",
        type=str,
        default="./datasets",
        help="Dataset root (e.g. ./datasets with cifar/ or ptbxl/ inside)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="ECG window length (default 1000 = full 10 s record at 100 Hz).",
    )
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", type=str, default="./output/baselines")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--ecg-task",
        type=str,
        default="superdiag",
        choices=["superdiag", "subdiag", "diag", "form", "rhythm", "all"],
        help="PTB-XL task (used when --task ecg).",
    )
    parser.add_argument(
        "--ecg-multilabel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="PTB-XL multi-label superclass targets + BCE loss + macro AUROC (paper protocol).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader, in_dim, n_cls, multilabel = _loaders(args)

    if args.model == "resnet1d":
        if args.task != "ecg":
            print("resnet1d is intended for ECG [B,T,C]. Use --task ecg.", file=sys.stderr)
            sys.exit(1)
        model = ResNet1DClassifier(in_channels=in_dim, num_classes=n_cls).to(device)
    else:
        model = TransformerSequenceClassifier(
            input_dim=in_dim, num_classes=n_cls, d_model=256, num_layers=4
        ).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    print(
        f"Baseline {args.model} | task={args.task} | "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    best_metric = float("-inf")
    select_metric = "val_macro_auc" if args.task == "ecg" else "val_acc"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch_baseline(model, train_loader, opt, device)
        va_loss, va_acc = evaluate_epoch_baseline(model, val_loader, device)
        sched.step()
        dt = time.time() - t0

        metrics: dict[str, float] = {
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
        }
        auc_msg = ""
        if args.task == "ecg":
            va_auc = evaluate_baseline_auc(model, val_loader, device, multilabel)
            metrics["val_macro_auc"] = va_auc
            auc_msg = f" | val macro-AUC: {va_auc:.4f}"
            if va_auc > best_metric:
                best_metric = va_auc
        else:
            if va_acc > best_metric:
                best_metric = va_acc

        print(
            f"[{epoch:03d}/{args.epochs}] train {tr_loss:.4f}/{tr_acc:.4f} | "
            f"val {va_loss:.4f}/{va_acc:.4f}{auc_msg} | {dt:.1f}s"
        )

        if metrics[select_metric] >= best_metric:
            path = os.path.join(args.output_dir, f"best_{args.model}_{args.task}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                },
                path,
            )
    print(f"Done. Best {select_metric}: {best_metric:.4f}")


if __name__ == "__main__":
    main()
