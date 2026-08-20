"""ENGRAM — ECG inference and evaluation.

Usage:
    python scripts/infer_ecg.py --checkpoint output/best_ecg.pt --ptbxl-test
    python scripts/infer_ecg.py --checkpoint output/best_ecg.pt --signal path/to/ecg.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from engram.data.ecg import _fit_window
from engram.data.paths import resolve_ptbxl_root
from engram.inference import load_model
from engram.training.loops import evaluate_macro_auc, evaluate_multilabel_auc

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_FULL = {
    "NORM": "Normal",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T-wave Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy",
}


def _normalize(signal: np.ndarray) -> np.ndarray:
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True) + 1e-8
    return (signal - mean) / std


@torch.no_grad()
def infer_signal(
    model,
    signal_path: str,
    device: torch.device,
    window_size: int | None = None,
) -> dict[str, float]:
    """Run inference on a single ECG signal.

    The signal may be [T, 12] or [12, T]. It is normalized and fitted to
    ``window_size`` (default: full signal length, or the checkpoint's
    modality window_size if available).
    """
    sig = np.load(signal_path).astype(np.float32)
    if sig.ndim == 2 and sig.shape[0] == 12:
        sig = sig.T  # [12, T] -> [T, 12]

    sig = _normalize(sig)

    if window_size is None:
        window_size = sig.shape[0]
    sig = _fit_window(sig, window_size)

    x = torch.from_numpy(sig).unsqueeze(0).to(device)  # [1, T, 12]
    out = model(x, modality="ecg")
    probs = F.softmax(out["logits"], dim=-1)[0]

    result = {cls: round(prob.item(), 6) for cls, prob in zip(CLASSES, probs)}
    print(f"\nECG Signal: {signal_path}")
    print("-" * 45)
    for cls, prob in result.items():
        bar = "█" * int(prob * 30)
        full = CLASS_FULL[cls]
        print(f"  {cls:6s} {full:28s} {prob * 100:5.1f}%  {bar}")
    return result


@torch.no_grad()
def eval_ptbxl(
    model,
    data_root: str,
    device: torch.device,
    batch_size: int = 128,
    window_size: int | None = None,
    task: str = "superdiag",
) -> dict[str, float]:
    """Evaluate the model on the PTB-XL test set.

    Uses macro one-vs-rest AUROC for all tasks (matching the paper protocol).
    Multi-label tasks report AUROC; the legacy single-label super-diagnostic
    task also reports accuracy for backward compatibility.
    """
    from engram.data.ecg import get_ecg_loaders

    root = resolve_ptbxl_root(data_root)
    multilabel = task != "superdiag"
    if window_size is None:
        # Prefer the window_size stored in the checkpoint config.
        window_size = getattr(model.cfg.modalities[0], "window_size", 1000)

    print("Loading PTB-XL test set...")
    _, _, test_loader = get_ecg_loaders(
        root=root,
        batch_size=batch_size,
        window_size=window_size,
        num_workers=4,
        multilabel=multilabel,
        task=task,
    )
    num_classes = test_loader.dataset.num_classes

    if multilabel:
        auc = evaluate_multilabel_auc(model, test_loader, device, "ecg")
        print(f"\nPTB-XL Test macro-AUROC ({task}): {auc:.4f}")
        return {"macro_auc": auc}

    auc = evaluate_macro_auc(model, test_loader, device, "ecg", num_classes)

    correct, total = 0, 0
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    total_loss = 0.0

    for x, labels in test_loader:
        x, labels = x.to(device), labels.to(device)
        out = model(x, modality="ecg", labels=labels)
        pred = out["logits"].argmax(dim=-1)

        B = x.size(0)
        correct += (pred == labels).sum().item()
        total += B
        total_loss += out["loss"].item() * B

        for c in range(num_classes):
            mask = labels == c
            per_class_correct[c] += (pred[mask] == labels[mask]).sum().item()
            per_class_total[c] += mask.sum().item()

    acc = correct / total if total > 0 else 0.0
    result = {
        "accuracy": acc,
        "macro_auc": auc,
        "loss": total_loss / total if total > 0 else 0.0,
    }

    print(f"\nPTB-XL Test Accuracy: {acc * 100:.2f}%  ({correct}/{total})")
    print(f"PTB-XL Test macro-AUROC: {auc:.4f}")
    print(f"Test Loss: {result['loss']:.4f}")
    print("-" * 50)
    for c in range(num_classes):
        cls_acc = per_class_correct[c] / per_class_total[c] * 100 if per_class_total[c] > 0 else 0
        full = CLASS_FULL[CLASSES[c]] if c < len(CLASSES) else f"class_{c}"
        print(
            f"  {CLASSES[c] if c < len(CLASSES) else f'cls{c}':6s} "
            f"{full:28s} {cls_acc:6.2f}%  ({per_class_correct[c]}/{per_class_total[c]})"
        )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ENGRAM ECG inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ptbxl-test", action="store_true", help="Evaluate on PTB-XL test set")
    parser.add_argument("--signal", type=str, default=None, help="Path to a single .npy signal")
    parser.add_argument("--data-root", type=str, default="./datasets")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument(
        "--ecg-task",
        type=str,
        default="superdiag",
        choices=["superdiag", "subdiag", "diag", "form", "rhythm", "all"],
        help="PTB-XL task group for evaluation",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=str, default=None, help="Write results to this JSON file")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model, cfg, _ckpt = load_model(args.checkpoint, device)

    result = {}
    if args.signal:
        result = infer_signal(model, args.signal, device, args.window_size)
    elif args.ptbxl_test:
        result = eval_ptbxl(
            model, args.data_root, device, args.batch_size, args.window_size, args.ecg_task
        )
    else:
        parser.print_help()
        sys.exit(1)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
