"""PRISM — ECG Inference Script

Kullanım:
    python scripts/infer_ecg.py --checkpoint output/best_ecg.pt --ptbxl-test
    python scripts/infer_ecg.py --checkpoint output/best_ecg.pt --signal path/to/ecg.npy
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from prism.config import ModalityConfig, PRISMConfig
from prism.model import PRISMForClassification

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_FULL = {
    "NORM": "Normal",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T-wave Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy",
}


def load_model(checkpoint_path: str, device: torch.device) -> PRISMForClassification:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg_data = ckpt.get("cfg")
    if isinstance(cfg_data, dict):
        from copy import deepcopy

        d = deepcopy(cfg_data)
        modalities = [ModalityConfig(**m) for m in d.pop("modalities", [])]
        cfg = PRISMConfig(**d, modalities=modalities)
    else:
        cfg = cfg_data

    model = PRISMForClassification(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_acc = ckpt.get("metrics", {}).get("val_acc", ckpt.get("val_acc", "?"))
    if isinstance(val_acc, float):
        print(f"Loaded checkpoint — epoch: {epoch}, val_acc: {val_acc:.4f}")
    else:
        print(f"Loaded checkpoint — epoch: {epoch}")

    return model


@torch.no_grad()
def infer_signal(model: PRISMForClassification, signal_path: str, device: torch.device):
    """Tek bir ECG sinyali üzerinde inference.

    signal: numpy array [T, 12] veya [12, T]
    """
    sig = np.load(signal_path).astype(np.float32)
    if sig.ndim == 2 and sig.shape[0] == 12:
        sig = sig.T  # [12, T] → [T, 12]

    # normalize
    mean = sig.mean(axis=0, keepdims=True)
    std = sig.std(axis=0, keepdims=True) + 1e-8
    sig = (sig - mean) / std

    # window
    window = 250
    if sig.shape[0] >= window:
        sig = sig[:window]
    else:
        pad = np.zeros((window - sig.shape[0], 12), dtype=np.float32)
        sig = np.concatenate([sig, pad], axis=0)

    x = torch.from_numpy(sig).unsqueeze(0).to(device)  # [1, T, 12]
    out = model(x, modality="ecg")
    probs = F.softmax(out["logits"], dim=-1)[0]

    print(f"\nECG Signal: {signal_path}")
    print("-" * 45)
    for i, (cls, prob) in enumerate(zip(CLASSES, probs)):
        bar = "█" * int(prob.item() * 30)
        full = CLASS_FULL[cls]
        print(f"  {cls:6s} {full:28s} {prob.item() * 100:5.1f}%  {bar}")


@torch.no_grad()
def eval_ptbxl(
    model: PRISMForClassification,
    data_root: str,
    device: torch.device,
    batch_size: int = 128,
    window_size: int = 250,
):
    """PTB-XL test seti üzerinde tam değerlendirme."""
    from prism.data.ecg import get_ecg_loaders

    print("Loading PTB-XL test set...")
    _, _, test_loader = get_ecg_loaders(
        root=data_root,
        batch_size=batch_size,
        window_size=window_size,
        num_workers=4,
    )

    correct, total = 0, 0
    per_class_correct = [0] * 5
    per_class_total = [0] * 5
    total_loss = 0.0

    for x, labels in test_loader:
        x, labels = x.to(device), labels.to(device)
        out = model(x, modality="ecg", labels=labels)
        pred = out["logits"].argmax(dim=-1)

        B = x.size(0)
        correct += (pred == labels).sum().item()
        total += B
        total_loss += out["loss"].item() * B

        for c in range(5):
            mask = labels == c
            per_class_correct[c] += (pred[mask] == labels[mask]).sum().item()
            per_class_total[c] += mask.sum().item()

    print(f"\nPTB-XL Test Accuracy: {correct / total * 100:.2f}%  ({correct}/{total})")
    print(f"Test Loss: {total_loss / total:.4f}")
    print("-" * 50)
    for c in range(5):
        acc = per_class_correct[c] / per_class_total[c] * 100 if per_class_total[c] > 0 else 0
        full = CLASS_FULL[CLASSES[c]]
        print(
            f"  {CLASSES[c]:6s} {full:28s} {acc:6.2f}%  ({per_class_correct[c]}/{per_class_total[c]})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ptbxl-test", action="store_true")
    parser.add_argument("--signal", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="./datasets/ptbxl")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=250)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    if args.signal:
        infer_signal(model, args.signal, device)
    elif args.ptbxl_test:
        eval_ptbxl(model, args.data_root, device, args.batch_size, args.window_size)
    else:
        print("Kullanım: --signal <path.npy> veya --ptbxl-test")


if __name__ == "__main__":
    main()
