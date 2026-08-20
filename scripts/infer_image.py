"""ENGRAM — Image inference and evaluation.

Usage:
    python scripts/infer_image.py --checkpoint output/best_image.pt --image path/to/image.png
    python scripts/infer_image.py --checkpoint output/best_image.pt --cifar-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from engram.data.image import patchify
from engram.inference import load_model

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _infer_patch_size(cfg) -> int:
    """Return the patch size from the image modality config, defaulting to 4."""
    for m in cfg.modalities:
        if m.name == "image" and m.patch_size is not None:
            return m.patch_size
    return 4


@torch.no_grad()
def infer_single(
    model, image_path: str, device: torch.device, patch_size: int | None = None
) -> dict[str, float]:
    """Run inference on a single image."""
    from PIL import Image
    from torchvision import transforms

    if patch_size is None:
        patch_size = _infer_patch_size(model.cfg)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    tf = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)  # [1, 3, 32, 32]
    x = patchify(x, patch_size)  # [1, 64, 3*patch_size^2]

    out = model(x, modality="image")
    probs = F.softmax(out["logits"], dim=-1)[0]  # [num_classes]
    top5 = probs.topk(min(5, probs.size(0)))

    result = {
        CIFAR10_CLASSES[idx.item()]: round(prob.item(), 6)
        for prob, idx in zip(top5.values, top5.indices)
    }
    print(f"\nImage: {image_path}")
    print("-" * 40)
    for cls, prob in result.items():
        print(f"  {cls:12s}  {prob * 100:6.2f}%")
    return result


@torch.no_grad()
def eval_cifar(
    model, data_root: str, device: torch.device, batch_size: int = 128
) -> dict[str, float]:
    """Evaluate the model on the CIFAR-10 test set."""
    from engram.data.image import PatchCollator
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    patch_size = _infer_patch_size(model.cfg)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    ds = datasets.CIFAR10(data_root, train=False, download=True, transform=tf)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=PatchCollator(patch_size=patch_size),
    )

    correct, total = 0, 0
    per_class_correct = [0] * len(CIFAR10_CLASSES)
    per_class_total = [0] * len(CIFAR10_CLASSES)

    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        out = model(x, modality="image")
        pred = out["logits"].argmax(dim=-1)

        correct += (pred == labels).sum().item()
        total += labels.size(0)

        for c in range(len(CIFAR10_CLASSES)):
            mask = labels == c
            per_class_correct[c] += (pred[mask] == labels[mask]).sum().item()
            per_class_total[c] += mask.sum().item()

    acc = correct / total if total > 0 else 0.0
    result = {"accuracy": acc}

    print(f"\nCIFAR-10 Test Accuracy: {acc * 100:.2f}%  ({correct}/{total})")
    print("-" * 40)
    for c in range(len(CIFAR10_CLASSES)):
        cls_acc = per_class_correct[c] / per_class_total[c] * 100 if per_class_total[c] > 0 else 0
        print(
            f"  {CIFAR10_CLASSES[c]:12s}  {cls_acc:6.2f}%  ({per_class_correct[c]}/{per_class_total[c]})"
        )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ENGRAM image inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--cifar-test", action="store_true", help="Evaluate on CIFAR-10 test set")
    parser.add_argument("--data-root", type=str, default="./datasets/cifar")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=None, help="Override patch size")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=str, default=None, help="Write results to this JSON file")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model, _cfg, _ckpt = load_model(args.checkpoint, device)

    result = {}
    if args.image:
        result = infer_single(model, args.image, device, args.patch_size)
    elif args.cifar_test:
        result = eval_cifar(model, args.data_root, device, args.batch_size)
    else:
        parser.print_help()
        sys.exit(1)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
