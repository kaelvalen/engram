"""PRISM — Image Inference Script

Kullanım:
    python infer_image.py --checkpoint output/best_image.pt --image path/to/image.png
    python infer_image.py --checkpoint output/best_image.pt --cifar-test   # CIFAR-10 test seti üzerinde toplu değerlendirme
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from prism.config import ModalityConfig, PRISMConfig
from prism.data.image import patchify
from prism.model import PRISMForClassification
from torchvision import transforms

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


def load_model(checkpoint_path: str, device: torch.device) -> PRISMForClassification:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg_data = ckpt.get("cfg")
    if isinstance(cfg_data, dict):
        from copy import deepcopy

        d = deepcopy(cfg_data)
        modalities = [ModalityConfig(**m) for m in d.pop("modalities", [])]
        cfg = PRISMConfig(**d, modalities=modalities)
    elif cfg_data is not None:
        cfg = cfg_data
    else:
        cfg = PRISMConfig(
            hidden_dim=256,
            num_heads=8,
            num_layers=12,
            modalities=[ModalityConfig(name="image", input_dim=48, num_classes=10)],
        )

    model = PRISMForClassification(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_acc = ckpt.get("val_acc", "?")
    print(
        f"Loaded checkpoint — epoch: {epoch}, val_acc: {val_acc:.4f}"
        if isinstance(val_acc, float)
        else f"Loaded checkpoint — epoch: {epoch}"
    )
    return model


@torch.no_grad()
def infer_single(
    model: PRISMForClassification, image_path: str, device: torch.device, patch_size: int = 4
):
    """Tek bir görüntü üzerinde inference."""
    from PIL import Image

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
    x = patchify(x, patch_size)  # [1, 64, 48]

    out = model(x, modality="image")
    probs = F.softmax(out["logits"], dim=-1)[0]  # [10]
    top5 = probs.topk(5)

    print(f"\nImage: {image_path}")
    print("-" * 40)
    for prob, idx in zip(top5.values, top5.indices):
        print(f"  {CIFAR10_CLASSES[idx.item()]:12s}  {prob.item() * 100:6.2f}%")


@torch.no_grad()
def eval_cifar(
    model: PRISMForClassification, data_root: str, device: torch.device, batch_size: int = 128
):
    """CIFAR-10 test seti üzerinde toplu değerlendirme."""
    from prism.data.image import PatchCollator
    from torch.utils.data import DataLoader
    from torchvision import datasets

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
        collate_fn=PatchCollator(patch_size=4),
    )

    correct, total = 0, 0
    per_class_correct = [0] * 10
    per_class_total = [0] * 10

    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        out = model(x, modality="image")
        pred = out["logits"].argmax(dim=-1)

        correct += (pred == labels).sum().item()
        total += labels.size(0)

        for c in range(10):
            mask = labels == c
            per_class_correct[c] += (pred[mask] == labels[mask]).sum().item()
            per_class_total[c] += mask.sum().item()

    print(f"\nCIFAR-10 Test Accuracy: {correct / total * 100:.2f}%  ({correct}/{total})")
    print("-" * 40)
    for c in range(10):
        acc = per_class_correct[c] / per_class_total[c] * 100 if per_class_total[c] > 0 else 0
        print(
            f"  {CIFAR10_CLASSES[c]:12s}  {acc:6.2f}%  ({per_class_correct[c]}/{per_class_total[c]})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None, help="Tek görüntü inference")
    parser.add_argument(
        "--cifar-test", action="store_true", help="CIFAR-10 test seti değerlendirme"
    )
    parser.add_argument("--data-root", type=str, default="./datasets/cifar")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    if args.image:
        infer_single(model, args.image, device)
    elif args.cifar_test:
        eval_cifar(model, args.data_root, device, args.batch_size)
    else:
        print("Kullanım: --image <path> veya --cifar-test")


if __name__ == "__main__":
    main()
