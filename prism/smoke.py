import torch

from prism.config import ModalityConfig, PRISMConfig
from prism.model import PRISMForClassification

cfg = PRISMConfig(
    hidden_dim=256,
    num_heads=8,
    num_layers=12,
    delta_every=4,
    modalities=[
        ModalityConfig(name="ecg",   input_dim=12,  num_classes=5),
        ModalityConfig(name="image", input_dim=48, num_classes=10),
    ]
)

model = PRISMForClassification(cfg)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

B, T = 2, 128

# ECG
ecg    = torch.randn(B, T, 12)
labels = torch.randint(0, 5, (B,))
out    = model(ecg, modality="ecg", labels=labels)
print(f"ECG   — logits: {out['logits'].shape}, loss: {out['loss'].item():.4f}")

# Image
img    = torch.randn(B, 64, 48)  # 64 patches × (4×4×3) CIFAR-style
labels = torch.randint(0, 10, (B,))
out    = model(img, modality="image", labels=labels)
print(f"Image — logits: {out['logits'].shape}, loss: {out['loss'].item():.4f}")
