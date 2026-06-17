from __future__ import annotations

import math

import torch
import torch.nn as nn


class TransformerSequenceClassifier(nn.Module):
    """Small causal Transformer classifier for ``[B, T, D]`` patch sequences."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        pe = torch.zeros(1, 4096, d_model)
        position = torch.arange(0, 4096, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None, modality: str | None = None
    ) -> dict:
        b, t, _ = x.shape
        h = self.proj(x) + self.pe[:, :t, :]
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        h = self.norm(h.mean(dim=1))
        logits = self.head(h)
        out: dict = {"logits": logits}
        if labels is not None:
            if labels.dim() == 2:
                # multi-hot targets: macro-AUROC protocol for PTB-XL
                out["loss"] = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            else:
                out["loss"] = nn.functional.cross_entropy(logits, labels)
        return out
