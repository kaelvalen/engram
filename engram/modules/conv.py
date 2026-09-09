from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Optional causal_conv1d fast path (CUDA-only).
_CAUSAL_CONV1D_FN = None
_CAUSAL_CONV1D_CHECKED = False


def _load_causal_conv1d():
    global _CAUSAL_CONV1D_FN, _CAUSAL_CONV1D_CHECKED
    if not _CAUSAL_CONV1D_CHECKED:
        _CAUSAL_CONV1D_CHECKED = True
        try:
            from causal_conv1d import causal_conv1d_fn

            _CAUSAL_CONV1D_FN = causal_conv1d_fn
        except Exception:
            _CAUSAL_CONV1D_FN = None
    return _CAUSAL_CONV1D_FN


class ShortCausalConv1d(nn.Module):
    """Short causal 1D convolution.

    Each token only attends to itself and previous (kernel_size - 1) tokens.
    Carries conv_state for streaming decoding.
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        # groups=dim: depthwise conv with minimal parameters
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, T, dim]
        conv_state: torch.Tensor | None = None,  # [B, dim, kernel_size-1]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        xt = x.transpose(1, 2)  # [B, dim, T]

        if conv_state is None:
            fn = _load_causal_conv1d()
            if x.is_cuda and fn is not None:
                try:
                    w = self.conv.weight.squeeze(1)
                    b = self.conv.bias
                    out = fn(xt, w, b, activation=None)
                    new_state = xt[:, :, -(self.kernel_size - 1) :]
                    return out.transpose(1, 2), new_state
                except Exception:
                    pass
            # Prefill: zero-padding on the left
            pad = self.kernel_size - 1
            xt_padded = F.pad(xt, (pad, 0))
        else:
            # Decode: append previous state to the left
            xt_padded = torch.cat([conv_state, xt], dim=2)

        # Updated state: last (kernel_size - 1) tokens
        new_state = xt_padded[:, :, -(self.kernel_size - 1) :]

        out = self.conv(xt_padded)  # [B, dim, T]
        out = out.transpose(1, 2)  # [B, T, dim]
        return out, new_state

    def empty_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.dim, self.kernel_size - 1, device=device, dtype=dtype)
