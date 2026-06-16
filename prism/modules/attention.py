from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ffn import SwiGLU
from .norm import RMSNorm


def _build_rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    """Standard rotary position embedding cos/sin caches: [T, head_dim]."""
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    ang = torch.outer(t, freqs)  # [T, half]
    emb = torch.cat([ang, ang], dim=-1)  # [T, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, Dh]; cos/sin: [T, Dh]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class SlidingWindowAttention(nn.Module):
    """Causal multi-head attention restricted to a sliding window of `window`
    past tokens (incl. self). With RoPE. This is the attention component used
    in Gated DeltaNet-H1 / H2 style hybrids, included here for ablation.
    """

    def __init__(self, hidden_dim: int, num_heads: int, window: int = 128):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window = window

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Cache RoPE and the sliding-window mask per (T, device) to avoid
        # rebuilding them on every forward pass.
        self._rope_cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self._mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _get_rope_cache(self, T: int, head_dim: int, device: torch.device):
        key = (T, device)
        if key not in self._rope_cache:
            self._rope_cache[key] = _build_rope_cache(T, head_dim, device)
        return self._rope_cache[key]

    def _get_window_mask(self, T: int, device: torch.device) -> torch.Tensor:
        key = (T, device)
        if key not in self._mask_cache:
            i = torch.arange(T, device=device).unsqueeze(1)
            j = torch.arange(T, device=device).unsqueeze(0)
            allowed = (j <= i) & (j > i - self.window)
            mask = torch.zeros(T, T, device=device)
            mask = mask.masked_fill(~allowed, float("-inf"))
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor, state=None):
        # KV-cache streaming is not implemented; reject a non-None state to avoid
        # silent correctness bugs in streaming use cases.
        if state is not None:
            raise NotImplementedError(
                "SlidingWindowAttention does not yet support a KV-cache state. "
                "Pass state=None for full-sequence classification."
            )
        B, T, _ = x.shape
        H, Dh = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).split(self.hidden_dim, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        cos, sin = self._get_rope_cache(T, Dh, x.device)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if T == 1:
            attn_mask = None  # single query attends to itself only
        else:
            attn_mask = self._get_window_mask(T, x.device).to(q.dtype)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
        return self.out_proj(o), state


class SWABlock(nn.Module):
    """SlidingWindowAttention + SwiGLU + RMSNorm residual block.

    No short conv: attention layers in hybrid SSMs use RoPE for position
    instead of a causal conv.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window: int = 128,
        ffn_expand: int = 2,
        dropout: float = 0.0,
        **_unused,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.attn = SlidingWindowAttention(hidden_dim, num_heads, window)
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, ffn_expand)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.ffn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, conv_state=None, mixer_state=None):
        r = x
        x_a, new_mixer = self.attn(self.norm1(x), mixer_state)
        x = r + self.dropout(x_a)
        x = x + self.ffn_dropout(self.ffn(self.norm2(x)))
        # conv_state passes through unchanged (attention has no conv).
        return x, conv_state, new_mixer
