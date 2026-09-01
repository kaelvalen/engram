from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ffn import SwiGLU
from .norm import RMSNorm


@functools.lru_cache(maxsize=8)
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


@functools.lru_cache(maxsize=8)
def _build_window_mask(T: int, S: int, window: int, device: torch.device) -> torch.Tensor:
    """Additive (T, S) causal sliding-window mask for a chunk of T queries whose
    keys are right-aligned with the chunk: the S keys are the chunk's own T keys
    preceded by S - T cached predecessors. Query i attends key s iff
    s <= i + (S - T) (causal) and i + (S - T) - s < window (sliding window).
    With S == T this is exactly the full-sequence causal window mask.
    """
    i = torch.arange(T, device=device).unsqueeze(1) + (S - T)
    j = torch.arange(S, device=device).unsqueeze(0)
    allowed = (j <= i) & (j > i - window)
    mask = torch.zeros(T, S, device=device)
    return mask.masked_fill(~allowed, float("-inf"))


@dataclass
class SWAState:
    """KV-cache state for streaming sliding-window attention.

    Holds the RoPE-applied keys and values of the last `window` tokens seen so
    far — covering absolute positions [pos - W, pos), where W <= window — plus
    `pos`, the number of tokens processed (absolute position of the next one).
    Feeding the state returned by one chunk into the next makes chunked (or
    token-by-token) decoding exactly equal a single full-sequence forward.
    Call `detach()` between chunks to cut the autograd graph.

    In the SGMS masked execution path (§3.4) the cache instead holds the last
    `window` *routed* keys/values and `pos` is a per-batch tensor counting
    routed tokens — the expert's own subsequence time axis.
    """

    k: torch.Tensor  # [B, H, W, Dh] RoPE-applied keys of the last <= window tokens
    v: torch.Tensor  # [B, H, W, Dh]
    pos: int | torch.Tensor  # tokens seen so far (int; per-batch tensor when masked)

    def detach(self) -> SWAState:
        pos = self.pos.detach() if isinstance(self.pos, torch.Tensor) else self.pos
        return SWAState(k=self.k.detach(), v=self.v.detach(), pos=pos)


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

    def _get_rope_cache(self, T: int, head_dim: int, device: torch.device):
        return _build_rope_cache(T, head_dim, device)

    def empty_state(self, batch_size: int, device, dtype) -> SWAState:
        """Zero-length cache at position 0 (stream start)."""
        H, Dh = self.num_heads, self.head_dim
        return SWAState(
            k=torch.zeros(batch_size, H, 0, Dh, device=device, dtype=dtype),
            v=torch.zeros(batch_size, H, 0, Dh, device=device, dtype=dtype),
            pos=0,
        )

    def forward(self, x: torch.Tensor, state: SWAState | None = None, write_mask=None):
        """Prefill/decode with optional KV-cache state (streaming).

        With ``state=None`` this is the original full-sequence path.  With a
        state, the chunk's queries attend to the cached window of previous
        RoPE-applied keys/values plus their own causal window — exactly equal
        to a single full-sequence forward (fp64-tested).

        ``write_mask`` (SGMS §3.4) switches to the masked execution path:
        the window slides over the routed subsequence only.
        """
        if write_mask is not None:
            return self._forward_masked(x, state, write_mask)
        B, T, _ = x.shape
        H, Dh = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).split(self.hidden_dim, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        pos0 = state.pos if state is not None else 0
        cos, sin = self._get_rope_cache(pos0 + T, Dh, x.device)
        cos = cos[pos0 : pos0 + T].to(q.dtype)
        sin = sin[pos0 : pos0 + T].to(q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if state is not None:
            k_all = torch.cat([state.k, k], dim=2)
            v_all = torch.cat([state.v, v], dim=2)
        else:
            k_all, v_all = k, v
        # Queries need at most window-1 predecessors; drop anything older.
        k_all = k_all[:, :, -(self.window + T - 1) :]
        v_all = v_all[:, :, -(self.window + T - 1) :]

        attn_mask = _build_window_mask(T, k_all.shape[2], self.window, x.device).to(q.dtype)
        o = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=attn_mask)
        o = o.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)

        new_state = SWAState(
            k=k_all[:, :, -self.window :].contiguous(),
            v=v_all[:, :, -self.window :].contiguous(),
            pos=pos0 + T,
        )
        return self.out_proj(o), new_state

    def _forward_masked(self, x: torch.Tensor, state: SWAState | None, write_mask: torch.Tensor):
        """SGMS §3.4 masked execution: the window slides over the ROUTED
        subsequence only. Non-routed tokens are never written to the KV cache
        and are masked out as keys; their query outputs are computed but
        meaningless (the caller gathers outputs at routed positions only).
        RoPE positions run along the routed subsequence — the expert's own
        time axis — so chunked streaming with state hand-off stays exactly
        equal to a single full-sequence masked forward (fp64-tested).
        """
        B, T, _ = x.shape
        H, Dh, W = self.num_heads, self.head_dim, self.window
        q, k, v = self.qkv(x).split(self.hidden_dim, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        m = write_mask.to(x.dtype)  # [B,T] in {0,1}
        pos_prev = (
            state.pos.to(x.device)
            if state is not None
            else torch.zeros(B, dtype=torch.long, device=x.device)
        )
        # Position along the expert's own stream: routed tokens take their
        # rank; unrouted tokens inherit the most recent routed position (or 0
        # before the first routed token).
        q_pos = (pos_prev.unsqueeze(1) + m.long().cumsum(dim=1) - 1).clamp(min=0)  # [B,T]

        cos_full, sin_full = self._get_rope_cache(int(q_pos.max()) + 1, Dh, x.device)
        cos = cos_full[q_pos].to(q.dtype).unsqueeze(1)  # [B,1,T,Dh]
        sin = sin_full[q_pos].to(q.dtype).unsqueeze(1)
        # per-batch RoPE (subsequence positions); _apply_rope expects a shared
        # [T, Dh] cache, so apply the rotation directly here.
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        if state is not None and state.k.shape[2] > 0:
            Wc = state.k.shape[2]
            valid_len = pos_prev.clamp(max=Wc)  # [B]
            slot = torch.arange(Wc, device=x.device)
            cache_valid = slot.unsqueeze(0) >= (Wc - valid_len.unsqueeze(1))  # [B,Wc]
            # right-aligned cache: slot s (>= Wc - valid_len) holds the routed
            # key at subsequence position pos_prev - (Wc - s)
            cache_pos = pos_prev.unsqueeze(1) - Wc + slot.unsqueeze(0)
            k_all = torch.cat([state.k.to(k.dtype), k], dim=2)
            v_all = torch.cat([state.v.to(v.dtype), v], dim=2)
        else:
            cache_valid = torch.zeros(B, 0, dtype=torch.bool, device=x.device)
            cache_pos = torch.zeros(B, 0, dtype=torch.long, device=x.device)
            k_all, v_all = k, v

        key_pos = torch.cat([cache_pos, q_pos], dim=1)  # [B,S]
        key_valid = torch.cat([cache_valid, m.bool()], dim=1)  # [B,S]

        S = key_pos.shape[1]
        # Explicit causality in k_all index space: a query may only attend
        # keys at or before its own position (subsequence-position comparison
        # alone would let unrouted queries see the *next* routed token, which
        # shares their q_pos).
        causal = torch.arange(S, device=x.device).unsqueeze(0) <= (
            S - T + torch.arange(T, device=x.device)
        ).unsqueeze(1)  # [T,S]
        allow = (
            key_valid.unsqueeze(1)
            & causal.unsqueeze(0)
            & (key_pos.unsqueeze(1) <= q_pos.unsqueeze(2))
            & ((q_pos.unsqueeze(2) - key_pos.unsqueeze(1)) < W)
        )  # [B,T,S]
        attn_mask = torch.zeros(B, T, S, device=x.device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~allow, float("-inf"))
        o = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=attn_mask.unsqueeze(1))
        o = o.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)

        # New cache: last W routed keys per batch (contiguous tail per stream).
        new_k = torch.zeros(B, H, W, Dh, device=x.device, dtype=k_all.dtype)
        new_v = torch.zeros(B, H, W, Dh, device=x.device, dtype=v_all.dtype)
        new_pos = pos_prev + m.long().sum(dim=1)
        for b in range(B):
            cand = key_valid[b].nonzero().flatten()
            if cand.numel() > 0:
                tail = cand[-W:]
                new_k[b, :, -tail.numel() :] = k_all[b, :, tail]
                new_v[b, :, -tail.numel() :] = v_all[b, :, tail]
        return self.out_proj(o), SWAState(k=new_k, v=new_v, pos=new_pos)


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
