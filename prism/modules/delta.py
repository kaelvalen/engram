from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import ShortCausalConv1d
from .ffn import SwiGLU
from .norm import RMSNorm, l2_normalize


@dataclass
class DeltaState:
    S: torch.Tensor  # [B, H, Dh, Dh] float32

    def detach(self) -> DeltaState:
        return DeltaState(S=self.S.detach())


class GatedDeltaRule(nn.Module):
    """Gated Delta Rule with matrix-valued associative memory.

    Per token, per head:
        S_t = α_t · [S_{t-1} - β_t · (S_{t-1} k_t) k_t^T] + β_t · v_t k_t^T
        o_t = S_t · q_t
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        chunk_size: int = 64,
        gate_bias_init: float = 4.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.qk_norm    = qk_norm
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.alpha_proj    = nn.Linear(hidden_dim, num_heads, bias=True)
        self.beta_proj     = nn.Linear(hidden_dim, num_heads, bias=True)
        self.out_gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj      = nn.Linear(hidden_dim, hidden_dim, bias=False)

        nn.init.constant_(self.alpha_proj.bias, gate_bias_init)
        nn.init.zeros_(self.beta_proj.bias)

    def empty_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> DeltaState:
        return DeltaState(S=torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim,
            device=device, dtype=torch.float32,
        ))

    def _project(self, x):
        B, T, _ = x.shape
        H, Dh   = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        if self.qk_norm:
            q = l2_normalize(q, dim=-1)
            k = l2_normalize(k, dim=-1)

        alpha = torch.sigmoid(self.alpha_proj(x)).transpose(1, 2)  # [B, H, T]
        beta  = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)   # [B, H, T]
        gate  = F.silu(self.out_gate_proj(x))                       # [B, T, hidden]

        return q, k, v, alpha, beta, gate

    @staticmethod
    def _recurrent(q, k, v, alpha, beta, S0):
        """Per-token recurrent reference. float32 internally."""
        B, H, T, Dh = q.shape
        S    = S0
        outs = []

        qf, kf, vf = q.float(), k.float(), v.float()
        af, bf     = alpha.float(), beta.float()

        for i in range(T):
            kt = kf[:, :, i]                              # [B, H, Dh]
            vt = vf[:, :, i]
            qt = qf[:, :, i]
            at = af[:, :, i].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            bt = bf[:, :, i].unsqueeze(-1).unsqueeze(-1)

            Sk      = torch.einsum("bhij,bhj->bhi", S, kt).unsqueeze(-1)  # [B,H,Dh,1]
            outer_k = kt.unsqueeze(-2)                                     # [B,H,1,Dh]
            outer_v = vt.unsqueeze(-1)                                     # [B,H,Dh,1]

            S  = at * (S - bt * (Sk @ outer_k)) + bt * (outer_v @ outer_k)
            ot = torch.einsum("bhij,bhj->bhi", S, qt)
            outs.append(ot)

        out = torch.stack(outs, dim=2)  # [B, H, T, Dh]
        return out.to(q.dtype), S

    def forward(
        self,
        x: torch.Tensor,
        state: DeltaState | None = None,
    ) -> tuple[torch.Tensor, DeltaState]:
        B, T, _ = x.shape
        q, k, v, alpha, beta, gate = self._project(x)

        S0 = state.S.float() if state is not None else \
             torch.zeros(B, self.num_heads, self.head_dim, self.head_dim,
                         device=x.device, dtype=torch.float32)

        if T == 1:
            o, S_new = self._recurrent(q, k, v, alpha, beta, S0)
        else:
            # chunked
            outs = []
            S = S0
            for start in range(0, T, self.chunk_size):
                end = min(start + self.chunk_size, T)
                chunk_o, S = self._recurrent(
                    q[:, :, start:end], k[:, :, start:end], v[:, :, start:end],
                    alpha[:, :, start:end], beta[:, :, start:end], S
                )
                outs.append(chunk_o)
            o     = torch.cat(outs, dim=2)
            S_new = S

        o = o.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
        o = o * gate
        o = self.out_proj(o)
        return o, DeltaState(S=S_new)


class DeltaBlock(nn.Module):
    """GatedDeltaRule + ShortCausalConv + SwiGLU + RMSNorm residual block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        chunk_size: int = 64,
        gate_bias_init: float = 4.0,
        conv_kernel_size: int = 4,
        ffn_expand: int = 2,
    ):
        super().__init__()
        self.norm1  = RMSNorm(hidden_dim)
        self.conv   = ShortCausalConv1d(hidden_dim, conv_kernel_size)
        self.delta  = GatedDeltaRule(hidden_dim, num_heads, qk_norm,
                                     chunk_size, gate_bias_init)
        self.norm2  = RMSNorm(hidden_dim)
        self.ffn    = SwiGLU(hidden_dim, ffn_expand)

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        delta_state: DeltaState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, DeltaState]:
        r   = x
        x_n = self.norm1(x)
        x_c, new_conv_state   = self.conv(x_n, conv_state)
        x_d, new_delta_state  = self.delta(x_c, delta_state)
        x   = r + x_d
        x   = x + self.ffn(self.norm2(x))
        return x, new_conv_state, new_delta_state
