from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import ShortCausalConv1d
from .ffn import SwiGLU
from .norm import RMSNorm


def parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Parallel prefix scan: h_t = a_t * h_{t-1} + b_t

    a: [B, H, T, N] complex  — per-step decay
    b: [B, H, T, N] complex  — per-step input
    returns h: [B, H, T, N] complex

    Uses work-efficient parallel scan (Blelloch 1990).
    All ops are batched tensor ops — no Python loop over T.
    """
    B, H, T, N = a.shape

    # pad to next power of 2
    T_pad = 1
    while T_pad < T:
        T_pad *= 2

    if T_pad > T:
        pad = T_pad - T
        a = torch.cat([a, torch.ones (B, H, pad, N, dtype=a.dtype, device=a.device)], dim=2)
        b = torch.cat([b, torch.zeros(B, H, pad, N, dtype=b.dtype, device=b.device)], dim=2)

    levels = int(math.log2(T_pad))

    # upsweep
    for d in range(levels):
        step      = 2 ** (d + 1)
        idx_r     = torch.arange(step - 1, T_pad, step, device=a.device)
        idx_l     = idx_r - 2 ** d
        a_l       = a[:, :, idx_l]
        b_l       = b[:, :, idx_l]
        a[:, :, idx_r] = a[:, :, idx_r] * a_l
        b[:, :, idx_r] = a[:, :, idx_r] * b_l + b[:, :, idx_r]

    # downsweep
    a = a.clone(); b = b.clone()
    a[:, :, -1] = torch.ones (B, H, N, dtype=a.dtype, device=a.device)
    b[:, :, -1] = torch.zeros(B, H, N, dtype=b.dtype, device=b.device)

    for d in range(levels - 1, -1, -1):
        step      = 2 ** (d + 1)
        idx_r     = torch.arange(step - 1, T_pad, step, device=a.device)
        idx_l     = idx_r - 2 ** d
        a_l_old   = a[:, :, idx_l].clone()
        b_l_old   = b[:, :, idx_l].clone()
        
        a[:, :, idx_l] = a[:, :, idx_r]
        b[:, :, idx_l] = b[:, :, idx_r]
        
        b[:, :, idx_r] = a[:, :, idx_r] * b_l_old + b[:, :, idx_r]
        a[:, :, idx_r] = a[:, :, idx_r] * a_l_old

    # shift right by 1 to get inclusive scan, trim padding
    h = torch.roll(b, 1, dims=2)
    h[:, :, 0] = 0
    # add b back: h_t = a_t * h_{t-1} + b_t
    # after exclusive scan h holds h_{t-1}, now compute h_t
    h = a * h + b  # this is b after upsweep which already holds cumulative product

    return h[:, :, :T]


class S4SSM(nn.Module):
    """Diagonal S4D-Complex SSM with input-dependent step size.

    Per head, per token:
        Δ_t  = softplus(Linear(x_t) + δ_bias)
        Ā    = exp(Δ_t · A)
        B̄    = expm1(Δ_t · A) / A · B
        h_t  = Ā ⊙ h_{t-1} + B̄ · u_t
        y_t  = 2 · Re(C* ⊙ h_t) + D · u_t

    Prefill uses parallel scan (fully parallel over T).
    Decode (T=1) uses single-step recurrence.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        state_mult: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.state_dim  = self.head_dim * state_mult

        H, Dh, N = num_heads, self.head_dim, self.state_dim

        self.in_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # A: diagonal complex, negative real part → stable
        A_log  = torch.linspace(math.log(1), math.log(state_mult * Dh), N).unsqueeze(0).repeat(H, 1)
        A_imag = math.pi * torch.arange(1, N + 1).unsqueeze(0).repeat(H, 1)
        self.A_log  = nn.Parameter(A_log)
        self.A_imag = nn.Parameter(A_imag)

        self.B_re = nn.Parameter(torch.randn(H, N) * 0.01)
        self.B_im = nn.Parameter(torch.randn(H, N) * 0.01)
        self.C_re = nn.Parameter(torch.randn(H, N) * 0.01)
        self.C_im = nn.Parameter(torch.randn(H, N) * 0.01)

        self.D = nn.Parameter(torch.ones(H, Dh))

        dt_init      = torch.exp(
            torch.rand(H, Dh) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        self.dt_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        inv_sp       = torch.log(torch.expm1(dt_init.reshape(-1)))
        self.dt_proj.bias.data.copy_(inv_sp)
        nn.init.zeros_(self.dt_proj.weight)

        self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _get_A(self) -> torch.Tensor:
        return torch.complex(-self.A_log.exp(), self.A_imag)  # [H, N]

    def _get_BC(self):
        return (
            torch.complex(self.B_re, self.B_im),  # [H, N]
            torch.complex(self.C_re, self.C_im),
        )

    def empty_state(self, batch_size, device, dtype):
        return torch.zeros(
            batch_size, self.num_heads, self.state_dim,
            dtype=torch.complex64, device=device,
        )

    def _step(self, x_t, h, A, Bc, Cc):
        """Single decode step. x_t: [B, hidden_dim]"""
        B = x_t.shape[0]
        H, Dh, N = self.num_heads, self.head_dim, self.state_dim

        u  = self.in_proj(x_t).view(B, H, Dh)
        # Note: We take mean over Dh to reduce to per-head scope.
        # This is a deliberate simplification for the multi-head SSM,
        # unlike Mamba which uses per-channel (D-specific) selective step sizes.
        dt = F.softplus(self.dt_proj(x_t)).view(B, H, Dh).mean(dim=-1)  # [B, H]

        dA    = dt.unsqueeze(-1).to(torch.complex64) * A.unsqueeze(0)    # [B, H, N]
        A_bar = torch.exp(dA)
        B_bar = torch.expm1(dA) / A.unsqueeze(0) * Bc.unsqueeze(0)

        u_c = u.mean(dim=-1).to(torch.complex64).unsqueeze(-1)           # [B, H, 1]
        h   = A_bar * h + B_bar * u_c                                    # [B, H, N]

        y = 2.0 * (Cc.unsqueeze(0).conj() * h).real.sum(dim=-1)         # [B, H]
        y = y.unsqueeze(-1) + self.D.unsqueeze(0) * u                   # [B, H, Dh]
        y = y.reshape(B, self.hidden_dim)
        return y, h

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        H, Dh, N = self.num_heads, self.head_dim, self.state_dim

        A  = self._get_A()
        Bc, Cc = self._get_BC()

        h0 = state.to(torch.complex64) if state is not None else \
             self.empty_state(B, x.device, x.dtype)

        # decode path: single step, no scan overhead
        if T == 1:
            y, h_new = self._step(x[:, 0], h0, A, Bc, Cc)
            y = y.unsqueeze(1)
            gate = F.silu(self.gate_proj(x))
            return self.out_proj(y * gate), h_new

        # Note: Mean over Dh is a deliberate simplification for the multi-head SSM,
        # reducing per-channel selectivity to per-head selectivity.
        # prefill path: parallel scan over T
        u  = self.in_proj(x).view(B, T, H, Dh).permute(0, 2, 1, 3)     # [B, H, T, Dh]
        dt = F.softplus(self.dt_proj(x)).view(B, T, H, Dh).permute(0, 2, 1, 3)
        dt_s = dt.mean(dim=-1, keepdim=True).to(torch.complex64)         # [B, H, T, 1]

        A_b  = A.unsqueeze(0).unsqueeze(2)    # [1, H, 1, N]
        Bc_b = Bc.unsqueeze(0).unsqueeze(2)
        Cc_b = Cc.unsqueeze(0).unsqueeze(2)

        dA    = dt_s * A_b                                                # [B, H, T, N]
        A_bar = torch.exp(dA)
        B_bar = torch.expm1(dA) / A_b * Bc_b

        u_c   = u.mean(dim=-1, keepdim=True).to(torch.complex64)         # [B, H, T, 1]
        b_seq = B_bar * u_c                                               # [B, H, T, N]

        # fold initial state into first timestep
        if state is not None:
            b_seq = b_seq.clone()
            b_seq[:, :, 0] = b_seq[:, :, 0] + A_bar[:, :, 0] * h0

        h = parallel_scan(A_bar.clone(), b_seq.clone())                  # [B, H, T, N]
        h_new = h[:, :, -1, :]                                           # [B, H, N]

        # output
        y = 2.0 * (Cc_b.conj() * h).real.sum(dim=-1)                    # [B, H, T]
        y = y.unsqueeze(-1) + self.D.unsqueeze(0).unsqueeze(2) * u      # [B, H, T, Dh]
        y = y.permute(0, 2, 1, 3).reshape(B, T, self.hidden_dim)

        gate = F.silu(self.gate_proj(x))
        y    = self.out_proj(y * gate)

        return y, h_new


class S4Block(nn.Module):
    """S4SSM + ShortCausalConv + SwiGLU + RMSNorm residual block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        state_mult: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        conv_kernel_size: int = 4,
        ffn_expand: int = 2,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.conv  = ShortCausalConv1d(hidden_dim, conv_kernel_size)
        self.ssm   = S4SSM(hidden_dim, num_heads, state_mult, dt_min, dt_max)
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn   = SwiGLU(hidden_dim, ffn_expand)

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        ssm_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r   = x
        x_n = self.norm1(x)
        x_c, new_conv_state = self.conv(x_n, conv_state)
        x_s, new_ssm_state  = self.ssm(x_c, ssm_state)
        x   = r + x_s
        x   = x + self.ffn(self.norm2(x))
        return x, new_conv_state, new_ssm_state
