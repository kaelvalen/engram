from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import ShortCausalConv1d
from .ffn import SwiGLU
from .norm import RMSNorm
from .scan import linear_recurrence, seq_recurrence


def _acc(x: torch.Tensor) -> torch.Tensor:
    """Promote bf16/fp16 to float32 for stable accumulation; keep float32/float64."""
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


class SSDMixer(nn.Module):
    """Mamba-2-style SSD (state-space duality) selective scan.

    Unlike the legacy S4D block this keeps **per-channel** state and does not
    average Δ or the input over the head dimension. Following Mamba-2:

        x_t : per-channel input          [B, T, H, P]
        Δ_t : per-head step size          [B, T, H]        (selective, > 0)
        B_t, C_t : per-head state vectors [B, T, H, N]      (selective)
        a_t = exp(Δ_t · A_h)             scalar decay per head, A_h < 0
        h_t = a_t · h_{t-1} + (Δ_t·x_t) ⊗ B_t      state [B, H, P, N]
        y_t = (h_t · C_t).sum(N) + D_h · x_t        output [B, T, H, P]

    The recurrence is solved with a parallel associative scan (prefill) or a
    single-step update (decode, T == 1). This is the primitive Mamba-3 builds on.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        state_dim: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        scan_backend: str = "auto",
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # P
        self.state_dim = state_dim  # N
        self.scan_backend = scan_backend

        H, P, N = num_heads, self.head_dim, state_dim

        self.in_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)  # → x values [.,H,P]
        self.B_proj = nn.Linear(hidden_dim, H * N, bias=False)  # selective B
        self.C_proj = nn.Linear(hidden_dim, H * N, bias=False)  # selective C

        # Δ: per-head, input-dependent. Bias initialised so softplus(bias) spans [dt_min, dt_max].
        self.dt_proj = nn.Linear(hidden_dim, H, bias=True)
        dt_init = torch.exp(
            torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_sp = torch.log(torch.expm1(dt_init))
        self.dt_proj.bias.data.copy_(inv_sp)
        nn.init.zeros_(self.dt_proj.weight)

        # A: scalar per head, negative & stable via A = -exp(A_log). Init in [1, 16].
        A_init = torch.empty(H).uniform_(1.0, 16.0)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(H))  # per-head skip

        self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _A(self) -> torch.Tensor:
        return -torch.exp(self.A_log)  # [H], negative

    def empty_state(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.state_dim,
            device=device, dtype=torch.float32,
        )

    def _project(self, x: torch.Tensor):
        B, T, _ = x.shape
        H, P, N = self.num_heads, self.head_dim, self.state_dim
        xv = self.in_proj(x).view(B, T, H, P)
        Bc = self.B_proj(x).view(B, T, H, N)
        Cc = self.C_proj(x).view(B, T, H, N)
        dt = F.softplus(self.dt_proj(x))  # [B, T, H] > 0
        a = torch.exp(dt * self._A())  # [B, T, H] decay in (0, 1)
        dBx = (dt.unsqueeze(-1) * xv).unsqueeze(-1) * Bc.unsqueeze(-2)  # [B,T,H,P,N]
        return xv, Cc, a, dBx

    def _step(self, x_t: torch.Tensor, h: torch.Tensor):
        """Decode step. x_t: [B, hidden_dim]; h: [B, H, P, N]."""
        B = x_t.shape[0]
        H, P, N = self.num_heads, self.head_dim, self.state_dim
        xv = self.in_proj(x_t).view(B, H, P)
        Bc = self.B_proj(x_t).view(B, H, N)
        Cc = self.C_proj(x_t).view(B, H, N)
        dt = F.softplus(self.dt_proj(x_t))  # [B, H]
        a = torch.exp(dt * self._A())  # [B, H]

        dBx = (dt.unsqueeze(-1) * xv).unsqueeze(-1) * Bc.unsqueeze(-2)  # [B,H,P,N]
        h = a.unsqueeze(-1).unsqueeze(-1) * h + dBx
        y = (h * Cc.unsqueeze(-2)).sum(-1)  # [B,H,P]
        y = y + self.D.view(1, H, 1) * xv
        return y.reshape(B, self.hidden_dim), h

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        B, T, _ = x.shape
        H, P, N = self.num_heads, self.head_dim, self.state_dim

        h0 = _acc(state) if state is not None else self.empty_state(B, x.device, x.dtype)

        if T == 1:
            y, h_new = self._step(x[:, 0], h0)
            y = y.unsqueeze(1)
            gate = F.silu(self.gate_proj(x))
            return self.out_proj(y * gate), h_new

        xv, Cc, a, dBx = self._project(x)  # dBx: [B,T,H,P,N], a: [B,T,H]

        # Arrange for the scan: flatten (B,H,P) as the batch, scan over T, N as channel.
        a_scan = a.permute(0, 2, 1).reshape(B * H, 1, T, 1).expand(B * H, P, T, N)
        a_scan = _acc(a_scan.reshape(B * H * P, T, N))
        b_scan = _acc(dBx.permute(0, 2, 3, 1, 4).reshape(B * H * P, T, N))

        # Fold initial state into the first timestep's input.
        if state is not None:
            h0_flat = h0.reshape(B * H * P, N)
            b_scan = b_scan.clone()
            b_scan[:, 0] = b_scan[:, 0] + a_scan[:, 0] * h0_flat

        h = linear_recurrence(a_scan, b_scan, self.scan_backend)  # [B*H*P, T, N]
        h = h.reshape(B, H, P, T, N).permute(0, 3, 1, 2, 4)  # [B,T,H,P,N]
        h_new = h[:, -1]  # [B,H,P,N]

        y = (h * Cc.unsqueeze(3)).sum(-1)  # [B,T,H,P]
        y = y + self.D.view(1, 1, H, 1) * xv
        y = y.reshape(B, T, self.hidden_dim).to(x.dtype)

        gate = F.silu(self.gate_proj(x))
        return self.out_proj(y * gate), h_new

    # Sequential reference (ground truth for equivalence tests; not on the hot path).
    def forward_reference(self, x: torch.Tensor, state: torch.Tensor | None = None):
        B, T, _ = x.shape
        H, P, N = self.num_heads, self.head_dim, self.state_dim
        h0 = _acc(state) if state is not None else self.empty_state(B, x.device, x.dtype)
        xv, Cc, a, dBx = self._project(x)
        a_scan = a.permute(0, 2, 1).reshape(B * H, 1, T, 1).expand(B * H, P, T, N)
        a_scan = _acc(a_scan.reshape(B * H * P, T, N))
        b_scan = _acc(dBx.permute(0, 2, 3, 1, 4).reshape(B * H * P, T, N))
        if state is not None:
            b_scan = b_scan.clone()
            b_scan[:, 0] = b_scan[:, 0] + a_scan[:, 0] * h0.reshape(B * H * P, N)
        h = seq_recurrence(a_scan, b_scan).reshape(B, H, P, T, N).permute(0, 3, 1, 2, 4)
        y = (h * Cc.unsqueeze(3)).sum(-1) + self.D.view(1, 1, H, 1) * xv
        y = y.reshape(B, T, self.hidden_dim).to(x.dtype)
        gate = F.silu(self.gate_proj(x))
        return self.out_proj(y * gate), h[:, -1]


class SSDBlock(nn.Module):
    """SSDMixer + ShortCausalConv + SwiGLU + RMSNorm residual block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        state_dim: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        conv_kernel_size: int = 4,
        ffn_expand: int = 2,
        scan_backend: str = "auto",
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.conv = ShortCausalConv1d(hidden_dim, conv_kernel_size)
        self.ssm = SSDMixer(hidden_dim, num_heads, state_dim, dt_min, dt_max, scan_backend)
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, ffn_expand)

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        ssm_state: torch.Tensor | None = None,
    ):
        r = x
        x_n = self.norm1(x)
        x_c, new_conv_state = self.conv(x_n, conv_state)
        x_s, new_ssm_state = self.ssm(x_c, ssm_state)
        x = r + x_s
        x = x + self.ffn(self.norm2(x))
        return x, new_conv_state, new_ssm_state
