from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import ShortCausalConv1d
from .ffn import SwiGLU
from .norm import RMSNorm, l2_normalize

logger = logging.getLogger(__name__)


def _acc(x: torch.Tensor) -> torch.Tensor:
    """Promote to an accumulation dtype: float32 for bf16/fp16, but keep
    float32/float64 untouched (so float64 gradcheck stays double-precision)."""
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


# Cache the FLA chunk_gated_delta_rule kernel (Triton, GPU-only). None if absent.
_FLA_FN = None
_FLA_CHECKED = False
_FLA_WARNED = False


def _load_fla():
    """Return fla.ops chunk_gated_delta_rule or None (cached)."""
    global _FLA_FN, _FLA_CHECKED
    if not _FLA_CHECKED:
        _FLA_CHECKED = True
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule

            _FLA_FN = chunk_gated_delta_rule
        except Exception:  # pragma: no cover - import path varies / GPU-only
            _FLA_FN = None
    return _FLA_FN


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
        backend: str = "reference",
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qk_norm = qk_norm
        self.chunk_size = chunk_size
        self.backend = backend

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.alpha_proj = nn.Linear(hidden_dim, num_heads, bias=True)
        self.beta_proj = nn.Linear(hidden_dim, num_heads, bias=True)
        self.out_gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        nn.init.constant_(self.alpha_proj.bias, gate_bias_init)
        nn.init.zeros_(self.beta_proj.bias)

    def empty_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> DeltaState:
        return DeltaState(
            S=torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                device=device,
                dtype=torch.float32,
            )
        )

    def _project(self, x):
        B, T, _ = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        if self.qk_norm:
            q = l2_normalize(q, dim=-1)
            k = l2_normalize(k, dim=-1)

        alpha = torch.sigmoid(self.alpha_proj(x)).transpose(1, 2)  # [B, H, T]
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)  # [B, H, T]
        gate = F.silu(self.out_gate_proj(x))  # [B, T, hidden]

        return q, k, v, alpha, beta, gate

    @staticmethod
    def _recurrent_naive(q, k, v, alpha, beta, S0):
        """Per-token recurrent reference. Kept for tests; not on the hot path."""
        B, H, T, Dh = q.shape
        S = S0
        outs = []

        qf, kf, vf = _acc(q), _acc(k), _acc(v)
        af, bf = _acc(alpha), _acc(beta)

        for i in range(T):
            kt = kf[:, :, i]  # [B, H, Dh]
            vt = vf[:, :, i]
            qt = qf[:, :, i]
            at = af[:, :, i].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            bt = bf[:, :, i].unsqueeze(-1).unsqueeze(-1)

            Sk = torch.einsum("bhij,bhj->bhi", S, kt).unsqueeze(-1)  # [B,H,Dh,1]
            outer_k = kt.unsqueeze(-2)  # [B,H,1,Dh]
            outer_v = vt.unsqueeze(-1)  # [B,H,Dh,1]

            S = at * (S - bt * (Sk @ outer_k)) + bt * (outer_v @ outer_k)
            ot = torch.einsum("bhij,bhj->bhi", S, qt)
            outs.append(ot)

        out = torch.stack(outs, dim=2)  # [B, H, T, Dh]
        return out.to(q.dtype), S

    @staticmethod
    def _step_one(q, k, v, alpha, beta, S0):
        """Closed-form single-token step (T == 1 inference path).

        Equivalent to _recurrent_naive for T==1 but without the Python loop
        and without per-step allocations.
        """
        # q,k,v: [B,H,1,Dh]  alpha,beta: [B,H,1]  S0: [B,H,Dh,Dh]
        qt = _acc(q[:, :, 0])  # [B,H,Dh]
        kt = _acc(k[:, :, 0])
        vt = _acc(v[:, :, 0])
        at = _acc(alpha[:, :, 0])  # [B,H]
        bt = _acc(beta[:, :, 0])

        Sk = torch.einsum("bhij,bhj->bhi", S0, kt)  # [B,H,Dh]
        u = bt.unsqueeze(-1) * vt - (at * bt).unsqueeze(-1) * Sk  # [B,H,Dh]

        Sq = torch.einsum("bhij,bhj->bhi", S0, qt)  # [B,H,Dh]
        kq = (kt * qt).sum(-1, keepdim=True)  # [B,H,1]
        o = at.unsqueeze(-1) * Sq + u * kq  # [B,H,Dh]

        a4 = at.unsqueeze(-1).unsqueeze(-1)  # [B,H,1,1]
        S_new = a4 * S0 + torch.einsum("bhi,bhj->bhij", u, kt)
        return o.unsqueeze(2).to(q.dtype), S_new

    @staticmethod
    def _recurrent_vectorized(q, k, v, alpha, beta, S0, chunk_size):
        """Vectorized chunkwise delta rule, mathematically equivalent to _recurrent_naive.

        Within each chunk we substitute u_t = β_t v_t - α_t β_t S_{t-1} k_t so the
        recurrence becomes S_t = α_t S_{t-1} + u_t k_t^T. Rescaling ũ_t = u_t / ᾱ_t
        (with ᾱ_t = Π_{i≤t} α_i) yields a triangular linear system in ũ that we
        solve in a single batched call. Output and state update are then two
        batched matmuls per chunk. T sequential matmuls → O(1) per chunk.
        """
        B, H, T, Dh = q.shape
        device = q.device
        qf, kf, vf = _acc(q), _acc(k), _acc(v)
        af, bf = _acc(alpha), _acc(beta)
        dtype = qf.dtype

        S = S0
        outs = []

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            C = end - start

            q_c = qf[:, :, start:end]  # [B,H,C,Dh]
            k_c = kf[:, :, start:end]
            v_c = vf[:, :, start:end]
            a_c = af[:, :, start:end]  # [B,H,C]
            b_c = bf[:, :, start:end]

            # ᾱ_t and 1/ᾱ_t via cumulative log-sum for numerical safety.
            log_a = torch.log(a_c.clamp(min=1e-12))
            cum_log = torch.cumsum(log_a, dim=-1)  # [B,H,C]
            alpha_bar = torch.exp(cum_log)  # [B,H,C]
            inv_alpha_bar = torch.exp(-cum_log)

            # Pairwise k inner products and causal masks (built once per chunk).
            kk = torch.einsum("bhcd,bhsd->bhcs", k_c, k_c)  # [B,H,C,C]
            mask_strict = torch.tril(torch.ones(C, C, device=device, dtype=dtype), diagonal=-1)
            mask_inc = torch.tril(torch.ones(C, C, device=device, dtype=dtype), diagonal=0)

            # System matrix L: L_{t,s} = β_t (k_t · k_s) for s<t, 0 elsewhere.
            # Combined with unit diagonal (via unitriangular=True) this is (I + L).
            L = b_c.unsqueeze(-1) * kk * mask_strict  # [B,H,C,C]

            # rhs row t: (β_t / ᾱ_t) v_t - β_t (S_0 k_t)
            Sk = torch.einsum("bhij,bhcj->bhci", S, k_c)  # [B,H,C,Dh]
            rhs = (b_c * inv_alpha_bar).unsqueeze(-1) * v_c - b_c.unsqueeze(-1) * Sk

            # Forward triangular solve for ũ.
            u_tilde = torch.linalg.solve_triangular(L, rhs, upper=False, unitriangular=True)

            # Output: o_t = ᾱ_t · [(S_0 q_t) + Σ_{s≤t} ũ_s (k_s · q_t)]
            Sq = torch.einsum("bhij,bhcj->bhci", S, q_c)  # [B,H,C,Dh]
            qk = torch.einsum("bhcd,bhsd->bhcs", q_c, k_c) * mask_inc
            attn = torch.einsum("bhcs,bhsd->bhcd", qk, u_tilde)
            o_chunk = alpha_bar.unsqueeze(-1) * (Sq + attn)
            outs.append(o_chunk)

            # State: S_C = ᾱ_C · [S_0 + Ũ^T K]
            UK = torch.einsum("bhcd,bhce->bhde", u_tilde, k_c)
            S = alpha_bar[:, :, -1].view(B, H, 1, 1) * (S + UK)

        out = torch.cat(outs, dim=2)
        return out.to(q.dtype), S

    def _forward_fla(self, q, k, v, alpha, beta, S0):
        """FLA Triton chunk_gated_delta_rule path (GPU-only, production backend).

        Maps our (alpha = per-step forget gate ∈ (0,1), beta = write gate) to
        FLA's op-level signature ``(q, k, v, g, beta, ...)``.

        IMPORTANT — `g` is the **per-step log decay** g_t = log α_t (NOT the
        cumulative sum): the op kernel forms the cumulative diagonal decays
        internally (chunk_local_cumsum). This matches the op-level API of FLA
        0.3.x (the version pinned in pyproject). Note the *layer-level*
        GatedDeltaNet in newer FLA takes a raw projection plus A_log/dt_bias and
        sets use_gate_in_kernel=True — a different, higher-level convention we do
        not use here. Because the exact mapping is version-dependent, this path
        is only trustworthy once tests/test_delta_equivalence.py passes on GPU;
        that test is the gate, not this comment.

        q/k are already l2-normalised in `_project`, so we pass scale=1.0 and do
        NOT enable in-kernel l2-norm (avoid double normalisation); the op default
        is no in-kernel norm. We keep the kwargs minimal so a signature change
        triggers the caller's graceful fallback rather than a silent miscompute.
        """
        fn = _load_fla()
        if fn is None:
            raise ImportError("FLA not available")
        # [B,H,T,Dh] → [B,T,H,Dh]; alpha,beta [B,H,T] → [B,T,H]
        qf = q.transpose(1, 2).contiguous()
        kf = k.transpose(1, 2).contiguous()
        vf = v.transpose(1, 2).contiguous()
        g = torch.log(alpha.transpose(1, 2).clamp(min=1e-12)).contiguous()
        bf = beta.transpose(1, 2).contiguous()
        o, S_new = fn(
            qf,
            kf,
            vf,
            g,
            bf,
            scale=1.0,
            initial_state=S0,
            output_final_state=True,
            head_first=False,
        )
        o = o.transpose(1, 2)  # [B,T,H,V] → [B,H,T,V]
        return o.to(q.dtype), S_new

    def forward(
        self,
        x: torch.Tensor,
        state: DeltaState | None = None,
    ) -> tuple[torch.Tensor, DeltaState]:
        global _FLA_WARNED
        B, T, _ = x.shape
        q, k, v, alpha, beta, gate = self._project(x)

        acc_dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
        S0 = (
            _acc(state.S)
            if state is not None
            else torch.zeros(
                B,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                device=x.device,
                dtype=acc_dtype,
            )
        )

        if T == 1:
            o, S_new = self._step_one(q, k, v, alpha, beta, S0)
        elif self.backend == "fla":
            try:
                o, S_new = self._forward_fla(q, k, v, alpha, beta, S0)
            except ImportError as e:
                # FLA is not installed; graceful fallback to the reference backend.
                if not _FLA_WARNED:
                    logger.warning(
                        "FLA delta backend unavailable (%s); falling back to the "
                        "reference chunked implementation.",
                        e,
                    )
                    _FLA_WARNED = True
                o, S_new = self._recurrent_vectorized(q, k, v, alpha, beta, S0, self.chunk_size)
            except RuntimeError as e:
                # CUDA OOM or kernel signature mismatch: surface loudly rather than
                # silently switching backends.
                logger.error("FLA delta backend failed at runtime: %s", e)
                raise
        else:
            o, S_new = self._recurrent_vectorized(q, k, v, alpha, beta, S0, self.chunk_size)

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
        backend: str = "reference",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.conv = ShortCausalConv1d(hidden_dim, conv_kernel_size)
        self.delta = GatedDeltaRule(
            hidden_dim, num_heads, qk_norm, chunk_size, gate_bias_init, backend
        )
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, ffn_expand)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.ffn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        delta_state: DeltaState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, DeltaState]:
        r = x
        x_n = self.norm1(x)
        x_c, new_conv_state = self.conv(x_n, conv_state)
        x_d, new_delta_state = self.delta(x_c, delta_state)
        x = r + self.dropout(x_d)
        x = x + self.ffn_dropout(self.ffn(self.norm2(x)))
        return x, new_conv_state, new_delta_state
