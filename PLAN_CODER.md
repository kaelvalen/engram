# CODING PLAN — unify repo narrative + wire surprise-gated routing (interface only)

**Audience:** the coder/agent that will implement this (Claude Code, another
agent, or a human with a terminal). Run steps in order; each has acceptance
criteria. Most steps are docs; only Step 2 touches runtime code, and it is a
non-breaking, default-off interface change.

**Scope:** documentation + one small router interface. **No** training runs, **no**
architecture redesign. This is a single sitting (a few hours), not a project.

**Non-goal:** fixing SGMS's spike-gate recall result. That is a separate, later
experiment that *depends on* Step 2 existing — not on it being trained yet.

---

## Step 0 — Decision gate (do this first, in writing, before any edit)

Two decisions block Steps 3/4. Resolve both at the top of this file, then proceed.

### 0a. Final project name (replaces `ENGRAM`)
Chosen name: **ENGRAM** (DECIDED). Banner tagline: *ENGRAM — modality-portable
hybrid linear-recurrent backbone with surprise-gated explicit memory.*
(Full naming rationale at the bottom of this file.)

Rename scope: **doc-level (cheap)** — DECIDED. Change only prose/titles:
- `README.md` title
- `paper/PAPER_DRAFT.md` title and the paper's subtitle already carrying the
  scientific claim ("modality-portable hybrid linear-recurrent backbone")
- Citation bibtex `title=` (keep the `github.com/kaelvalen/engram` URL as-is
  until the repo itself is renamed on GitHub — a separate decision)

Python package **stays `engram` internally** for now. Defer the full package
rename until ENGRAM is proven (a paper / after the surprise experiment). This
keeps risk near zero.

### 0b. SABER's role
DECIDED: `engram.saber.saber.SurpriseEstimator` becomes an *input* to SGMS's
router (Step 2). SABER is no longer a standalone "experimental" effort — it is
the surprise-signal supplier inside SGMS.

---

## Step 1 — EXPERIMENTS.md: record the merge decision + surprise hypothesis

**File:** `EXPERIMENTS.md`

Replace the standalone framing of the `## SGMS spike status (2026-07-18, RTX
5060)` section (which currently reads as if SGMS and SABER are unrelated
efforts) by adding, inside/above it, an explicit **Hypothesis / Next**
paragraph stating:

- SGMS's router (`sgms/router.py`) routes on `z_t = W_r h_t` alone — a plain
  learned linear projection with no "how well is the recurrent state guessing
  this token?" signal (Switch-Transformer style).
- Hypothesis: the first spike-gate failure (SGMS recall 0.023 vs GDR-only
  0.096 @64 — see the existing table) is partly a **weak-routing-signal**
  problem, not purely a capacity problem.
- Planned test: feed `SurpriseEstimator.forward(z_t, z_hat_t)` output — a
  clamped, normalized per-token scalar, `[B, T]` — into the router as an extra
  per-token feature, then re-run the identical spike-gate protocol (3 seeds ×
  8000 steps, MQAR 8-pairs @ T=64, `configs/sgms/spike.yaml`) and check whether
  it closes the gap to GDR-only.
- Be explicit that this is **a hypothesis and a planned experiment, not a
  result** — do not claim improvement before a run exists.

Also note the already-agreed honesty items that stay: parameter matching
(sgms 4.48 M vs baselines 3.16 M) is not yet honored; "modality-portable" ≠
"single-set-of-weights."

**Acceptance:** a reader of `EXPERIMENTS.md` knows (1) the surprise-gating idea
is *planned, not run*, and (2) how it will be tested.

---

## Step 2 — Router interface: accept an optional surprise feature

Runtime code. Make the router *capable* of consuming a per-token surprise signal
without wiring the full SABER encoder/predictor pipeline yet. Default must be
**off** so every existing test keeps its exact output.

**Files:** `sgms/router.py`, `sgms/config.py`, `sgms/block.py`, new
`tests/sgms/test_router_surprise.py`.

### 2a. `sgms/router.py` — `TokenRouter`
- Add constructor param `surprise_scale: float = 0.0`. Store it.
  - `0.0` (default) ⇒ behavior is byte-for-byte identical to today — this is
    the "off" switch, and it avoids a separate boolean flag.
- Change `forward` signature to
  `forward(self, h, exclude=None, surprise: torch.Tensor | None = None)`.
- In the `learned` branch, after `z = F.linear(h, self.weight, self.router_bias)`
  and after the `exclude` mask, add:
  ```python
  if surprise is not None and self.surprise_scale != 0.0:
      # surprise: [B, T] normalized scalar per token (engram.saber SurpriseEstimator).
      z = z + self.surprise_scale * surprise.unsqueeze(-1)  # [B, T] -> [B, T, 1]
  ```
  Put the comment in; don't leave the choice of learnable-vs-fixed unexplained.
  (Fixed scalar scale, config-driven, is simpler for v1 — note that.)
- Keep `uniform` and `random` modes untouched (they ignore surprise).

### 2b. `sgms/config.py` — `SGMSConfig`
- Add field `router_surprise_scale: float = 0.0` with a one-line docstring.
- No `__post_init__` validation needed beyond a non-negative check (optional).

### 2c. `sgms/block.py` — `SGMSBlock`
- Thread it through so a caller *can* pass surprise without a big routing
  refactor:
  - Pass `surprise_scale=cfg.router_surprise_scale` when constructing `TokenRouter`.
  - Add optional `surprise: torch.Tensor | None = None` to `SGMSBlock.forward`
    and forward it to `self.router(x, exclude=drop_idx, surprise=surprise)`.
  - Default `None` keeps all existing callers unchanged.

### 2d. New test — `tests/sgms/test_router_surprise.py`
- **(a) Regression guard:** with `surprise_scale=0.0`, `router.forward(h,
  surprise=some_tensor)` yields the same `logits`/`indices`/`gates` as
  `router.forward(h)` (allclose, exact values).
- **(b) Shape/gradient flow:** with `surprise_scale > 0`, `logits` differs from
  the no-surprise case, gradient flows back to `router.weight`, and output
  shapes are `[B, T, K]` etc.
- **(c) Mode isolation:** `uniform`/`random` modes are unaffected by a passed
  `surprise` (they already don't read it) — assert no shape crash.

**Acceptance:** `pytest tests/sgms/` is green, including the new file, with zero
changes to any existing test's expected values.

> What Step 2 does **not** do (deliberately): no SABER encoder/predictor is wired
> into SGMS training, and no spike-gate run is launched. Generating `surprise` at
> train time (a lightweight standalone predictor vs. the full `SABER` stack) is
> a decision for the later experiment task, not this refactor.

---

## Step 3 — README.md: one story instead of three

**File:** `README.md`

Restructure top-to-bottom into a single narrative, replacing the current
three-way split (core / SGMS / SABER as separate top-level sections):

```
# <NEW_NAME> — <one-line claim>
> (existing summary blockquote, lightly edited)

## What's the claim     (existing "core" content, unchanged)
## Architecture          (existing, unchanged)
## Status                (existing status block, unchanged)

## SGMS: learned per-token routing      (was its own top-level section)
    - existing SGMS content
    - NEW subsection "Surprise-gated routing (in progress)": one paragraph.
      Points to the EXPERIMENTS.md Step-1 addition. States status honestly:
      interface exists (Step 2); experiment NOT yet run.

## Testing / Repo layout / Honest scope / Citation   (existing, unchanged)
```

Delete the standalone `## SABER (experimental)` section entirely — its content
moves into the SGMS subsection above. The `engram/saber/` *directory* does not
move; only its framing changes. Replace the "SABER is experimental / not wired
into the benchmark matrix" label with the concrete roadmap line
("surprise-gated routing — interface in Step 2, experiment pending").

**Acceptance:** README reads as one flow —
`<NAME> = backbone core (ready, verified) → SGMS + surprise routing (active
development, the architectural contribution) → kernel layer re-usable as
standalone infrastructure (FLA-style)`. No orphaned SABER header remains.

---

## Step 4 — Naming pass (apply Step-0a decision)

**Cheap option checklist:**
- `README.md` title line
- `paper/PAPER_DRAFT.md` title
- Citation bibtex `title=` (keep the `github.com/kaelvalen/engram` URL as-is
  unless the repo itself is renamed on GitHub — a separate decision)

**Full option checklist (only if chosen in 0a):**
- `pyproject.toml`: `name`, `description`, `[project.scripts]`, and
  `[tool.setuptools.packages.find].include`
- Rename `engram/` → `<new_name>/`
- Global replace: `from engram.` → `from <new_name>.`, `import engram` →
  `import <new_name>` across `sgms/`, `tests/`, `scripts/`, `train.py`
- `flake.nix` / `.envrc` if they reference the package name
- Re-run Step 5 after this (this is the step most likely to break silently)

**Acceptance:** no stragglers of the old name in prose or imports.

---

## Step 5 — Verify nothing broke

```bash
pytest                                  # expect 270+ passed, same as before the plan
python scripts/validate_ptbxl_tasks.py  # unaffected, sanity check anyway
```

If Step 4 used the full rename, also grep for stragglers:

```bash
grep -rn "\bengram\." --include="*.py" .        # nothing outside intentional refs
grep -rn "ENGRAM" README.md paper/              # no orphaned old name in prose
```

---

## Naming candidates (Step-0a background)

The architecture: a linear-recurrent (SSD+Delta) compressed state **plus** an
explicit memory whose write/route decision is gated by prediction surprise —
"predictable → stay in the recurrent average; surprising → write to memory."
The name should carry that "what goes where" idea. (Known collision to avoid:
`HAM`, *Hybrid Associative Memory*, is taken by a 2026 paper — don't use the
HAM family.)

| Name | Why |
|---|---|
| **ENGRAM** (recommended) | Neuroscience: "the physical trace of a memory." A surprising token leaves an engram (explicit memory); a predictable one dissolves into the recurrent average. Single word, strong image, says exactly what the mechanism does. |
| **SIEVE** | What separates keeps-in-memory from passes-through — literally the router's job. Plain, physical, "primitive." |
| **THRESH** | Both "threshold" (the routing decision boundary) and "thresh/thresh" (winnow grain from chaff). Rustic, first-principles. |
| **KAIROS** | Greek "the right/moments" — the router's real question is *when* to write. More philosophical, less techy-acronym. |
| **SAGE** | "Surprise-Adaptive Gated Experts." Closest to a classic ML-paper acronym; hooks directly into SGMS's MoE roots. |

**Recommendation:** ENGRAM. If none fit, propose another one-liner with the
same "route-by-surprise → imprint" metaphor (e.g. **Imprint**, **Etch**,
**Trace**) before falling back to SAGE.

---

## Step 6 — Design-review rework (post commit 024a09a / c056960)

Resolutions from the design review. Applies on top of Steps 1–5; the router
interface and the experiment protocol were both reworked.

### 6.1 Discovery: the original scalar-broadcast surprise was a no-op at top_k=1
`z = W_r h + scale · surprise.unsqueeze(-1)` added the *same* scalar to all K
expert logits. Softmax and argmax are shift-invariant, so at `top_k=1` it
changed **nothing**: not the decision, not the gate, not even the softmax —
only the raw `logits` field (feeding L_bal/L_z). Empirically confirmed
(`selection_same=True, gates_same=True, probs_same=True`). The committed
"surprise-gated routing interface" therefore did not gate routing.

### 6.2 Fix: per-expert surprise weight (implemented, tests pass)
- `TokenRouter` now has `surprise_weight: Parameter(shape [K])`, default zeros
  ⇒ inert, backward-compatible (all prior tests unchanged).
- Forward (learned, scale>0):
  `z = W_r h + surprise_scale · surprise_weight · surprise.unsqueeze(-1)`.
  Expert-dependent ⇒ can change the decision.
- `SGMSConfig.router_surprise_scale` stays the master on/off gate.
- Tests `tests/sgms/test_router_surprise.py` rewritten: inert defaults, per-expert
  logit-shift formula, **decision can actually flip**, grad reaches
  `router.weight` and `surprise_weight`, wrong-shape raises, modes ignore.
- Note: with `top_k=1` and no straight-through, `surprise_weight` still gets no
  task gradient *through the gate path* (Switch limitation) — that is precisely
  why the experiment is staged (6.4).

### 6.3 Signal source (decided): lightweight standalone predictor, type (a)
A small causal predictor over the router's pre-norm hidden stream:
`ĥ_t = P(h_{t-1})`, `P` = small MLP with an **EMA copy** as the stable surprise
baseline (mirrors `engram/saber/saber.py` `Predictor`); online `P` backprop trains
toward `h_t` (stop-grad); surprise = normalized `|h_t − ĥ_t|`.
- **Chosen over (b)** (deviation from EMA of past, `|h_t − EMA(h_{<t})|`): (b) is
  local volatility, not *prediction error*, and the research question is what the
  recurrent state fails to compress. (a) matches SABER and the literature.
- **Full SABER stack is NOT used in this experiment** — kept separate so the
  routing question is not confounded by slot-memory machinery.
- **Causality requirement:** `P` sees only `h_{<t}` by construction. Add a
  leakage test: perturbing `h_{>t}` must not change `surprise_t`; assert the
  EMA/online predictor built from a shifted stream matches the causal reference.

### 6.4 Staged experiment protocol (avoid conflating quantity with learnability)
All on the MQAR spike protocol (3 seeds × 8000 steps @ T=64), laptop-class:
0. **Fixed-scale probe** — freeze router, sweep per-expert `surprise_weight` /
   `surprise_scale`. "Is the signal useful at all?" Cheap insurance; if no
   setting moves recall, redesign the predictor before any learnability work.
1. **Learned, `top_k=1` + straight_through** — apples-to-apples vs the existing
   negative baseline (same `top_k`). `surprise_weight` trains.
2. **Full learned `top_k=2`** — higher capacity. **Requires the control
   condition**: `top_k=2` without surprise (zeroed) alongside `top_k=2 + surprise`.
   If both beat baseline, the gain is from `k`, not surprise.

Report `mean ± std` over seeds; discount the efficiency claim until gathered
execution (§3.4, v2) lands — dense masking scales compute with `K`, not `top_k`.

### 6.5 Provenance note
`write_strength = sigmoid(γ·surprise)` in `engram/saber/saber.py` is
**pre-existing** SABER code (commit 9b4c724), not part of these commits. It is
the explicit-memory write gate; the router path is the additive per-expert
feature above. Keep these two mechanisms distinct in any paper/write-up.

---

## Step 7 — `sgms/surprise.py`: lightweight causal standalone predictor (spec for coder)

### 7.1 Wiring decision (revised — layer-local, acyclic; option (b))
The naive "final-hidden, single global surprise fed to all routers" is a
**depth-cycle, not a time-cycle**: `surprise = f(h_final)` but `h_final` is only
produced *by* the routers that `surprise` is meant to drive — circular across
depth, uncomputable in one forward pass (forces the two-pass option (a)).
REJECTED for this experiment.

Chosen: **(b) layer-local.** Each `SGMSBlock` owns its own `HiddenSurprisePredictor`
that predicts that block's **own input** `x_i` (the pre-norm hidden entering the
block — the same signal the router's `W_r` reads): `ĥ_{i,t} = P_i(x_{i,t-1})`,
`surprise_i = normalize(|x_{i,t} − ĥ_{i,t}|)`, fed **only to that block's router**
— exactly the existing layer-local interface
`self.router(x, exclude, surprise)` (single pass, no two-forward).

Acyclicity: `surprise_i` uses only `x_i`, which is already causally produced
before the router runs in the same forward. Cross-layer coupling is the ordinary
DAG of a deep network (`surprise_1 → x_1 → x_2 → surprise_2`), not a cycle.
Compute is not doubled.

Alternative explicitly rejected: (a) two-pass (plain forward → surprise → rerun
with surprise). It doubles compute and changes semantics ("surprise from a clean
pass") — a deliberate heavier experiment, not the default.

### 7.2 File / class / signature
**File:** `sgms/surprise.py` — new module.

```python
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F

@dataclass
class SurprisePredictorConfig:
    hidden_dim: int
    predictor_hidden_dim: int = 64          # small MLP capacity
    ema_decay: float = 0.999                 # stable baseline lag
    surprise_mu_lambda: float = 0.99         # running mean decay
    surprise_sigma_lambda: float = 0.99      # running var decay
    surprise_eps_min: float = 1e-6
    surprise_eps_scale: float = 1.0
    surprise_max: float = 3.0

class HiddenSurprisePredictor(nn.Module):
    """ĥ_t = P(x_{t-1}); surprise = normalized |x_t - ĥ_t| (EMA baseline).
    Layer-local (option (b)): operates on ONE block's own input x_i and feeds
    only that block's router. Acyclic + single pass by construction.

    Lightweight, causal, SABER-free: no LatentEncoder. Mirrors
    engram/saber/saber.py Predictor + SurpriseEstimator patterns only.
    """
    def __init__(self, cfg: SurprisePredictorConfig):
        # online predictor, strictly past-only input
        self.predictor = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.predictor_hidden_dim, cfg.hidden_dim),
        )
        # EMA shadows of ALL predictor params (stable surprise baseline, lags online)
        self._ema_shadows = [p.detach().clone() for p in self.predictor.parameters()]
        # running surprise statistics (mu/sigma buffers), updated ONLY in training
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))

    @torch.no_grad()
    def update_ema(self):
        # shadow = decay*shadow + (1-decay)*param, elementwise per parameter
        ...

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """x: [B, T, D] this block's own input (pre-norm hidden).
        Returns (surprise [B, T], aux {h_hat, pred_loss})."""
        # Causal shift: pred_t uses x_{t-1}; t=0 predicts from zeros (<start>).
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        x_hat = self._predict_with_ema(x_prev)              # stable baseline
        abs_diff = (x - x_hat).abs().mean(dim=-1)           # [B,T]
        if self.training:                                    # update mu/sigma only in train
            ...
        eps = torch.clamp(self.cfg.eps_scale * self.sigma.sqrt(), min=self.cfg.eps_min)
        surprise = ((abs_diff - self.mu) / (self.sigma.sqrt() + eps)).clamp(0, self.cfg.surprise_max)
        pred_online = self.predictor(x_prev)                 # online params train toward x
        pred_loss = F.mse_loss(pred_online, x.detach())      # JEPA-style, stop-grad target
        return surprise, {"h_hat": x_hat, "pred_loss": pred_loss}
```

Design notes for the coder:
- **Ownership:** `SGMSBlock` owns one `HiddenSurprisePredictor` when enabled
  (config `router_surprise_scale > 0`), built per-layer. In `SGMSBlock.forward`,
  compute `surprise_i, aux = self.surprise_predictor(x)` from the block's own
  input `x` and pass to `self.router(x, ..., surprise=surprise_i)`. The external
  `surprise=` override on the block stays for probe/ablation (manual/manual signal
  injection); the internal predictor is the default source. Per-layer predictors
  are **independent by default**; a shared predictor across layers is an optional
  config choice (default off for isolation).
- EMA shadows: one deep copy per online parameter; `update_ema()` blends each.
  Prediction ALWAYS uses the EMA shadows (stable baseline); the online `predictor`
  is what backprop trains toward `x` via `pred_loss` (stop-grad on target).
- `surprise_i` is a pure deterministic function of `x_{i,t-1}, x_{i,t}` in eval
  mode (mu/sigma frozen) — this is what makes 7.3 testable.
- Trainer (later task): call each layer's `update_ema()` each step and add the
  sum of per-layer `pred_loss` (weighted) to the SGMS objective. This step ships
  the module + test only.

### 7.3 Causality / leakage test — `tests/sgms/test_surprise_predictor.py` (pure unit, no GPU)
The test exercises one layer's predictor over its own input `x [B,T,D]`; the
same assertion holds per layer. `x` is any causal hidden tensor the layer owns.
1. **Future-invariance:** build `x [B,T,D]`, `p.eval()` (freeze mu/sigma), compute
   `surprise_full`. Perturb strictly-future positions (`x_bad[:, t0:] += noise`).
   Assert `torch.equal(surprise_full[:, :t0], surprise_bad[:, :t0])` and that
   `surprise_bad[:, t0:]` differs. Proves no future leakage into `surprise_t`.
2. **Strict shift correctness:** changing `x_t` changes `surprise_t` but NOT
   `surprise_{t-1}` (pred at `t-1` used `x_{t-2}`, not `x_{t-1}`). Assert single
   nonzero per-row effect.
3. **Shape/sanity:** `surprise.shape == [B, T]`, values in `[0, surprise_max]`,
   finite; `h_hat` shares `x`'s shape.
4. **Determinism in eval:** two identical `eval()` forwards give
   `torch.equal` surprise (buffers not updated).

### 7.4 Acceptance before Stage 0 (MQAR spike)
`pytest tests/sgms/test_surprise_predictor.py` green AND `pytest tests/sgms/` green
with the module present but **unwired** (no trainer/experiment changes in this
step). Only then wire surprise generation + run Stage 0.
