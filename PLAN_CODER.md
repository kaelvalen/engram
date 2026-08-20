# CODING PLAN — unify repo narrative + wire surprise-gated routing (interface only)

**Audience:** the coder/agent that will implement this (Claude Code, another
agent, or a human with a terminal). Run steps in order; each has acceptance
criteria. Most steps are docs; only Step 2 touches runtime code, and it is a
non-breaking, default-off interface change.

**Scope:** documentation + one small router interface. **No** training runs, **no**
architecture redesign. This is a single sitting (a few hours), not a project.

**Non-goal:** fixing MoM's spike-gate recall result. That is a separate, later
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
DECIDED: `engram.saber.saber.SurpriseEstimator` becomes an *input* to MoM's
router (Step 2). SABER is no longer a standalone "experimental" effort — it is
the surprise-signal supplier inside MoM.

---

## Step 1 — EXPERIMENTS.md: record the merge decision + surprise hypothesis

**File:** `EXPERIMENTS.md`

Replace the standalone framing of the `## MoM spike status (2026-07-18, RTX
5060)` section (which currently reads as if MoM and SABER are unrelated
efforts) by adding, inside/above it, an explicit **Hypothesis / Next**
paragraph stating:

- MoM's router (`mom/router.py`) routes on `z_t = W_r h_t` alone — a plain
  learned linear projection with no "how well is the recurrent state guessing
  this token?" signal (Switch-Transformer style).
- Hypothesis: the first spike-gate failure (MoM recall 0.023 vs GDR-only
  0.096 @64 — see the existing table) is partly a **weak-routing-signal**
  problem, not purely a capacity problem.
- Planned test: feed `SurpriseEstimator.forward(z_t, z_hat_t)` output — a
  clamped, normalized per-token scalar, `[B, T]` — into the router as an extra
  per-token feature, then re-run the identical spike-gate protocol (3 seeds ×
  8000 steps, MQAR 8-pairs @ T=64, `configs/mom/spike.yaml`) and check whether
  it closes the gap to GDR-only.
- Be explicit that this is **a hypothesis and a planned experiment, not a
  result** — do not claim improvement before a run exists.

Also note the already-agreed honesty items that stay: parameter matching
(mom 4.48 M vs baselines 3.16 M) is not yet honored; "modality-portable" ≠
"single-set-of-weights."

**Acceptance:** a reader of `EXPERIMENTS.md` knows (1) the surprise-gating idea
is *planned, not run*, and (2) how it will be tested.

---

## Step 2 — Router interface: accept an optional surprise feature

Runtime code. Make the router *capable* of consuming a per-token surprise signal
without wiring the full SABER encoder/predictor pipeline yet. Default must be
**off** so every existing test keeps its exact output.

**Files:** `mom/router.py`, `mom/config.py`, `mom/block.py`, new
`tests/mom/test_router_surprise.py`.

### 2a. `mom/router.py` — `TokenRouter`
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

### 2b. `mom/config.py` — `MoMConfig`
- Add field `router_surprise_scale: float = 0.0` with a one-line docstring.
- No `__post_init__` validation needed beyond a non-negative check (optional).

### 2c. `mom/block.py` — `MoMBlock`
- Thread it through so a caller *can* pass surprise without a big routing
  refactor:
  - Pass `surprise_scale=cfg.router_surprise_scale` when constructing `TokenRouter`.
  - Add optional `surprise: torch.Tensor | None = None` to `MoMBlock.forward`
    and forward it to `self.router(x, exclude=drop_idx, surprise=surprise)`.
  - Default `None` keeps all existing callers unchanged.

### 2d. New test — `tests/mom/test_router_surprise.py`
- **(a) Regression guard:** with `surprise_scale=0.0`, `router.forward(h,
  surprise=some_tensor)` yields the same `logits`/`indices`/`gates` as
  `router.forward(h)` (allclose, exact values).
- **(b) Shape/gradient flow:** with `surprise_scale > 0`, `logits` differs from
  the no-surprise case, gradient flows back to `router.weight`, and output
  shapes are `[B, T, K]` etc.
- **(c) Mode isolation:** `uniform`/`random` modes are unaffected by a passed
  `surprise` (they already don't read it) — assert no shape crash.

**Acceptance:** `pytest tests/mom/` is green, including the new file, with zero
changes to any existing test's expected values.

> What Step 2 does **not** do (deliberately): no SABER encoder/predictor is wired
> into MoM training, and no spike-gate run is launched. Generating `surprise` at
> train time (a lightweight standalone predictor vs. the full `SABER` stack) is
> a decision for the later experiment task, not this refactor.

---

## Step 3 — README.md: one story instead of three

**File:** `README.md`

Restructure top-to-bottom into a single narrative, replacing the current
three-way split (core / MoM / SABER as separate top-level sections):

```
# <NEW_NAME> — <one-line claim>
> (existing summary blockquote, lightly edited)

## What's the claim     (existing "core" content, unchanged)
## Architecture          (existing, unchanged)
## Status                (existing status block, unchanged)

## MoM: learned per-token routing      (was its own top-level section)
    - existing MoM content
    - NEW subsection "Surprise-gated routing (in progress)": one paragraph.
      Points to the EXPERIMENTS.md Step-1 addition. States status honestly:
      interface exists (Step 2); experiment NOT yet run.

## Testing / Repo layout / Honest scope / Citation   (existing, unchanged)
```

Delete the standalone `## SABER (experimental)` section entirely — its content
moves into the MoM subsection above. The `engram/saber/` *directory* does not
move; only its framing changes. Replace the "SABER is experimental / not wired
into the benchmark matrix" label with the concrete roadmap line
("surprise-gated routing — interface in Step 2, experiment pending").

**Acceptance:** README reads as one flow —
`<NAME> = backbone core (ready, verified) → MoM + surprise routing (active
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
  `import <new_name>` across `mom/`, `tests/`, `scripts/`, `train.py`
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
| **SAGE** | "Surprise-Adaptive Gated Experts." Closest to a classic ML-paper acronym; hooks directly into MoM's MoE roots. |

**Recommendation:** ENGRAM. If none fit, propose another one-liner with the
same "route-by-surprise → imprint" metaphor (e.g. **Imprint**, **Etch**,
**Trace**) before falling back to SAGE.
