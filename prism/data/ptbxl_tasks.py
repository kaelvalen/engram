"""PTB-XL task → label mapping (pure, pandas-free, fully unit-testable).

PTB-XL is benchmarked over six task groups (Strodthoff et al. 2020;
helme/ecg_ptbxl_benchmarking). Each maps a record's SCP codes to a different
label vocabulary; all are **multi-label**:

    superdiag — 5 diagnostic superclasses (NORM/MI/STTC/CD/HYP)
    subdiag   — diagnostic subclasses
    diag      — diagnostic statements
    form      — form statements
    rhythm    — rhythm statements
    all       — every SCP statement

These functions operate on a plain dict view of ``scp_statements.csv``:

    scp = {code: {"diagnostic": 1, "form": 0, "rhythm": 0,
                  "diagnostic_subclass": "...", "diagnostic_class": "..."}, ...}

so they need no pandas (the dataset passes ``scp_df.to_dict("index")``). The
label *vocabulary* is derived from the full statement table — not from any data
split — so train/val/test share identical class indices.
"""

from __future__ import annotations

from collections.abc import Iterable

TASKS = ("superdiag", "subdiag", "diag", "form", "rhythm", "all")
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def _flag(row: dict, key: str) -> bool:
    """True if a boolean SCP column (diagnostic/form/rhythm) is set (1/1.0/True)."""
    v = row.get(key)
    try:
        return float(v) == 1.0
    except (TypeError, ValueError):
        return bool(v)


def _str(row: dict, key: str) -> str | None:
    """Non-empty, non-NaN string field (e.g. diagnostic_subclass)."""
    v = row.get(key)
    if v is None:
        return None
    if isinstance(v, float):  # NaN
        return None
    s = str(v).strip()
    return s or None


def task_vocab(scp: dict[str, dict], task: str) -> list[str]:
    """Sorted, split-independent class vocabulary for a task."""
    if task == "superdiag":
        return list(SUPERCLASSES)  # canonical fixed order (matches the literature)
    if task == "all":
        return sorted(scp.keys())
    if task in ("diag", "form", "rhythm"):
        key = "diagnostic" if task == "diag" else task
        return sorted(c for c, r in scp.items() if _flag(r, key))
    if task == "subdiag":
        subs = {
            _str(r, "diagnostic_subclass")
            for r in scp.values()
            if _flag(r, "diagnostic")
        }
        return sorted(s for s in subs if s)
    raise ValueError(f"Unknown PTB-XL task {task!r}; valid: {TASKS}")


def record_labels(scp: dict[str, dict], codes: Iterable[str], task: str) -> list[str]:
    """Class names present in one record under a task (order-preserving, deduped)."""
    out: list[str] = []
    for code in codes:
        row = scp.get(code)
        if row is None:
            continue
        if task == "all":
            out.append(code)
        elif task == "diag" and _flag(row, "diagnostic"):
            out.append(code)
        elif task in ("form", "rhythm") and _flag(row, task):
            out.append(code)
        elif task == "subdiag" and _flag(row, "diagnostic"):
            sub = _str(row, "diagnostic_subclass")
            if sub:
                out.append(sub)
        elif task == "superdiag" and _flag(row, "diagnostic"):
            sup = _str(row, "diagnostic_class")
            if sup in SUPERCLASSES:
                out.append(sup)
    seen: list[str] = []
    for x in out:
        if x not in seen:
            seen.append(x)
    return seen


def multi_hot(labels: Iterable[str], vocab: list[str]) -> list[float]:
    """Multi-hot vector over ``vocab`` for the given label names."""
    index = {c: i for i, c in enumerate(vocab)}
    vec = [0.0] * len(vocab)
    for name in labels:
        if name in index:
            vec[index[name]] = 1.0
    return vec
