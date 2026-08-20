"""Evaluation metrics. Dependency-free macro AUROC for the PTB-XL story.

PTB-XL is benchmarked with macro-averaged AUROC (one-vs-rest), not accuracy
(Strodthoff et al. 2020). xresnet1d101 reaches ~0.928 macro AUC on the 5-class
super-diagnostic task; matching that within bootstrap CI is the paper's bar.
"""

from __future__ import annotations

import torch


def binary_auroc(scores: torch.Tensor, positive: torch.Tensor) -> float | None:
    """Rank-based binary AUROC (Mann–Whitney U). Returns None if the class is
    degenerate (all-positive or all-negative). Ties handled via average ranks.
    """
    scores = scores.detach().float().flatten()
    positive = positive.detach().bool().flatten()
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    # average ranks (1-indexed) to handle ties correctly
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    # resolve ties: assign the mean rank within each group of equal scores
    uniq, inv, counts = torch.unique(sorted_scores, return_inverse=True, return_counts=True)
    cum = torch.cumsum(counts, 0)
    start = cum - counts
    mean_rank = (start + cum + 1).float() / 2.0  # 1-indexed average rank per group
    ranks[order] = mean_rank[inv]
    rank_pos = ranks[positive].sum()
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def roc_auc_ovr_macro(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged one-vs-rest AUROC for single-label multiclass logits.

    logits: [N, C]; labels: [N] integer class ids. Averages per-class binary
    AUROC over classes that are present (non-degenerate).
    """
    aucs = []
    for c in range(num_classes):
        auc = binary_auroc(logits[:, c], labels == c)
        if auc is not None:
            aucs.append(auc)
    if not aucs:
        return float("nan")
    return sum(aucs) / len(aucs)


def bootstrap_auroc_ci(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    multilabel: bool | None = None,
    num_classes: int | None = None,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Macro AUROC point estimate + bootstrap CI, matching the Strodthoff et al.
    PTB-XL protocol (report e.g. ``0.928(05)`` = mean ± 0.005 half-width).

    Resamples the N examples with replacement ``n_resamples`` times. Returns
    ``(point, lo, hi)`` where [lo, hi] is the central ``ci`` interval; the
    half-width is ``(hi - lo) / 2``.

    - multi-label: scores/targets are [N, C] (targets in {0,1}).
    - single-label: scores are [N, C] logits/probs, targets are [N] class ids
      (set ``multilabel=False`` and pass ``num_classes``).
    """
    if multilabel is None:
        multilabel = targets.dim() == 2

    def macro(idx):
        if multilabel:
            return multilabel_auroc_macro(scores[idx], targets[idx])
        return roc_auc_ovr_macro(scores[idx], targets[idx], num_classes or scores.shape[1])

    n = scores.shape[0]
    g = torch.Generator().manual_seed(seed)
    point = macro(torch.arange(n))
    samples = []
    for _ in range(n_resamples):
        idx = torch.randint(0, n, (n,), generator=g)
        val = macro(idx)
        if val == val:  # skip NaN (degenerate resample)
            samples.append(val)
    samples = sorted(samples)
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    lo = samples[max(0, int(lo_q * len(samples)))]
    hi = samples[min(len(samples) - 1, int(hi_q * len(samples)))]
    return point, lo, hi


def multilabel_auroc_macro(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Macro AUROC for multi-label targets (PTB-XL all/diag/form/rhythm tasks).

    scores, targets: [N, C] with targets in {0,1}. This is the metric shape the
    full multi-label PTB-XL loader should feed (see EXPERIMENTS.md).
    """
    aucs = []
    for c in range(scores.shape[1]):
        auc = binary_auroc(scores[:, c], targets[:, c])
        if auc is not None:
            aucs.append(auc)
    if not aucs:
        return float("nan")
    return sum(aucs) / len(aucs)
