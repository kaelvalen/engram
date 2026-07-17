"""Expert knockout (spec §7.3): force-exclude one expert at inference
(router renormalised) and quantify the resulting metric asymmetry.

Expected signature: GDR knockout → MQAR collapse, minor effect on smoothing
probes; SSD knockout → degradation on smooth-signal tasks.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def mqar_accuracy_metric(model, ids, labels, knockout=None) -> float:
    """Recall accuracy at scored positions, optionally under knockout."""
    out = model(ids, knockout=knockout) if knockout else model(ids)
    pred = out["logits"].argmax(-1)
    scored = labels != -100
    return float((pred[scored] == labels[scored]).float().mean())


def knockout_evaluation(
    model,
    ids: torch.Tensor,
    labels: torch.Tensor,
    metric_fn,
    experts: tuple[str, ...] = ("ssd", "gdr"),
    layers: list[int] | None = None,
) -> dict:
    """Baseline metric + per-expert knockout metrics with deltas.

    ``metric_fn(model, ids, labels, knockout)`` — knockout is the
    {layer: {expert}} dict MoMLM.forward understands.
    """
    num_layers = getattr(model, "cfg").num_layers
    layers = list(range(num_layers)) if layers is None else layers
    report = {"baseline": {"accuracy": metric_fn(model, ids, labels, None)}}
    for name in experts:
        ko = {layer: {name} for layer in layers}
        acc = metric_fn(model, ids, labels, ko)
        report[name] = {
            "accuracy": acc,
            "delta": report["baseline"]["accuracy"] - acc,
        }
    return report
