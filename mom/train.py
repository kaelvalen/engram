"""MoM training loop (spec §5).

Minimal, deterministic trainer for the spike: synthetic MQAR (§5.1), AdamW +
cosine schedule with 3% warmup (§5.2), task loss + stability objectives
(§3.7), and first-class routing statistics logged every eval (§6.1):
per-layer utilization, routing entropy, gate confidence, flip rate between
adjacent evals, and the minimum utilization collapse detector (§6.4).

Usage:
    python -m mom.train --config configs/mom/spike.yaml [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .baselines import build_model
from .config import MoMConfig
from .losses import mom_auxiliary_loss
from .router import routing_stats
from .tasks.mqar import MQARConfig, make_mqar_batch


def _set_seeds(seed: int):
    torch.manual_seed(seed)


def _lr_at(step: int, total: int, base: float, warmup_frac: float) -> float:
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * t))


def evaluate(model, task_cfg: MQARConfig, batch_size: int, seed: int, device="cpu") -> float:
    """Recall accuracy at scored (post-query) positions on a fixed batch."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    ids, labels = make_mqar_batch(task_cfg, batch_size, g, device)
    with torch.no_grad():
        logits = model(ids)["logits"]
    # logits[t] predicts token t+1: compare against labels shifted by one.
    pred = logits[:, :-1].argmax(-1)
    tgt = labels[:, 1:]
    scored = tgt != -100
    model.train()
    return float((pred[scored] == tgt[scored]).float().mean())


def _routing_log(routings, prev_indices: list | None) -> tuple[dict, list]:
    """First-class routing metrics (§6.1) + flip rate between evals."""
    stats = routing_stats(routings)
    indices = [r.indices.detach().clone() for r in routings]
    if prev_indices is not None:
        flips = [
            float((a != b).float().mean()) for a, b in zip(indices, prev_indices, strict=False)
        ]
        stats["flip_rate"] = sum(flips) / len(flips) if flips else 0.0
    for layer in stats["layers"]:
        layer["utilization"] = layer["utilization"].tolist()
    return stats, indices


def train_one(config: dict, seed: int, device: str = "cpu") -> dict:
    """One seeded run. Returns history + final metrics (spec §5.2 determinism)."""
    _set_seeds(seed)
    cfg = MoMConfig.from_dict(config["model"])
    task_cfg = MQARConfig(**{k: v for k, v in config["task"].items() if k != "name"})
    optim = config["optim"]
    steps = optim["steps"]
    eval_every = config.get("eval_every", max(1, steps // 10))

    kind = config["model"].get("kind", "mom")
    model = build_model(kind, cfg, task_cfg.vocab_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=optim["lr"], weight_decay=optim["weight_decay"], betas=(0.9, 0.95)
    )
    data_gen = torch.Generator().manual_seed(seed + 1)

    use_amp = bool(optim.get("bf16", False))
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    is_mom = hasattr(model, "blocks") and kind in ("mom", "B4", "B5")
    history = []
    prev_indices = None
    t0 = time.time()
    for step in range(steps):
        lr = _lr_at(step, steps, optim["lr"], optim["warmup_frac"])
        for g in opt.param_groups:
            g["lr"] = lr

        ids, labels = make_mqar_batch(task_cfg, optim["batch_size"], data_gen, device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(ids)
            logits = out["logits"]
            task_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            if is_mom:
                aux, parts = mom_auxiliary_loss(out["routings"], cfg.lambda_bal, cfg.lambda_z)
                loss = task_loss + aux
                bal, z = float(parts["bal"]), float(parts["z"])
            else:
                loss = task_loss
                bal = z = 0.0
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, optim["grad_clip"])
        opt.step()

        record = {
            "step": step,
            "task_loss": float(task_loss.detach()),
            "bal": bal,
            "z": z,
            "lr": lr,
        }
        if (step + 1) % eval_every == 0 or step == steps - 1:
            record["accuracy"] = evaluate(
                model, task_cfg, optim["batch_size"], seed=999, device=device
            )
            if is_mom:
                with torch.no_grad():
                    eval_ids, _ = make_mqar_batch(
                        task_cfg, optim["batch_size"], torch.Generator().manual_seed(999), device
                    )
                    eval_out = model(eval_ids)
                stats, prev_indices = _routing_log(eval_out["routings"], prev_indices)
                record.update(stats)
            else:
                # baselines carry no router: keep the log schema consistent
                record["layers"] = []
                record["min_utilization"] = None
        history.append(record)

    summary = {
        "seed": seed,
        "kind": kind,
        "history": history,
        "final": history[-1],
        "param_report": model.param_report() if hasattr(model, "param_report") else {},
        "wall_time_s": round(time.time() - t0, 2),
    }
    # Context generalization (spike gate §6.4: recall @ 4k). Evaluated at the
    # same num_pairs with longer contexts than training.
    accuracy_by_context = {}
    for ctx in config.get("eval_contexts", []):
        ctx_cfg = MQARConfig(
            vocab_size=task_cfg.vocab_size, num_pairs=task_cfg.num_pairs, seq_len=ctx
        )
        accuracy_by_context[str(ctx)] = evaluate(
            model, ctx_cfg, optim["batch_size"], seed=999, device=device
        )
    summary["accuracy_by_context"] = accuracy_by_context
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="MoM spike trainer (MQAR)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="override; default: config seeds")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seeds = [args.seed] if args.seed is not None else config.get("seeds", [0])
    log_dir = Path(config.get("log_dir", "output/mom")) / config.get("experiment", "run")
    log_dir.mkdir(parents=True, exist_ok=True)

    finals = []
    for seed in seeds:
        summary = train_one(config, seed, device=args.device)
        out_file = log_dir / f"seed{seed}.json"
        out_file.write_text(json.dumps(summary, indent=2))
        acc = summary["final"].get("accuracy", float("nan"))
        min_util = summary["final"].get("min_utilization")
        min_util_s = f"{min_util:.3f}" if min_util is not None else "—"
        print(
            f"[{config.get('experiment')}] seed={seed} acc={acc:.4f} "
            f"task_loss={summary['final']['task_loss']:.4f} min_util={min_util_s} "
            f"params={summary['param_report'].get('total', 0):,} → {out_file}"
        )
        finals.append(acc)

    if len(finals) > 1:
        t = torch.tensor(finals)
        print(f"mean ± std accuracy over {len(finals)} seeds: {t.mean():.4f} ± {t.std():.4f}")


if __name__ == "__main__":
    main()
