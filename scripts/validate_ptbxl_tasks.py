"""Validate PTB-XL task vocabularies against the real scp_statements.csv.

Closes the EXPERIMENTS.md submission TODO: the per-task class counts must
match the published benchmark (Strodthoff et al. 2020):

    superdiag 5 · subdiag 23 · diag 44 · form 19 · rhythm 12 · all 71

Also cross-checks every SCP code used by ptbxl_database.csv records against
the statement table, so a dataset-update surprise (renamed/dropped codes)
fails loudly here instead of silently degrading labels.

Usage:
    python scripts/validate_ptbxl_tasks.py [--data-root datasets/ptbxl]
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

from engram.data.ptbxl_tasks import TASKS, record_labels, task_vocab

EXPECTED = {
    "superdiag": 5,
    "subdiag": 23,
    "diag": 44,
    "form": 19,
    "rhythm": 12,
    "all": 71,
}


def _load_scp(path: Path) -> dict[str, dict]:
    """scp_statements.csv → {code: row-dict} (pandas-free, like the mapping)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # first column is the unnamed code index
        code_key = reader.fieldnames[0]
        return {row.pop(code_key): row for row in reader}


def _record_codes(path: Path) -> list[set[str]]:
    """scp_codes column of every record in ptbxl_database.csv."""
    codes = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            codes.append(set(ast.literal_eval(row["scp_codes"]).keys()))
    return codes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets/ptbxl")
    args = ap.parse_args()
    root = Path(args.data_root)

    scp = _load_scp(root / "scp_statements.csv")
    print(f"scp_statements.csv: {len(scp)} codes\n")

    ok = True
    print(f"{'task':<10} {'vocab':>6} {'expected':>9}  status")
    for task in TASKS:
        vocab = task_vocab(scp, task)
        match = len(vocab) == EXPECTED[task]
        ok &= match
        print(f"{task:<10} {len(vocab):>6} {EXPECTED[task]:>9}  {'OK' if match else 'MISMATCH'}")

    records = _record_codes(root / "ptbxl_database.csv")
    unknown = sorted({c for codes in records for c in codes if c not in scp})
    print(f"\nptbxl_database.csv: {len(records)} records")
    if unknown:
        ok = False
        print(f"ERROR: {len(unknown)} codes missing from scp_statements.csv: {unknown}")
    else:
        print("all record SCP codes resolve against scp_statements.csv")

    # Label coverage per task: how many records receive ≥1 label.
    for task in TASKS:
        covered = sum(1 for codes in records if record_labels(scp, codes, task))
        print(f"  records with ≥1 {task:<10} label: {covered:>6} / {len(records)}")

    print("\nRESULT:", "OK — vocabularies match the benchmark" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
