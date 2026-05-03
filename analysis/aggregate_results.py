"""
Aggregate per-seed CSVs into the headline and ablation tables.

Reads `results/sweep/<method>/<attacker>/seed<S>/episodes.csv`, computes
mean and standard deviation of per-episode losses across the 5 seeds,
and emits a CSV with one row per (method, attacker) cell. Output
matches the row/column structure of Table 4 (tab:headline) in the
manuscript.

For ablations (Table 5), pass `--root results/ablations` and the
script will aggregate the same way, with ablation variants in place
of methods.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np


def collect_returns(method_dir: Path) -> Dict[str, List[float]]:
    """Returns {attacker_name: [seed-mean returns]}"""
    attackers: Dict[str, List[float]] = {}
    for attacker_dir in sorted(method_dir.iterdir()):
        if not attacker_dir.is_dir():
            continue
        per_seed_means: List[float] = []
        for seed_dir in sorted(attacker_dir.iterdir()):
            csv_path = seed_dir / "episodes.csv"
            if not csv_path.exists():
                logging.warning("missing %s", csv_path)
                continue
            with open(csv_path) as f:
                rdr = csv.DictReader(f)
                returns = [float(r["return"]) for r in rdr]
            if returns:
                per_seed_means.append(float(np.mean(returns)))
        if per_seed_means:
            attackers[attacker_dir.name] = per_seed_means
    return attackers


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/sweep", help="results/sweep or results/ablations")
    p.add_argument("--out", default="results/sweep/aggregated.csv")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found")

    rows = []
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        per_attacker = collect_returns(method_dir)
        for attacker, seed_means in per_attacker.items():
            arr = np.asarray(seed_means)
            # Episode "loss" = negative reward sum (lower is worse per paper §6.1).
            losses = -arr
            rows.append({
                "method": method,
                "attacker": attacker,
                "n_seeds": int(arr.size),
                "loss_mean": float(losses.mean()),
                "loss_std": float(losses.std(ddof=1) if arr.size > 1 else 0.0),
                "return_mean": float(arr.mean()),
                "return_std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
            })

    if not rows:
        raise RuntimeError(f"no results found under {root}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logging.info("wrote %s (%d rows)", out_path, len(rows))

    # Pretty-print to stdout in the layout of the paper table.
    methods = sorted(set(r["method"] for r in rows))
    attackers = sorted(set(r["attacker"] for r in rows))
    print("\nFinal episode loss (mean +/- std over seeds)")
    header = "  attacker      | " + " | ".join(f"{m:>14s}" for m in methods)
    print(header)
    print("-" * len(header))
    for atk in attackers:
        cells = []
        for m in methods:
            match = next((r for r in rows if r["method"] == m and r["attacker"] == atk), None)
            if match is None:
                cells.append(f"{'-':>14s}")
            else:
                cells.append(f"{match['loss_mean']:>7.1f} +/- {match['loss_std']:<3.1f}")
        print(f"  {atk:<13s} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
