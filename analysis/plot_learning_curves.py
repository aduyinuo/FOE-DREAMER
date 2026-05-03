"""
Figure 5: learning curves across baselines and FOE-Dreamer.

Reads `results/<method>/seed<S>/metrics.csv` produced by the trainers
and plots episode loss versus training day with mean and standard
deviation bands across seeds, one panel per attacker profile.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_method(method_dir: Path) -> Dict[str, np.ndarray]:
    """Returns {seed_name: [(env_step, ep_return), ...]}"""
    seeds: Dict[str, np.ndarray] = {}
    for seed_dir in sorted(method_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        csv_path = seed_dir / "metrics.csv"
        if not csv_path.exists():
            continue
        rows = []
        with open(csv_path) as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append((int(r.get("env_step", 0)), float(r.get("ep_return", 0.0))))
        if rows:
            seeds[seed_dir.name] = np.asarray(rows)
    return seeds


def aggregate(curves: Dict[str, np.ndarray], n_bins: int = 200) -> tuple:
    """Bins env_step into n_bins, returns (bin_centers, mean_loss, std_loss)."""
    if not curves:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    all_steps = np.concatenate([c[:, 0] for c in curves.values()])
    edges = np.linspace(all_steps.min(), all_steps.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    per_seed = []
    for c in curves.values():
        binned = []
        for i in range(n_bins):
            mask = (c[:, 0] >= edges[i]) & (c[:, 0] < edges[i + 1])
            if mask.any():
                binned.append(-float(c[mask, 1].mean()))   # loss = -return
            else:
                binned.append(np.nan)
        per_seed.append(binned)
    arr = np.asarray(per_seed, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return centers, mean, std


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    p.add_argument("--methods", nargs="+",
                   default=["foe_dreamer", "rainbow", "iqn"])
    p.add_argument("--attackers", nargs="+", default=["avoidant", "enticed"])
    p.add_argument("--out", default="results/figures/figure5_learning_curves.png")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.")

    fig, axes = plt.subplots(1, len(args.attackers), figsize=(6 * len(args.attackers), 4), sharey=True)
    if len(args.attackers) == 1:
        axes = [axes]

    for ax, attacker in zip(axes, args.attackers):
        for method in args.methods:
            method_dir = Path(args.root) / method / attacker
            curves = load_method(method_dir)
            if not curves:
                logging.warning("no curves for %s/%s", method, attacker)
                continue
            x, mean, std = aggregate(curves)
            ax.plot(x, mean, label=method)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_title(f"Attacker: {attacker}")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Episode loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    logging.info("wrote %s", out)


if __name__ == "__main__":
    main()
