"""
Multi-seed sweep orchestrator.

Runs the full set of (method, attacker, seed) configurations reported
in Table 4 of the manuscript:

  methods   : foe_dreamer, rainbow, iqn, sleep, random, heuristic
  attackers : avoidant, enticed
  seeds     : 0..N-1 (default 5 seeds)

For each configuration, evaluates the trained policy (or runs the
non-learning baseline) for `--n-episodes` episodes against the
specified attacker profile and writes per-episode losses to
`results/sweep/<method>/<attacker>/seed<S>/episodes.csv`. The raw
per-seed CSVs are the per-seed episode losses promised in §6.1.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from itertools import product
from pathlib import Path
from typing import List


METHODS = ["foe_dreamer", "rainbow", "iqn", "sleep", "random", "heuristic"]
ATTACKERS = ["avoidant", "enticed"]


def run_one(
    method: str, attacker: str, seed: int,
    scenario: str, n_episodes: int, out_root: Path, checkpoint_root: str,
) -> int:
    out = out_root / method / attacker / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if method == "foe_dreamer":
        ckpt = Path(checkpoint_root) / f"seed{seed}" / f"ckpt_step{200_000:09d}.pt"
        cmd = [
            "python", "-m", "interaction.eval",
            "--checkpoint", str(ckpt),
            "--scenario", scenario,
            "--n-episodes", str(n_episodes),
            "--seed", str(seed),
            "--out-dir", str(out),
        ]
    elif method in ("rainbow", "iqn"):
        ckpt = Path(checkpoint_root) / method / f"seed{seed}" / "final.pt"
        cmd = [
            "python", "-m", "baselines.run_baseline",
            "--baseline", method,
            "--scenario", scenario,
            "--n-episodes", str(n_episodes),
            "--seed", str(seed),
            "--out-dir", str(out),
        ]
    else:
        cmd = [
            "python", "-m", "baselines.run_baseline",
            "--baseline", method,
            "--scenario", scenario,
            "--n-episodes", str(n_episodes),
            "--seed", str(seed),
            "--out-dir", str(out),
        ]

    logging.info("RUN %s/%s/seed%d: %s", method, attacker, seed, " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--attackers", nargs="+", default=ATTACKERS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--scenario", default="envs/scenarios/scenario_small.yaml")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--out-root", default="results/sweep")
    parser.add_argument("--checkpoint-root", default="results/foe_dreamer")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    for method, attacker, seed in product(args.methods, args.attackers, range(args.seeds)):
        if args.dry_run:
            logging.info("dry-run: %s/%s/seed%d", method, attacker, seed)
            continue
        rc = run_one(method, attacker, seed, args.scenario, args.n_episodes, out_root, args.checkpoint_root)
        if rc != 0:
            failures.append(f"{method}/{attacker}/seed{seed}")

    summary = {
        "methods": args.methods,
        "attackers": args.attackers,
        "seeds": args.seeds,
        "n_episodes": args.n_episodes,
        "failures": failures,
    }
    with open(out_root / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if failures:
        logging.error("%d failures: %s", len(failures), failures)
    logging.info("sweep complete: %s", out_root)


if __name__ == "__main__":
    main()
