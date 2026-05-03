"""
Baseline runner.

Plays a baseline policy in the cyber-defense environment for N
episodes, dumping per-episode returns to a CSV. Used to populate the
columns of Table 4 in the manuscript.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from envs import make_env
from interaction.rollout import collect_episode

from .heuristic import ExpertHeuristic
from .random_policy import RandomBaseline
from .sleep import SleepPolicy


def build_baseline(name: str, num_hosts: int, seed: int) -> Any:
    if name == "sleep":
        return SleepPolicy(num_hosts=num_hosts)
    if name == "random":
        return RandomBaseline(num_hosts=num_hosts, seed=seed)
    if name == "heuristic":
        return ExpertHeuristic(
            num_hosts=num_hosts,
            public_host_indices=[0, 1],         # PublicWeb1, PublicWeb2
            internal_host_indices=[2, 3, 4],    # NTP, DB, WEB
            seed=seed,
        )
    raise ValueError(f"unknown baseline: {name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", choices=["sleep", "random", "heuristic"], required=True)
    p.add_argument("--scenario", default="envs/scenarios/scenario_small.yaml")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results/baselines")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out_dir) / args.baseline
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(args.scenario)
    policy = build_baseline(args.baseline, num_hosts=env.num_hosts, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    rows = []
    returns = []
    for i in range(args.n_episodes):
        ep, stats = collect_episode(
            env, policy, action_dim=env.action_space.n,
            epsilon=0.0, rng=rng, seed=args.seed + i,
        )
        rows.append({
            "baseline": args.baseline,
            "episode": i,
            "return": stats.episode_return,
            "length": stats.episode_length,
        })
        returns.append(stats.episode_return)

    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary = {
        "baseline": args.baseline,
        "n_episodes": args.n_episodes,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
