"""
Ablation runner (§6.3 of the manuscript).

Trains and evaluates the two ablation variants reported in Table 5:

  - Full--Factor          : factored world model, no opponent loss
                            (set train.beta_opponent = 0.0)
  - Full--Factor--Opponent: collapsed latent (no factoring, no opponent)
                            (set model.u_dim = 0 and train.beta_opponent = 0.0)

For each variant and each seed, runs the same `interaction.train`
driver with overrides applied to the config, and writes results
under `results/ablations/<variant>/seed<S>/`.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

import yaml


VARIANTS = {
    "full_factor": {
        "train": {"beta_opponent": 0.0},
    },
    "full_factor_opponent": {
        "model": {"u_dim": 0},
        "train": {"beta_opponent": 0.0},
    },
}


def write_variant_config(base_path: str, variant: str) -> Path:
    with open(base_path) as f:
        cfg = yaml.safe_load(f) or {}
    overrides = VARIANTS[variant]
    for section, kvs in overrides.items():
        cfg.setdefault(section, {}).update(kvs)
    cfg["run_name"] = f"ablation_{variant}"
    out = Path(tempfile.mkstemp(prefix=f"foe_dreamer_{variant}_", suffix=".yaml")[1])
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/foe_dreamer.yaml")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--out-root", default="results/ablations")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        cfg_path = write_variant_config(args.config, variant)
        logging.info("variant=%s config=%s", variant, cfg_path)
        for seed in range(args.seeds):
            cmd = [
                "python", "-m", "interaction.train",
                "--config", str(cfg_path),
                "--run-name", f"{variant}/seed{seed}",
                "--seed", str(seed),
            ]
            if args.total_steps is not None:
                cmd += ["--total-steps", str(args.total_steps)]
            logging.info("RUN %s", " ".join(cmd))
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
