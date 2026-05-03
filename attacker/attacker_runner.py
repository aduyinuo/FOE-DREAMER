"""
Per-host attacker runner.

Selects an attacker profile by `--type` and runs it. Two profiles ship:

  - avoidant : conservative, withdraws on positive deception indicators
  - enticed  : opportunistic, tolerates ambiguous indicators
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .base import Attacker, AttackerProfile
from .avoidant import AvoidantAttacker
from .enticed import EnticedAttacker


ATTACKER_TYPES = {
    "avoidant": AvoidantAttacker,
    "enticed": EnticedAttacker,
}


def _default_profile_path(type_name: str) -> str:
    return str(Path(__file__).parent / "profiles" / f"{type_name}.yaml")


def build_attacker(type_name: str, profile_path: str, seed: int) -> Attacker:
    if type_name not in ATTACKER_TYPES:
        raise ValueError(
            f"unknown attacker type: {type_name!r} (valid: {sorted(ATTACKER_TYPES)})"
        )
    profile = AttackerProfile.from_yaml(profile_path)
    cls = ATTACKER_TYPES[type_name]
    return cls(profile=profile, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type", choices=sorted(ATTACKER_TYPES), required=True,
        help="attacker profile to run",
    )
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    profile_path = args.profile or _default_profile_path(args.type)
    attacker = build_attacker(args.type, profile_path, args.seed)
    attacker.run()


if __name__ == "__main__":
    main()
