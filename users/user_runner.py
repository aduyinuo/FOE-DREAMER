"""
Per-host user simulator entry point.

Loads a persona by name and runs the simulator loop.

  - u_rand   : memoryless per-step sampling
  - u_burst  : ON/OFF dynamics with heavy-tail ON durations
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .base import User, UserProfile
from .u_rand import UrandUser
from .u_burst import UburstUser


PERSONAS = {
    "u_rand": UrandUser,
    "u_burst": UburstUser,
}


def _default_profile_path(persona_name: str) -> str:
    return str(Path(__file__).parent / "personas" / f"{persona_name}.yaml")


def build_user(persona_name: str, profile_path: str, seed: int) -> User:
    if persona_name not in PERSONAS:
        raise ValueError(
            f"unknown persona: {persona_name!r} (valid: {sorted(PERSONAS)})"
        )
    profile = UserProfile.from_yaml(profile_path)
    cls = PERSONAS[persona_name]
    return cls(profile=profile, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persona", choices=sorted(PERSONAS), required=True,
        help="user persona to run",
    )
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    profile_path = args.profile or _default_profile_path(args.persona)
    user = build_user(args.persona, profile_path, args.seed)
    user.run()


if __name__ == "__main__":
    main()
