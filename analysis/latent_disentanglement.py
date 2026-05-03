"""
Latent disentanglement diagnostics (§6.4 of the manuscript).

Produces:

  Table 6 (R^2 user prediction)         : results/disentanglement/r2_user.csv
  Table 7 (action prediction accuracy)  : results/disentanglement/action_pred.csv
  Figure 6 (opponent embedding t-SNE)   : results/figures/figure6_opponent_embeddings.png
  Figure 7 (action accuracy over time)  : results/figures/figure7_action_accuracy.png

Inputs:
  --checkpoint  : a FOE-Dreamer checkpoint produced by interaction.train
  --rollout-dir : directory with .npz rollouts (obs, action, attacker_action,
                  attacker_target, user_burst_phase). One file per episode.

The R^2 score is computed by training a linear regressor from (z, u)
to user_burst_phase; both columns of the table report on holdout
episodes. Action-prediction accuracy is read from the opponent
decoder; t-SNE is taken on the tau embeddings.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise SystemExit("torch is required for the disentanglement diagnostics.")


def _try_import_sklearn():
    try:
        from sklearn.linear_model import Ridge
        from sklearn.manifold import TSNE
        from sklearn.metrics import r2_score
        return Ridge, TSNE, r2_score
    except ImportError:
        raise SystemExit("scikit-learn is required. `pip install scikit-learn`.")


def load_rollouts(rollout_dir: Path) -> List[Dict[str, np.ndarray]]:
    rollouts = []
    for path in sorted(rollout_dir.glob("*.npz")):
        rollouts.append(dict(np.load(path)))
    if not rollouts:
        raise FileNotFoundError(f"no .npz rollouts under {rollout_dir}")
    return rollouts


def encode_rollouts(checkpoint_path: str, rollouts: List[Dict[str, np.ndarray]]):
    """Returns z, u, tau, user_phase, attacker_action arrays concatenated across episodes."""
    torch = _try_import_torch()
    from interaction.eval import load_modules
    from envs import make_env

    env = make_env("envs/scenarios/scenario_small.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modules = load_modules(checkpoint_path, env, device)
    rssm = modules["rssm"]; encoder = modules["encoder"]
    opponent = modules["opponent"]

    zs, us, taus, user_phases, atk_actions = [], [], [], [], []
    for ep in rollouts:
        obs = torch.from_numpy(ep["obs"]).float().to(device)
        action = torch.from_numpy(ep["action"]).float().to(device)
        T, B = obs.shape[0], 1
        obs = obs.unsqueeze(1) if obs.dim() == 2 else obs
        action = action.unsqueeze(1) if action.dim() == 2 else action

        with torch.no_grad():
            embed = encoder(obs.reshape(T, -1))
            embed = embed.reshape(T, B, -1)
            nonterm = torch.ones(T, B, 1, device=device)
            init = rssm.init_state(B, device)
            _, posterior = rssm.rollout_observation(T, embed, action, nonterm, init)
            zs.append(posterior.z_stoch.squeeze(1).cpu().numpy())
            us.append(posterior.u_stoch.squeeze(1).cpu().numpy())
            # tau over sliding windows.
            K = opponent.K_window
            ep_taus = []
            for t in range(K, T):
                tau = opponent.encode(
                    obs[t - K:t], action[t - K:t]
                ).squeeze(0).cpu().numpy()
                ep_taus.append(tau)
            taus.append(np.asarray(ep_taus))
        user_phases.append(ep.get("user_burst_phase", np.zeros(T, dtype=np.int64)))
        atk_actions.append(ep["attacker_action"])
    return (
        np.concatenate(zs, axis=0),
        np.concatenate(us, axis=0),
        np.concatenate(taus, axis=0),
        np.concatenate(user_phases, axis=0),
        np.concatenate(atk_actions, axis=0),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rollout-dir", required=True)
    p.add_argument("--out-dir", default="results/disentanglement")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_out = Path("results/figures")
    fig_out.mkdir(parents=True, exist_ok=True)

    Ridge, TSNE, r2_score = _try_import_sklearn()
    rollouts = load_rollouts(Path(args.rollout_dir))
    z, u, tau, user_phase, atk_action = encode_rollouts(args.checkpoint, rollouts)

    # ---- Table 6: R^2 user prediction ----
    rows = []
    for label, feat in [("z (controllable)", z), ("u (exogenous)", u), ("z+u", np.concatenate([z, u], axis=1))]:
        # train/test split 80/20
        n = feat.shape[0]
        cut = int(0.8 * n)
        reg = Ridge(alpha=1.0).fit(feat[:cut], user_phase[:cut].astype(float))
        r2 = r2_score(user_phase[cut:].astype(float), reg.predict(feat[cut:]))
        rows.append({"feature": label, "r2_user": float(r2)})
    with open(out / "r2_user.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "r2_user"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logging.info("Table 6: r2_user.csv written")

    # ---- Table 7: action prediction accuracy from tau ----
    # Use a simple linear classifier over tau -> attacker_action.
    from sklearn.linear_model import LogisticRegression
    n = min(tau.shape[0], atk_action.shape[0])
    feat = tau[:n]; targ = atk_action[:n]
    cut = int(0.8 * n)
    clf = LogisticRegression(max_iter=1000, multi_class="auto").fit(feat[:cut], targ[:cut])
    acc = float((clf.predict(feat[cut:]) == targ[cut:]).mean())
    with open(out / "action_pred.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "action_pred_acc"])
        w.writeheader()
        w.writerow({"feature": "tau", "action_pred_acc": acc})
    logging.info("Table 7: action_pred.csv written (acc=%.3f)", acc)

    # ---- Figure 6: t-SNE of tau ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required. `pip install matplotlib`.")

    sample = min(2000, tau.shape[0])
    idx = np.random.default_rng(0).choice(tau.shape[0], sample, replace=False)
    tsne = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(tau[idx])
    plt.figure(figsize=(5, 5))
    colors = atk_action[idx]
    plt.scatter(tsne[:, 0], tsne[:, 1], c=colors, cmap="tab10", s=4, alpha=0.7)
    plt.title("Opponent embedding (tau) t-SNE")
    plt.tight_layout()
    plt.savefig(fig_out / "figure6_opponent_embeddings.png", dpi=200)
    logging.info("Figure 6: figure6_opponent_embeddings.png written")

    # ---- Figure 7: action accuracy over training (placeholder bin: per-window accuracy) ----
    win = 200
    accs = []
    for s in range(0, n - win, win):
        sub = clf.predict(feat[s:s + win])
        accs.append(float((sub == targ[s:s + win]).mean()))
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(accs)) * win, accs, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Action prediction accuracy")
    plt.title("Opponent action prediction over training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_out / "figure7_action_accuracy.png", dpi=200)
    logging.info("Figure 7: figure7_action_accuracy.png written")


if __name__ == "__main__":
    main()
