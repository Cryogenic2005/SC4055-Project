"""Inspect CartPole checkpoints for value-scale stability and policy collapse.

Prints compact diagnostics for `best.pkl` and `final.pkl` in one checkpoint directory:
  - output weight scale (`w_out`) for VQC models
  - Q-value mean/std/range over sampled states
  - greedy action share (near 50/50 is weak; skewed but stable can be better)

Usage
-----
python analysis/cartpole/inspect_weights.py \
  --checkpoints runs/checkpoints/CartPole-v1_vqc_0
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sc4055_project.qrl.agent import make_q_fn
from sc4055_project.qrl.config import Config
from sc4055_project.qrl.noise import parse_noise
from sc4055_project.tasks.cartpole.agent import encode_obs

N_SAMPLES = 512


def sample_observations(n: int, seed: int = 0) -> np.ndarray:
    """Sample broad CartPole states (not on-policy visitation)."""
    rng = np.random.default_rng(seed)
    # CartPole observations: x, x_dot, theta, theta_dot.
    # Ranges here are intentionally wider than termination bounds for stress-testing.
    x = rng.uniform(-4.8, 4.8, n)
    xdot = rng.uniform(-5.0, 5.0, n)
    theta = rng.uniform(-0.418, 0.418, n)
    thetadot = rng.uniform(-5.0, 5.0, n)
    return np.stack([x, xdot, theta, thetadot], axis=1).astype(np.float32)


def report(path: Path, cfg: Config, params, obs: np.ndarray, noise: str) -> None:
    q_fn = make_q_fn(cfg, parse_noise(noise), seed=cfg.seed)
    encoded = encode_obs(obs)
    q = np.asarray(q_fn(params, jnp.asarray(encoded), jax.random.PRNGKey(0)))
    share = np.bincount(q.argmax(axis=1), minlength=cfg.n_actions) / len(q)

    print(f"\n== {path.name}")
    if isinstance(params, dict) and "w_out" in params:
        w_out = np.asarray(params["w_out"])
        print(f"   w_out                    {np.round(w_out, 3)}")
        print(f"   |w_out| max              {np.abs(w_out).max():.3f}")
    if isinstance(params, dict) and "lam" in params:
        lam = np.asarray(params["lam"])
        print(f"   lam mean / std           {lam.mean():.3f} / {lam.std():.3f}")

    print(f"   Q mean / std             {q.mean():+.3f} / {q.std():.3f}")
    print(f"   Q range                  {q.min():+.3f} .. {q.max():+.3f}")
    print(f"   greedy action share      {' '.join(f'{s:.0%}' for s in share)}")

    q_gap_ratio = np.abs(q[:, 0] - q[:, 1]) / np.maximum(
        np.mean(np.abs(q), axis=1), 1e-8
    )
    print(f"   median |Q_L-Q_R| / mean|Q|  {np.median(q_gap_ratio):.4f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoints",
        type=Path,
        required=True,
        help="a run checkpoint directory containing config.json and best/final.pkl",
    )
    args = p.parse_args()

    cfg_path = args.checkpoints / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path}")

    cfg = Config.load(cfg_path)
    obs = sample_observations(N_SAMPLES, seed=cfg.seed)

    found = False
    for name in ("best.pkl", "final.pkl"):
        path = args.checkpoints / name
        if not path.exists():
            continue
        found = True
        with open(path, "rb") as fh:
            params = pickle.load(fh)
        report(path, cfg, params, obs, cfg.noise)

    if not found:
        raise SystemExit(f"no best.pkl or final.pkl in {args.checkpoints}")


if __name__ == "__main__":
    main()
