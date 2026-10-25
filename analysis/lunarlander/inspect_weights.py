"""Is the Q-function diverging? Inspect a trained LunarLander checkpoint.

A DQN whose value function has run away looks *exactly* like an agent that never
learned: the greedy policy is near-random, so the score sits at random-policy level.
The difference is visible in the weights, and this script prints it.

What to look for (numbers from the 25 Oct diagnosis):

    |w_out|           should be O(1).  It was ~50 while the TD targets were O(1),
                      i.e. the value function was ~30x outside its own target range.
    Q mean / std      should be O(1) for reward_scale=0.01.  It was +21 +- 15.
    saturated         fraction of encoder units pinned at |tanh| > 0.99.  Was 19%
                      with raw observations; 0.1% after scaling in encode_obs().
    action share      the greedy action histogram.  Near-uniform (25/25/25/25) means
                      the policy is not state-dependent; a usable policy is skewed.

Usage
-----
    python analysis/lunarlander/inspect_weights.py \
        --checkpoints runs/checkpoints/LunarLander-v3_vqc_0

Compares best.pkl (selected on a greedy validation eval) with final.pkl, which is
usually where the late-training collapse shows up.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sc4055_project.qrl.config import default_config_for
from sc4055_project.qrl.noise import parse_noise
from sc4055_project.tasks.lunarlander.agent import encode_obs, make_q_fn

N_SAMPLES = 256


def sample_observations(n: int, seed: int = 0) -> np.ndarray:
    """States spread over LunarLander's observation ranges, not over its visit distribution."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(-1.5, 1.5, n),      # x
            rng.uniform(-1.5, 1.5, n),      # y
            rng.uniform(-5.0, 5.0, n),      # vx
            rng.uniform(-5.0, 5.0, n),      # vy
            rng.uniform(-np.pi, np.pi, n),  # angle
            rng.uniform(-5.0, 5.0, n),      # angular velocity
            rng.integers(0, 2, n),          # left leg contact
            rng.integers(0, 2, n),          # right leg contact
        ],
        axis=1,
    ).astype(np.float32)


def report(path: Path, params, q_fn, obs: np.ndarray) -> None:
    encoded = encode_obs(obs)
    pre_activation = encoded @ np.asarray(params["enc_w"]) + np.asarray(params["enc_b"])
    features = np.tanh(pre_activation)
    q = np.asarray(q_fn(params, jnp.asarray(encoded), jax.random.PRNGKey(0)))
    share = np.bincount(q.argmax(axis=1), minlength=4) / len(q)

    print(f"\n== {path.name}")
    print(f"   w_out                    {np.round(np.asarray(params['w_out']), 2)}")
    print(f"   |w_out| max              {np.abs(np.asarray(params['w_out'])).max():.2f}")
    print(f"   Q mean / std             {q.mean():+.2f} / {q.std():.2f}")
    print(f"   Q range                  {q.min():+.2f} .. {q.max():+.2f}")
    print(f"   encoder saturated        {np.mean(np.abs(features) > 0.99):.1%}")
    print(f"   greedy action share      {' '.join(f'{s:.0%}' for s in share)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", type=Path, required=True,
                   help="a run's checkpoint directory (contains best.pkl / final.pkl)")
    p.add_argument("--layers", type=int, default=None, help="default: the task's default depth")
    args = p.parse_args()

    override = {} if args.layers is None else {"n_layers": args.layers}
    cfg = default_config_for("LunarLander-v3", **override)
    q_fn = make_q_fn(cfg, parse_noise("none"), seed=0)
    obs = sample_observations(N_SAMPLES)

    found = False
    for name in ("best.pkl", "final.pkl"):
        path = args.checkpoints / name
        if not path.exists():
            continue
        found = True
        with open(path, "rb") as fh:
            report(path, pickle.load(fh), q_fn, obs)
    if not found:
        raise SystemExit(f"no best.pkl or final.pkl in {args.checkpoints}")


if __name__ == "__main__":
    main()
