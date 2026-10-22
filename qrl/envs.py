"""Gymnasium environment helpers and per-task observation encoders.

The encoder maps a raw observation to the vector the model sees:
  basis   FrozenLake state index -> 4 bits in {0, 1}; the circuit applies RX(pi * bit).
  arctan  CartPole / Acrobot features -> arctan(x) in (-pi/2, pi/2)  (Skolik et al.).
  linear  LunarLander: identity here; the trainable 8 -> 4 encoder lives in the model.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from qrl.config import TASKS


def make_env(env_id: str, seed: int | None = None, max_steps: int | None = None, **kw) -> gym.Env:
    if env_id == "FrozenLake-v1":
        # Slippery is the registered default; keep it so the 0.70 threshold means something.
        kw.setdefault("is_slippery", True)
    env = gym.make(env_id, max_episode_steps=max_steps, **kw)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def encode_obs(env_id: str, obs) -> np.ndarray:
    """Return a float32 vector of length TASKS[env_id]['n_qubits'] (or obs_dim for MLPs)."""
    enc = TASKS[env_id]["encoding"]
    if enc == "basis":
        s = int(obs)
        bits = [(s >> i) & 1 for i in range(4)]
        return np.asarray(bits, dtype=np.float32)
    if enc == "arctan":
        return np.arctan(np.asarray(obs, dtype=np.float32))
    if enc == "linear":
        return np.asarray(obs, dtype=np.float32)
    raise ValueError(enc)


def one_hot_obs(env_id: str, obs) -> np.ndarray:
    """What the classical MLP baselines see. FrozenLake gets a 16-way one-hot so the
    MLP is not handed a smarter encoding than the VQC."""
    if env_id == "FrozenLake-v1":
        v = np.zeros(16, dtype=np.float32)
        v[int(obs)] = 1.0
        return v
    return np.asarray(obs, dtype=np.float32)


def random_baseline(env_id: str, n_episodes: int = 100, seed: int = 0) -> float:
    """Mean undiscounted return of a uniform-random policy (R_random in the plan)."""
    env = make_env(env_id, seed=seed)
    rng = np.random.default_rng(seed)
    rets = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
            total += r
            done = term or trunc
        rets.append(total)
    env.close()
    return float(np.mean(rets))
