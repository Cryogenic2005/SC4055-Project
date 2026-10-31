from __future__ import annotations

import numpy as np

from sc4055_project.qrl.config import Config
from sc4055_project.qrl.dqn import train_agent


def encode_obs(obs: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """CartPole encoder: arctan feature map per observation dimension."""
    return np.arctan(np.asarray(obs, dtype=np.float32)).astype(np.float32)


def train(cfg: Config, quick: bool = False) -> dict:
    out = train_agent(
        cfg,
        state_shape=(cfg.obs_dim,),
        state_encode_fn=encode_obs,
        reward_augment_fn=None,
        quick=quick,
    )

    print(out)
    return out
