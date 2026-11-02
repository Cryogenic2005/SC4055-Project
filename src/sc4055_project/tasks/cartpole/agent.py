from __future__ import annotations

import numpy as np

from sc4055_project.qrl.agent import TaskAdapter


def encode_obs(obs: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """CartPole encoder: arctan feature map per observation dimension."""
    return np.arctan(np.asarray(obs, dtype=np.float32)).astype(np.float32)


TASK_ADAPTER = TaskAdapter(
    encode_obs=encode_obs,
    state_shape=(4,),
)
