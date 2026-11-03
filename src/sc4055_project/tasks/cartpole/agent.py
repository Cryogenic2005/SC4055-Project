from __future__ import annotations

import numpy as np

from sc4055_project.qrl.agent import TaskAdapter

# CartPole observations are continuous and have very different dynamic ranges.
# Dividing by these ranges keeps each feature in roughly [-1, 1] before the
# arctan feature map, which prevents the rotation encoding from saturating early.
CARTPOLE_OBS_SCALE = np.array([2.4, 2.0, 0.209, 2.0], dtype=np.float32)


def encode_obs(obs: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32)
    x = np.clip(x / CARTPOLE_OBS_SCALE, -1.0, 1.0)
    return np.arctan(x).astype(np.float32)


TASK_ADAPTER = TaskAdapter(
    encode_obs=encode_obs,
    state_shape=(4,),
)
