from __future__ import annotations

from functools import partial

import numpy as np

from sc4055_project.qrl.agent import TaskAdapter
from sc4055_project.qrl.config import Config
from sc4055_project.qrl.dqn import train_agent


def state_to_bits(s: int | np.ndarray, n_qubits: int) -> np.ndarray:
    """Convert an integer state to a binary representation (bit array)."""
    s = np.asarray(s, dtype=np.int64)
    bits = np.stack([(s >> i) & 1 for i in range(n_qubits)], axis=-1)
    return bits.astype(np.float32)


def shape_reward(r: float, terminated: bool) -> float:
    """Chen et al. 2020 shaping for FrozenLake: +1 goal, -0.2 hole, -0.01 per step.
    Used only for the replay buffer; logged returns stay raw."""
    if terminated:
        return 1.0 if r > 0 else -0.2
    return -0.01


def make_task_adapter(cfg: Config) -> TaskAdapter:
    state_encode_fn = partial(state_to_bits, n_qubits=cfg.n_qubits)  # pyright: ignore[reportArgumentType]
    return TaskAdapter(
        encode_obs=state_encode_fn,
        reward_augment_fn=shape_reward,
    )


def train(cfg: Config, quick: bool = False) -> dict:
    adapter = make_task_adapter(cfg)
    out = train_agent(
        cfg,
        adapter=adapter,
        quick=quick,
    )

    print(out)
    return out
