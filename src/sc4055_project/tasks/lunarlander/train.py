"""LunarLander training entry point.

The task-specific model and optimizer logic live in `tasks/lunarlander/agent.py` as a
`TaskAdapter`. The shared DQN loop in `qrl.dqn` handles the replay buffer, epsilon-greedy
behaviour, target updates, checkpointing and greedy evaluation; task code only wires in
its own observation encoding and model definition.
"""

from __future__ import annotations

from sc4055_project.qrl.config import Config
from sc4055_project.qrl.dqn import train_agent

from .agent import TASK_ADAPTER


def train(cfg: Config, quick: bool = False) -> dict:
    out = train_agent(
        cfg,
        adapter=TASK_ADAPTER,
        quick=quick,
    )

    print(out)
    return out
