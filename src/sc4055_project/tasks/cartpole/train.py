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
