from __future__ import annotations

import json
from pathlib import Path

from sc4055_project.qrl.dqn import train_agent

from .agent import TASK_ADAPTER


def train(cfg, quick=False):
    if cfg.env_id != "Acrobot-v1":
        raise ValueError("Acrobot trainer requires Acrobot-v1")
    result = train_agent(cfg, adapter=TASK_ADAPTER, quick=quick)
    path = Path(cfg.run_dir) / "checkpoints" / cfg.run_name / "training_summary.json"
    # quick runs are explicitly labelled and must not be pooled with experiments.
    path.write_text(json.dumps(dict(result, quick=quick), indent=2), encoding="utf-8")
    return result
