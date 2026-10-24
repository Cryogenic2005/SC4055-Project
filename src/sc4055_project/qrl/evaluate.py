"""Shared evaluation harness. OWNER: P3.

Contract:

    from qrl.evaluate import evaluate_checkpoint
    evaluate_checkpoint(cfg: Config, checkpoint: str, conditions: list[str] = NOISE_MENU,
                        n_episodes: int = 100) -> None
        # runs n_episodes greedy episodes per condition and appends rows to
        # runs/eval/<task>_<model>_<seed>.csv via qrl.logging_utils.EvalLogger,
        # always including the eval_noise_type="none" reference block.

The per-condition device comes from qrl.noise.make_device(n_qubits, parse_noise(cond)).
A single-task version for FrozenLake is in tasks/frozenlake/evaluate.py (P2).
"""

from __future__ import annotations

from .config import Config
from .noise import NOISE_MENU


def evaluate_checkpoint(
    cfg: Config,
    checkpoint: str,
    conditions: list[str] = NOISE_MENU,
    n_episodes: int = 100,
) -> None:
    raise NotImplementedError(
        "P3: shared evaluation harness not implemented yet; see tasks/frozenlake/evaluate.py"
    )
