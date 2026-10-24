"""Shared DQN training loop. OWNER: P1.

Contract (so that every task's train script looks the same):

    from qrl.dqn import train
    train(cfg: Config) -> dict   # writes runs/train/<run_name>.csv via qrl.logging_utils.TrainLogger,
                                 # saves checkpoints to runs/checkpoints/<run_name>/ep{N}.pkl and final.pkl,
                                 # returns {"final_score": mean of last 100 returns, "solved_at": episode or None}

Requirements from the plan:
* jax.jit the update step and jax.vmap over the batch (fast path is the default).
* optax.multi_transform with separate learning rates for input scaling (lr_input),
  circuit angles (lr_circuit) and output weights (lr_output).
* Never combine shots with backprop: if cfg.noise is shots=N, use diff_method="parameter-shift".
* Log wall-clock per episode (TrainLogger does this).

The FrozenLake-only loop in tasks/frozenlake/train.py is the reference to generalise.
"""

from __future__ import annotations

from .config import Config


def train(cfg: Config) -> dict:
    raise NotImplementedError(
        "P1: shared DQN loop not implemented yet; see tasks/frozenlake/train.py"
    )
