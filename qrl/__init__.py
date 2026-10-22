"""Shared JAX + PennyLane codebase for the noisy quantum RL project (SC4055).

Modules
-------
config        Config dataclass: one dict-like object drives every run.
envs          Gymnasium wrappers, per-task observation encoders, random baseline.
noise         NoiseSpec, presets, make_device(), apply_noise(), unit conversions.
agent         VQC / MLP / hybrid models as pure functions of a params pytree.
dqn           Replay buffer + DQN training loop (jit + optax.multi_transform).
evaluate      Greedy evaluation of a checkpoint under one noise condition.
logging_utils CSV logger whose columns match docs/analysis_spec columns.
"""

from qrl.config import Config  # noqa: F401
