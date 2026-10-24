"""Day-1 smoke test: run this on every laptop and on Colab before doing anything else.

    python scripts/smoke_test.py

Checks: pinned versions import, a jitted 4-qubit QNode runs fast, default.mixed works,
all four Gymnasium environments (incl. Box2D LunarLander) reset, and a 5-episode
FrozenLake training run writes a CSV with the agreed columns.
"""

from __future__ import annotations

import tempfile
import time
import warnings
from pathlib import Path


def test_versions():
    import jax
    import pennylane as qml

    assert qml.__version__.startswith("0.45"), qml.__version__
    assert jax.__version__.startswith("0.10"), jax.__version__


def test_jitted_qnode():
    """Check that a jitted 4-qubit QNode runs fast."""
    import jax
    import jax.numpy as jnp

    from sc4055_project.qrl.noise import parse_noise
    from sc4055_project.tasks.frozenlake.agent import (
        init_params,
        make_q_fn,
        state_to_bits,
    )

    # Initialization: 5 layers, 4 qubits, 3 params per qubit, 4 output weights
    params = init_params(5, jax.random.PRNGKey(0))
    q = make_q_fn(5, parse_noise("none"))
    bits = jnp.asarray(state_to_bits(jnp.arange(16)))  # pyright: ignore[reportArgumentType]

    # Flush compilation overhead: run once before timing
    q(params, bits, jax.random.PRNGKey(0)).block_until_ready()

    t = time.perf_counter()
    for _ in range(20):
        q(params, bits, jax.random.PRNGKey(0)).block_until_ready()

    ms = (time.perf_counter() - t) / 20 * 1000
    if ms >= 50:
        warnings.warn(
            f"QNode execution is slow: {ms:.1f} ms per batch of 16 states "
            "(expected below 50 ms; check whether JIT is effective)",
            RuntimeWarning,
            stacklevel=2,
        )


def test_mixed_device():
    import jax
    import jax.numpy as jnp

    from sc4055_project.qrl.noise import parse_noise
    from sc4055_project.tasks.frozenlake.agent import (
        init_params,
        make_q_fn,
        state_to_bits,
    )

    params = init_params(2, jax.random.PRNGKey(0))
    q = make_q_fn(2, parse_noise("preset=heron2026"))
    out = q(params, jnp.asarray(state_to_bits(jnp.arange(4))), jax.random.PRNGKey(0))  # pyright: ignore[reportArgumentType]
    assert out.shape == (4, 4)


def test_envs():
    """Check that all four Gymnasium environments are accessible."""
    import gymnasium as gym

    for e in ["FrozenLake-v1", "CartPole-v1", "Acrobot-v1", "LunarLander-v3"]:
        env = gym.make(e)
        env.reset(seed=0)
        env.close()


def test_tiny_training_run():
    import csv

    from sc4055_project.frozenlake import main
    from sc4055_project.qrl.logging_utils import TRAIN_COLUMNS

    with tempfile.TemporaryDirectory() as d:
        main([
            "train",
            "--seed",
            "0",
            "--episodes",
            "10",
            "--layers",
            "2",
            "--run-dir",
            d,
            "--quick",
        ])
        csvs = list(Path(d, "train").glob("*.csv"))
        assert csvs, "No CSV written"

        with open(csvs[0]) as f:
            rows = list(csv.DictReader(f))

        assert list(rows[0].keys()) == TRAIN_COLUMNS, rows[0].keys()
