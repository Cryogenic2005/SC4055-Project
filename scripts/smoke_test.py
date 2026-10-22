"""Day-1 smoke test: run this on every laptop and on Colab before doing anything else.

    python scripts/smoke_test.py

Checks: pinned versions import, a jitted 4-qubit QNode runs fast, default.mixed works,
all four Gymnasium environments (incl. Box2D LunarLander) reset, and a 5-episode
FrozenLake training run writes a CSV with the agreed columns.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ok = True


def check(name, fn):
    global ok
    t = time.perf_counter()
    try:
        msg = fn()
        print(f"  [ok]   {name:40s} {msg or ''}  ({time.perf_counter()-t:.1f}s)")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] {name:40s} {type(e).__name__}: {e}")


def versions():
    import jax, pennylane as qml, gymnasium, optax
    assert qml.__version__.startswith("0.45"), qml.__version__
    assert jax.__version__.startswith("0.10"), jax.__version__
    return f"pennylane {qml.__version__}, jax {jax.__version__}, gymnasium {gymnasium.__version__}, optax {optax.__version__}"


def jitted_qnode():
    import jax, jax.numpy as jnp
    from qrl.noise import parse_noise
    from tasks.frozenlake.agent import init_params, make_q_fn, state_to_bits
    params = init_params(5, jax.random.PRNGKey(0))
    q = make_q_fn(5, parse_noise("none"))
    bits = jnp.asarray(state_to_bits(jnp.arange(16)))
    q(params, bits, jax.random.PRNGKey(0)).block_until_ready()
    t = time.perf_counter()
    for _ in range(20):
        q(params, bits, jax.random.PRNGKey(0)).block_until_ready()
    ms = (time.perf_counter() - t) / 20 * 1000
    assert ms < 50, f"{ms:.1f} ms per batch of 16: you are probably on the slow non-jit path"
    return f"{ms:.2f} ms per batch of 16 states (5 layers)"


def mixed_device():
    import jax, jax.numpy as jnp
    from qrl.noise import parse_noise
    from tasks.frozenlake.agent import init_params, make_q_fn, state_to_bits
    params = init_params(2, jax.random.PRNGKey(0))
    q = make_q_fn(2, parse_noise("preset=heron2026"))
    out = q(params, jnp.asarray(state_to_bits(jnp.arange(4))), jax.random.PRNGKey(0))
    assert out.shape == (4, 4)
    return "default.mixed + Heron-2026 preset"


def envs():
    import gymnasium as gym
    names = []
    for e in ["FrozenLake-v1", "CartPole-v1", "Acrobot-v1", "LunarLander-v3"]:
        env = gym.make(e); env.reset(seed=0); env.close()
        names.append(f"{e}(thr={env.spec.reward_threshold})")
    return " ".join(names)


def tiny_training_run():
    import csv
    from tasks.frozenlake.train import main
    from qrl.logging_utils import TRAIN_COLUMNS
    with tempfile.TemporaryDirectory() as d:
        out = main(["--seed", "0", "--episodes", "10", "--layers", "2", "--run-dir", d, "--quick"])
        csvs = list(Path(d, "train").glob("*.csv"))
        assert csvs, "no CSV written"
        rows = list(csv.DictReader(open(csvs[0])))
        assert list(rows[0].keys()) == TRAIN_COLUMNS, rows[0].keys()
    return f"{len(rows)} episodes logged, {out['n_params']} params"


if __name__ == "__main__":
    print("SC4055 smoke test")
    check("pinned versions", versions)
    check("jitted 4-qubit QNode (fast path)", jitted_qnode)
    check("density-matrix device + preset", mixed_device)
    check("gymnasium envs incl. Box2D", envs)
    check("FrozenLake 6-episode DQN + CSV", tiny_training_run)
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
