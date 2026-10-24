import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sc4055_project.qrl.noise import parse_noise
from sc4055_project.tasks.frozenlake.agent import (
    count_params,
    init_params,
    make_q_fn,
    state_to_bits,
)


def test_bits():
    assert state_to_bits(0).tolist() == [0, 0, 0, 0]
    assert state_to_bits(15).tolist() == [1, 1, 1, 1]
    assert state_to_bits(np.array([1, 2])).tolist() == [[1, 0, 0, 0], [0, 1, 0, 0]]


@pytest.mark.parametrize(
    "noise",
    [
        "none",
        "coherent=0.1",
        "depolarizing=0.01",
        "readout=0.02",
        "preset=heron2026",
        "shots=100",
    ],
)
def test_q_shape_and_jit(noise):
    params = init_params(2, jax.random.PRNGKey(0))
    q_fn = make_q_fn(2, parse_noise(noise), seed=0)
    bits = jnp.asarray(state_to_bits(np.arange(3)))
    q = q_fn(params, bits, jax.random.PRNGKey(1))
    assert q.shape == (3, 4)
    assert jnp.all(jnp.isfinite(q))
    assert count_params(params) == 2 * 4 * 3 + 2 * 4 + 4


def test_gradient_flows():
    params = init_params(2, jax.random.PRNGKey(0))
    q_fn = make_q_fn(2, parse_noise("none"), seed=0)
    bits = jnp.asarray(state_to_bits(np.arange(4)))
    g = jax.grad(lambda p: q_fn(p, bits, jax.random.PRNGKey(0)).sum())(params)
    assert float(jnp.abs(g["theta"]).sum()) > 0


def test_noisy_q_is_contracted():
    params = init_params(2, jax.random.PRNGKey(0))
    bits = jnp.asarray(state_to_bits(np.arange(16)))
    clean = make_q_fn(2, parse_noise("none"))(params, bits, jax.random.PRNGKey(0))
    noisy = make_q_fn(2, parse_noise("depolarizing=0.1"))(
        params, bits, jax.random.PRNGKey(0)
    )
    # Q = w z with w=1: depolarizing shrinks expvals toward 0 => |Q| shrinks
    assert float(jnp.abs(noisy).mean()) < float(jnp.abs(clean).mean())
