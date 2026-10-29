import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sc4055_project.qrl.config import default_config_for
from sc4055_project.qrl.logging_utils import TRAIN_COLUMNS
from sc4055_project.qrl.noise import parse_noise
from sc4055_project.tasks.lunarlander.agent import (
    count_params,
    encode_obs,
    init_params,
    make_q_fn,
)
from sc4055_project.tasks.lunarlander.train import train


def _cfg(model="vqc", **kw):
    return default_config_for("LunarLander-v3", model, **kw)


def _obs(batch=3, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(-1.5, 1.5, (batch, 8)), dtype=jnp.float32)


def test_encode_obs():
    v = encode_obs(np.zeros(8))
    assert v.shape == (8,)
    assert v.dtype == np.float32


@pytest.mark.parametrize(
    "model,n_params",
    [
        ("vqc", 104),
        ("encoder_only", 56),
        ("mlp_small", 108),
        ("mlp_large", 4996),
    ],
)
def test_param_counts(model, n_params):
    params = init_params(_cfg(model), jax.random.PRNGKey(0))
    assert count_params(params) == n_params


def test_mlp_small_is_parameter_matched():
    hybrid = count_params(init_params(_cfg("vqc"), jax.random.PRNGKey(0)))
    small = count_params(init_params(_cfg("mlp_small"), jax.random.PRNGKey(0)))
    assert abs(small - hybrid) / hybrid < 0.05


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
    cfg = _cfg()
    params = init_params(cfg, jax.random.PRNGKey(0))
    q_fn = make_q_fn(cfg, parse_noise(noise), seed=0)
    obs = _obs(3)
    q = q_fn(params, obs, jax.random.PRNGKey(1))
    assert q.shape == (3, 4)
    assert bool(jnp.all(jnp.isfinite(q)))


def test_gradient_flows_through_encoder_and_circuit():
    cfg = _cfg()
    params = init_params(cfg, jax.random.PRNGKey(0))
    q_fn = make_q_fn(cfg, parse_noise("none"), seed=0)
    obs = _obs(4)
    g = jax.grad(lambda p: q_fn(p, obs, jax.random.PRNGKey(0)).sum())(params)
    assert float(jnp.abs(g["theta"]).sum()) > 0
    assert float(jnp.abs(g["enc_w"]).sum()) > 0


def test_noisy_q_is_contracted():
    cfg = _cfg()
    params = init_params(cfg, jax.random.PRNGKey(0))
    obs = _obs(16)
    clean = make_q_fn(cfg, parse_noise("none"))(params, obs, jax.random.PRNGKey(0))
    noisy = make_q_fn(cfg, parse_noise("depolarizing=0.1"))(
        params, obs, jax.random.PRNGKey(0)
    )
    assert float(jnp.abs(noisy).mean()) < float(jnp.abs(clean).mean())


@pytest.mark.parametrize("model", ["encoder_only", "mlp_small", "mlp_large"])
def test_classical_models_shape_and_ignore_quantum_noise(model):
    cfg = _cfg(model)
    params = init_params(cfg, jax.random.PRNGKey(0))
    obs = _obs(5)
    clean = make_q_fn(cfg, parse_noise("none"))(params, obs, jax.random.PRNGKey(0))
    noisy = make_q_fn(cfg, parse_noise("depolarizing=0.1"))(
        params, obs, jax.random.PRNGKey(0)
    )
    assert clean.shape == (5, 4)
    assert bool(jnp.allclose(clean, noisy))


def test_encoder_only_is_key_independent():
    cfg = _cfg("encoder_only")
    params = init_params(cfg, jax.random.PRNGKey(0))
    q_fn = make_q_fn(cfg, parse_noise("coherent=0.2"))
    obs = _obs(2)
    a = q_fn(params, obs, jax.random.PRNGKey(0))
    b = q_fn(params, obs, jax.random.PRNGKey(7))
    assert bool(jnp.allclose(a, b))


def test_unknown_model_rejected():
    cfg = _cfg()
    cfg.model = "no_such_model"
    with pytest.raises(ValueError):
        init_params(cfg, jax.random.PRNGKey(0))


def test_tiny_training_run_writes_spec_columns(tmp_path: Path):
    cfg = _cfg(
        n_episodes=2,
        max_steps=32,
        learning_starts=4,
        batch_size=4,
        eval_episodes=0,
        checkpoint_every=0,
        log_every=1,
        run_dir=str(tmp_path),
    )
    out = train(cfg)
    assert out["n_params"] == 104

    csvs = sorted(Path(tmp_path, "train").glob("*.csv"))
    assert len(csvs) == 1
    with open(csvs[0], newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert list(rows[0].keys()) == TRAIN_COLUMNS
    assert rows[0]["task"] == "LunarLander-v3"
    assert rows[0]["model"] == "vqc"
    assert rows[0]["experiment"] == "A"
    # the checkpoint's config.json round-trips (incl. reward_scale)
    from sc4055_project.qrl.config import Config

    ckpt_cfg = Config.load(Path(tmp_path, "checkpoints", cfg.run_name, "config.json"))
    assert ckpt_cfg.reward_scale == cfg.reward_scale
