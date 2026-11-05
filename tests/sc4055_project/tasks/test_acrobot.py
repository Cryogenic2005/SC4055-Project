import csv
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sc4055_project.qrl.agent import count_params
from sc4055_project.qrl.logging_utils import EVAL_COLUMNS, TRAIN_COLUMNS
from sc4055_project.qrl.noise import parse_noise
from sc4055_project.tasks.acrobot.agent import TASK_ADAPTER, encode_obs, task_config
from sc4055_project.tasks.acrobot.evaluate import evaluate
from sc4055_project.tasks.acrobot.train import train


def test_encoder_preserves_six_features_and_rejects_invalid_input():
    obs = np.array([1, 0, -1, 0, 4, -9], dtype=np.float32)
    np.testing.assert_allclose(encode_obs(obs), np.arctan(obs))
    with pytest.raises(ValueError):
        encode_obs(np.zeros(4))
    with pytest.raises(ValueError):
        encode_obs([np.nan] * 6)


def test_parameter_matching_and_gradient_flow():
    cfg = task_config()
    key = jax.random.PRNGKey(1)
    params = TASK_ADAPTER.init_params(cfg, key)
    small = TASK_ADAPTER.init_params(task_config("mlp_small"), key)
    assert count_params(params) == 99
    assert count_params(small) == 103
    q = TASK_ADAPTER.make_q_fn(cfg, parse_noise("none"))
    x = jnp.asarray([encode_obs([1, 0, 1, 0, 0.5, -0.5])])
    assert q(params, x, key).shape == (1, 3)
    grads = jax.grad(lambda p: q(p, x, key).sum())(params)
    for name in ("theta", "lam", "w_out"):
        assert bool(jnp.all(jnp.isfinite(grads[name])))
        assert float(jnp.abs(grads[name]).sum()) > 0


@pytest.mark.parametrize("condition", ["depolarizing=0.01", "shots=10"])
def test_noisy_forward_pass(condition):
    cfg = task_config(n_layers=1)
    key = jax.random.PRNGKey(0)
    params = TASK_ADAPTER.init_params(cfg, key)
    q = TASK_ADAPTER.make_q_fn(cfg, parse_noise(condition))
    values = q(params, jnp.ones((1, 6)) * 0.1, key)
    assert values.shape == (1, 3)
    assert bool(jnp.all(jnp.isfinite(values)))


def test_train_reload_and_deterministic_evaluation(tmp_path):
    cfg = task_config(
        "mlp_small",
        n_episodes=2,
        max_steps=8,
        batch_size=2,
        learning_starts=2,
        train_every=1,
        checkpoint_every=1,
        eval_episodes=1,
        run_dir=str(tmp_path),
    )
    result = train(cfg)
    with open(result["best_checkpoint"], "rb") as handle:
        params = pickle.load(handle)
    first = evaluate(cfg, params, ["readout=0.02"], 2, str(tmp_path), 30000)
    second = evaluate(cfg, params, ["readout=0.02"], 2, str(tmp_path), 30000)
    assert first == second
    assert set(first) == {"none", "readout=0.02"}
    with (tmp_path / "train" / f"{cfg.run_name}.csv").open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == TRAIN_COLUMNS
        assert len(list(reader)) == 2
    with (tmp_path / "eval" / f"{cfg.run_name}.csv").open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == EVAL_COLUMNS
        rows = list(reader)
        assert len(rows) == 4  # A rerun must not append duplicate episodes.
        assert all(row["experiment"] == "B" for row in rows)


def test_noisy_training_is_experiment_c():
    assert task_config(noise="depolarizing=0.01").experiment == "C"


def test_compiled_probability_circuit_matches_shared_circuit():
    from sc4055_project.tasks.acrobot.inference import (
        probability_function,
        z_eigenvalues,
    )

    cfg = task_config()
    key = jax.random.PRNGKey(7)
    params = TASK_ADAPTER.init_params(cfg, key)
    probabilities = probability_function(cfg, params)
    signs = z_eigenvalues(cfg.n_qubits, cfg.n_actions)
    reference = TASK_ADAPTER.make_q_fn(cfg, parse_noise("none"))
    for obs in ([1, 0, 1, 0, 0, 0], [0.4, -0.6, -0.5, 0.7, 4, -8]):
        x = encode_obs(obs)
        analytic = np.asarray(probabilities(x)) @ signs * np.asarray(params["w_out"])
        np.testing.assert_allclose(
            analytic,
            np.asarray(reference(params, x[None], key)[0]),
            atol=2e-6,
            rtol=2e-6,
        )


def test_finite_shots_retain_sampling_variance_and_joint_correlations():
    from sc4055_project.tasks.acrobot.inference import sampled_q, z_eigenvalues

    # Bell-state marginals must be correlated, not independently sampled.
    probs = np.array([0.5, 0, 0, 0.5])
    signs = z_eigenvalues(2, 2)
    rng = np.random.default_rng(17)
    values = np.array(
        [sampled_q(probs, np.ones(2), signs, 10, rng) for _ in range(1000)]
    )
    np.testing.assert_array_equal(values[:, 0], values[:, 1])
    assert abs(values[:, 0].mean()) < 0.05
    assert 0.06 < values[:, 0].var() < 0.14


def test_eval_resumes_completed_episodes_and_rejects_protocol_changes(
    tmp_path, monkeypatch
):
    from sc4055_project.tasks.acrobot import evaluate as module

    cfg = task_config("mlp_small", max_steps=4)
    params = TASK_ADAPTER.init_params(cfg, jax.random.PRNGKey(0))
    first = module.evaluate(cfg, params, ["none"], 2, str(tmp_path))

    def must_not_run(*args, **kwargs):
        raise AssertionError("Completed evaluation was executed again")

    monkeypatch.setattr(module, "make_env", must_not_run)
    assert module.evaluate(cfg, params, ["none"], 2, str(tmp_path)) == first
    with pytest.raises(ValueError, match="protocol differs"):
        module.evaluate(cfg, params, ["none"], 3, str(tmp_path))
