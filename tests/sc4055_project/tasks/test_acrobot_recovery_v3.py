import pickle
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sc4055_project.qrl.agent import make_q_fn as reference
from sc4055_project.qrl.dqn import train_agent
from sc4055_project.qrl.envs import make_env
from sc4055_project.qrl.noise import parse_noise
from sc4055_project.tasks.acrobot.agent import TASK_ADAPTER, task_config
from sc4055_project.tasks.acrobot.pauli_backend import make_q_fn


@pytest.mark.parametrize("reupload", [True, False])
def test_exact_noisy_values_and_gradients(reupload):
    cfg = task_config(data_reuploading=reupload)
    key = jax.random.PRNGKey(18)
    p = TASK_ADAPTER.init_params(cfg, key)
    x = jax.random.normal(key, (2, 6)) * 0.5
    spec = parse_noise("depolarizing=0.01")
    fast, old = make_q_fn(cfg, spec), reference(cfg, spec)
    np.testing.assert_allclose(fast(p, x, key), old(p, x, key), atol=1e-10, rtol=1e-9)
    weights = jnp.array([[0.7, -0.4, 1.2], [-0.1, 1.1, 0.3]])
    grads = [
        jax.jit(jax.grad(lambda p, fn=fn: (fn(p, x, key) * weights).sum()))(p)
        for fn in (fast, old)
    ]
    for a, b in zip(jax.tree.leaves(grads[0]), jax.tree.leaves(grads[1])):
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize(
    "model,noise", [("mlp_small", "none"), ("vqc", "depolarizing=0.01")]
)
def test_interrupted_training_matches_uninterrupted(tmp_path, model, noise):
    settings = dict(
        n_episodes=4,
        max_steps=5,
        batch_size=2,
        learning_starts=2,
        buffer_size=20,
        checkpoint_every=1,
        eval_episodes=1,
    )
    cfg = task_config(model, noise=noise, run_dir=str(tmp_path / "resume"), **settings)

    class Interrupt(gym.Wrapper):
        count = 0

        def reset(self, **kwargs):
            self.count += 1
            if self.count == 3:
                raise RuntimeError("simulated disconnect")
            return self.env.reset(**kwargs)

    env = Interrupt(make_env("Acrobot-v1", seed=0, max_steps=5))
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        train_agent(cfg, adapter=TASK_ADAPTER, env=env, resume_training=True)
    train_agent(cfg, adapter=TASK_ADAPTER, resume_training=True)
    full = task_config(model, noise=noise, run_dir=str(tmp_path / "full"), **settings)
    train_agent(full, adapter=TASK_ADAPTER, resume_training=True)

    def snapshot(c):
        with (Path(c.run_dir) / "checkpoints" / c.run_name / "resume.pkl").open(
            "rb"
        ) as handle:
            return pickle.load(handle)

    resumed, expected = snapshot(cfg), snapshot(full)
    for name in ("params", "target", "opt_state", "key"):
        for a, b in zip(
            jax.tree.leaves(resumed[name]), jax.tree.leaves(expected[name])
        ):
            np.testing.assert_array_equal(a, b)
    assert resumed["recent"] == expected["recent"]
    assert resumed["steps"] == expected["steps"]
