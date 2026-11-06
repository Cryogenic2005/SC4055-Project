from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sc4055_project.qrl.agent import TaskAdapter
from sc4055_project.qrl.agent import init_params as shared_init_params
from sc4055_project.qrl.agent import make_q_fn as reference_q_fn
from sc4055_project.qrl.config import default_config_for


def task_config(model="vqc", **overrides):
    """Acrobot-specific choices, shared by training and all classical controls.

    Four VQC layers have 99 parameters; the 6->10->3 control has 103.
    Scale rewards only inside TD updates; logs and evaluation use raw rewards.
    These are starting hyperparameters, not a claim that the task is solved.
    """
    settings = dict(
        n_layers=4,
        n_episodes=600,
        batch_size=32,
        learning_starts=1000,
        buffer_size=50_000,
        train_every=4,
        target_update_every=100,
        epsilon_decay_episodes=400,
        epsilon_end=0.05,
        lr_output=0.01,
        reward_scale=0.01,
        double_dqn=True,
        grad_clip=1.0,
        checkpoint_every=50,
        eval_episodes=10,
    )
    settings.update(overrides)
    if model not in ("vqc", "mlp_small", "mlp_large"):
        raise ValueError("Acrobot supports vqc, mlp_small and mlp_large")
    return default_config_for("Acrobot-v1", model, **settings)


def encode_obs(obs):
    """The same six arctan features for VQC and both classical controls."""
    x = np.asarray(obs, dtype=np.float32)
    if x.shape != (6,) or not np.isfinite(x).all():
        raise ValueError("Acrobot observation must be six finite numbers")
    return np.arctan(x).astype(np.float32)


def init_params(cfg, key):
    if cfg.model != "mlp_small":
        return shared_init_params(cfg, key)
    # Parameter matching is to the default four-layer VQC only. Counts are logged
    # for any depth ablation; changing depth does not silently change this control.
    params = {}
    for i, (nin, nout) in enumerate(((6, 10), (10, 3)), 1):
        key, k = jax.random.split(key)
        scale = jnp.sqrt(2.0 / nin) if i == 1 else 0.01 * jnp.sqrt(1.0 / nin)
        params[f"w{i}"] = scale * jax.random.normal(k, (nin, nout))
        params[f"b{i}"] = jnp.zeros(nout)
    return params


def make_optimizer(cfg, params):
    rates = {"theta": cfg.lr_circuit, "lam": cfg.lr_input, "w_out": cfg.lr_output}
    transforms = {}
    for name in params:
        adam = optax.adam(rates.get(name, cfg.lr_mlp))
        transforms[name] = (
            optax.chain(optax.clip_by_global_norm(cfg.grad_clip), adam)
            if cfg.grad_clip > 0
            else adam
        )
    return optax.multi_transform(transforms, {name: name for name in params})


def make_q_fn(cfg, spec, **kwargs):
    if cfg.model == "vqc" and spec.kind == "depolarizing":
        from .pauli_backend import make_q_fn as pauli_q_fn

        return pauli_q_fn(cfg, spec, **kwargs)
    return reference_q_fn(cfg, spec, **kwargs)


TASK_ADAPTER = TaskAdapter(
    init_params=init_params,
    make_q_fn=make_q_fn,
    encode_obs=encode_obs,
    make_optimizer=make_optimizer,
    state_shape=(6,),
)
