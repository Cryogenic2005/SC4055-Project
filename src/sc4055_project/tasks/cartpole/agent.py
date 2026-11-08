from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sc4055_project.qrl.agent import (
    TaskAdapter,
    _build_circuit,
    _forward_mlp,
    _perturb_parameters,
    init_params,
)
from sc4055_project.qrl.config import Config
from sc4055_project.qrl.noise import NoiseSpec

# CartPole observations are continuous and have very different dynamic ranges.
# Dividing by these ranges keeps each feature in roughly [-1, 1] before the
# arctan feature map, which prevents the rotation encoding from saturating early.
CARTPOLE_OBS_SCALE = np.array([2.4, 2.0, 0.209, 2.0], dtype=np.float32)


def encode_obs(obs: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32)
    x = np.clip(x / CARTPOLE_OBS_SCALE, -1.0, 1.0)
    return np.arctan(x).astype(np.float32)


def _init_params(cfg: Config, key, **kwargs) -> dict[str, Any]:
    """Return a pytree of trainable parameters for the model.

    Args:
        cfg: Config object with model hyperparameters.
        key: JAX PRNGKey for random initialization.
        **kwargs: Config overrides (n_layers, n_qubits, n_actions, data_reuploading).

    Returns:
        A dictionary representing the model parameters.
    """
    n_actions = kwargs.get("n_actions", cfg.n_actions)

    params = init_params(cfg, key, **kwargs)
    if cfg.model == "vqc":
        params["b_out"] = jnp.zeros((n_actions,))

    return params


def _make_optimizer(cfg: Config, params=None):
    """Default optimizer matches the shared VQC and MLP models."""
    labels = {
        "theta": "circuit",
        "lam": "input",
        "w_out": "output_w",
        "b_out": "output_b",
    }
    if isinstance(params, dict):
        labels = {k: labels.get(k, "mlp") for k in params}

    def _adam(lr):
        adam = optax.adam(lr)
        if cfg.grad_clip <= 0 or cfg.grad_clip is None:
            return adam

        return optax.chain(optax.clip_by_global_norm(cfg.grad_clip), adam)

    return optax.multi_transform(
        {
            "circuit": _adam(cfg.lr_circuit),
            "input": _adam(cfg.lr_input),
            "output_w": _adam(cfg.lr_output),
            "output_b": _adam(cfg.lr_output_b),
            "mlp": _adam(cfg.lr_mlp),
        },
        labels,
    )


def _make_q_fn(cfg: Config, spec: NoiseSpec, *, jit: bool = True, **kwargs):
    if cfg.model in ("mlp_small", "mlp_large"):

        def batched_mlp(params, x, key):
            del key
            return _forward_mlp(params, x)

        return jax.jit(batched_mlp) if jit else batched_mlp

    circuit = _build_circuit(
        n_layers=kwargs.get("n_layers", cfg.n_layers),
        n_qubits=kwargs.get("n_qubits", cfg.n_qubits),
        n_actions=kwargs.get("n_actions", cfg.n_actions),
        spec=spec,
        reupload=kwargs.get("data_reuploading", cfg.data_reuploading),
        seed=kwargs.get("seed", cfg.seed),
    )

    sigma = spec.coherent_sigma

    def single(params, x, key):
        theta, lam = _perturb_parameters(params, key, sigma)
        z = jnp.stack(circuit(x, theta, lam))
        return params["w_out"] * z + params["b_out"]

    def batched(params, x, key):
        keys = jax.random.split(key, x.shape[0])
        return jax.vmap(single, in_axes=(None, 0, 0))(params, x, keys)

    def batched_shots(params, x, key):
        # Finite-shot devices sample internally. Avoid vmap and JIT.
        keys = jax.random.split(key, x.shape[0])
        return jnp.stack([single(params, x[i], keys[i]) for i in range(x.shape[0])])

    if spec.shots:
        return batched_shots

    return jax.jit(batched) if jit else batched


TASK_ADAPTER = TaskAdapter(
    encode_obs=encode_obs,
    state_shape=(4,),
    init_params=_init_params,
    make_optimizer=_make_optimizer,
    make_q_fn=_make_q_fn,
)
