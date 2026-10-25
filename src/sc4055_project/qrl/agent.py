"""Shared model interface. OWNER: P1 (core QAgent).

Every model is a pair of pure functions so that the same DQN loop can train a VQC,
an MLP or a hybrid:

    params = init_params(cfg, key)            # a JAX pytree
    q      = q_values(params, x, spec, key)   # x: encoded obs (batch, d) -> (batch, n_actions)

`spec` is a qrl.noise.NoiseSpec. Channel noise is inserted with qrl.noise.apply_noise
inside the QNode; coherent noise is applied with qrl.noise.perturb_angles outside it.

A working single-task reference implementation lives in tasks/frozenlake/agent.py
(P2). P1 should generalise that into this module (data re-uploading, trainable
input scaling, RX/RY/RZ layers with a CZ ring, Z readouts with trainable output
weights, and the MLP / hybrid variants), keeping the two-function interface above
so the P2 code, the P3 harness and the P4 baselines all plug in unchanged.
"""

from typing import Any

import jax
import jax.numpy as jnp

from .config import Config


def init_params(cfg: Config, key, **kwargs) -> dict[str, Any]:
    """Return a pytree of trainable parameters for the model.

    Args:
        cfg: Config object with model hyperparameters.
        key: JAX PRNGKey for random initialization.
        **kwargs: Config overrides (n_layers, n_qubits, n_actions, data_reuploading).

    Returns:
        A dictionary representing the model parameters.
    """
    n_layers = kwargs.get("n_layers", cfg.n_layers)
    n_qubits = kwargs.get("n_qubits", cfg.n_qubits)
    n_actions = kwargs.get("n_actions", cfg.n_actions)
    data_reuploading = kwargs.get("data_reuploading", cfg.data_reuploading)

    # Should be set through __post_init__ in Config
    assert n_qubits is not None, "cfg.n_qubits must be set for init_params"

    k1, _k2 = jax.random.split(key)
    return {
        "theta": jax.random.uniform(
            k1, (n_layers, n_qubits, 3), minval=0.0, maxval=2 * jnp.pi
        ),
        "lam": jnp.ones((n_layers if data_reuploading else 1, n_qubits)),
        "w_out": jnp.ones((n_actions,)),
    }


def q_values(params, x, spec, key=None):
    raise NotImplementedError(
        "P1: shared QAgent not implemented yet; see tasks/frozenlake/agent.py"
    )


def count_params(params) -> int:
    return int(sum(p.size for p in jax.tree_util.tree_leaves(params)))
