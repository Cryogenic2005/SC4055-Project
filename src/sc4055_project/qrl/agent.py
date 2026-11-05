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

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import pennylane as qml

from .config import Config
from .noise import NoiseSpec, apply_noise, apply_readout_noise, make_device


def _identity(x):
    return x


@dataclass(frozen=True)
class TaskAdapter:
    """Hook points for task-specific models and training behaviour.

    The common Q-learning loop only needs a few pieces from a task: how to create the
    parameter pytree, how to build the Q-function, and (optionally) how to build a
    custom optimizer / update step. Tasks can override these without copying the full
    engineering code from `qrl.dqn`.
    """

    init_params: Callable[[Config, jax.Array], Any] = field(default=None)  # type: ignore[assignment]
    make_q_fn: Callable[..., Any] = field(default=None)  # type: ignore[assignment]
    make_optimizer: Callable[..., Any] | None = None
    build_update_fn: Callable[..., Any] | None = None
    count_params: Callable[[Any], int] = field(default=None)  # type: ignore[assignment]
    encode_obs: Callable[[Any], Any] = field(default=_identity)
    reward_augment_fn: Callable[[float, bool], float] | None = None
    state_shape: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.init_params is None:
            object.__setattr__(self, "init_params", init_params)
        if self.make_q_fn is None:
            object.__setattr__(self, "make_q_fn", make_q_fn)
        if self.count_params is None:
            object.__setattr__(self, "count_params", count_params)


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

    if cfg.model == "mlp_small":
        return _mlp_init(key, (cfg.obs_dim, 8, n_actions))
    if cfg.model == "mlp_large":
        return _mlp_init(key, (cfg.obs_dim, *cfg.mlp_large_hidden, n_actions))

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


def make_q_fn(cfg: Config, spec: NoiseSpec, *, jit: bool = True, **kwargs):
    """Create a batched Q-value function.

    Args:
        cfg: Config object with model hyperparameters.
        spec: Noise configuration.
        jit: Whether to JIT-compile the batched function.

    Kwargs:
        Overrides for config parameters

    Returns:
        Function q_fn(params, x, key) -> (batch, n_actions).
    """
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
        return params["w_out"] * z

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


def count_params(params) -> int:
    return int(sum(p.size for p in jax.tree_util.tree_leaves(params)))


def _build_circuit(
    n_layers: int,
    n_qubits: int,
    n_actions: int,
    spec: NoiseSpec,
    reupload: bool = True,
    seed: int | None = None,
    diff_method: str | None = None,
):
    """Construct the QNode for one encoded observation."""
    assert n_layers > 0, "n_layers must be positive."
    assert n_qubits >= 2, "n_qubits must be at least 2."
    assert 1 <= n_actions <= n_qubits, "n_actions must be between 1 and n_qubits."

    dev = make_device(n_qubits, spec, seed=seed)

    if diff_method is None:
        diff_method = "parameter-shift" if dev.shots else "backprop"

    def encode(bits, lam_l):
        for i in range(n_qubits):
            qml.RX(jnp.pi * bits[i] * lam_l[i], wires=i)
            apply_noise(spec, [i])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def circuit(x, theta, lam):
        encode(x, lam[0])

        for layer in range(n_layers):
            if reupload and layer > 0:
                encode(x, lam[layer])

            for qubit in range(n_qubits):
                qml.RX(theta[layer, qubit, 0], wires=qubit)
                apply_noise(spec, [qubit])

                qml.RY(theta[layer, qubit, 1], wires=qubit)
                apply_noise(spec, [qubit])

                qml.RZ(theta[layer, qubit, 2], wires=qubit)
                apply_noise(spec, [qubit])

            for qubit in range(n_qubits):
                neighbour = (qubit + 1) % n_qubits
                qml.CZ(wires=[qubit, neighbour])
                apply_noise(spec, [qubit, neighbour])

        apply_readout_noise(spec, range(n_actions))
        return [qml.expval(qml.PauliZ(qubit)) for qubit in range(n_actions)]

    return circuit


def _perturb_parameters(params, key, sigma: float):
    """Apply coherent over-rotation noise to circuit and encoding angles."""
    if sigma <= 0:
        return params["theta"], params["lam"]

    k_theta, k_lam = jax.random.split(key)

    theta_noise = jax.random.normal(k_theta, params["theta"].shape) * sigma
    lam_noise = jax.random.normal(k_lam, params["lam"].shape) * sigma / jnp.pi

    theta = params["theta"] + theta_noise
    lam = params["lam"] + lam_noise

    return theta, lam


def _mlp_init(key, sizes: tuple[int, ...]) -> dict[str, jnp.ndarray]:
    params: dict[str, jnp.ndarray] = {}
    for i, (n_in, n_out) in enumerate(itertools.pairwise(sizes), start=1):
        key, k = jax.random.split(key)
        std = (
            jnp.sqrt(2.0 / n_in) if i < len(sizes) - 1 else 0.01 * jnp.sqrt(1.0 / n_in)
        )
        params[f"w{i}"] = std * jax.random.normal(k, (n_in, n_out))
        params[f"b{i}"] = jnp.zeros((n_out,))
    return params


def _forward_mlp(params, x):
    y = x
    n_layers = sum(1 for k in params if k.startswith("w"))
    for i in range(1, n_layers + 1):
        y = y @ params[f"w{i}"] + params[f"b{i}"]
        if i < n_layers:
            y = jax.nn.relu(y)
    return y
