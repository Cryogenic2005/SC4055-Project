"""FrozenLake VQC agent (owner: P2). Chen et al. 2020 / Skolik et al. 2022 style.

Circuit on 4 qubits:
    basis encoding   RX(pi * b_i) for the 4 bits of the state index
    L variational layers, each: RX, RY, RZ on every qubit, then a CZ ring
    (optional data re-uploading: the encoding is repeated before every layer)
    readout          <Z_i> for i in range(4)  -> one Q-value per action
    Q_a = w_out[a] * <Z_a>                      (trainable output scale)

    NOTE: Skolik's Q = w (<Z>+1)/2 is non-negative at init and cannot represent the
    negative shaped rewards (-0.2 hole, -0.01 step) without flipping the sign of w for
    every state; with it, training stalled at 0 on the deterministic map. Q = w <Z> fixed it.

Pure-functional: `init_params` builds the pytree, `make_q_fn` builds a jitted
batched function  q(params, bits, key) -> (batch, 4).  Noise is inserted with
qrl.noise.apply_noise after every gate when a density-matrix device is used.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml

from sc4055_project.qrl.noise import (
    NoiseSpec,
    apply_noise,
    apply_readout_noise,
    make_device,
)

N_QUBITS = 4
N_ACTIONS = 4


def init_params(n_layers: int, key, reupload: bool = True) -> dict:
    k1, k2 = jax.random.split(key)
    return {
        "theta": jax.random.uniform(
            k1, (n_layers, N_QUBITS, 3), minval=0.0, maxval=2 * jnp.pi
        ),
        "lam": jnp.ones((n_layers if reupload else 1, N_QUBITS)),  # input scaling
        "w_out": jnp.ones((N_ACTIONS,)),
    }


def count_params(params) -> int:
    return int(sum(p.size for p in jax.tree_util.tree_leaves(params)))


def build_circuit(
    n_layers: int,
    spec: NoiseSpec,
    reupload: bool = True,
    seed: int | None = None,
    diff_method: str | None = None,
):
    """Return a QNode f(bits, theta, lam) -> list of 4 expvals."""
    dev = make_device(N_QUBITS, spec, seed=seed)
    if diff_method is None:
        # Real hardware cannot backprop: with shots use parameter-shift, else backprop.
        diff_method = "parameter-shift" if dev.shots else "backprop"

    def encode(bits, lam_l):
        for i in range(N_QUBITS):
            qml.RX(jnp.pi * bits[i] * lam_l[i], wires=i)
            apply_noise(spec, [i])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def circuit(bits, theta, lam):
        encode(bits, lam[0])
        for l in range(n_layers):
            if reupload and l > 0:
                encode(bits, lam[l])
            for i in range(N_QUBITS):
                qml.RX(theta[l, i, 0], wires=i)
                apply_noise(spec, [i])
                qml.RY(theta[l, i, 1], wires=i)
                apply_noise(spec, [i])
                qml.RZ(theta[l, i, 2], wires=i)
                apply_noise(spec, [i])
            for i in range(N_QUBITS):
                qml.CZ(wires=[i, (i + 1) % N_QUBITS])
                apply_noise(spec, [i, (i + 1) % N_QUBITS])
        apply_readout_noise(spec, range(N_ACTIONS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_ACTIONS)]

    return circuit


def make_q_fn(
    n_layers: int,
    spec: NoiseSpec,
    reupload: bool = True,
    seed: int | None = None,
    jit: bool = True,
):
    """Batched Q-function. Coherent over-rotation is drawn per call from `key`.
    With shots the device samples internally, so we skip jit (device RNG is stateful)."""
    circuit = build_circuit(n_layers, spec, reupload, seed=seed)
    sigma = spec.coherent_sigma

    def single(bits, params, key):
        theta, lam = params["theta"], params["lam"]
        if sigma > 0:
            k1, k2 = jax.random.split(key)
            theta = theta + sigma * jax.random.normal(k1, theta.shape)
            lam = (
                lam + sigma * jax.random.normal(k2, lam.shape) / jnp.pi
            )  # perturb encoding angles too
        z = jnp.stack(circuit(bits, theta, lam))
        return params["w_out"] * z

    def batched(params, bits, key):
        keys = jax.random.split(key, bits.shape[0])
        return jax.vmap(single, in_axes=(0, None, 0))(bits, params, keys)

    def batched_shots(params, bits, key):
        # Finite shots: the device samples internally, so neither jit nor vmap applies.
        keys = jax.random.split(key, bits.shape[0])
        return jnp.stack([
            single(bits[i], params, keys[i]) for i in range(bits.shape[0])
        ])

    if spec.shots:
        return batched_shots
    return jax.jit(batched) if jit else batched


def state_to_bits(s: int | np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.int64)
    return np.stack([(s >> i) & 1 for i in range(N_QUBITS)], axis=-1).astype(np.float32)
