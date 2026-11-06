"""Exact Pauli-transfer simulation for analytic local depolarizing noise.

Stores all 4**n Pauli expectations, not sampled trajectories. Local isotropic
depolarization multiplies X/Y/Z by 1-4p/3. CZ is a signed permutation. This is
the same channel as the reference density-matrix circuit, in a real basis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _cz_map(n, a, b):
    paulis = [
        np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.diag([1, -1]),
    ]
    basis = [np.kron(x, y) for x in paulis for y in paulis]
    cz = np.diag([1, 1, 1, -1])
    mapping, signs = [], []
    for p in basis:
        coefficients = np.array([np.trace(q @ cz @ p @ cz).real / 4 for q in basis])
        k = int(np.argmax(abs(coefficients)))
        mapping.append(k)
        signs.append(coefficients[k])
    index = np.arange(4**n)
    wa, wb = 4 ** (n - 1 - a), 4 ** (n - 1 - b)
    da, db = index // wa % 4, index // wb % 4
    pair = da * 4 + db
    mapped = np.asarray(mapping)[pair]
    permutation = index + (mapped // 4 - da) * wa + (mapped % 4 - db) * wb
    return permutation, np.asarray(signs, np.float32)[pair], da, db


def _rotation(angle, axis):
    c, s = jnp.cos(angle), jnp.sin(angle)
    matrix = jnp.eye(4)
    i, k = ((2, 3), (3, 1), (1, 2))[axis]
    return matrix.at[i, i].set(c).at[k, k].set(c).at[i, k].set(-s).at[k, i].set(s)


def make_q_fn(cfg, spec, *, jit=True, **kwargs):
    if (
        cfg.model != "vqc"
        or spec.kind != "depolarizing"
        or spec.shots
        or spec.coherent_sigma
        or spec.readout_p
        or spec.amp_damp
        or spec.t1_us is not None
    ):
        raise ValueError("Pauli backend supports analytic depolarizing VQC only")
    n, layers = cfg.n_qubits, cfg.n_layers
    attenuation = 1 - 4 * spec.p1 / 3
    maps = []
    for a in range(n):
        permutation, signs, da, db = _cz_map(n, a, (a + 1) % n)
        scale = (1 - 4 * spec.p2 / 3) ** ((da != 0).astype(int) + (db != 0).astype(int))
        maps.append((jnp.asarray(permutation), jnp.asarray(signs * scale)))
    initial = np.array([1.0, 0.0, 0.0, 1.0], np.float32)
    for _ in range(n - 1):
        initial = np.kron(initial, [1.0, 0.0, 0.0, 1.0]).astype(np.float32)
    initial = jnp.asarray(initial.reshape((4,) * n))
    readout = jnp.asarray([3 * 4 ** (n - 1 - a) for a in range(cfg.n_actions)])

    def single(params, x):
        def layer_step(state, inputs):
            layer, theta, lam = inputs
            encode = jnp.logical_or(layer == 0, cfg.data_reuploading)
            for wire in range(n):
                angle = theta[wire, 0] + jnp.where(
                    encode, jnp.pi * x[wire] * lam[wire], 0.0
                )
                rotation = (
                    _rotation(theta[wire, 2], 2)
                    @ _rotation(theta[wire, 1], 1)
                    @ _rotation(angle, 0)
                )
                # Depolarization commutes with every unitary on this wire.
                damping = jnp.where(encode, attenuation**4, attenuation**3)
                rotation = (
                    rotation * jnp.array([1.0, damping, damping, damping])[:, None]
                )
                state = jnp.moveaxis(
                    jnp.tensordot(rotation, state, axes=(1, wire)), 0, wire
                )
            flat = state.reshape(-1)
            for permutation, factors in maps:
                flat = flat[permutation] * factors
            return flat.reshape((4,) * n), None

        lam = (
            params["lam"]
            if cfg.data_reuploading
            else jnp.repeat(params["lam"], layers, axis=0)
        )
        final, _ = jax.lax.scan(
            layer_step,
            initial.astype(params["theta"].dtype),
            (jnp.arange(layers), params["theta"], lam),
        )
        return final.reshape(-1)[readout] * params["w_out"]

    def batched(params, x, key):
        del key
        return jax.vmap(single, in_axes=(None, 0))(params, x)

    return jax.jit(batched) if jit else batched
