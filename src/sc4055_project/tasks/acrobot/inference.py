"""Compiled exact probabilities followed by finite-shot Z-basis sampling.

Evaluation only. Multinomial sampling retains joint Z correlations and uses
the requested shot budget; it does not substitute exact expectations for shots.
The underlying gate sequence matches qrl.agent._build_circuit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml


def probability_function(cfg, params):
    dev = qml.device("default.qubit", wires=cfg.n_qubits)

    @qml.qnode(dev, interface="jax")
    def probabilities(x):
        def encode(layer):
            for wire in range(cfg.n_qubits):
                qml.RX(jnp.pi * x[wire] * params["lam"][layer, wire], wires=wire)

        encode(0)
        for layer in range(cfg.n_layers):
            if cfg.data_reuploading and layer > 0:
                encode(layer)
            for wire in range(cfg.n_qubits):
                qml.RX(params["theta"][layer, wire, 0], wires=wire)
                qml.RY(params["theta"][layer, wire, 1], wires=wire)
                qml.RZ(params["theta"][layer, wire, 2], wires=wire)
            for wire in range(cfg.n_qubits):
                qml.CZ(wires=[wire, (wire + 1) % cfg.n_qubits])
        return qml.probs(wires=range(cfg.n_qubits))

    return jax.jit(probabilities)


def z_eigenvalues(n_qubits, n_actions):
    states = np.arange(2**n_qubits)
    return np.stack(
        [1 - 2 * ((states >> (n_qubits - 1 - i)) & 1) for i in range(n_actions)], axis=1
    )


def sampled_q(probabilities, weights, signs, shots, rng):
    p = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(p).all() or float(p.min()) < -1e-5:
        raise FloatingPointError("Invalid quantum probabilities")
    p = np.maximum(p, 0)
    p /= p.sum()
    counts = rng.multinomial(shots, p)
    return np.asarray(weights) * (counts @ signs / shots)
