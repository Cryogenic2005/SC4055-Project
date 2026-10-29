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

import numpy as np

N_QUBITS = 4


def state_to_bits(s: int | np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.int64)
    return np.stack([(s >> i) & 1 for i in range(N_QUBITS)], axis=-1).astype(np.float32)
