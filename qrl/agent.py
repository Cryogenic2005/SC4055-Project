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


def init_params(cfg, key):
    raise NotImplementedError("P1: shared QAgent not implemented yet; see tasks/frozenlake/agent.py")


def q_values(params, x, spec, key=None):
    raise NotImplementedError("P1: shared QAgent not implemented yet; see tasks/frozenlake/agent.py")


def count_params(params) -> int:
    import jax
    return int(sum(p.size for p in jax.tree_util.tree_leaves(params)))
