"""LunarLander-v3 agent (owner: P4). Hybrid VQC plus its ablations and MLP baselines.

The plan's hybrid model, ``model="vqc"``:

    observation (8 floats)
      -> trainable linear encoder 8 -> 4, then tanh  (bounded in (-1, 1))
      -> 4-qubit VQC: RX(pi * x_i * lam_i) angle encoding,
         L layers of RX, RY, RZ on every qubit with a CZ ring
         (data re-uploading repeats the encoding before every layer),
      -> <Z_i> for i in range(4), Q_a = w_out[a] * <Z_a>.

The other ``qrl.config.MODELS`` variants share the same Q-function signature so the
DQN loop, the evaluation harness and the analysis scripts never branch on the model:

    encoder_only   the same 8 -> 4 encoder, then a trainable 4 -> 4 linear layer.
                   This is the plan's ablation: it asks whether the VQC carries any
                   weight beyond the classical encoder.
    mlp_small      parameter-matched MLP (8 -> 8 -> 4, 108 params vs 104 for the
                   4-layer hybrid), so "VQC vs same-size classical net" is fair.
    mlp_large      2 x 64 ReLU MLP (8 -> 64 -> 64 -> 4, 4996 params).

Noise only enters the quantum layer, so ``encoder_only`` and the MLPs ignore the
``NoiseSpec``; evaluating them under the noise menu would repeat the same number.

Pure-functional, as required by ``qrl/agent.py``: ``init_params`` builds a pytree and
``make_q_fn`` returns  q(params, obs, key) -> (batch, n_actions).  Channel noise is
inserted through ``qrl.noise.apply_noise`` after every gate, coherent over-rotation is
drawn from ``key``, and finite shots fall back to a Python loop (never jit, never
backprop) exactly like the FrozenLake reference.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from sc4055_project.qrl.agent import make_q_fn as make_vqc_q_fn
from sc4055_project.qrl.config import Config
from sc4055_project.qrl.noise import (
    NoiseSpec,
)

N_QUBITS = 4
N_ACTIONS = 4
OBS_DIM = 8

# The readout <Z> is bounded, so ``w_out`` sets the Q-value range. ``train.py`` scales the
# TD target by Config.reward_scale (0.01 for LunarLander), which puts rewards in [-1, 1];
# the initial output weight therefore stays O(1). Raising it (e.g. for --reward-scale 1.0)
# makes early targets far below the initial Q-values and destabilises training.
W_OUT_INIT = 1.0


# --------------------------------------------------------------------------- #
# per-model parameter trees
# --------------------------------------------------------------------------- #
def _mlp_init(key, sizes: tuple[int, ...]) -> dict:
    params: dict[str, jnp.ndarray] = {}
    for i, (n_in, n_out) in enumerate(itertools.pairwise(sizes), start=1):
        key, k = jax.random.split(key)
        # He-style scaling for the ReLU layers; the final layer starts small.
        std = (
            jnp.sqrt(2.0 / n_in) if i < len(sizes) - 1 else 0.01 * jnp.sqrt(1.0 / n_in)
        )
        params[f"w{i}"] = std * jax.random.normal(k, (n_in, n_out))
        params[f"b{i}"] = jnp.zeros((n_out,))
    return params


def init_params(cfg: Config, key) -> dict:
    """Build the pytree for ``cfg.model``. See the module docstring for the shapes."""
    k_enc, k_theta = jax.random.split(key)
    if cfg.model == "vqc":
        n_layers = cfg.n_layers
        reupload = cfg.data_reuploading
        return {
            "enc_w": jax.random.uniform(
                k_enc, (OBS_DIM, N_QUBITS), minval=-1.0, maxval=1.0
            )
            * jnp.sqrt(6.0 / (OBS_DIM + N_QUBITS)),
            "enc_b": jnp.zeros((N_QUBITS,)),
            "theta": jax.random.uniform(
                k_theta, (n_layers, N_QUBITS, 3), minval=0.0, maxval=2 * jnp.pi
            ),
            "lam": jnp.ones((n_layers if reupload else 1, N_QUBITS)),
            "w_out": jnp.full((N_ACTIONS,), W_OUT_INIT),
        }
    if cfg.model == "encoder_only":
        return {
            "enc_w": jax.random.uniform(
                k_enc, (OBS_DIM, N_QUBITS), minval=-1.0, maxval=1.0
            )
            * jnp.sqrt(6.0 / (OBS_DIM + N_QUBITS)),
            "enc_b": jnp.zeros((N_QUBITS,)),
            "W": 0.01 * jax.random.normal(k_theta, (N_QUBITS, N_ACTIONS)),
            "b": jnp.zeros((N_ACTIONS,)),
        }
    if cfg.model == "mlp_small":
        return _mlp_init(key, (OBS_DIM, 8, N_ACTIONS))
    if cfg.model == "mlp_large":
        return _mlp_init(key, (OBS_DIM, *cfg.mlp_large_hidden, N_ACTIONS))
    raise ValueError(f"unknown model {cfg.model!r}")


def count_params(params) -> int:
    return int(sum(p.size for p in jax.tree_util.tree_leaves(params)))


# --------------------------------------------------------------------------- #
# Q-functions
# --------------------------------------------------------------------------- #
def _encode_features(params, obs):
    """8 -> 4 classical encoder: tanh keeps the angle-encoding inputs bounded."""
    return jnp.tanh(obs @ params["enc_w"] + params["enc_b"])


def _forward_mlp(params, obs):
    x = obs
    n_layers = sum(1 for k in params if k.startswith("w"))
    for i in range(1, n_layers + 1):
        x = x @ params[f"w{i}"] + params[f"b{i}"]
        if i < n_layers:
            x = jax.nn.relu(x)
    return x


def make_q_fn(
    cfg: Config,
    spec: NoiseSpec,
    seed: int | None = None,
    jit: bool = True,
):
    """Batched Q-function q(params, obs, key) -> (batch, n_actions).

    ``obs`` is the raw 8-float observation (``qrl.envs.encode_obs`` is the identity for
    LunarLander). ``key`` drives coherent over-rotation and is ignored by the classical
    models. Finite shots disable jit/vmap because the device samples internally.
    """
    if cfg.model == "vqc":
        circuit = make_vqc_q_fn(cfg, spec, seed=seed, jit=jit)
        sigma = spec.coherent_sigma

        def single(obs, params, key):
            x = _encode_features(params, obs)
            theta, lam = params["theta"], params["lam"]
            if sigma > 0:
                k1, k2 = jax.random.split(key)
                theta = theta + sigma * jax.random.normal(k1, theta.shape)
                lam = lam + sigma * jax.random.normal(k2, lam.shape) / jnp.pi
            return params["w_out"] * jnp.stack(circuit(x, theta, lam))

        def batched(params, obs, key):
            keys = jax.random.split(key, obs.shape[0])
            return jax.vmap(single, in_axes=(0, None, 0))(obs, params, keys)

        def batched_shots(params, obs, key):
            keys = jax.random.split(key, obs.shape[0])
            return jnp.stack([
                single(obs[i], params, keys[i]) for i in range(obs.shape[0])
            ])

        if spec.shots:
            return batched_shots
        return jax.jit(batched) if jit else batched

    if cfg.model == "encoder_only":

        def batched(params, obs, key):
            x = _encode_features(params, obs)
            return x @ params["W"] + params["b"]

    elif cfg.model in ("mlp_small", "mlp_large"):

        def batched(params, obs, key):
            return _forward_mlp(params, obs)

    else:
        raise ValueError(f"unknown model {cfg.model!r}")

    return jax.jit(batched) if jit else batched


# LunarLander observations arrive on wildly different scales (x,y in +-1.5 but the
# angle in +-pi and velocities in +-5). Fed raw into a tanh encoder, most of the state
# space saturates the squashing: 19% of encoder units sat at |tanh| > 0.99 and distinct
# states collapsed onto the same four features. Dividing by the documented range puts
# every input in roughly [-1, 1] so tanh operates in its linear region. Applied in
# encode_obs(), i.e. before every model, so the classical baselines see the same inputs.
OBS_SCALE = np.array([1.5, 1.5, 5.0, 5.0, np.pi, 5.0, 1.0, 1.0], dtype=np.float32)


def encode_obs(obs) -> np.ndarray:
    """Raw LunarLander observation -> float32 (8,), scaled to roughly [-1, 1].

    See OBS_SCALE: without this the tanh encoder saturates and the VQC never sees a
    usable feature vector.
    """
    return np.asarray(obs, dtype=np.float32) / OBS_SCALE
