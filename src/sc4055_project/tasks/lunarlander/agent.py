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
import optax

from sc4055_project.qrl.agent import TaskAdapter
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
        vqc_q_fn = make_vqc_q_fn(cfg, spec, seed=seed, jit=jit)

        def vqc_batched(params, obs, key):
            x = _encode_features(params, obs)

            # Delegate circuit evaluation and noise handling.
            return vqc_q_fn(params, x, key)

        return vqc_batched

    if cfg.model == "encoder_only":

        def encoder_batched(params, obs, key):
            x = _encode_features(params, obs)
            return x @ params["W"] + params["b"]

        return jax.jit(encoder_batched) if jit else encoder_batched

    if cfg.model in ("mlp_small", "mlp_large"):

        def mlp_batched(params, obs, key):
            return _forward_mlp(params, obs)

        return jax.jit(mlp_batched) if jit else mlp_batched

    raise ValueError(f"unknown model {cfg.model!r}")


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


# parameter-name -> optimizer group
_GROUP = {
    "theta": "circuit",
    "lam": "input",
    "w_out": "output",
    "enc_w": "encoder",
    "enc_b": "encoder",
    "W": "encoder",
    "b": "encoder",
}


def _optimizer_labels(params) -> dict:
    return {name: _GROUP.get(name, "mlp") for name in params}


def make_optimizer(cfg: Config, params):
    def adam(lr: float):
        if cfg.grad_clip > 0:
            return optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(lr))
        return optax.adam(lr)

    return optax.multi_transform(
        {
            "circuit": adam(cfg.lr_circuit),
            "input": adam(cfg.lr_input),
            "output": adam(cfg.lr_output),
            "encoder": adam(cfg.lr_mlp),
            "mlp": adam(cfg.lr_mlp),
        },
        _optimizer_labels(params),
    )


def build_update_fn(cfg: Config, q_fn, opt, *, jit: bool = True):
    def loss_fn(p, tgt, s, a, r, s2, d, k):
        k1, k2, k3 = jax.random.split(k, 3)
        q = q_fn(p, s, k1)
        q_sa = jnp.take_along_axis(q, a[:, None], axis=1)[:, 0]
        if cfg.double_dqn:
            next_action = jnp.argmax(q_fn(p, s2, k2), axis=1)
            q_next = jnp.take_along_axis(
                jax.lax.stop_gradient(q_fn(tgt, s2, k3)), next_action[:, None], axis=1
            )[:, 0]
        else:
            q_next = jax.lax.stop_gradient(q_fn(tgt, s2, k2)).max(axis=1)
        y = r + cfg.gamma * (1.0 - d) * q_next
        return optax.huber_loss(q_sa, y).mean()

    grad_fn = jax.value_and_grad(loss_fn)

    def update(p, o, tgt, batch, k):
        loss, g = grad_fn(p, tgt, *batch, k)
        upd, o = opt.update(g, o, p)
        return optax.apply_updates(p, upd), o, loss

    return jax.jit(update) if jit else update


TASK_ADAPTER = TaskAdapter(
    init_params=init_params,
    make_q_fn=make_q_fn,
    make_optimizer=make_optimizer,
    build_update_fn=build_update_fn,
    count_params=count_params,
    encode_obs=encode_obs,
)
