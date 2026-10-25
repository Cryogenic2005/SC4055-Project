"""DQN training loop for LunarLander-v3 (owner: P4).

Same shape as the FrozenLake reference (``tasks/frozenlake/train.py``): replay buffer,
epsilon-greedy behaviour policy, a period-50 target network, per-group Adam learning
rates (circuit / input scaling / output / encoder) and the shared loggers, so the
LunarLander CSVs drop straight into ``analysis/analyze.py``.

Two LunarLander-specific choices, both recorded in ``config.json``:

* ``reward_scale`` (default 0.01) rescales the TD target only. LunarLander pays
  +-100 on the terminal step while the readout is a bounded <Z>, so an unscaled target
  would need an output weight two orders of magnitude above its initial value. The
  *logged* episode return is always the raw Gymnasium return, so the analysis sees the
  same numbers as every other task.
* ``w_out`` starts at 10 (``agent.W_OUT_INIT``) rather than 1, which puts the initial
  Q-values in the range the scaled rewards actually visit.
"""

from __future__ import annotations

import pickle
import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sc4055_project.qrl.config import Config
from sc4055_project.qrl.envs import make_env
from sc4055_project.qrl.logging_utils import TrainLogger
from sc4055_project.qrl.noise import parse_noise

from .agent import count_params, encode_obs, init_params, make_q_fn

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
    # Separate learning rates per parameter group (TFQ / Skolik recipe).
    return optax.multi_transform(
        {
            "circuit": optax.adam(cfg.lr_circuit),
            "input": optax.adam(cfg.lr_input),
            "output": optax.adam(cfg.lr_output),
            "encoder": optax.adam(cfg.lr_mlp),
            "mlp": optax.adam(cfg.lr_mlp),
        },
        _optimizer_labels(params),
    )


class Replay:
    def __init__(self, size: int, obs_dim: int, rng: np.random.Generator):
        self.s = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros(size, np.int32)
        self.r = np.zeros(size, np.float32)
        self.s2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros(size, np.float32)
        self.n = 0
        self.i = 0
        self.size = size
        self.rng = rng

    def add(self, s, a, r, s2, d):
        (
            self.s[self.i],
            self.a[self.i],
            self.r[self.i],
            self.s2[self.i],
            self.d[self.i],
        ) = (s, a, r, s2, d)
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, b: int):
        idx = self.rng.integers(0, self.n, size=b)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


def greedy_eval(cfg: Config, q_fn, params, env, n_episodes: int, key) -> float:
    """Mean raw return of the greedy policy over n_episodes (used to pick best.pkl)."""
    if n_episodes <= 0:
        return float("nan")
    rets = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        done, ret = False, 0.0
        while not done:
            key, k = jax.random.split(key)
            a = int(jnp.argmax(q_fn(params, encode_obs(s)[None], k)[0]))
            s, r, term, trunc, _ = env.step(a)
            ret += r
            done = term or trunc
        rets.append(ret)
    return float(np.mean(rets))


def _epsilon(cfg: Config, ep: int) -> float:
    if cfg.epsilon_decay_rate:
        return max(cfg.epsilon_end, cfg.epsilon_start * cfg.epsilon_decay_rate**ep)
    return max(
        cfg.epsilon_end,
        cfg.epsilon_start
        - (cfg.epsilon_start - cfg.epsilon_end) * ep / cfg.epsilon_decay_episodes,
    )


def train(cfg: Config, quick: bool = False) -> dict:
    spec = parse_noise(cfg.noise)
    key = jax.random.PRNGKey(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    key, k_init = jax.random.split(key)
    params = init_params(cfg, k_init)
    target = params
    q_fn = make_q_fn(cfg, spec, seed=cfg.seed)
    opt = make_optimizer(cfg, params)
    opt_state = opt.init(params)

    def loss_fn(p, tgt, s, a, r, s2, d, k):
        k1, k2 = jax.random.split(k)
        q = q_fn(p, s, k1)
        q_sa = jnp.take_along_axis(q, a[:, None], axis=1)[:, 0]
        q_next = jax.lax.stop_gradient(q_fn(tgt, s2, k2)).max(axis=1)
        y = r + cfg.gamma * (1.0 - d) * q_next
        return optax.huber_loss(q_sa, y).mean()

    grad_fn = jax.value_and_grad(loss_fn)

    def update(p, o, tgt, batch, k):
        loss, g = grad_fn(p, tgt, *batch, k)
        upd, o = opt.update(g, o, p)
        return optax.apply_updates(p, upd), o, loss

    if not spec.shots:
        update = jax.jit(update)

    env = make_env(cfg.env_id, seed=cfg.seed, max_steps=cfg.max_steps, **cfg.env_kwargs)
    eval_env = make_env(
        cfg.env_id, seed=cfg.seed + 10_000, max_steps=cfg.max_steps, **cfg.env_kwargs
    )
    best_greedy, best_ep = -np.inf, None
    buf = Replay(cfg.buffer_size, cfg.obs_dim, rng)
    run_dir = Path(cfg.run_dir)
    ckpt_dir = run_dir / "checkpoints" / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(ckpt_dir / "config.json")
    n_params = count_params(params)
    logger = TrainLogger(
        run_dir / "train" / f"{cfg.run_name}.csv",
        task=cfg.env_id,
        model=cfg.model,
        experiment=cfg.experiment,
        train_noise=cfg.noise,
        seed=cfg.seed,
        n_params=n_params,
    )
    recent = deque(maxlen=100)
    solved_at = None
    steps = 0
    grad_steps = 0
    t0 = time.perf_counter()
    print(
        f"[{cfg.run_name}] {n_params} params, noise={spec.label}, reward_scale={cfg.reward_scale}"
    )

    for ep in range(cfg.n_episodes):
        eps = _epsilon(cfg, ep)
        s, _ = env.reset()
        s_enc = encode_obs(s)
        done, ret = False, 0.0
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(cfg.n_actions))
            else:
                key, k = jax.random.split(key)
                a = int(jnp.argmax(q_fn(params, s_enc[None], k)[0]))
            s2, r, term, trunc, _ = env.step(a)
            s2_enc = encode_obs(s2)
            r_train = float(r) * cfg.reward_scale
            buf.add(s_enc, a, r_train, s2_enc, float(term))
            s_enc, ret, done, steps = s2_enc, ret + r, term or trunc, steps + 1
            if buf.n >= cfg.learning_starts and steps % cfg.train_every == 0:
                key, k = jax.random.split(key)
                batch = buf.sample(cfg.batch_size)
                params, opt_state, _loss = update(params, opt_state, target, batch, k)
                grad_steps += 1
                if grad_steps % cfg.target_update_every == 0:
                    target = params
        recent.append(ret)
        logger.episode(ep, ret)
        avg = float(np.mean(recent))
        if solved_at is None and len(recent) == 100 and avg >= cfg.threshold:
            solved_at = ep
        if ep % cfg.log_every == 0 or ep == cfg.n_episodes - 1:
            print(
                f"ep {ep:5d}  return {ret:7.1f}  avg100 {avg:7.1f}  eps {eps:.2f}  "
                f"{time.perf_counter() - t0:7.1f}s"
            )
        if cfg.checkpoint_every and ep % cfg.checkpoint_every == 0 and ep > 0:
            with open(ckpt_dir / f"ep{ep}.pkl", "wb") as fh:
                pickle.dump(jax.device_get(params), fh)
            key, k = jax.random.split(key)
            g = greedy_eval(cfg, q_fn, params, eval_env, cfg.eval_episodes, k)
            if g >= best_greedy:  # >= so an equally good later checkpoint wins
                best_greedy, best_ep = g, ep
                with open(ckpt_dir / "best.pkl", "wb") as fh:
                    pickle.dump(jax.device_get(params), fh)
            print(
                f"   [eval ep {ep}] greedy mean over {cfg.eval_episodes} eps = {g:.1f}  "
                f"(best {best_greedy:.1f} @ ep {best_ep})"
            )
        if quick and ep >= 5:
            break

    with open(ckpt_dir / "final.pkl", "wb") as fh:
        pickle.dump(jax.device_get(params), fh)
    key, k = jax.random.split(key)
    final_greedy = greedy_eval(cfg, q_fn, params, eval_env, cfg.eval_episodes, k)
    if not (ckpt_dir / "best.pkl").exists() or final_greedy >= best_greedy:
        best_greedy, best_ep = final_greedy, cfg.n_episodes - 1
        with open(ckpt_dir / "best.pkl", "wb") as fh:
            pickle.dump(jax.device_get(params), fh)
    logger.close()
    env.close()
    eval_env.close()
    out = {
        "final_score": float(np.mean(recent)),
        "solved_at": solved_at,
        "final_greedy": final_greedy,
        "best_greedy": best_greedy,
        "best_ep": best_ep,
        "wall_s": time.perf_counter() - t0,
        "n_params": n_params,
        "checkpoint": str(ckpt_dir / "final.pkl"),
        "best_checkpoint": str(ckpt_dir / "best.pkl"),
    }
    print(out)
    return out
