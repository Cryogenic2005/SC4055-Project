"""Train the FrozenLake VQC agent with DQN (owner: P2).

Examples
    python tasks/frozenlake/train.py --seed 0                       # Experiment A
    python tasks/frozenlake/train.py --seed 0 --noise depolarizing=0.01   # Experiment C
    python tasks/frozenlake/train.py --seed 0 --noise shots=100     # parameter-shift, no backprop
    python tasks/frozenlake/train.py --episodes 50 --quick          # smoke run

Writes runs/train/FrozenLake-v1_vqc_<seed>[_<noise>].csv (analysis-spec columns),
runs/checkpoints/<run_name>/{config.json, ep*.pkl, final.pkl}.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qrl.config import Config, default_config_for  # noqa: E402
from qrl.envs import make_env  # noqa: E402
from qrl.logging_utils import TrainLogger  # noqa: E402
from qrl.noise import parse_noise  # noqa: E402
from tasks.frozenlake.agent import init_params, make_q_fn, count_params, state_to_bits  # noqa: E402


class Replay:
    def __init__(self, size: int, rng: np.random.Generator):
        self.s = np.zeros((size, 4), np.float32); self.a = np.zeros(size, np.int32)
        self.r = np.zeros(size, np.float32); self.s2 = np.zeros((size, 4), np.float32)
        self.d = np.zeros(size, np.float32); self.n = 0; self.i = 0; self.size = size; self.rng = rng

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.size; self.n = min(self.n + 1, self.size)

    def sample(self, b: int):
        idx = self.rng.integers(0, self.n, size=b)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


def make_optimizer(cfg: Config):
    # Separate learning rates per parameter group (TFQ / Skolik recipe).
    labels = {"theta": "circuit", "lam": "input", "w_out": "output"}
    return optax.multi_transform(
        {"circuit": optax.adam(cfg.lr_circuit), "input": optax.adam(cfg.lr_input),
         "output": optax.adam(cfg.lr_output)},
        labels)


def train(cfg: Config, quick: bool = False) -> dict:
    spec = parse_noise(cfg.noise)
    key = jax.random.PRNGKey(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    key, k_init = jax.random.split(key)
    params = init_params(cfg.n_layers, k_init, cfg.data_reuploading)
    target = params
    q_fn = make_q_fn(cfg.n_layers, spec, cfg.data_reuploading, seed=cfg.seed)
    opt = make_optimizer(cfg)
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

    env = make_env(cfg.env_id, seed=cfg.seed, max_steps=cfg.max_steps)
    buf = Replay(cfg.buffer_size, rng)
    run_dir = Path(cfg.run_dir)
    ckpt_dir = run_dir / "checkpoints" / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(ckpt_dir / "config.json")
    n_params = count_params(params)
    logger = TrainLogger(run_dir / "train" / f"{cfg.run_name}.csv", task=cfg.env_id, model=cfg.model,
                         experiment=cfg.experiment, train_noise=cfg.noise, seed=cfg.seed, n_params=n_params)
    recent = deque(maxlen=100)
    solved_at = None
    steps = 0
    grad_steps = 0
    t0 = time.perf_counter()
    print(f"[{cfg.run_name}] {n_params} params, noise={spec.label}, device={'mixed' if spec.needs_mixed else 'qubit'}")

    for ep in range(cfg.n_episodes):
        eps = max(cfg.epsilon_end, cfg.epsilon_start - (cfg.epsilon_start - cfg.epsilon_end) * ep / cfg.epsilon_decay_episodes)
        s, _ = env.reset()
        sb = state_to_bits(s)
        done, ret = False, 0.0
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(4))
            else:
                key, k = jax.random.split(key)
                a = int(jnp.argmax(q_fn(params, sb[None], k)[0]))
            s2, r, term, trunc, _ = env.step(a)
            s2b = state_to_bits(s2)
            buf.add(sb, a, r, s2b, float(term))
            sb, ret, done, steps = s2b, ret + r, term or trunc, steps + 1
            if buf.n >= cfg.learning_starts and steps % cfg.train_every == 0:
                key, k = jax.random.split(key)
                batch = buf.sample(cfg.batch_size)
                params, opt_state, loss = update(params, opt_state, target, batch, k)
                grad_steps += 1
                if grad_steps % cfg.target_update_every == 0:
                    target = params
        recent.append(ret)
        logger.episode(ep, ret)
        avg = float(np.mean(recent))
        if solved_at is None and len(recent) == 100 and avg >= cfg.threshold:
            solved_at = ep
        if ep % cfg.log_every == 0 or ep == cfg.n_episodes - 1:
            print(f"ep {ep:5d}  return {ret:5.1f}  avg100 {avg:.3f}  eps {eps:.2f}  {time.perf_counter()-t0:7.1f}s")
        if cfg.checkpoint_every and ep % cfg.checkpoint_every == 0 and ep > 0:
            with open(ckpt_dir / f"ep{ep}.pkl", "wb") as fh:
                pickle.dump(jax.device_get(params), fh)
        if quick and ep >= 5:
            break

    with open(ckpt_dir / "final.pkl", "wb") as fh:
        pickle.dump(jax.device_get(params), fh)
    logger.close()
    env.close()
    out = {"final_score": float(np.mean(recent)), "solved_at": solved_at,
           "wall_s": time.perf_counter() - t0, "n_params": n_params, "checkpoint": str(ckpt_dir / "final.pkl")}
    print(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", default="none", help="none | depolarizing=p | coherent=sigma | readout=p | shots=N | preset=name")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--no-reupload", action="store_true")
    ap.add_argument("--quick", action="store_true", help="stop after a few episodes (smoke test)")
    a = ap.parse_args(argv)
    cfg = default_config_for("FrozenLake-v1", "vqc", seed=a.seed, noise=a.noise, run_dir=a.run_dir,
                             **({"n_layers": a.layers} if a.layers else {}),
                             **({"n_episodes": a.episodes} if a.episodes else {}),
                             data_reuploading=not a.no_reupload)
    return train(cfg, quick=a.quick)


if __name__ == "__main__":
    main()
