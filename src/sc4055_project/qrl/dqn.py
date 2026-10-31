from __future__ import annotations

import json
import pickle
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .agent import count_params, init_params, make_q_fn
from .config import Config
from .envs import make_env
from .logging_utils import TrainLogger
from .noise import parse_noise


class ReplayBuffer:
    def __init__(
        self, size: int, state_shape: tuple[int, ...], rng: np.random.Generator
    ):
        self.state_shape = state_shape
        self.s = np.zeros((size, *state_shape), np.float32)
        self.a = np.zeros(size, np.int32)
        self.r = np.zeros(size, np.float32)
        self.s2 = np.zeros((size, *state_shape), np.float32)
        self.d = np.zeros(size, np.float32)
        self.n = 0
        self.i = 0
        self.size = size
        self.rng = rng

    def add(self, s, a, r, s2, d):
        if s.shape != self.state_shape:
            raise ValueError(
                f"State shape mismatch: expected {self.state_shape}, got {s.shape}"
            )

        if s2.shape != self.state_shape:
            raise ValueError(
                f"Next state shape mismatch: expected {self.state_shape}, got {s2.shape}"
            )

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


def make_optimizer(cfg: Config):
    # Separate learning rates per parameter group (TFQ / Skolik recipe).
    labels = {"theta": "circuit", "lam": "input", "w_out": "output"}
    return optax.multi_transform(
        {
            "circuit": optax.adam(cfg.lr_circuit),
            "input": optax.adam(cfg.lr_input),
            "output": optax.adam(cfg.lr_output),
        },
        labels,
    )


def get_update_fn(q_fn, opt, gamma, *, jit=True):
    def loss_fn(p, tgt, s, a, r, s2, d, k):
        k1, k2 = jax.random.split(k)
        q = q_fn(p, s, k1)
        q_sa = jnp.take_along_axis(q, a[:, None], axis=1)[:, 0]
        q_next = jax.lax.stop_gradient(q_fn(tgt, s2, k2)).max(axis=1)
        y = r + gamma * (1.0 - d) * q_next
        return optax.huber_loss(q_sa, y).mean()

    grad_fn = jax.value_and_grad(loss_fn)

    def update(p, o, tgt, batch, k):
        loss, g = grad_fn(p, tgt, *batch, k)
        upd, o = opt.update(g, o, p)
        return optax.apply_updates(p, upd), o, loss

    if jit:
        return jax.jit(update)

    return update


def train_agent(
    cfg: Config,
    *,
    state_shape: tuple[int, ...],
    state_encode_fn: Callable[[Any], np.ndarray],
    reward_augment_fn: Callable[[float, bool], float] | None = None,
    env=None,
    eval_env=None,
    quick: bool = False,
) -> dict:
    """Train a DQN agent on the given environment and return training statistics.

    Allows for custom state encoding and reward shaping functions to be provided.

    Args:
        cfg: Configuration object containing hyperparameters and settings.
        state_shape: Shape of the encoded state space.
        state_encode_fn: Function to encode states before feeding into the model.
        reward_augment_fn: Optional function to augment rewards for training.
        env: Optional environment instance for training.
            If None, a new environment is created.
        eval_env: Optional environment instance for evaluation.
            If None, a new environment is created.
        quick: If True, runs a quick training session for testing purposes.

    Returns:
        Training metadata
    """

    # Setup random seeds and RNGs.
    key = jax.random.PRNGKey(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    spec = parse_noise(cfg.noise)

    # Initialize model, optimizer, and target network.
    key, k_init = jax.random.split(key)
    params = init_params(cfg, k_init)
    target = params

    q_fn = make_q_fn(cfg, spec)
    opt = make_optimizer(cfg)
    opt_state = opt.init(params)

    # Get the update function for training
    update_fn = get_update_fn(q_fn, opt, cfg.gamma, jit=not bool(spec.shots))

    # Setup directories for training info
    run_dir = Path(cfg.run_dir)
    ckpt_dir = run_dir / "checkpoints" / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(ckpt_dir / "config.json")
    (ckpt_dir / "metadata.json").write_text(json.dumps(run_metadata(), indent=2))

    # Logger for training progress
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

    # Setup environments for training and evaluation
    created_env = env is None
    if env is None:
        env = make_env(
            cfg.env_id, seed=cfg.seed, max_steps=cfg.max_steps, **cfg.env_kwargs
        )

    created_eval_env = eval_env is None
    if eval_env is None:
        eval_env = make_env(
            cfg.env_id,
            seed=cfg.seed + 10_000,
            max_steps=cfg.max_steps,
            **cfg.env_kwargs,
        )

    # Buffer for experience replay
    buf = ReplayBuffer(cfg.buffer_size, state_shape, rng)

    # Return history
    recent = deque(maxlen=100)

    eps = cfg.epsilon_start
    solved_at = None
    steps, grad_steps = 0, 0
    best_greedy, best_ep = -np.inf, None
    t0 = time.perf_counter()
    print(
        f"[{cfg.run_name}] {n_params} params, noise={spec.label}, device={'mixed' if spec.needs_mixed else 'qubit'}"
    )

    n_episodes = cfg.n_episodes if not quick else 5
    for ep in range(n_episodes):
        s, _ = env.reset()
        if state_encode_fn:
            s = state_encode_fn(s)

        done, ret = False, 0.0
        while not done:
            # Epsilon-greedy action selection
            if rng.random() < eps:
                a = int(rng.integers(cfg.n_actions))
            else:
                key, k = jax.random.split(key)
                a = int(jnp.argmax(q_fn(params, s[None], k)[0]))

            # Step the environment
            s2, r, term, trunc, _ = env.step(a)
            r = float(r)  # Just for typing
            if state_encode_fn:
                s2 = state_encode_fn(s2)

            # Augmnent reward for training only
            r_train = r
            if cfg.reward_shaping and reward_augment_fn:
                r_train = reward_augment_fn(r, term)

            # Update
            buf.add(s, a, r_train, s2, float(term))
            s = s2
            ret += r
            done = term or trunc
            steps += 1

            # Train the model if enough experience has been collected
            if buf.n >= cfg.learning_starts and steps % cfg.train_every == 0:
                key, k = jax.random.split(key)

                # Sample a batch from the replay buffer and perform an update
                batch = buf.sample(cfg.batch_size)
                params, opt_state, _ = update_fn(params, opt_state, target, batch, k)

                # Update the target network periodically
                grad_steps += 1
                if grad_steps % cfg.target_update_every == 0:
                    target = params

        # Log the episode returns
        recent.append(ret)
        logger.episode(ep, ret)

        # Check if the environment is solved
        avg = float(np.mean(recent))
        if solved_at is None and len(recent) == 100 and avg >= cfg.threshold:
            solved_at = ep

        if _should_print_train_info(cfg, ep):
            print(
                f"ep {ep:5d}",
                f"return {ret:5.1f}",
                f"avg100 {avg:.3f}",
                f"eps {eps:.2f}",
                f"{time.perf_counter() - t0:7.1f}s",
                sep="  ",
            )

        eps = update_epsilon(cfg, ep)

        if _should_eval(cfg, ep, n_episodes):
            # Save epoch checkpoint
            save_checkpoint(params, ckpt_dir / f"ep{ep}.pkl")
            if ep == cfg.n_episodes - 1:
                save_checkpoint(params, ckpt_dir / "final.pkl")

            key, k = jax.random.split(key)
            g = greedy_eval(
                q_fn,
                params,
                eval_env,
                cfg.eval_episodes,
                k,
                state_encode_fn=state_encode_fn,
            )
            if g >= best_greedy:
                # >= so a later, equally good checkpoint wins (more training)
                best_greedy, best_ep = g, ep
                save_checkpoint(params, ckpt_dir / "best.pkl")
                with open(ckpt_dir / "best.pkl", "wb") as fh:
                    pickle.dump(jax.device_get(params), fh)

            print(
                f"   [eval ep {ep}] greedy mean over {cfg.eval_episodes}",
                f"eps = {g:.3f}  (best {best_greedy:.3f} @ ep {best_ep})",
            )

    logger.close()

    # Don't close user environment if it was passed in
    if created_env:
        env.close()

    if created_eval_env:
        eval_env.close()

    return {
        "final_score": float(np.mean(recent)),
        "solved_at": solved_at,
        "final_greedy": g,
        "best_greedy": best_greedy,
        "best_ep": best_ep,
        "wall_s": time.perf_counter() - t0,
        "n_params": n_params,
        "checkpoint": str(ckpt_dir / "final.pkl"),
        "best_checkpoint": str(ckpt_dir / "best.pkl"),
    }


def update_epsilon(cfg: Config, episode: int) -> float:
    eps = cfg.epsilon_start

    if cfg.epsilon_decay_rate:
        # Exponential decay
        eps *= cfg.epsilon_decay_rate**episode
        eps = max(cfg.epsilon_end, eps)
    else:
        # Linear decay
        eps_range = cfg.epsilon_end - cfg.epsilon_start
        progress = min(episode / cfg.epsilon_decay_episodes, 1.0)
        eps += eps_range * progress

    return eps


def save_checkpoint(params, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(jax.device_get(params), fh)


def greedy_eval(q_fn, params, env, n_episodes: int, key, *, state_encode_fn) -> float:
    """Mean raw return of the greedy policy over n_episodes (used to pick best.pkl)."""
    if n_episodes <= 0:
        return float("nan")

    ret = 0.0
    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            key, k = jax.random.split(key)
            a = int(jnp.argmax(q_fn(params, state_encode_fn(s)[None], k)[0]))
            s, r, term, trunc, _ = env.step(a)

            ret += r
            done = term or trunc

    return ret / n_episodes


def run_metadata() -> dict:
    """Code and package versions, saved next to config.json so results trace to code."""
    import platform
    import subprocess

    import gymnasium
    import pennylane

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return {
        "git_commit": commit or None,
        "python": platform.python_version(),
        "pennylane": pennylane.__version__,
        "jax": jax.__version__,
        "optax": optax.__version__,
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
    }


def _should_print_train_info(cfg: Config, ep: int) -> bool:
    return ep % cfg.log_every == 0 or ep == cfg.n_episodes - 1


def _should_eval(cfg: Config, ep: int, n_episodes: int) -> bool:
    if ep == n_episodes - 1:
        return True

    return bool(cfg.checkpoint_every) and ep % cfg.checkpoint_every == 0 and ep > 0
