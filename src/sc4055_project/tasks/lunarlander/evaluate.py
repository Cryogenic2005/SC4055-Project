"""Experiment B evaluation for LunarLander-v3 checkpoints (owner: P4).

Runs the trained checkpoint greedily for ``n_episodes`` under every noise condition and
appends rows with the analysis-spec columns to ``runs/eval/<run_name>.csv``. The
``none`` condition is the reference point of the sweep; keep it first.

Only the ``vqc`` hybrid has a quantum layer, so noise conditions are meaningful for it
only. The classical variants (``encoder_only``, ``mlp_small``, ``mlp_large``) ignore the
noise spec and would repeat the same number under every condition; pass
``conditions=["none"]`` for those to keep the eval CSV honest.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sc4055_project.qrl.config import Config
from sc4055_project.qrl.envs import make_env
from sc4055_project.qrl.logging_utils import EvalLogger
from sc4055_project.qrl.noise import NOISE_MENU, parse_noise

from .agent import encode_obs, make_q_fn


def evaluate(
    cfg: Config,
    params,
    conditions=NOISE_MENU,
    n_episodes: int = 100,
    run_dir: str = "runs",
) -> dict:
    logger = EvalLogger(
        Path(run_dir) / "eval" / f"{cfg.run_name}.csv",
        task=cfg.env_id,
        model=cfg.model,
        experiment=cfg.experiment,
        train_noise=cfg.noise,
        seed=cfg.seed,
        append=True,
    )
    results = {}
    for cond in conditions:
        spec = parse_noise(cond)
        q_fn = make_q_fn(cfg, spec, seed=cfg.seed + 1000)
        env = make_env(
            cfg.env_id,
            seed=cfg.seed + 1000,
            max_steps=cfg.max_steps,
            **cfg.env_kwargs,
        )
        key = jax.random.PRNGKey(cfg.seed + 1000)
        rets = []
        for ep in range(n_episodes):
            s, _ = env.reset()
            done, ret = False, 0.0
            while not done:
                key, k = jax.random.split(key)
                a = int(jnp.argmax(q_fn(params, encode_obs(s)[None], k)[0]))
                s, r, term, trunc, _ = env.step(a)
                ret += r
                done = term or trunc
            rets.append(ret)
            logger.episode(spec.eval_type, spec.eval_level, ep, ret)
        env.close()
        results[cond] = float(np.mean(rets))
        print(f"{cond:22s} mean return {results[cond]:.1f}")
    logger.close()
    return results
