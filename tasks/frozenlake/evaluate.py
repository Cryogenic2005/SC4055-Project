"""Evaluate a trained FrozenLake checkpoint under the noise menu (owner: P2).

    python tasks/frozenlake/evaluate.py --checkpoint runs/checkpoints/FrozenLake-v1_vqc_0/final.pkl
    python tasks/frozenlake/evaluate.py --checkpoint ... --conditions none depolarizing=0.01 preset=heron2026

Appends 100 greedy episodes per condition to runs/eval/FrozenLake-v1_vqc_<seed>.csv
(Experiment B when the checkpoint was trained noise-free, C otherwise).
Until P3's shared harness (qrl/evaluate.py) lands, this file is the reference.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qrl.config import Config  # noqa: E402
from qrl.envs import make_env  # noqa: E402
from qrl.logging_utils import EvalLogger  # noqa: E402
from qrl.noise import NOISE_MENU, parse_noise  # noqa: E402
from tasks.frozenlake.agent import make_q_fn, state_to_bits  # noqa: E402


def evaluate(cfg: Config, params, conditions=NOISE_MENU, n_episodes: int = 100, run_dir: str = "runs") -> dict:
    logger = EvalLogger(Path(run_dir) / "eval" / f"{cfg.env_id}_{cfg.model}_{cfg.seed}.csv", task=cfg.env_id,
                        model=cfg.model, experiment=cfg.experiment, train_noise=cfg.noise, seed=cfg.seed, append=True)
    results = {}
    for cond in conditions:
        spec = parse_noise(cond)
        q_fn = make_q_fn(cfg.n_layers, spec, cfg.data_reuploading, seed=cfg.seed + 1000)
        env = make_env(cfg.env_id, seed=cfg.seed + 1000)
        key = jax.random.PRNGKey(cfg.seed + 1000)
        rets = []
        for ep in range(n_episodes):
            s, _ = env.reset()
            done, ret = False, 0.0
            while not done:
                key, k = jax.random.split(key)
                a = int(jnp.argmax(q_fn(params, state_to_bits(s)[None], k)[0]))
                s, r, term, trunc, _ = env.step(a)
                ret += r; done = term or trunc
            rets.append(ret)
            logger.episode(spec.eval_type, spec.eval_level, ep, ret)
        env.close()
        results[cond] = float(np.mean(rets))
        print(f"{cond:22s} mean return {results[cond]:.3f}")
    logger.close()
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--conditions", nargs="*", default=NOISE_MENU)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--run-dir", default="runs")
    a = ap.parse_args(argv)
    ckpt = Path(a.checkpoint)
    cfg = Config.load(ckpt.parent / "config.json")
    with open(ckpt, "rb") as fh:
        params = pickle.load(fh)
    return evaluate(cfg, params, a.conditions, a.episodes, a.run_dir)


if __name__ == "__main__":
    main()
