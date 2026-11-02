"""Train and evaluate the CartPole-v1 VQC agent with DQN (owner: P1).

Examples:
```
cartpole train --seed 0
cartpole train --seed 0 --noise depolarizing=0.01
cartpole train --seed 0 --model mlp_small
cartpole train --seed 0 --episodes 50 --quick
cartpole evaluate --checkpoint runs/checkpoints/CartPole-v1_vqc_0/best.pkl
cartpole evaluate --checkpoint ... --conditions none depolarizing=0.01 preset=heron2026
```
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from sc4055_project.qrl.config import MODELS, Config, default_config_for
from sc4055_project.qrl.noise import NOISE_MENU
from sc4055_project.tasks.cartpole import evaluate, train

TRAIN_DOC = """\
Writes runs/train/CartPole-v1_<model>_<seed>[_<noise>].csv and
runs/checkpoints/<run_name>/{config.json, metadata.json, ep*.pkl, best.pkl, final.pkl}.\
"""

EVAL_DOC = """\
Appends greedy evaluation episodes to runs/eval/<run_name>.csv.
Experiment label is B for noise-free training checkpoints, C otherwise.\
"""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="cartpole",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = ap.add_subparsers(dest="command", required=True)

    train_ap = subparsers.add_parser(
        "train", help="Train a CartPole agent with DQN", description=TRAIN_DOC
    )
    train_ap.add_argument("--seed", type=int, default=0)
    train_ap.add_argument("--model", choices=MODELS, default="vqc")
    train_ap.add_argument(
        "--noise",
        default="none",
        help="none | depolarizing=p | coherent=sigma | readout=p | shots=N | preset=name",
    )
    train_ap.add_argument("--layers", type=int, default=None)
    train_ap.add_argument("--episodes", type=int, default=None)
    train_ap.add_argument("--run-dir", default="runs")
    train_ap.add_argument(
        "--eps-decay",
        type=int,
        default=None,
        help="linear epsilon decay length in episodes (overrides multiplicative)",
    )
    train_ap.add_argument(
        "--lr", type=float, default=None, help="circuit learning rate"
    )
    train_ap.add_argument(
        "--no-reupload",
        action="store_true",
        help="disable data re-uploading",
    )
    train_ap.add_argument(
        "--quick", action="store_true", help="stop after a few episodes (smoke test)"
    )

    eval_ap = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained CartPole checkpoint under the noise menu",
        description=EVAL_DOC,
    )
    eval_ap.add_argument("--checkpoint", required=True)
    eval_ap.add_argument("--conditions", nargs="*", default=NOISE_MENU)
    eval_ap.add_argument("--episodes", type=int, default=100)
    eval_ap.add_argument("--run-dir", default="runs")

    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)

    if a.command == "evaluate":
        ckpt = Path(a.checkpoint)
        cfg = Config.load(ckpt.parent / "config.json")
        with open(ckpt, "rb") as fh:
            params = pickle.load(fh)
        evaluate(cfg, params, a.conditions, a.episodes, a.run_dir)

    elif a.command == "train":
        overrides = {}
        if a.layers:
            overrides["n_layers"] = a.layers
        if a.episodes:
            overrides["n_episodes"] = a.episodes
        if a.lr:
            overrides["lr_circuit"] = a.lr
        if a.eps_decay:
            overrides.update(
                epsilon_decay_episodes=a.eps_decay, epsilon_decay_rate=None
            )
        if a.no_reupload:
            overrides["data_reuploading"] = False

        cfg = default_config_for(
            "CartPole-v1",
            a.model,
            seed=a.seed,
            noise=a.noise,
            run_dir=a.run_dir,
            **overrides,
        )
        train(cfg, quick=a.quick)

    else:
        raise ValueError(f"Unknown command {a.command}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
