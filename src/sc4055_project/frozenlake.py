"""Train and evaluate the FrozenLake VQC agent with DQN (owner: P2)."""

import argparse
import pickle
from pathlib import Path

from sc4055_project.qrl.config import Config, default_config_for
from sc4055_project.qrl.noise import NOISE_MENU
from sc4055_project.tasks.frozenlake import evaluate, train

TRAIN_DOC = """\
Examples
    python tasks/frozenlake/train.py --seed 0                       # Experiment A
    python tasks/frozenlake/train.py --seed 0 --noise depolarizing=0.01   # Experiment C
    python tasks/frozenlake/train.py --seed 0 --noise shots=100     # parameter-shift, no backprop
    python tasks/frozenlake/train.py --seed 0 --slippery            # stochastic map (threshold 0.70)
    python tasks/frozenlake/train.py --episodes 50 --quick          # smoke run

Writes runs/train/FrozenLake-v1_vqc_<seed>[_<noise>].csv (analysis-spec columns),
runs/checkpoints/<run_name>/{config.json, ep*.pkl, final.pkl}.\
"""

EVAL_DOC = """\
Examples
    python tasks/frozenlake/evaluate.py --checkpoint runs/checkpoints/FrozenLake-v1_vqc_0/final.pkl
    python tasks/frozenlake/evaluate.py --checkpoint ... --conditions none depolarizing=0.01 preset=heron2026

Appends 100 greedy episodes per condition to runs/eval/FrozenLake-v1_vqc_<seed>.csv
(Experiment B when the checkpoint was trained noise-free, C otherwise).
Until P3's shared harness (qrl/evaluate.py) lands, this file is the reference.\
"""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="frozenLake",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = ap.add_subparsers(dest="command", required=True)

    train_ap = subparsers.add_parser(
        "train", help="Train the FrozenLake VQC agent with DQN", description=TRAIN_DOC
    )
    train_ap.add_argument("--seed", type=int, default=0)
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
        "--no-shaping", action="store_true", help="disable Chen-style reward shaping"
    )
    train_ap.add_argument(
        "--lr", type=float, default=None, help="circuit learning rate"
    )
    train_ap.add_argument(
        "--reupload",
        action="store_true",
        help="data re-uploading (default off for basis encoding)",
    )
    train_ap.add_argument(
        "--quick", action="store_true", help="stop after a few episodes (smoke test)"
    )
    train_ap.add_argument(
        "--slippery",
        action="store_true",
        help="use the stochastic registered env (is_slippery=True); default is the deterministic Chen et al. map",
    )

    eval_ap = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained FrozenLake checkpoint under the noise menu",
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
        cfg = default_config_for(
            "FrozenLake-v1",
            "vqc",
            seed=a.seed,
            noise=a.noise,
            run_dir=a.run_dir,
            **({"n_layers": a.layers} if a.layers else {}),
            **({"n_episodes": a.episodes} if a.episodes else {}),
            **(
                {"epsilon_decay_episodes": a.eps_decay, "epsilon_decay_rate": None}
                if a.eps_decay
                else {}
            ),
            **({"reward_shaping": False} if a.no_shaping else {}),
            **({"lr_circuit": a.lr} if a.lr else {}),
            **({"data_reuploading": True} if a.reupload else {}),
            tag="slip" if a.slippery else "",
        )
        if a.slippery:
            cfg.env_kwargs = {"is_slippery": True}

        train(cfg, quick=a.quick)

    else:
        raise ValueError(f"Unknown command {a.command}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
