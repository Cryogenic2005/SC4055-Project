"""Train and evaluate the LunarLander-v3 agent with DQN (owner: P4).

    lunarlander train --seed 0                                  # hybrid, Experiment A
    lunarlander train --seed 0 --model encoder_only             # ablation
    lunarlander train --seed 0 --model mlp_small                # parameter-matched MLP
    lunarlander train --seed 0 --model mlp_large                # 2x64 MLP
    lunarlander train --seed 0 --noise depolarizing=0.01        # Experiment C (not in P4's protocol)
    lunarlander train --seed 0 --episodes 20 --quick            # smoke run
    lunarlander evaluate --checkpoint runs/checkpoints/LunarLander-v3_vqc_0/best.pkl
    lunarlander evaluate --checkpoint ... --conditions none depolarizing=0.01 preset=heron2026
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from sc4055_project.qrl.config import MODELS, Config, default_config_for
from sc4055_project.qrl.noise import NOISE_MENU
from sc4055_project.tasks.lunarlander import evaluate, train

TRAIN_DOC = """\
The hybrid (``vqc``) is the plan's model: an 8 -> 4 tanh encoder feeding a 4-qubit VQC
with 4 layers, <Z> readouts and trainable output weights. ``encoder_only`` replaces the
VQC with a 4 -> 4 linear layer; ``mlp_small``/``mlp_large`` are the classical baselines.

Writes runs/train/LunarLander-v3_<model>_<seed>[_<noise>].csv (analysis-spec columns) and
runs/checkpoints/<run_name>/{config.json, ep*.pkl, best.pkl, final.pkl}.

The logged return is always the raw Gymnasium return; ``--reward-scale`` only rescales
the TD target (see tasks/lunarlander/train.py).\
"""

EVAL_DOC = """\
Appends 100 greedy episodes per condition to runs/eval/<run_name>.csv (Experiment B
when the checkpoint was trained noise-free). Keep ``none`` first: it is the reference
point of every sweep. Noise only affects the ``vqc`` hybrid; for the classical variants
pass ``--conditions none``.\
"""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="lunarlander",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = ap.add_subparsers(dest="command", required=True)

    train_ap = subparsers.add_parser(
        "train", help="Train a LunarLander agent with DQN", description=TRAIN_DOC
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
        "--reward-scale",
        type=float,
        default=None,
        help="multiplier for the TD target only (default 0.01; logged returns stay raw)",
    )
    train_ap.add_argument("--lr", type=float, default=None, help="circuit learning rate")
    train_ap.add_argument(
        "--eps-decay",
        type=int,
        default=None,
        help="linear epsilon decay length in episodes (overrides the config's value)",
    )
    train_ap.add_argument(
        "--no-reupload",
        action="store_true",
        help="disable data re-uploading (re-uploading is the hybrid's default)",
    )
    train_ap.add_argument(
        "--quick", action="store_true", help="stop after a few episodes (smoke test)"
    )

    eval_ap = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained LunarLander checkpoint under the noise menu",
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
        conditions = list(a.conditions or ["none"])
        evaluate(cfg, params, conditions, a.episodes, a.run_dir)

    elif a.command == "train":
        overrides = {}
        if a.layers:
            overrides["n_layers"] = a.layers
        if a.episodes:
            overrides["n_episodes"] = a.episodes
        if a.lr:
            overrides["lr_circuit"] = a.lr
        if a.eps_decay:
            overrides.update(epsilon_decay_episodes=a.eps_decay, epsilon_decay_rate=None)
        if a.no_reupload:
            overrides["data_reuploading"] = False
        if a.reward_scale is not None:
            overrides["reward_scale"] = a.reward_scale

        cfg = default_config_for(
            "LunarLander-v3",
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
