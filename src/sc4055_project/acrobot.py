"""Train and evaluate P3's Acrobot VQC and classical controls."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from sc4055_project.qrl.config import Config
from sc4055_project.qrl.noise import NOISE_MENU
from sc4055_project.tasks.acrobot.agent import task_config
from sc4055_project.tasks.acrobot.evaluate import evaluate
from sc4055_project.tasks.acrobot.train import train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--model", choices=("vqc", "mlp_small", "mlp_large"), default="vqc")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--noise", default="none")
    tr.add_argument("--run-dir", default="runs")
    tr.add_argument("--tag", default="")
    tr.add_argument(
        "--config", help="JSON configuration (takes precedence over CLI defaults)"
    )
    for flag, kind in (
        ("episodes", int),
        ("layers", int),
        ("train-every", int),
        ("batch-size", int),
        ("learning-starts", int),
        ("checkpoint-every", int),
        ("eval-episodes", int),
        ("max-steps", int),
        ("lr-output", float),
        ("lr-circuit", float),
        ("lr-mlp", float),
        ("reward-scale", float),
    ):
        tr.add_argument("--" + flag, type=kind)
    tr.add_argument("--quick", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--conditions", nargs="+", default=NOISE_MENU)
    ev.add_argument("--episodes", type=int, default=100)
    ev.add_argument("--run-dir", default="runs")
    ev.add_argument("--eval-seed", type=int, default=20000)
    args = parser.parse_args(argv)
    if args.command == "train":
        if args.config:
            cfg = Config.load(args.config)
        else:
            fields = (
                "train_every",
                "batch_size",
                "learning_starts",
                "checkpoint_every",
                "eval_episodes",
                "max_steps",
                "lr_output",
                "lr_circuit",
                "lr_mlp",
                "reward_scale",
            )
            overrides = {
                f: getattr(args, f) for f in fields if getattr(args, f) is not None
            }
            if args.episodes is not None:
                overrides["n_episodes"] = args.episodes
            if args.layers is not None:
                overrides["n_layers"] = args.layers
            if any(
                overrides.get(f, 1) < 1
                for f in (
                    "n_episodes",
                    "n_layers",
                    "train_every",
                    "batch_size",
                    "learning_starts",
                )
            ):
                parser.error("Episode, layer, batch and update counts must be positive")
            cfg = task_config(
                args.model,
                seed=args.seed,
                noise=args.noise,
                run_dir=args.run_dir,
                tag=args.tag,
                **overrides,
            )
        print(json.dumps(train(cfg, quick=args.quick), indent=2), flush=True)
    else:
        checkpoint = Path(args.checkpoint)
        cfg = Config.load(checkpoint.parent / "config.json")
        # Only load trusted local checkpoints produced by the team's training code.
        with checkpoint.open("rb") as handle:
            params = pickle.load(handle)
        evaluate(
            cfg, params, args.conditions, args.episodes, args.run_dir, args.eval_seed
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
