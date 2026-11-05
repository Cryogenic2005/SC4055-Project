"""Sequential Acrobot A/B/C runner. No training is started without an explicit stage.

From the repository root:
    python analysis/acrobot/run_experiments.py A --models vqc --seeds 0
    python analysis/acrobot/run_experiments.py B --seeds 0
    python analysis/acrobot/run_experiments.py C --seeds 0 1 2

Existing training logs are never silently overwritten. Use another --run-dir for
a new protocol. Checkpoints for evaluation must be locally trusted team files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def source_hashes(root):
    files = [*(root / "src").rglob("*.py"), root / "pyproject.toml"]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("A", "B", "C"))
    ap.add_argument(
        "--models",
        nargs="+",
        choices=("vqc", "mlp_small", "mlp_large"),
        default=["vqc", "mlp_small", "mlp_large"],
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--run-dir", default="runs/acrobot_formal")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if min(args.episodes, args.eval_episodes) < 1:
        ap.error("Episode budgets must be positive")
    seeds = (
        args.seeds
        if args.seeds is not None
        else list(range(3 if args.stage == "C" else 5))
    )
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        ap.error("Seeds must be distinct nonnegative integers")
    root = Path(__file__).resolve().parents[2]
    out = Path(args.run_dir).resolve()
    commands = []
    for seed in seeds:
        models = args.models if args.stage == "A" else ["vqc"]
        for model in models:
            name = f"Acrobot-v1_{model}_{seed}"
            if args.stage == "C":
                name += "_depolarizing0.01"
            checkpoint = out / "checkpoints" / name / "best.pkl"
            if args.stage in ("A", "C"):
                if (out / "train" / f"{name}.csv").exists():
                    raise FileExistsError(
                        f"Training already exists: {name}; use another --run-dir"
                    )
                cmd = [
                    sys.executable,
                    "-u",
                    "-m",
                    "sc4055_project.acrobot",
                    "train",
                    "--model",
                    model,
                    "--seed",
                    str(seed),
                    "--episodes",
                    str(args.episodes),
                    "--run-dir",
                    str(out),
                ]
                if args.stage == "C":
                    cmd += ["--noise", "depolarizing=0.01"]
                commands.append(cmd)
            elif not args.dry_run and not checkpoint.is_file():
                raise FileNotFoundError(f"Run A first: {checkpoint}")
            # A includes an independent clean evaluation for every model. B is the
            # full noisy sweep on clean-trained VQCs; C compares at matched noise.
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "sc4055_project.acrobot",
                "evaluate",
                "--checkpoint",
                str(checkpoint),
                "--episodes",
                str(args.eval_episodes),
                "--run-dir",
                str(out),
            ]
            conditions = args.conditions
            if conditions is None and args.stage == "A":
                conditions = ["none"]
            if conditions is None and args.stage == "C":
                conditions = ["none", "depolarizing=0.01"]
            if conditions is not None:
                cmd += ["--conditions", *conditions]
            commands.append(cmd)
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        manifest = out / f"manifest_{args.stage}.json"
        if manifest.exists():
            ap.error(f"Manifest already exists: {manifest}; choose another --run-dir")
        manifest.write_text(
            json.dumps(
                {
                    "arguments": vars(args),
                    "seeds": seeds,
                    "source_sha256": source_hashes(root),
                    "commands": commands,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    for cmd in commands:
        print(subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
