"""Experiment C: does training under the deployment noise pay off?

The claim this script tests
---------------------------
Experiment B trains a VQC agent with no noise and then scores it under noise.
That is a train/deploy mismatch: the agent optimises J(no noise) but we grade it
on J(noise).  Here we retrain the *same* agent with noise switched on during
training -- QuantumNAT's "noise injection" (Wang et al., DAC 2022) -- and ask a
narrow, falsifiable question:

    at the deployment noise level, does the noise-trained agent score higher
    than the clean-trained agent?

Nothing else changes: same circuit, same hyper-parameters, same seeds, same
evaluation conditions.  `train_noise` is the only free variable, and the scale-0
arm *is* Experiment B, so the comparison is self-contained.

Why sweep the scale instead of running one arm
----------------------------------------------
Too little training noise and nothing happens; too much and the circuit can no
longer represent a useful policy, so the agent becomes uselessly "robust".  The
prediction is a hump: return at the deployment noise peaks when the training
noise *matches* deployment and falls off on both sides.  One arm cannot tell
"no effect" apart from "wrong scale", which is why the grid has >= 3 scales.

Usage
-----
    python analysis/noise_aware_sweep.py --env FrozenLake-v1 \
        --base depolarizing=0.02 --scales 0 0.25 1 4 \
        --seeds 0 1 2 --episodes 600 --eval-episodes 30

Outputs
-------
    <out>/summary.csv            one row per (train_noise, seed, eval condition)
    <out>/eval/<arm>_seed<k>/    the untouched per-episode rows of each arm
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from importlib import import_module
from pathlib import Path

from sc4055_project.qrl.config import Config, default_config_for

# One entry per supported task.  Each task package exposes train(cfg) -> dict and
# evaluate(cfg, params, conditions=..., n_episodes=..., run_dir=...), so this script
# never has to know which task it is running.
TASK_PACKAGES = {
    "FrozenLake-v1": "sc4055_project.tasks.frozenlake",
    "CartPole-v1": "sc4055_project.tasks.cartpole",
    "LunarLander-v3": "sc4055_project.tasks.lunarlander",
}

# The condition list plotted as one curve.  Depolarizing is the axis we sweep; the
# rest are fixed reference points (readout error, then the hardware presets).
SWEEP_CONDITIONS = ["none"] + [f"depolarizing={p:g}" for p in (5e-4, 1e-3, 5e-3, 1e-2, 5e-2)]


def scale_noise(base: str, scale: float) -> str:
    """`base` with its numeric level multiplied by `scale`; scale 0 means no noise.

    The noise menu is a string language (see qrl/noise.py), so scaling an arm is
    string arithmetic: "depolarizing=0.02" at scale 0.25 -> "depolarizing=0.005".
    """
    if base in ("", "none") or scale == 0:
        return "none"
    kind, level = base.split("=", 1)
    return f"{kind}={float(level) * scale:g}"


def noise_scale(noise: str, base: str) -> float:
    """Inverse of scale_noise: how many multiples of `base` is `noise`?"""
    if noise == "none":
        return 0.0
    return float(noise.split("=", 1)[1]) / float(base.split("=", 1)[1])


def make_cfg(env_id: str, seed: int, noise: str, base: str, args) -> Config:
    """One arm's Config: task defaults, then only the knobs this experiment needs."""
    cfg = default_config_for(env_id, model="vqc", noise=noise, seed=seed, n_episodes=args.episodes)
    cfg.run_dir = str(args.out / "runs")   # keep checkpoints/CSVs out of the shared runs/
    cfg.train_every = args.train_every
    cfg.log_every = max(args.episodes // 4, 1)
    cfg.checkpoint_every = 0   # keep final.pkl only; no mid-run checkpointing
    cfg.eval_episodes = 0      # evaluation happens once, below, through evaluate()
    cfg.tag = f"T{noise_scale(noise, base):g}"
    return cfg


def row_path(out: Path, noise: str, seed: int) -> Path:
    """One JSON per (arm, seed). Written the moment an arm finishes, so a sweep that
    dies after six hours keeps everything it already paid for and can be resumed."""
    return out / "rows" / f"{noise.replace('=', '')}_seed{seed}.json"


def save_row(row: dict, out: Path) -> None:
    path = row_path(out, row["train_noise"], row["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2))


def load_rows(out: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((out / "rows").glob("*.json"))]


def run_arm(env_id: str, seed: int, noise: str, base: str, args, out: Path) -> dict | None:
    """Train one arm, then evaluate its final checkpoint under every condition."""
    task = import_module(TASK_PACKAGES[env_id])
    cfg = make_cfg(env_id, seed, noise, base, args)

    if args.resume and row_path(out, noise, seed).exists():
        print(f"  seed {seed}  train_noise={noise:20s} (already done, skipped)", flush=True)
        return None

    t0 = time.perf_counter()
    result = task.train(cfg)
    train_s = time.perf_counter() - t0

    with open(result["checkpoint"], "rb") as fh:
        params = pickle.load(fh)

    # Each arm gets its own run_dir: some tasks name the eval CSV without the noise
    # tag, so writing every arm into one directory would interleave their rows.
    returns = task.evaluate(
        cfg,
        params,
        conditions=args.conditions,
        n_episodes=args.eval_episodes,
        run_dir=str(out / "eval" / f"{cfg.tag}_seed{seed}"),
    )
    deployment = scale_noise(base, 1.0)
    print(
        f"  seed {seed}  train_noise={noise:20s} ({train_s:5.1f}s)  "
        f"return@deploy={returns.get(deployment, float('nan')):8.3f}",
        flush=True,
    )
    return {
        "env": env_id,
        "seed": seed,
        "train_noise": noise,
        "train_scale": noise_scale(noise, base),
        "train_seconds": round(train_s, 1),
        "final_train_return": result["final_score"],
        "returns": returns,
    }


def write_summary(rows: list[dict], path: Path) -> None:
    """Long format, one row per (arm, seed, condition): easy to group or to plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["env", "seed", "train_noise", "train_scale", "train_seconds",
                    "final_train_return", "eval_condition", "mean_return"])
        for r in rows:
            for cond, value in r["returns"].items():
                w.writerow([r["env"], r["seed"], r["train_noise"], r["train_scale"],
                            r["train_seconds"], round(r["final_train_return"], 4),
                            cond, round(value, 4)])


def print_table(rows: list[dict], conditions: list[str], base: str) -> None:
    """Mean return per (arm, condition), averaged over seeds: the headline table."""
    arms = sorted({r["train_noise"] for r in rows}, key=lambda n: noise_scale(n, base))
    print(f"\n{'train noise':21s}" + "".join(f"{short(c):>13s}" for c in conditions))
    for arm in arms:
        cells = []
        for cond in conditions:
            vals = [r["returns"][cond] for r in rows if r["train_noise"] == arm]
            cells.append(f"{sum(vals) / len(vals):13.3f}")
        print(f"{arm:21s}" + "".join(cells))


def short(condition: str) -> str:
    """Column label for the printed table: 'depolarizing=0.005' -> 'dep=0.005'."""
    return condition.replace("depolarizing=", "dep=")[:12]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--env", default="FrozenLake-v1", choices=sorted(TASK_PACKAGES))
    p.add_argument("--base", default="depolarizing=0.02",
                   help="the noise expected at deployment; arms are this, scaled")
    p.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.25, 1.0, 4.0],
                   help="training-noise multipliers of --base; 0 = clean (Experiment B)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--train-every", type=int, default=1)
    p.add_argument("--conditions", nargs="+", default=SWEEP_CONDITIONS)
    p.add_argument("--out", type=Path, default=Path("analysis_out/noise_aware"))
    p.add_argument("--resume", action="store_true",
                   help="skip arms whose rows/*.json already exists (use after a crash)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    deployment = scale_noise(args.base, 1.0)
    arms = [scale_noise(args.base, s) for s in args.scales]

    print(f"env={args.env}  deployment={deployment}  arms={arms}")
    print(f"seeds={args.seeds}  episodes={args.episodes}  "
          f"eval={args.eval_episodes} eps x {len(args.conditions)} conditions\n")

    for noise in arms:
        for seed in args.seeds:
            row = run_arm(args.env, seed, noise, args.base, args, args.out)
            if row is not None:
                save_row(row, args.out)

    # Rebuilt from rows/ so a resumed sweep still produces the full table.
    rows = load_rows(args.out)
    write_summary(rows, args.out / "summary.csv")
    print_table(rows, args.conditions, args.base)
    print(f"\nwrote {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
