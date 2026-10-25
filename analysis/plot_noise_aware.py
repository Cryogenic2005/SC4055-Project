"""Figures for the noise-aware sweep produced by analysis/noise_aware_sweep.py.

Reads <out>/summary.csv and draws two panels:

  left   success rate vs. evaluation noise, one line per training arm
         -> does the arm trained at the deployment noise win where it matters?
  right  the "operating window" of each arm: the band of noise levels where it
         still succeeds.  This is the compact statement of the result -- noise-aware
         training does not just lift the curve, it moves the window.

Usage
-----
    python analysis/plot_noise_aware.py --summary analysis_out/fl_sweep/summary.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

CLEAN_X = 0.0    # the "none" condition is drawn at zero noise
SUCCESS = 1/3    # "still works" = at least one of the three seeds still solves it


def eval_level(condition: str) -> float:
    """'none' -> 0.0, 'depolarizing=0.01' -> 0.01."""
    return 0.0 if condition == "none" else float(condition.split("=", 1)[1])


def load(summary: Path) -> dict[str, dict[str, list[float]]]:
    """{train_noise: {eval_condition: [mean_return per seed]}}"""
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(summary, newline="") as fh:
        for row in csv.DictReader(fh):
            data[row["train_noise"]][row["eval_condition"]].append(float(row["mean_return"]))
    return data


def arm_scale(train_noise: str, base_level: float) -> float:
    """0.0 for the clean arm, otherwise the multiplier of the deployment noise."""
    return 0.0 if train_noise == "none" else float(train_noise.split("=", 1)[1]) / base_level


def window(curve: dict[str, float]) -> tuple[float, float] | None:
    """Lowest and highest noise level at which the arm still succeeds."""
    ok = [eval_level(c) for c, v in curve.items() if v >= SUCCESS]
    return (min(ok), max(ok)) if ok else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", type=Path, default=Path("analysis_out/fl_sweep/summary.csv"))
    p.add_argument("--out", type=Path, default=None, help="PNG path (default: next to summary)")
    p.add_argument("--env", default="FrozenLake-v1")
    p.add_argument("--base-level", type=float, default=0.01,
                   help="the deployment noise level the arms were scaled from")
    args = p.parse_args()

    data = load(args.summary)
    arms = sorted(data, key=lambda a: arm_scale(a, args.base_level))
    dep_conditions = sorted(
        (c for c in next(iter(data.values())) if c.startswith("depolarizing=")),
        key=eval_level,
    )

    # mean success over seeds per (arm, condition)
    mean = {a: {c: float(np.mean(v)) for c, v in data[a].items()} for a in arms}

    fig, (ax_curve, ax_window) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    # ---- left: the sweep -----------------------------------------------------
    for i, arm in enumerate(arms):
        xs = [CLEAN_X] + [eval_level(c) for c in dep_conditions]
        ys = [mean[arm]["none"]] + [mean[arm][c] for c in dep_conditions]
        colour = plt.cm.viridis(i / max(len(arms) - 1, 1))
        ax_curve.plot(xs, ys, "o-", color=colour, lw=2, ms=5,
                      label=f"trained at {arm} ({arm_scale(arm, args.base_level):g}x)")
        # seed spread as a faint band, so n=3 is visible rather than hidden
        spread = [np.ptp(data[arm]["none"])] + [np.ptp(data[arm][c]) for c in dep_conditions]
        ax_curve.fill_between(
            xs,
            np.clip(np.subtract(ys, spread), 0, 1),
            np.clip(np.add(ys, spread), 0, 1),
            color=colour,
            alpha=0.12,
        )
    ax_curve.axvline(0.01, color="0.4", ls="--", lw=1)
    ax_curve.text(0.0104, 0.55, "deployment noise", color="0.35", fontsize=9, rotation=90)
    ax_curve.set_xlim(-0.001, 0.032)
    ax_curve.set_xlabel("evaluation depolarizing noise  (left-most point = noiseless)")
    ax_curve.set_ylabel("greedy success rate")
    ax_curve.set_ylim(-0.05, 1.05)
    ax_curve.set_title("Training noise sets where the agent works")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(fontsize=8, loc="lower left", framealpha=0.9)

    # ---- right: operating windows -------------------------------------------
    for i, arm in enumerate(arms):
        band = window(mean[arm])
        colour = plt.cm.viridis(i / max(len(arms) - 1, 1))
        if band is None:
            ax_window.text(0.001, i, "never succeeds", va="center", fontsize=9, color="0.4")
            continue
        lo, hi = band[0], band[1]
        ax_window.broken_barh([(lo, hi - lo)], (i - 0.32, 0.64), color=colour, alpha=0.85)
    ax_window.axvline(0.01, color="0.4", ls="--", lw=1)
    ax_window.set_xlim(-0.001, 0.032)
    ax_window.set_yticks(range(len(arms)))
    ax_window.set_yticklabels([f"{arm_scale(a, args.base_level):g}x" for a in arms])
    ax_window.set_ylabel("training noise (x deployment)")
    ax_window.set_xlabel("tested noise levels where at least one seed still solves")
    ax_window.set_title("Noise-aware training moves the window, not just the level")
    ax_window.grid(alpha=0.25, axis="x")

    fig.suptitle(f"{args.env}: does training under noise help at the noise you deploy into?", y=1.02)
    fig.tight_layout()
    out = args.out or args.summary.with_name("noise_aware.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")

    # ---- the same numbers, as text ------------------------------------------
    print(f"\n{'training arm':>14s}  {'success at':>10s}  {'operating window':>26s}")
    for arm in arms:
        w = window(mean[arm])
        span = "never" if w is None else f"{w[0]:g} .. {w[1]:g}"
        print(f"{arm_scale(arm, args.base_level):>13g}x  {mean[arm]['none']:>10.2f}  {span:>26s}")


if __name__ == "__main__":
    main()
