"""Summarise CartPole training curves from runs/train/*.csv.

Writes one row per run with:
  - max rolling-100 return (best100)
  - final rolling-100 return (final100)
  - first solved episode (rolling-100 >= threshold)
  - total episodes and wall-clock seconds

Usage
-----
python analysis/cartpole/summarize_training.py \
  --glob "runs/train/CartPole-v1_*.csv" \
  --out analysis_out/cartpole/training_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

THRESHOLD = 475.0


def solved_episode(m100: pd.Series, threshold: float) -> int | None:
    hits = m100[m100 >= threshold]
    if hits.empty:
        return None
    return int(hits.index[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default="runs/train/CartPole-v1_*.csv")
    p.add_argument(
        "--out", type=Path, default=Path("analysis_out/cartpole/training_summary.csv")
    )
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    args = p.parse_args()

    rows: list[dict] = []
    for path in sorted(Path().glob(args.glob)):
        df = pd.read_csv(path)
        if df.empty:
            continue

        m100 = df["return"].rolling(100, min_periods=1).mean()
        solved = solved_episode(m100, args.threshold)

        rows.append({
            "run_file": path.name,
            "task": str(df["task"].iloc[0]),
            "model": str(df["model"].iloc[0]),
            "train_noise": str(df["train_noise"].iloc[0]),
            "seed": int(df["seed"].iloc[0]),
            "episodes": len(df),
            "final_return": float(df["return"].iloc[-1]),
            "best100": float(m100.max()),
            "final100": float(m100.iloc[-1]),
            "solved_ep": solved,
            "n_params": int(df["n_params"].iloc[-1]),
            "wall_s": float(df["wall_s"].iloc[-1]),
        })

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    if out.empty:
        print("no files matched")
        return

    print(
        out[
            [
                "run_file",
                "model",
                "train_noise",
                "seed",
                "best100",
                "final100",
                "solved_ep",
            ]
        ].to_string(index=False)
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
