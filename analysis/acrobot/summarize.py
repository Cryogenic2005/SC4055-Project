"""Summarize real Acrobot CSVs without importing JAX; no invented values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(run_dir):
    root = Path(run_dir)
    train_rows = []
    for path in sorted((root / "train").glob("Acrobot-v1_*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        rolling = df["return"].rolling(100, min_periods=100).mean()
        solved = df.loc[rolling >= -100, "episode"]
        train_rows.append(
            dict(
                model=df.model.iloc[0],
                seed=int(df.seed.iloc[0]),
                train_noise=df.train_noise.iloc[0],
                episodes=len(df),
                final100=float(df["return"].tail(100).mean())
                if len(df) >= 100
                else None,
                best100=float(rolling.max()) if len(df) >= 100 else None,
                solved_episode=int(solved.iloc[0]) if len(solved) else None,
                n_params=int(df.n_params.iloc[0]),
                wall_s=float(df.wall_s.iloc[-1]),
            )
        )
    eval_rows = []
    keys = [
        "model",
        "seed",
        "train_noise",
        "experiment",
        "eval_noise_type",
        "eval_noise_level",
    ]
    for path in sorted((root / "eval").glob("Acrobot-v1_*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        if df.duplicated([*keys, "episode"]).any():
            raise ValueError(f"Duplicate evaluation episodes: {path}")
        for group, values in df.groupby(keys, dropna=False):
            row = dict(zip(keys, group, strict=True))
            row.update(episodes=len(values), mean_return=float(values["return"].mean()))
            eval_rows.append(row)
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_csv(output / "training_per_seed.csv", index=False)
    evaluations = pd.DataFrame(eval_rows)
    evaluations.to_csv(output / "evaluation_per_seed.csv", index=False)
    if not evaluations.empty:
        # Training seed is the replicate, not the individual evaluation episode.
        aggregates = (
            evaluations.groupby(
                [
                    "model",
                    "train_noise",
                    "experiment",
                    "eval_noise_type",
                    "eval_noise_level",
                ],
                dropna=False,
            )
            .mean_return.agg(["count", "mean", "std"])
            .reset_index()
        )
        aggregates.to_csv(output / "evaluation_across_seeds.csv", index=False)
        print(aggregates.to_string(index=False))
    print(
        json.dumps(
            {
                "training_runs": len(train_rows),
                "evaluation_groups": len(eval_rows),
                "summary_dir": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    summarize(parser.parse_args().run_dir)
