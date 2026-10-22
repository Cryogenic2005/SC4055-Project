"""CSV loggers. Column names are the contract with analysis/analyze.py
(see docs/analysis_spec.md). Do not rename columns here without telling P3.

Training log  runs/train/<task>_<model>_<seed>.csv
    task, model, experiment, train_noise, seed, episode, return, n_params, wall_s
Eval log      runs/eval/<task>_<model>_<seed>.csv
    task, model, experiment, train_noise, seed, eval_noise_type, eval_noise_level, episode, return
Random        runs/random.csv
    task, r_random
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

TRAIN_COLUMNS = ["task", "model", "experiment", "train_noise", "seed",
                 "episode", "return", "n_params", "wall_s"]
EVAL_COLUMNS = ["task", "model", "experiment", "train_noise", "seed",
                "eval_noise_type", "eval_noise_level", "episode", "return"]
RANDOM_COLUMNS = ["task", "r_random"]


class CsvLogger:
    """Append-only CSV writer that writes a header once and flushes every row,
    so a Colab disconnect loses at most one episode."""

    def __init__(self, path: str | Path, columns: list[str], append: bool = False):
        self.path = Path(path)
        self.columns = columns
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not (append and self.path.exists() and self.path.stat().st_size > 0)
        self._fh = open(self.path, "a" if append else "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=columns, extrasaction="raise")
        if new_file:
            self._w.writeheader()
            self._fh.flush()

    def log(self, **row) -> None:
        missing = [c for c in self.columns if c not in row]
        if missing:
            raise KeyError(f"missing columns {missing}")
        self._w.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TrainLogger(CsvLogger):
    def __init__(self, path, *, task, model, experiment, train_noise, seed, n_params, append=False):
        super().__init__(path, TRAIN_COLUMNS, append=append)
        self.fixed = dict(task=task, model=model, experiment=experiment,
                          train_noise=train_noise, seed=seed, n_params=n_params)
        self.t0 = time.perf_counter()

    def episode(self, episode: int, ret: float) -> None:
        self.log(episode=episode, **{"return": float(ret)},
                 wall_s=round(time.perf_counter() - self.t0, 3), **self.fixed)


class EvalLogger(CsvLogger):
    def __init__(self, path, *, task, model, experiment, train_noise, seed, append=False):
        super().__init__(path, EVAL_COLUMNS, append=append)
        self.fixed = dict(task=task, model=model, experiment=experiment,
                          train_noise=train_noise, seed=seed)

    def episode(self, eval_noise_type: str, eval_noise_level: str, episode: int, ret: float) -> None:
        self.log(eval_noise_type=eval_noise_type, eval_noise_level=eval_noise_level,
                 episode=episode, **{"return": float(ret)}, **self.fixed)


def write_random_baseline(path: str | Path, task: str, r_random: float) -> None:
    """Upsert one task's row in runs/random.csv."""
    path = Path(path)
    rows: dict[str, float] = {}
    if path.exists():
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows[r["task"]] = float(r["r_random"])
    rows[task] = r_random
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RANDOM_COLUMNS)
        w.writeheader()
        for t, v in sorted(rows.items()):
            w.writerow({"task": t, "r_random": v})
