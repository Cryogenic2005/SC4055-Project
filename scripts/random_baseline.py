"""Measure R_random (mean return of a uniform-random policy over 100 episodes).

    python scripts/random_baseline.py FrozenLake-v1          # your own task
    python scripts/random_baseline.py --all                  # all four

Writes/updates runs/random.csv with columns task,r_random (analysis spec).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qrl.config import TASKS, default_config_for  # noqa: E402
from qrl.envs import random_baseline  # noqa: E402
from qrl.logging_utils import write_random_baseline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default="runs/random.csv")
    a = ap.parse_args()
    tasks = list(TASKS) if a.all else [a.task]
    for t in tasks:
        # Use the same map/kwargs as the task's training config (FrozenLake: deterministic).
        r = random_baseline(t, a.episodes, **default_config_for(t).env_kwargs)
        write_random_baseline(a.out, t, r)
        print(f"{t:16s} R_random = {r:.3f}")
