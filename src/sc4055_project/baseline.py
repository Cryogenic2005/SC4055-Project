"""Measure R_random (mean return of a uniform-random policy over 100 episodes).

    python src/sc4055_project/baseline.py FrozenLake-v1          # your own task
    python src/sc4055_project/baseline.py --all                  # all four

Writes/updates runs/random.csv with columns task,r_random (analysis spec).
"""

import argparse

from sc4055_project.qrl.config import TASKS, default_config_for
from sc4055_project.qrl.envs import random_baseline
from sc4055_project.qrl.logging_utils import write_random_baseline


def run_task(task: str, episodes: int, output_file: str = "runs/random.csv"):
    # Use the same map/kwargs as the task's training config (FrozenLake: deterministic).
    r = random_baseline(task, episodes, **default_config_for(task).env_kwargs)
    write_random_baseline(output_file, task, r)
    print(f"{task:16s} R_random = {r:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", type=str, choices=TASKS.keys())
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default="runs/random.csv")
    a = ap.parse_args()

    if a.all:
        tasks = list(TASKS)
    elif a.task:
        tasks = [a.task]
    else:
        raise ValueError("Either --all or a task name must be specified")

    for task in tasks:
        run_task(task, a.episodes, a.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
