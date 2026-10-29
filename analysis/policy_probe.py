"""How much noise does each trained policy tolerate, state by state?

A single greedy episode on FrozenLake is one bit: solved or not.  That is far too
coarse to compare training arms with three seeds, so this script measures the same
thing at two finer resolutions, using the environment's own transition table:

  solved   fraction of the 11 decision states from which the greedy policy still
           reaches the goal.  A single episode always starts at S, so this is the
           "robustness over starting positions" version of the usual success rate,
           and it moves in 1/11 steps instead of all-or-nothing.
  agree    fraction of those states where the greedy action is an optimal action.
           This is what `solved` is made of, and it degrades before `solved` does,
           so it shows the degradation earlier.

Both are cheap: 16 forward passes per noise level, then pure numpy rollouts.

Usage
-----
    python analysis/policy_probe.py --runs analysis_out/fl_sweep/runs
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sc4055_project.qrl.config import default_config_for
from sc4055_project.qrl.noise import parse_noise
from sc4055_project.tasks.frozenlake.agent import make_q_fn, state_to_bits

N_STATES = 16


def frozen_lake_tables(n_states: int = N_STATES):
    """The 4x4 map's transition table, taken from the env rather than hard-coded."""
    env = gym.make("FrozenLake-v1", is_slippery=False)
    transitions = env.unwrapped.P  # type: ignore[attr-defined]
    env.close()
    return transitions


def optimal_q(transitions, n_states: int = N_STATES, gamma: float = 0.999) -> np.ndarray:
    """Q*(s, a) by value iteration, for the `agree` measure."""
    v = np.zeros(n_states)
    for _ in range(2000):
        q = np.zeros((n_states, 4))
        for s in range(n_states):
            for a, branches in transitions[s].items():
                q[s, a] = sum(p * (r + gamma * (0.0 if done else v[s2]))
                              for p, s2, r, done in branches)
        if np.max(np.abs(q.max(axis=1) - v)) < 1e-12:
            v = q.max(axis=1)
            break
        v = q.max(axis=1)
    return q


def decision_states(transitions, n_states: int = N_STATES) -> list[int]:
    """States where a choice still matters: not a hole, not the goal."""
    return [
        s for s in range(n_states)
        if not all(done for branches in transitions[s].values() for _, _, _, done in branches)
    ]


def solved_from(q: np.ndarray, transitions, start: int, max_steps: int = 100) -> bool:
    """Roll the greedy policy out of `start` using the transition table."""
    state, total = start, 0.0
    for _ in range(max_steps):
        action = int(np.argmax(q[state]))
        _, state, reward, done = max(transitions[state][action], key=lambda b: b[0])
        total += reward
        if done:
            return total > 0
    return False


def probe(params, cfg, transitions, q_star, states, level: float) -> tuple[float, float]:
    """(solved fraction, optimal-action agreement) at one evaluation noise level."""
    spec = parse_noise("none" if level == 0 else f"depolarizing={level:g}")
    q_fn = make_q_fn(cfg.n_layers, spec, cfg.data_reuploading, seed=0)
    bits = state_to_bits(np.arange(N_STATES))
    q = np.asarray(q_fn(params, jnp.asarray(bits), jax.random.PRNGKey(0)))

    solved = float(np.mean([solved_from(q, transitions, s) for s in states]))
    best = q.argmax(axis=1)
    # FrozenLake has ties (two equally short paths), so "optimal" means within 1e-9.
    optimal = np.abs(q_star.max(axis=1)[:, None] - q_star) < 1e-9
    agree = float(np.mean([optimal[s, best[s]] for s in states]))
    return solved, agree


def checkpoint_for(run_root: Path, noise: str, seed: int, tag: str) -> Path:
    """Where the sweep left this arm's final weights."""
    name = f"FrozenLake-v1_vqc_{seed}"
    if noise != "none":
        name += "_" + noise.replace("=", "")
    return run_root / "checkpoints" / f"{name}_{tag}" / "final.pkl"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("analysis_out/fl_sweep/runs"))
    p.add_argument("--arms", nargs="+", default=["none", "depolarizing=0.0025",
                                                 "depolarizing=0.005", "depolarizing=0.01",
                                                 "depolarizing=0.02"])
    p.add_argument("--tags", nargs="+", default=None,
                   help="arm tag per --arms entry (default: T<scale> from the 0.01 base)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--levels", type=float, nargs="+",
                   default=[0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03])
    p.add_argument("--out", type=Path, default=Path("analysis_out/fl_sweep/policy_probe.csv"))
    args = p.parse_args()

    cfg = default_config_for("FrozenLake-v1")
    transitions = frozen_lake_tables()
    q_star = optimal_q(transitions)
    states = decision_states(transitions)
    tags = args.tags or [
        f"T{(0.0 if a == 'none' else float(a.split('=')[1]) / 0.01):g}" for a in args.arms
    ]

    rows = []
    for arm, tag in zip(args.arms, tags, strict=True):
        for seed in args.seeds:
            path = checkpoint_for(args.runs, arm, seed, tag)
            if not path.exists():
                print(f"  (missing {path.name}, skipped)")
                continue
            with open(path, "rb") as fh:
                params = pickle.load(fh)
            for level in args.levels:
                solved, agree = probe(params, cfg, transitions, q_star, states, level)
                rows.append(dict(train_noise=arm, scale=float(tag[1:]), seed=seed,
                                 eval_noise=level, solved=solved, agree=agree))
        print(f"  probed {arm}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")

    # ---- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    for i, arm in enumerate(args.arms):
        sub = [r for r in rows if r["train_noise"] == arm]
        if not sub:
            continue
        colour = plt.cm.viridis(i / max(len(args.arms) - 1, 1))
        for axis, key in zip(axes, ("solved", "agree")):
            xs = sorted({r["eval_noise"] for r in sub})
            ys = [np.mean([r[key] for r in sub if r["eval_noise"] == x]) for x in xs]
            axis.plot(xs, ys, "o-", color=colour, lw=2, ms=4,
                      label=f"trained at {arm} ({sub[0]['scale']:g}x deployment)")
    for axis, title in zip(axes, ("Behaviour: solved from how many starts?",
                                  "Mechanism: is the action still optimal?")):
        axis.axvline(0.01, color="0.4", ls="--", lw=1)
        axis.set_xlabel("evaluation depolarizing noise")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.25)
        axis.set_title(title, fontsize=10)
    axes[0].set_ylabel(f"fraction of {len(states)} starting states solved")
    axes[1].set_ylabel(f"fraction of {len(states)} states with an optimal action")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Training under noise keeps the policy right at higher noise levels", y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    png = args.out.with_name("policy_probe.png")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
