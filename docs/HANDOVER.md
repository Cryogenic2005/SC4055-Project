# Handover: noise-aware training work (P4)

Written 1 Nov 2026 for whoever picks this up next — a teammate, or another agent with no
memory of these sessions. It is the working log: what was done, what was learned, what is
still wrong, and where the bodies are buried.

Read this first, then `docs/REPORT_noise_aware.md` (the results write-up).

---

## 1. The project, in one paragraph

SC4055 group project. The research question is **"how much hardware-like noise can a
variational-quantum-circuit RL agent tolerate, and does training under noise win any of it
back?"** Four tasks of increasing size share one codebase and one noise menu: FrozenLake
(P2), CartPole (P1), Acrobot (P3), LunarLander (P4 — this work). Protocol: Experiment A =
clean training, B = clean agent evaluated under noise, C = retrained under noise.
Details: `README.md`, `docs/team_plan.md`, `docs/analysis_spec.md`, `docs/run_matrix.md`.

My assignment was LunarLander (P4) plus the shared MLP baselines and environment wrappers.
The noise-aware experiment (C) was originally marked "evaluation only, too expensive" for
LunarLander; most of what follows is about why that was right, and what happened when we
did it anyway.

---

## 2. Repo map — where everything lives

```
src/sc4055_project/
  qrl/noise.py            single entry point for all noise; string language
  qrl/config.py           Config dataclass + per-task defaults (default_config_for)
  qrl/agent.py            SHARED generic model interface  <- P1's refactor, not merged
  qrl/dqn.py              SHARED generic training loop    <- P1's refactor, not merged
  qrl/envs.py             make_env, encode_obs, one_hot_obs, random_baseline
  tasks/lunarlander/      P4's task: agent.py, train.py, evaluate.py, README
  tasks/frozenlake/       P2's task (train/evaluate still per-task on main)
  qrl/logging_utils.py    CSV loggers; column names are a contract with analysis/

analysis/
  noise_aware_sweep.py    Experiment C runner: trains an arm, evaluates it, resume-safe
  policy_probe.py         FrozenLake per-state probe (solved / optimal-action agreement)
  plot_noise_aware.py     figures straight from summary.csv
  lunarlander/
    README.md             the four LunarLander bugs, with before/after evidence
    inspect_weights.py    "is my Q-function diverging?" — prints w_out, Q range, saturation
    watchdog.py           supervises a long sweep; restarts on stall
    shutdown_when_done.py arms a shutdown for when the sweep finishes

docs/
  REPORT_noise_aware.md   the extensive results report
  HANDOVER.md             this file
  noise_aware_findings.md FrozenLake Experiment C write-up + the Exp B reproducibility warning
  lunarlander_noise_aware_results.md  LunarLander Experiment C (negative) with all 6 seeds
  p2_frozenlake_results.md            P2's own write-up (not mine)

analysis_out/
  fl_sweep/               FrozenLake Experiment C: rows, checkpoints, figures, summary
  lunarlander_sweep/      LunarLander Experiment C: rows, logs, summary
  p2_frozenlake/          P2's committed raw data
```

---

## 3. Timeline of what was done

**25 Oct — FrozenLake first.**
Read QuantumNAT end to end. Implemented its noise-injection idea as
`analysis/noise_aware_sweep.py` (arms = training-noise levels, everything else fixed).
Ran 5 arms × 3 seeds. Found the hump: each arm peaks at its own training noise; matched
noise lifts solved-from-all-starts 0.61 → 0.91; tolerance 0.0125 → 0.015 → 0.025 while
clean performance falls 1.00 → 0.79 → 0.52. Wrote `analysis/policy_probe.py` to measure
*why* (optimal-action agreement, 29% vs 97%). Committed `598aaf1`.

**25–26 Oct — LunarLander: it wasn't learning at all.**
Spent an hour trying to run Experiment C there and discovered the clean agent could not
learn LunarLander (1000 episodes, worse than random). Inspected the saved weights and found
the value function had diverged: `|w_out| ≈ 50` against TD targets of O(1). Traced it to
`lr_output = 0.1`. Fixed three things (readout LR, observation scaling, ε decay) → the
agent learned to −110. Then found maximisation bias still collapsing it late; added Double
DQN → −14.6, best checkpoint at the end. Committed `598aaf1`, `2bb8e18` (rebased over P2's
merge `1434238`).

**29–30 Oct — P2's numbers don't reproduce.**
Re-ran P2's Experiment B on current code: their committed seed-0 checkpoint still gives
0.00 at p = 0.01 (so the eval code is fine), but fresh training gives 1.00, 0.00, 0.00,
0.00, 1.00 — 0.40, not 0.00. The cliff sits on the test point. Documented in
`docs/noise_aware_findings.md`, committed `d8cbc4c`.

**30 Oct 22:47 – 31 Oct 05:29 — the 16-hour LunarLander Experiment C.**
Measured the real cost first (2.1 ms per clean gradient step vs 49.5 ms per noisy one, 24×)
and sized the run: 3 clean seeds + 3 seeds trained at `depolarizing=0.005`, 1000 episodes,
five evaluation conditions. Watched by `watchdog.py` (restarts on stall), shut down
automatically at 05:29. Result: **negative** — trained arm −401 vs clean −358 at the
matched noise, both below random. Committed `b650e54`, `956a668`.

**31 Oct — reviewed P1's branches.**
`p1/Cartpole` and `p1/implement-generics` refactor everything into `qrl/agent.py` +
`qrl/dqn.py` and add CartPole. They preserve `OBS_SCALE` and the config flags, **but the
new `qrl/dqn.py` ignores `cfg.double_dqn` and `cfg.grad_clip`** — see §6.

**1 Nov — report + this handover.** Committed the resume support for the sweep (`140a637`).

---

## 4. Findings you can rely on

Full detail and tables in `docs/REPORT_noise_aware.md`. One line each:

1. **Noise-aware training works, at the noise it was trained for.** FrozenLake matched
   noise: 0.61 → 0.91 solved-from-all-starts; 1/3 → 3/3 seeds. Replicated by P2 (5 seeds).
2. **It is a trade.** Tolerance 0.0125 → 0.025 at 2× training noise, clean 1.00 → 0.52.
3. **Under-training is worse than nothing.** 0.25× scores 0.00 at deployment vs 0.33 clean.
4. **Mechanism = Q-margin.** At p = 0.02: 29% optimal actions clean-trained, 97% at 2×.
5. **LunarLander: negative.** 104 parameters cannot learn the task and absorb the noise.
6. **Exp B's 0.00 at p = 0.01 is a coin flip** — 0.40 on fresh seeds. Quote distributions.
7. **LunarLander's four bugs** were optimisation, not capacity: `mlp_large` with 48× the
   parameters did *worse*.

---

## 5. Current state (as of 1 Nov)

* `main` is clean and pushed: `140a637`.
* Nothing is running. No background jobs, no armed watchdogs, no pending shutdown.
* All results are committed under `analysis_out/` (about 1.5 MB total; checkpoints are
  tiny pickles, so the raw data is in the repo and the figures render on GitHub).
* `pytest` → 50 passed, 1 skipped (the skip is P2's Qiskit cross-check, needs the
  `qiskit` extra). `ruff check .` → 1 pre-existing warning in
  `tests/sc4055_project/qrl/test_noise.py:100` (unused `spec`), not mine, not blocking.
* CI runs on every push to main: pytest blocks, pyright/ruff are warnings only.

---

## 6. Traps (read before touching anything)

**(a) P1's generic loop drops the Double DQN fix.**
`qrl/dqn.py` on `p1/implement-generics` and `p1/Cartpole` computes
`q_fn(tgt, s2, k).max(axis=1)` and never reads `cfg.double_dqn` or `cfg.grad_clip`. The
config still sets both for LunarLander, so they will look enabled while doing nothing, and
LunarLander will silently go back to collapsing after episode 750. Port the two-line change
*before* merging, then re-run one seed to confirm it still lands.

**(b) Same seed ≠ same policy.**
Identical `Config` and seed produced −14.6 in one run and −61 in another, differing only in
whether periodic evaluation consumed the RNG stream. Any change to evaluation cadence,
logging, or loop order reshuffles every policy. Do not compare runs across code versions
without re-running the baseline.

**(c) Never run two JAX training processes at once on this machine.**
Observed twice: two concurrent runs both stalled at ~60 s of logged wall time and made no
further progress until killed. Run sequentially, or accept the stall.

**(d) Noisy simulation is 24× slower.** Measured: 2.1 ms per clean gradient step, 49.5 ms
with `depolarizing=0.005` (`default.mixed` density matrix). Budget ~4 h per 1000-episode
LunarLander arm; FrozenLake is ~2 min per arm because its episodes are ~10 steps.

**(e) The two FrozenLake metrics disagree in resolution.**
Single greedy episode = 1 bit per seed (values move in thirds). Solved-from-all-starts =
11 bits. Only the latter can carry a claim with 3 seeds. `policy_probe.py` implements it by
reading the transition table from the env — do not hard-code the 4×4 map.

**(f) LunarLander is a seed lottery.** 1 of 3 seeds learns. Means across seeds describe no
actual agent; always show the per-seed split.

**(g) `best.pkl`, not `final.pkl`.** `train.py` selects the best checkpoint on a greedy
validation eval. With 10 episodes the standard error is ~30 points on LunarLander, so use
≥20 episodes for selection or it picks on noise.

**(h) Windows/PowerShell.** `git commit -F <file>` for multi-line messages (here-strings
work); PowerShell `.Replace()` with a literal `\n` writes a backslash-n, not a newline —
use the editor tools for multi-line edits.

**(i) Judge nothing by the stdout log.** Python buffers; the training CSV is flushed every
episode. Both long runs needed `python -u` and were monitored through `runs/train/*.csv`.

---

## 7. What was *not* done (do not assume it exists)

* **No normalisation, no quantisation.** Only noise injection is implemented. The reasons
  are in `docs/REPORT_noise_aware.md` §2: normalisation needs a batch that a one-step-at-a-
  time RL loop does not have; quantising Q-values destroys the argmax when the four values
  are within ~0.06.
* **No hardware.** Everything is simulated (`default.qubit` / `default.mixed`).
* **No policy-gradient experiments.** All DQN.
* **P1's branches were read, not modified, not merged.**
* **No Acrobot or CartPole noise-aware runs from me** — those are P3's and P1's.

---

## 8. If you have an hour: what I would do next

1. **Port Double DQN into `qrl/dqn.py`** and add a test that the two target paths differ.
   (~30 min; prevents a silent regression.)
2. **Settle the Exp B reporting question** with P2: either quote the seed distribution, or
   move the Experiment C comparison point to p = 0.05 where every seed is 0.00.
3. **Test the "errors compound" hypothesis** on LunarLander: apply noise to a fixed
   fraction of gates, see whether survival scales with the number of noisy steps. ~20 min
   on an existing checkpoint, and it converts the §5.3 explanation from plausible to tested.

If you have a day: implement EMA normalisation on the measurement features and measure how
much of the FrozenLake gain survives without a true batch. That is the missing two-thirds
of QuantumNAT and the only part of the paper's theory not yet tested in an RL setting.

---

## 9. Environment notes

* Windows 11, 16 cores, CPU-only JAX. `.venv\Scripts\python.exe` (Python 3.12).
* jax 0.10.2, pennylane 0.45.1, gymnasium 1.4.0, optax 0.2.8, matplotlib 3.11.2.
* Box2D needs `pip install swig` before `gymnasium[box2d]`.
* `qrl/noise.py` documents a live PennyLane bug: `default.mixed(readout_prob=p)` raises
  `TypeError` on analytic measurements, so readout error is applied as an explicit
  `BitFlip` on the measured wires instead. Do not "fix" that by passing `readout_prob`.
