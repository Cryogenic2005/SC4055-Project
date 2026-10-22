# Results analysis spec

How the team's results get turned into the report's tables and figures. Follows the protocol in *Noisy quantum RL team plan.md* (Experiments A, B, C; normalized score; noise tolerance).

## 1. What every logger must write

Everyone writes plain CSV, one row per episode. Same column names on every task, or the aggregation breaks.

**Training log** (`runs/train/<task>_<model>_<seed>.csv`), Experiments A and C:

| column | example | notes |
|---|---|---|
| task | CartPole-v1 | Gymnasium id: FrozenLake-v1, CartPole-v1, Acrobot-v1, LunarLander-v3 |
| model | vqc | vqc, mlp_small (parameter-matched), mlp_large, encoder_only (LunarLander ablation) |
| experiment | A | A = noise-free training, C = noise-aware training |
| train_noise | none | `none`, or `depolarizing=0.01`, `coherent=0.1`, `preset=heron2026` |
| seed | 0 | |
| episode | 0..N | |
| return | 187.0 | undiscounted episode return |
| n_params | 62 | trainable parameter count |
| wall_s | 41.3 | seconds since run start (for the cost table) |

**Evaluation log** (`runs/eval/<task>_<model>_<seed>.csv`), Experiments B and C, 100 greedy episodes per condition:

| column | example | notes |
|---|---|---|
| task, model, experiment, train_noise, seed | | as above; B = trained noise-free, evaluated under noise |
| eval_noise_type | depolarizing | none, shots, coherent, depolarizing, readout, preset |
| eval_noise_level | 0.01 | the number from the noise menu; `0` for none; preset name for presets |
| episode | 0..99 | |
| return | | |

Always include an `eval_noise_type=none` block, which is the noise-free reference point for every sweep.

**Random baseline** (`runs/random.csv`): columns `task,r_random`, mean return of a uniform-random policy over 100 episodes, measured by each task owner.

## 2. Metrics

- **Normalized score** = (R − R_random) / (R_threshold − R_random). Thresholds: FrozenLake 0.70, CartPole 475 (registered value; docs say 500, we state 475), Acrobot −100, LunarLander 200.
- **Final score** per seed = mean return over the last 100 training episodes. Reported as mean across seeds with a 95% percentile-bootstrap CI (2,000 resamples).
- **Solved** = the 100-episode rolling mean reaches the threshold; we report the fraction of seeds that solve and the median episode at which they do.
- **Noise tolerance** = the largest noise level that keeps 90% of the noise-free normalized score, by log-linear interpolation between tested levels. Flagged "below min tested" or "above max tested" when the sweep does not bracket it.
- **C vs B**: difference in mean normalized score at the same noise level, with Welch's t-test and Mann–Whitney U as a distribution-free check. With 5 seeds per arm, Mann–Whitney's smallest possible two-sided p is 0.008, so we report effect sizes and CIs first and p-values second, and do not correct away results.

## 3. Outputs (`analyze.py`)

| Output | Content | Report use |
|---|---|---|
| `table_training.csv` | Experiment A/C final score, CI, solved fraction, episodes to solve, params, wall time per task × model | Main results table; VQC vs both MLPs |
| `fig1_learning_curves.png` | Normalized learning curves, mean ± 95% CI band, one panel per task | Experiment A figure |
| `fig2_noise_sweep.png` | **Headline**: normalized score vs noise strength, one line per task, one panel per noise type | Experiment B |
| `table_eval.csv` | Every (task, model, noise condition) point incl. presets | Appendix; preset rows go in a small Heron-2026 / NISQ-2022 / stress table |
| `fig3_tolerance_vs_qubits.png` + `table_tolerance.csv` | Noise tolerance vs qubit count | Scaling argument |
| `fig4_c_vs_b.png` + `table_c_vs_b.csv` | Noise-aware vs noise-free training at the same noise | Experiment C |

Run: `python analyze.py --train "runs/train/*.csv" --eval "runs/eval/*.csv" --random runs/random.csv --out results/`

Test without real data: `python make_dummy_data.py dummy` then point `analyze.py` at `dummy/`. The dummy numbers are invented and must never appear in the report.

## 4. Claims to check once real data arrives

1. Does CartPole reproduce Skolik's depolarizing threshold (solved at p ≤ 0.01, failed at p = 0.1)?
2. Does tolerance fall from FrozenLake/CartPole (4q) to Acrobot (6q)? Note LunarLander is 4 qubits in the hybrid design, so it tests task difficulty, not circuit size.
3. Is shot budget a bigger loss than Heron-2026 gate noise?
4. Does noise-aware training (C) recover score over B, and at which levels?
5. Does the parameter-matched MLP match the VQC, and does the large MLP beat both?
6. For LunarLander, does the encoder-only ablation match the hybrid? If so, the VQC is not carrying the result.
7. Any non-monotone effects (noise helping), reported as found.

## 5. Checked against the SC4055 brief

The analysis feeds the **Results and critical analysis** criterion (20%) directly, and supports **Implementation and experimental design** (25%) and **Report and code quality** (5%). The brief asks for:

- **Results presented and interpreted, with limitations.** Every figure above comes with CIs over seeds, and the report's results section should end with a limitations list: simulator only (no hardware), 5 seeds per arm, noise models are approximations of Heron calibration data, LunarLander's VQC is a small part of a hybrid model, and the classical baselines beat the VQC wherever they do.
- **Code that reproduces the reported results.** Every number and figure in the report must come out of `analyze.py` run on the committed CSVs, with no hand-edited numbers. Commit the raw `runs/` CSVs (or a link to them) next to the code, and record the config and seed of each run in its filename or a sidecar JSON.
- **The video's "what the results mean" segment** uses fig2 (noise sweep) and fig3 (tolerance vs qubits) as its two slides, plus one line from `table_training.csv` comparing the VQC to the MLPs.
