# Acrobot Experiment C: completed noise-aware training

P3 / Wang Jianning. Retrieved from the private Google Drive run on 6 November
2026, after all three training seeds and six evaluation groups completed.

## Protocol

- Acrobot-v1, 6 qubits, 4 layers, 99 parameters; seeds 0, 1, 2.
- Depolarizing p=0.01 during 600 training episodes per seed. Each gate is
  followed by the same per-wire local channel defined in `qrl/noise.py`.
- Same batch size 32, update every 4 environment steps, Double DQN, reward
  scaling and other Acrobot hyperparameters as A; no reduced episode horizon.
- Best checkpoint selected using 10-episode validation under training noise.
  Independent evaluation uses 100 episodes per seed in each of two conditions:
  clean and depolarizing p=0.01, with the same environment-seed protocol as B.
- Exact analytic Pauli-transfer simulation accelerates local depolarization.
  It stores all 4^6 Pauli expectations; it does not remove noise, truncate the
  state, or approximate expectations with sampled trajectories. Tests compare
  outputs and gradients with the original PennyLane density-matrix backend.
- The v3 source is isolated from A/B's v2 run by a separate source/protocol lock
  and output directory. Numerical equivalence does not imply identical RL
  trajectories after floating-point changes; version differences are disclosed.

## Results

Higher raw return is better. Each table entry is a 100-episode seed mean.

| Seed | B: clean training, noisy evaluation | C: noisy training, noisy evaluation | C minus B | C: clean evaluation |
|---|---:|---:|---:|---:|
| 0 | -500.00 | -156.42 | +343.58 | -413.72 |
| 1 | -500.00 | -307.45 | +192.55 | -500.00 |
| 2 | -500.00 | -296.00 | +204.00 | -500.00 |
| Mean | -500.00 | -253.29 | +246.71 | -471.24 |
| Sample SD | 0.00 | 84.09 | 84.09 | 49.81 |

For the same three seed IDs, B's clean evaluation mean is -353.06. C's clean
mean is -471.24, a paired mean change of **-118.18**. The full five-seed B clean
mean is -345.48; it is not used as the baseline of the paired comparison.

All three C training logs contain exactly episodes 0--599. None reaches the
raw rolling-100 training threshold of -100. Final training rolling means are
-183.27, -279.97 and -331.87; these are distinct from held-out evaluation.
Training times are 80.15, 96.78 and 99.10 minutes (4.60 hours total), excluding
independent evaluation and environment setup.

## Interpretation and limits

Noise-aware training improves matched-noise evaluation for all three tested
seed IDs, recovering performance from the -500 timeout floor. Its poorer
clean evaluation indicates dependence on the training noise condition. This
supports a task/configuration-specific robustness tradeoff, not general
hardware-noise tolerance, quantum advantage, or statistical significance.
Only one training noise level and three training replicates were tested.
There is no noisy classical-control experiment here. The clean VQC baseline
is already weak; floor effects limit interpretation of B's failures.

## Files and validation

- `training_per_seed.csv`: three 600-episode training summaries.
- `evaluation_per_seed.csv`: six 100-episode evaluation means.
- `evaluation_across_seeds.csv`: two means and sample SDs (ddof=1).
- `paired_BC_common_seeds.csv`: derived comparison restricted to seeds 0--2.

The original 1,800 training rows and 600 evaluation rows were downloaded and
checked for coverage, duplicate episode keys, means, SDs, rolling scores and
solve status. Raw logs, checkpoints and resume snapshots stay in the private
Drive folder; its sharing settings were not changed:
https://drive.google.com/drive/folders/1qKdueYh5IxwTdPNJ5eDRvAxMC9o4n6iF

Source summary IDs: training `1cpl53pLVwCEUClu3mkPmN3YmwuGpkOr1`, per-seed
evaluation `1HP10UwpgakgwihQCHAakD8E0HhxPMiQm`, aggregate evaluation
`1wfrvkLUg72mCCTY7pZ1N-xSf68RbrkUO`. The separate v3 run finished at about
05:56 Singapore time on 6 November 2026.

See [ABC figures and reproduction](../overview/README.md).
