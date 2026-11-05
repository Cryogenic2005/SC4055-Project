# P3: Acrobot-v1

## Recovery v2 (5 Nov 2026): use the new notebook instead of the old runner

`analysis/acrobot/recoverable.py` requires mounted Google Drive, checks source
and protocol hashes, skips completed training seeds, and writes all outputs
directly to Drive. Evaluation persists a condition record after EVERY episode
and resumes those records. A partially finished training seed restarts from
its beginning; replay/optimizer/RNG training state is not resumed.

Finite-shot evaluation now compiles the same unitary circuit to probabilities
and samples the joint Z distribution with a multinomial draw, retaining the
requested shot count and inter-qubit sampling correlations. Tests compare the
probability circuit's expectations to the shared circuit and check sampling
variance/correlation. This is not a switch to exact-expectation evaluation.
The sampling RNG differs from v1; do not merge v1 and v2 shot records as if
they were the same run. New runtime tests must pass in Colab before the
notebook launches the full experiment. The older instructions below describe
v1 only; its overwriting evaluation and non-resumable stage runner are obsolete.

Local branch `p3/acrobot`, based on P1's `p1/Cartpole` at `c41f7c1`.
Not pushed or merged. Main still has a stub shared trainer: do not copy this
task alone to main without the P1 framework.

## Initial configuration (not yet demonstrated to solve the task)

Six arctan observation features feed six qubits. Four re-uploading layers use
the shared RX(pi*x*lambda), RX/RY/RZ and CZ-ring circuit. Three Z expectations
times trainable output weights give Q values for three actions.

| Model | Architecture | Parameters |
|---|---|---|
| VQC | 6 qubits, 4 layers | 99 |
| mlp_small | 6->10->3 ReLU | 103 (4.0% above VQC) |
| mlp_large | 6->64->64->3 ReLU | 4,803 |

All models receive the same arctan features. Reward scaling 0.01 affects TD
updates only, never logged returns. Double DQN, replay 50,000, batch 32,
learning starts 1,000, train every 4 steps, target update every 100 gradient
steps. Circuit/input LR 0.001, output LR 0.01, MLP LR 0.001. Task optimizer
applies per-group gradient clipping at 1.0. Default 600 episodes, epsilon
1.0->0.05 over 400 episodes. Validation every 50 episodes, 10 episodes per
check; best validation checkpoint is independently evaluated.

These are engineering starting points, not measured successful settings.
Tune on labelled pilots, then freeze the A/C configuration and budget.

## Smoke and pilot (Python 3.12, after installing project)

```bash
python -m pytest tests/sc4055_project/tasks/test_acrobot.py -q
acrobot train --quick --max-steps 20 --learning-starts 8 --batch-size 4 --eval-episodes 1 --run-dir runs/acrobot_smoke
acrobot train --model mlp_large --seed 0 --episodes 100 --run-dir runs/acrobot_pilot_mlp
acrobot train --model vqc --seed 0 --episodes 100 --run-dir runs/acrobot_pilot_vqc
acrobot evaluate --checkpoint runs/acrobot_pilot_vqc/checkpoints/Acrobot-v1_vqc_0/best.pkl --conditions none depolarizing=0.01 --episodes 20 --run-dir runs/acrobot_pilot_vqc
```

The smoke's 20-step horizon is debugging only. Formal experiments must use
the normal 500-step horizon. A 100-episode pilot is not a formal result.

## Formal protocol (after a clean pilot learns)

```bash
# A: VQC and both MLPs, 5 seeds, plus independent clean evaluation
python analysis/acrobot/run_experiments.py A --seeds 0 1 2 3 4
# B: five clean VQCs, 17 conditions, 100 greedy episodes per condition
python analysis/acrobot/run_experiments.py B --seeds 0 1 2 3 4
# C: depolarizing=0.01 training, 3 seeds, clean and matched evaluation
python analysis/acrobot/run_experiments.py C --seeds 0 1 2
python analysis/acrobot/summarize.py runs/acrobot_formal
```

Preview any stage with `--dry-run`. Stages default to 600 training episodes;
use the same `--episodes` budget in A and C. Run sequentially. The stage
runner refuses to overwrite existing training logs; it does not resume
training. Restart interrupted seeds in a new directory and download results
between runs because Colab runtime storage is temporary.

## Evaluation and reporting

- Testing seeds start at 20000, separate from validation seeds. Episode
  environment seeds are paired across conditions. Every sweep includes none.
- Re-evaluation overwrites this run's CSV; full B replaces A's clean-only
  evaluation with a superset. Different evaluation protocols need different
  output directories. Only load trusted checkpoints.
- CLI accepts `--config path/to/config.json` to repeat a saved training config.
  In this mode the saved configuration takes precedence over other flags.
- Stage manifests save exact commands and source SHA-256 hashes. Training
  saves configuration, package versions, checkpoints and a summary.
- Solved: raw rolling mean over 100 training episodes >= -100. Report held-out
  greedy returns separately. A lucky single episode is not solved.
- Training seeds are replicates, not individual evaluation episodes. Compare
  C vs B at depolarizing=0.01, and do not claim qubit count alone caused a
  cross-task difference (task difficulty and training quality also vary).

## Verification status, 4 Nov 2026

Windows application control blocks the pinned JAX native DLL on this machine.
No Acrobot training result has been measured here. Use the supplied Colab
notebook and source archive; it runs the task tests before experiments.
Runtime checks must not be labelled passed until they execute successfully.
