# Acrobot Experiment A: completed clean-training results

Owner: P3 / Wang Jianning. Results retrieved from the connected Google Drive
on 5 November 2026 (Singapore time). This directory is a fixed snapshot of A,
not a live B/C summary.

## Scope and provenance

- Acrobot-v1, six qubits, four re-uploading layers; DQN with Double-DQN targets.
- VQC, small MLP and large MLP; seeds 0--4; **600 training episodes each**.
- Environment time limit 500 steps. Raw environment rewards are reported;
  reward scaling 0.01 applies only inside the TD update.
- Independent clean evaluation: **100 greedy episodes per trained seed**, using
  the best checkpoint selected by periodic 10-episode validation, not final.pkl.
- Training solve criterion: rolling mean over 100 episodes >= -100.
- Random-policy baseline: -500 over 100 episodes (seed 0).
- Exact model/training defaults: `tasks/acrobot/agent.py:task_config`.
- The Colab v2 source bundle is frozen inside the notebook. Its ZIP SHA-256 is
  `7367e92e04f8d56e86f655abcc6bc99bace2e03dd2f437884ae012340f7d5cbf`.
- Drive contains `STAGE_A_COMPLETE.json`. All 15 training rows have 600 episodes;
  all 15 clean-evaluation rows have 100 episodes.

The evaluation logger uses experiment label **B** even for the clean reference
evaluated immediately after A. These 15 `noise_type=none`, `train_noise=none`
rows are the clean baseline, not evidence that the full noise sweep is complete.

## Main results

Higher return is better. Standard deviation is across five training seeds,
not across individual evaluation episodes.

| Model | Parameters | Held-out greedy mean +/- SD | Mean final-100 training return | Training solves | Mean training minutes/seed |
|---|---:|---:|---:|---:|---:|
| VQC | 99 | -345.480 +/- 104.644 | -393.944 | 0/5 | 28.97 |
| Small MLP | 103 | -226.046 +/- 113.756 | -228.560 | 0/5 | 3.17 |
| Large MLP | 4,803 | -147.880 +/- 74.476 | -172.608 | 0/5 | 3.52 |

Per-seed held-out returns:

| Seed | VQC | Small MLP | Large MLP |
|---|---:|---:|---:|
| 0 | -218.78 | -270.31 | -101.49 |
| 1 | -348.96 | -233.70 | -132.45 |
| 2 | -491.44 | -389.14 | -113.12 |
| 3 | -389.67 | -122.36 | -112.73 |
| 4 | -278.55 | -114.72 | -279.61 |

## Interpretation and limits

The large MLP has the best mean score. The VQC is worse on average than the
similarly sized small MLP and took substantially longer on the observed CPU
runtime. No model reaches the project's training solve criterion on any seed;
none of these held-out seed means reaches -100 either. Some best validation
scores were above -100, which must not be reported as independent solved results.

This is evidence about these architectures, defaults and 600-episode budgets,
not a general claim that quantum RL cannot learn Acrobot. The five-seed spread
is large. Hardware/runner and Drive I/O affect timing, so time comparisons are
descriptive. Weak clean VQC performance and the -500 floor limit what later
noise comparisons can establish. B/C must be analysed separately when complete.

## Files and reproduction

- `training_per_seed.csv`: unmodified training summary retrieved from Drive.
- `evaluation_per_seed.csv`: unmodified 15-row clean-evaluation summary.
- `evaluation_across_seeds.csv`: unmodified five-seed aggregation.
- Raw per-episode CSVs, checkpoints, configs, source-hash protocol and progress
  records remain in the team's private Drive folder:
  https://drive.google.com/drive/folders/1GIkojoUei326GRGgKIXSb7d2gNjsev-1
  Access is permission-controlled; this commit does not change sharing.

To execute the persist-first pipeline, open
`notebooks/SC4055_Acrobot_Recovery_v2.ipynb` in Colab and mount your own Drive.
It checks storage and runs tests before A/B/C. Completed training seeds are
skipped; evaluation resumes from saved episodes. An interrupted training seed
restarts from its beginning because replay/optimizer state is not checkpointed.

To rebuild summaries from the raw Drive run directory in that runtime:

```bash
python analysis/acrobot/summarize.py /content/drive/MyDrive/SC4055_Acrobot_Recovery/experiment_v2
```

## Validation recorded

The user-run Colab v2 notebook reported **9 tests passed in 83.22 seconds**
before these experiments. Local syntax and Ruff checks pass. The final local
verification loaded JAX successfully: **53 tests passed and 1 was skipped** in
the full run; eight other tests encountered Windows temporary-directory
permission errors. Re-running exactly those eight with TEMP/TMP and pytest's
base temporary directory inside the workspace gave **8 passed**. Thus the
combined verification covers **61 passing tests and 1 skipped test**, across
two runs, rather than claiming a single clean full-suite invocation. The
earlier JAX application-control failure did not recur in this verification.
The branch also includes the latest main's CartPole test-import fix.
No Actions training workflow is included or triggered by this branch push.
