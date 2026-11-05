# Acrobot Experiment B: completed noise sweep

P3 / Wang Jianning. Fixed snapshot retrieved from Google Drive on 5 November
2026, after `STAGE_B_COMPLETE.json` was written. Experiment C is in progress
and is not included in these tables.

## Protocol and coverage

- Five clean-trained VQC checkpoints (seeds 0--4), selected in Experiment A.
- Six qubits, four layers, 99 trainable parameters; no retraining in B.
- **17 conditions per checkpoint, 100 greedy episodes per condition**:
  **85 condition/seed groups, 8,500 evaluation episodes**.
- Evaluation environment seeds are paired across conditions, distinct from
  training-time validation seeds. All reported returns are raw environment
  rewards; a timeout at the normal 500-step horizon gives -500.
- Noise placement and conventions are defined by `qrl/noise.py`. Presets are
  simulated hardware-style models, not measurements on actual hardware.
- Finite shots use the v2 joint Z-basis multinomial sampler on compiled exact
  circuit probabilities, retaining finite-shot variability and correlations.
  These are not exact-expectation results and are not the old v1 sampler's RNG
  sequence. All results here come from the v2 run.
- The clean reference is reused from A. Its five seed means match A exactly.

## Results

Higher return is better. Each row is the mean and sample SD of five seed-level
means; the 100 episodes within a seed are not independent training replicates.

| Evaluation condition | Mean return | Across-seed SD |
|---|---:|---:|
| None | -345.480 | 104.644 |
| Shots = 1,000 | -474.630 | 41.323 |
| Shots = 100 | -495.550 | 5.218 |
| Shots = 10 | -497.826 | 1.171 |
| Coherent sigma = 0.05 | -489.234 | 17.424 |
| Coherent sigma = 0.1 | -496.026 | 7.388 |
| Coherent sigma = 0.2 | -497.552 | 2.601 |
| Depolarizing p = 0.001 | -497.848 | 3.415 |
| Depolarizing p = 0.01 | -500.000 | 0.000 |
| Depolarizing p = 0.05 | -500.000 | 0.000 |
| Depolarizing p = 0.1 | -500.000 | 0.000 |
| Readout p = 0.01 | -345.480 | 104.644 |
| Readout p = 0.02 | -345.480 | 104.644 |
| Readout p = 0.05 | -345.480 | 104.644 |
| Preset: Heron-2026 | -495.064 | 11.037 |
| Preset: NISQ-2022 | -500.000 | 0.000 |
| Preset: Stress | -500.000 | 0.000 |

## Interpretation and limitations

1. Finite shots substantially degrade the clean policy, with lower mean
   performance as the shot budget decreases in this sweep.
2. Coherent and gate-channel errors bring these agents close to the failure
   floor. At depolarizing p >= 0.01, all five seed means are -500.
3. Symmetric readout noise preserves every seed's clean return at these tested
   levels. With analytic expectations and Q_a = w_a <Z_a>, this channel scales
   all Q values by the same positive factor (1-2p), preserving greedy argmax.
   This observation does not establish immunity to general measurement errors.
4. The clean agents already perform poorly (-345.48 mean vs a -100 solve
   threshold). The -500 floor masks differences among several noisy conditions.
   Report this limitation rather than interpreting similar failure scores as
   similar robustness. B cannot establish a quantum advantage or a causal
   dependence on qubit count across different tasks.
5. B tests deployment sensitivity only. Whether noisy training helps requires
   the separate Experiment C comparison at the same evaluation noise.

## Data and reproducibility

- `evaluation_per_seed.csv`: exactly the 85 VQC rows for clean training;
  classical clean-reference rows were filtered out of the live shared summary.
- `evaluation_across_seeds.csv`: the corresponding 17 aggregated VQC rows.
- All values are copied without numerical edits. Coverage, uniqueness, episode
  counts, five-seed means/SDs and matching A clean references were verified.
- Per-episode CSVs, per-condition saved records, checkpoint hashes and protocol
  metadata remain in the private Drive run directory:
  https://drive.google.com/drive/folders/1GIkojoUei326GRGgKIXSb7d2gNjsev-1
  This commit does not alter sharing or upload credentials/checkpoints.
- Source summaries were fetched from Drive files
  `1RfdPvnobrN7n4-xBTviHG4CkYwXXuM3Y` and
  `1vwdXHlxckKsYdiQtzAFHvzC17Czyh3Rw`, modified at 14:30:54 UTC
  on 5 November 2026 (22:30:54 Singapore time).

The Colab recovery notebook runs the shared 17-condition menu and writes each
completed evaluation episode to Drive. To regenerate live summaries:

```bash
python analysis/acrobot/summarize.py /content/drive/MyDrive/SC4055_Acrobot_Recovery/experiment_v2
```

The live summaries may later include C. This committed B snapshot stays fixed;
filter by model=vqc, train_noise=none and experiment=B when reproducing it.
