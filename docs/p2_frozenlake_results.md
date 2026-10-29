# P2 FrozenLake results: Experiments A, B and C (29 Oct 2026)

All P2 experiments are done except the two exclusions at the bottom. Everything ran on CPU.

Raw CSVs, checkpoints (best/final, `config.json`, `run_meta.json`), tables and figures are in `analysis_out/p2_frozenlake/`. The tables and figures were made with P3's draft `analyze.py`:
`python analyze.py --train 'analysis_out/p2_frozenlake/runs/train/*.csv' --eval 'analysis_out/p2_frozenlake/runs/eval/*.csv' --random analysis_out/p2_frozenlake/runs/random.csv --out analysis_out/p2_frozenlake`.
The runs were produced on the pre-refactor layout (commit 427241b + these fixes). On the refactored package, re-evaluating seed 0 gives identical results (none 1.00, depolarizing 0.01 0.00, Heron 1.00, readout 0.05 1.00), and the Qiskit cross-check passes.

## Code changes

1. **2-qubit noise conversion fixed.** `gate_error_to_depol_p` now inverts exactly for a DepolarizingChannel(p) on each wire: ε = d/(d+1)·(1−(1−p)ⁿ). That gives 2Q p ≈ 0.625ε. The old eps/2 understated 2Q noise by about 20%. Heron p2 is now 1.88e-3 (was 1.5e-3), NISQ p2 is 6.27e-3 (was 5e-3), and stress scales the same way. A new test checks this against the channel's process fidelity, and Qiskit agrees (below).
2. **Exp B/C labels fixed.** `evaluate.py` writes `experiment=B` for noise-free checkpoints and `C` for noise-trained ones, into `runs/eval/<run_name>.csv`. Before this, `analyze.py` crashed on real data. P3's shared `qrl/evaluate.py` should copy this rule.
3. **Run metadata.** `train.py` saves `run_meta.json` (git commit and package versions) next to `config.json`.
4. **Qiskit cross-check** (`sc4055_project.qiskit_crosscheck`, CLI `qiskit-crosscheck`, needs the `qiskit` extra).
5. Doc fixes: best.pkl in the docstrings, and the 68-parameter count in `config.py`.

## Qiskit cross-check: ALL PASS (`analysis_out/p2_frozenlake/qiskit_crosscheck.json`)

| Check | Result |
|---|---|
| Gate error → channel, verified with Qiskit `average_gate_fidelity` | 1Q 3e-4, 1Q 1e-3, 2Q 3e-3 and 2Q 1e-2 all recovered exactly |
| FrozenLake circuit (trained seed 0, all 16 states) under `preset=heron2026`, compared with an equivalent Aer `NoiseModel` (density matrix) | max \|Δ⟨Z⟩\| = 1.9e-7, while the noise itself moves ⟨Z⟩ by up to 0.076 |
| Aer sampled readout error vs the analytic (1−2p) | 0.9702 vs 0.9700 |
| Same `NoiseModel` through `qml.from_qiskit_noise` + `qml.add_noise`, readout re-added by hand | max \|Δ⟨Z⟩\| = 1.9e-12. The importer warned that it dropped readout, as the plan predicted |

## Results (success rate = mean return; random policy 0.015)

### A. Noise-free training, 5 seeds
All 5 seeds solved, with a median of 223 episodes to solve and a final mean of 0.99 (95% CI 0.98–1.00). Each run took about 75 s on one CPU core.

### B. Noise-free agents evaluated under noise (best.pkl, 100 episodes × 5 seeds, corrected presets)

| Condition | Success | | Condition | Success |
|---|---|---|---|---|
| none | 1.00 | | depolarizing 0.001 | 1.00 |
| shots 1000 / 100 / 10 | 1.00 / 0.83 / 0.16 | | **depolarizing 0.01 / 0.05 / 0.1** | **0.00 / 0.00 / 0.00** |
| coherent σ 0.05 / 0.1 / 0.2 | 0.99 / 0.70 / 0.18 | | readout 0.01 / 0.02 / 0.05 | 1.00 / 1.00 / 1.00 |
| preset Heron-2026 | 0.80 | | preset NISQ-2022 / stress | 0.40 / 0.00 |

Noise tolerance (largest level that keeps 90% of the clean score) is depolarizing p ≈ 0.0013, coherent σ ≈ 0.06, and readout above 0.05, the largest level tested.

### C. Noise-aware training (5 seeds each), evaluated at the training noise

| Training noise | C at own noise | B at same noise | Gain | C evaluated noise-free |
|---|---|---|---|---|
| depolarizing 0.01 | **1.00** | 0.00 | **+1.00** (Mann–Whitney p = 0.004) | 0.20 |
| Heron-2026 | **1.00** | 0.80 | +0.20 (p = 0.42) | 0.60 |
| coherent σ 0.1 | 0.81 | 0.70 | +0.11 (p = 0.35) | 0.40 |

### Findings for the report
1. **Depolarizing is the killer, readout is harmless.** Readout error scales every ⟨Z⟩ by the same (1−2p), so the greedy action never changes. Depolarizing after every gate shrinks the four Q-values by different amounts. The clean agent's Q-values are close together (0.51–0.57 at the start state), so its argmax flips. Tolerance is about p = 0.001, a tenth of Skolik's CartPole boundary.
2. **Noise-aware training recovers everything at depolarizing 0.01 (0 → 100%).** This is the clearest effect in the project. Heron and coherent gains are positive but not significant with 5 seeds.
3. **New finding: noise-trained agents become worse without noise.** Agents trained under depolarizing 0.01 succeed only 20% of the time on a noise-free simulator, and Heron-trained agents 60%. The policy has adapted to the shrunken Q-values, so it doesn't transfer back to clean conditions. Worth a sentence in the analysis and a line in the video.
4. **Shot budget matters as much as realistic gate noise.** 100 shots (0.83) costs about the same as the Heron-2026 preset (0.80), and 10 shots (0.16) is far worse. This answers analysis claim 3.

## Live IBM Heron numbers: not updated
I can't read live calibration data without an IBM Quantum login. Public 2025–26 figures put Heron r3 (ibm_pittsburgh) at a best-case 2Q error of about 5×10⁻⁴, with T1 ≈ 300 µs and T2 ≈ 370 µs ([Gambetta on X](https://x.com/jaygambetta/status/1925017189959991338); [PostQuantum](https://postquantum.com/industry-news/ibm-heron-r3-pittsburgh/)). The plan cites a median 2Q error of 1.52×10⁻³. Our preset (2Q 3×10⁻³, T1 150, T2 100) is therefore a conservative, r2-era Heron. I left it unchanged because changing it affects every member's results. If the team wants a "Heron r3" preset, it's a one-line addition and FrozenLake would take about 10 minutes to re-run. Someone with an IBM login can read the medians from the [IBM QPU information page](https://quantum.cloud.ibm.com/docs/en/guides/qpu-information).

## Not done
- **MLP baselines (Exp A)** are P4's component and need P4's code.
- **Shot-noise training** is infeasible: parameter-shift needs 2 × 68 circuit runs per gradient per sample, and it managed 10–20 episodes in 90 minutes. That's days per seed at any shot count. Report it as a limitation; shot noise is still covered by evaluation (Exp B).

## Open decision for the team
FrozenLake's normalized score is 1.44 when solved, because the deterministic map reaches 1.0 against the slippery-map threshold of 0.70. Either clip it at 1, use threshold 1.0 for this map, or explain it in the caption.
