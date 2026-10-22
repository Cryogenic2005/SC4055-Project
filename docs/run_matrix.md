# Run matrix for Experiments A, B and C

Draft by the Task execution thread, 22 Oct 2026. Derived from "Noisy quantum RL team plan.md". Every number below comes from that plan; adjust here when the protocol is frozen (M1, Wed 28 Oct).

## Fixed settings for every run

- Stack: `pennylane==0.45.*`, `jax[cpu]==0.10.2`, optax, `gymnasium[box2d]` (install `swig` first).
- Device: `default.qubit` + `jax.jit`/`vmap` for noise-free, shot and coherent runs; `default.mixed` only when channel noise is on.
- Never combine shots with backprop. Shot-noise training uses parameter-shift.
- Seeds: 0–4 (extend to 0–9 on FrozenLake and CartPole if time allows).
- "Solved" = mean of last 100 episodes vs `env.spec.reward_threshold` (CartPole 475, FrozenLake 0.70, Acrobot −100, LunarLander 200).
- Normalized score = (R − R_random) / (R_threshold − R_random), R_random measured over 100 random episodes per env.
- Log wall-clock time per run; checkpoint every N episodes (Colab drops).

## Experiment A: noise-free training (deadline M2, Sun 1 Nov)

| Task | Owner | Models | Seeds | Runs | Est. time per seed |
|---|---|---|---|---|---|
| FrozenLake-v1 | P2 | VQC, MLP-matched, MLP-large | 5 | 15 | seconds–minutes |
| CartPole-v1 | P1 | VQC, MLP-matched, MLP-large | 5 | 15 | 1–3 min (jitted) |
| Acrobot-v1 | P3 | VQC, MLP-matched, MLP-large | 5 | 15 | 5–15 min |
| LunarLander-v3 (hybrid 8→4) | P4 | VQC, MLP-matched, MLP-large, encoder-only ablation | 5 | 20 | 10–20 min + loop overhead |
| **Total** | | | | **65** | |

Also: 4 random-policy baselines (100 episodes each) for R_random.
LunarLander go/no-go on Sun 1 Nov: if not learning, keep the best checkpoint and its learning curve and move on.

## Experiment B: train noise-free, evaluate under noise (deadline M3, Wed 4 Nov)

Each trained VQC checkpoint (4 tasks × 5 seeds = 20) runs 100 greedy episodes under each condition:

| Family | Levels | Count |
|---|---|---|
| Clean reference | – | 1 |
| Shots | 1000, 100, 10 | 3 |
| Coherent over-rotation σ (rad) | 0.05, 0.1, 0.2 | 3 |
| Depolarizing p | 0.001, 0.01, 0.05, 0.1 | 4 |
| Readout p | 0.01, 0.02, 0.05 | 3 |
| Presets | Heron-2026, NISQ-2022, Stress (10× Heron) | 3 |
| **Total per checkpoint** | | **17** |

20 checkpoints × 17 conditions = **340 evaluation jobs** (34,000 episodes). Forward passes only. Cost is dominated by density-matrix conditions on Acrobot (6 qubits); LunarLander hybrid stays at 4 qubits. If an 8-qubit LunarLander fallback is used, expect ~1–1.5 h per noise level, so restrict it to clean, depol 0.01, Heron-2026 and shots 1000.

## Experiment C: noise-aware training (deadline M3, Wed 4 Nov)

| Task | Conditions | Seeds | Runs | Est. time per seed |
|---|---|---|---|---|
| FrozenLake-v1 | depol 0.01, σ = 0.1, Heron-2026 (+ shot-noise training with parameter-shift, P2) | 5 | 15 (+5) | 2–3× noise-free |
| CartPole-v1 | depol 0.01, σ = 0.1, Heron-2026 | 5 | 15 | 2–3 min |
| Acrobot-v1 | one condition (suggest depol 0.01) | 3 | 3 | ~50 min per 100k steps |
| LunarLander-v3 | none (evaluation only) | – | 0 | – |
| **Total** | | | **33 (+5)** | |

Each C run is then evaluated at its own training noise level and compared with the matching Experiment B cell.

## Output layout (proposed, to agree with the P3 harness)

```
results/
  train/{env}/{model}/{noise}/seed{k}/  metrics.csv  checkpoint.pkl  config.json
  eval/{env}/{model}/seed{k}/{condition}.csv
```

`metrics.csv` columns: episode, return, steps, epsilon, loss, wall_clock_s.
Eval CSV columns: env, model, seed, train_noise, eval_condition, episode, return.

## Compute split suggestion

Each owner runs their own task's A and C runs. Experiment B is a single batched script (P3 harness) that any machine can run per task. Spread seeds across four Colab sessions; Acrobot noisy runs are the longest pole.
