# Noise-aware training for variational quantum RL agents

**Status:** draft report of the Experiment C work (P4 / LunarLander + shared analysis).
Last updated 1 Nov 2026. Supersedes nothing; complements `docs/p2_frozenlake_results.md`.

---

## 1. Summary

We adapt **QuantumNAT** (Wang et al., DAC 2022) — noise injection, post-measurement
normalisation and post-measurement quantisation — from quantum *classification* to a
quantum **reinforcement learning** loop, and test whether training an agent under the
noise it will be deployed with buys anything.

**Headline claims, in order of evidential strength:**

1. **Noise-aware training recovers most of what noise destroys, but only at the noise it
   was trained for.** FrozenLake, matched noise p = 0.01: the fraction of starting states
   the greedy policy solves rises from **0.61 to 0.91**, and the number of seeds that
   solve the maze goes from **1/3 to 3/3**. Two independent datasets agree (ours, 3 seeds;
   P2's, 5 seeds at 0.00 → 1.00).
2. **It is a trade, not an upgrade.** Every arm peaks at its own training noise.
   Training at twice the deployment noise doubles the tolerable noise (0.0125 → 0.025)
   and halves clean performance (1.00 → 0.52). Training *below* deployment is worse than
   no noise at all (0.00 vs 0.33 at deployment).
3. **The mechanism is Q-value margin.** At p = 0.02 the clean-trained policy still picks
   an optimal action in 29% of decision states; the policy trained at that noise picks one
   in 97%.
4. **The method has a boundary.** On LunarLander it buys nothing: 104 parameters cannot
   learn the task *and* absorb the noise. This is a statement about the model/task/budget
   combination, not about quantum circuits.
5. **Experiment B's baseline is not reproducible to two decimals.** P2 report 0.00 at
   p = 0.01; on current code two of five fresh seeds survive (0.40). The cliff sits on the
   test point.

What we cannot claim: anything about real hardware, scaling past 6 qubits, quantum
advantage, or that the effect transfers to policy-gradient methods (everything here is
DQN).

---

## 2. Background: the method being adapted

QuantumNAT's three stages, and what happens when you point them at an RL loop:

| Stage | In the paper | In our RL loop | Implemented? |
|---|---|---|---|
| **Noise injection** | Sample Pauli errors per gate from the vendor noise model, re-sample every step | Same, via `qrl/noise.py`; density-matrix simulation makes this *exact*, not Monte-Carlo | **yes** |
| **Post-measurement normalisation** | Per-qubit standardisation across a batch of inputs, to cancel the affine noise map $y \mapsto \gamma y + \beta_x$ | A DQN steps one environment at a time — there is **no batch at inference** | **no** |
| **Post-measurement quantisation** | Uniform quantisation + a centroid loss, to snap small errors back | Quantising the Q-values would destroy the argmax resolution FrozenLake depends on (its four Q-values are within ~0.06) | **no** |

So this report tests **one third** of QuantumNAT — the part that needs no changes to the
inference interface. The other two are left as follow-ups (§8), for a reason that is
itself a finding: the paper's normalisation argument depends on a batch, and an RL agent
evaluated one state at a time does not have one.

### The theory that motivates injection

For a single-qubit Z-basis measurement, any noise channel acts on the readout as an affine
map, $E_z = \gamma E_z^\ast + \beta_x$, with $\gamma$ independent of the input and $\beta$
input-dependent. Uniform noise is therefore a *common scale factor* on every Q-value —
and an argmax does not care about a common scale. What breaks the policy is the
input-dependent part $\beta_x$, which reorders the actions state by state.

Injection works by forcing the parameters to be a near-fixed-point of the channel: the
agent learns to put its decision in directions the channel does not rotate.

---

## 3. Setup

### Tasks and models

| | FrozenLake-v1 | LunarLander-v3 |
|---|---|---|
| observation | 16 discrete states → 4 bits | 8 continuous, scaled by `OBS_SCALE` |
| model | 4 qubits, 5 layers, 68 parameters | 4 qubits, 4 layers, 104 parameters, 8→4 tanh encoder |
| readout | $\langle Z_i\rangle$ per qubit, $Q_a = w_{\text{out}}[a]\,\langle Z_a\rangle$ | same |
| training | DQN, γ = 0.9, 600 episodes | DQN + Double DQN, γ = 0.99, 1000 episodes |
| random policy | 0.015 success | ≈ −150 return |

### Noise model

`qrl/noise.py` is the single entry point; noise is described by a string so it can be
logged verbatim (`none`, `shots=N`, `coherent=σ`, `depolarizing=p`, `readout=p`,
`preset=heron2026|nisq2022|stress`).

* `depolarizing=p` applies PennyLane's `DepolarizingChannel(p)` after **every gate**, on
  every wire the gate touches. 1-qubit gates get one channel, CZ gets two.
* Per-application Bloch-vector factor is $(1 - 4p/3)$; with ~100 applications on
  FrozenLake and ~96 on LunarLander, $p = 0.005$ leaves 62% of the signal and $p = 0.01$
  leaves 38%.
* `readout=p` is a symmetric bit flip on the measured wire, applied just before
  `qml.expval`; it scales every $\langle Z\rangle$ by $(1-2p)$.

---

## 4. Experiment C on FrozenLake (the positive result)

### Design

Five arms differing **only** in the depolarising noise present during training:
0, 0.25×, 0.5×, 1×, 2× of a deployment level of p = 0.01. Three seeds each, 600 episodes,
identical circuit and hyper-parameters. Arm 0 *is* Experiment B, so the comparison is
self-contained.

Two measurements, deliberately:

* **single start** — the usual greedy episode from S. One bit per seed, coarse.
* **all starts** — the greedy policy rolled out from all 11 states where a decision
  matters, using the environment's own transition table. Eleven bits instead of one, and
  what the conclusions rest on.

### Result 1 — success at the deployment noise peaks at 1×

Single greedy episode from S, mean over 3 seeds, evaluated at p = 0.01:

| training noise | 0× | 0.25× | 0.5× | **1×** | 2× |
|---|---|---|---|---|---|
| success at p = 0.01 | 0.33 | 0.00 | 0.33 | **1.00** | 0.67 |

Matching the training noise takes the agent from 1 of 3 seeds solving to 3 of 3.
Training well below deployment (0.25×) is **worse than no noise at all**: it perturbs the
circuit without buying tolerance.

### Result 2 — the whole curve shifts

Fraction of the 11 decision states solved, mean over 3 seeds:

| arm | p=0 | 0.005 | 0.0075 | 0.01 | 0.015 | 0.02 | 0.025 | 0.03 |
|---|---|---|---|---|---|---|---|---|
| 0× | **1.00** | 0.79 | 0.64 | 0.61 | 0.24 | 0.06 | 0.06 | 0.06 |
| 0.25× | 0.97 | 0.82 | 0.73 | 0.36 | 0.21 | 0.00 | 0.00 | 0.00 |
| 0.5× | 0.33 | 0.67 | 0.67 | 0.58 | 0.48 | 0.48 | 0.36 | 0.33 |
| **1×** | 0.79 | 0.88 | **0.91** | **0.91** | 0.64 | 0.30 | 0.03 | 0.03 |
| 2× | 0.52 | 0.61 | 0.73 | 0.73 | 0.91 | **0.97** | 0.61 | 0.00 |

Per-seed noise tolerance (largest tested level keeping ≥ 50% of starts solvable), median:

| arm | 0× | 0.25× | 0.5× | 1× | 2× |
|---|---|---|---|---|---|
| tolerance | 0.0125 | 0.0075 | 0.0075 | 0.015 | **0.025** |

Two things to read off:

* **Each arm peaks at the noise it was trained on.** 1× peaks at 0.0075–0.01, 2× peaks at
  0.015–0.02. This is the hump QuantumNAT's own ablation predicts.
* **The upper edge grows with training noise while the clean end shrinks.** That is the
  trade, quantified.

### Result 3 — the mechanism is the action ranking, not the signal size

Uniform depolarising shrinks all Q-values by the same factor, and an argmax is invariant
to a common scale. What breaks the policy is the input-dependent part of the distortion.
Measured directly as agreement with the optimal action over the 11 decision states:

* at p = 0.02: clean-trained policy **29%** optimal, 2×-trained policy **97%**.

This is the same quantity QuantumNAT's normalisation step is designed to remove; we
measured it, but did not implement the removal (see §8).

---

## 5. LunarLander: four bugs, then a negative result

### 5.1 What was wrong with the training setup

The hybrid (8→4 tanh encoder → 4-qubit VQC, 104 parameters) appeared unable to learn at
all: 1000 episodes ended worse than a random policy. Inspecting the weights showed the
value function had **diverged**, not that the circuit lacked capacity.

| # | bug | evidence | fix |
|---|---|---|---|
| 1 | `lr_output = 0.1` under Adam moves each readout weight ~0.1 per step | `\|w_out\|` reached **±50** against TD targets of O(1); Q-values ±43 | **1e-2**: `\|w_out\|` → 2.7, Q ±1 |
| 2 | raw observations (angle ±π, velocities ±5) fed into `tanh` | **19%** of encoder units pinned at \|x\| > 0.99, aliasing distinct states | `OBS_SCALE` in `encode_obs`: saturation → **0.1%** |
| 3 | ε decayed to 0.01 by episode 400 | policy went greedy before the Q-function settled and never recovered | decay over 800 episodes, floor 0.05 |
| 4 | no Double DQN | maximisation bias: `max_a Q_target` is biased upward, and the bias bootstraps | `cfg.double_dqn`, `cfg.grad_clip` |

Effect of the four fixes, one seed each, greedy evaluation:

| | best greedy (30 episodes) |
|---|---|
| as found | −149 (≈ random, −150) |
| after bugs 1–3 | −110 |
| after bug 4 (Double DQN) | **−14.6**, best checkpoint at episode 999 |

Baselines at the identical budget (1 seed, 1000 episodes, clean), best greedy:

| model | params | best greedy |
|---|---|---|
| `mlp_small` 8→8→4 | 108 | **−54** |
| `encoder_only` 8→4→4 | 56 | −113 |
| VQC hybrid | 104 | −149 (pre-fix) |
| `mlp_large` 2×64 | 4996 | −427 (worst; diverged) |

The last row is the important one: **48× the parameters did worse**, so parameter count was
never the binding constraint — optimisation was.

### 5.2 Experiment C on LunarLander: negative

3 seeds clean, 3 seeds trained at `depolarizing=0.005`, 1000 episodes, greedy evaluation
over 20 episodes at five noise levels. Random ≈ −150.

| arm | seed | none | 0.0025 | **0.005** | 0.01 | 0.02 |
|---|---|---|---|---|---|---|
| clean | 0 | **−61** | −548 | −549 | −529 | −564 |
| clean | 1 | −431 | −405 | −397 | −496 | −464 |
| clean | 2 | −267 | −129 | −129 | −126 | −126 |
| trained 0.005 | 0 | −241 | −300 | **−189** | −456 | −512 |
| trained 0.005 | 1 | −348 | −638 | **−693** | −138 | −136 |
| trained 0.005 | 2 | −164 | −238 | **−322** | −386 | −531 |
| **mean, clean** | | −253 | −361 | **−358** | −384 | −385 |
| **mean, trained** | | −251 | −392 | **−401** | −326 | −393 |

At the matched noise the trained arm is **−401 against −358**: slightly worse, and both
are far below random play.

Three things happened:

1. **Only one of three clean seeds learned the task** (−61). The others ended below random.
2. **That one good policy was destroyed by the first noise level tested**: −61 → −548 at
   p = 0.0025, where the circuit still retains ~79% of its signal.
3. **Training under noise never rebuilt it.** The noise-trained arms landed between −190
   and −690 with no coherent dependence on the evaluation noise (seed 1 scored −693 at its
   own training level and −138 at twice that). That is a degenerate solution, not a policy.

### 5.3 Why it fails there and works on FrozenLake

Three compounding reasons, only one of which is about the circuit:

1. **Control tasks amplify small errors; mazes do not.** FrozenLake is 6–14 decisions
   long; a wrong step usually still reaches the goal or ends the episode. LunarLander is
   200–1000 steps of closed-loop control, where per-step errors compound — a 1% flip rate
   integrates to `0.99^300 ≈ 5%` over an episode. That is why the maze agent survives at
   38% signal and the lander dies at 79%.
2. **The policy had no margin to spend.** Robustness is a budget you can only spend if the
   clean policy has slack. On FrozenLake all seeds converge to the same strongly-separated
   solution; on LunarLander only one seed out of three learned anything, and it was found
   by luck rather than converged to, so its Q-gaps are small.
3. **No spare capacity for redundancy.** Encoding a decision so it survives the channel
   costs parameters. 104 parameters behind an 8→4 bottleneck can learn the task or absorb
   the noise, not both.

---

## 6. Reproducibility: Experiment B's baseline is a coin flip

P2's table (`docs/p2_frozenlake_results.md`) reports a clean-trained agent at
`depolarizing=0.01` scoring **0.00** across 5 seeds, which is what makes the Experiment C
gain look like +1.00. On current code that number is not stable:

| measurement | result at p = 0.01 |
|---|---|
| P2's committed seed-0 checkpoint, re-evaluated now | 0.00 (matches their table) |
| fresh training, seeds 0–4, same config | 1.00, 0.00, 0.00, 0.00, 1.00 → **0.40** |
| fresh training, seeds 0–2, P2's checkpoint-selection settings | 1.00, 0.00, 0.00 |

Re-evaluating their checkpoint reproduces 0.00, so the evaluation code is sound — the
difference is *which policy training lands on*. The cliff sits exactly on the test point,
so any change to the code or the RNG stream flips it.

**Consequence:** quote the baseline as a distribution, or move the comparison point to
where it is unambiguously 0 for every seed. The C side is the robust half: training under
0.01 gave 1.00 at 0.01 for 5/5 of P2's seeds and in our re-run.

---

## 7. Limitations

* **Simulator only.** No hardware. `depolarizing` stands in for a calibration-derived model.
* **3 seeds** on our FrozenLake sweep, 5 on P2's. The binary success numbers move in
  thirds; only the all-starts measure has enough resolution to carry claims.
* **One third of QuantumNAT implemented.** Normalisation and quantisation were not
  (reasons in §2; follow-ups in §8).
* **One noise family in training.** Training used depolarising only; readout, coherent and
  shot noise were evaluation-only.
* **LunarLander's clean arm is a seed lottery** (1/3 learn), so its mean is an artefact of
  two failed seeds. Quote the per-seed split.
* **DQN only.** The method is untested with policy-gradient or actor-critic agents.

---

## 8. Follow-ups, in the order I would do them

1. **Port Double DQN into the generic loop.** P1's `qrl/dqn.py` (branches
   `p1/implement-generics`, `p1/Cartpole`) still uses plain `max(axis=1)` and ignores
   `cfg.double_dqn` / `cfg.grad_clip`. Merging as-is silently reverts the LunarLander fix
   while the config still says `double_dqn=True`.
2. **Implement normalisation the RL way.** There is no batch at inference, so the paper's
   exact cancellation is unavailable. Running per-qubit mean/std (EMA) restores the mean
   correction but not exactness — and measuring how much of the gain that keeps is a
   result in itself. Note the replay-batch assumption: states in a replay batch are
   heterogeneous, so $\mathrm{Var}(\beta_x)$ within the batch is larger than in i.i.d.
   classification, which weakens normalisation — consecutive states from the same episode
   would be a *better* batch. That inversion is worth an ablation.
3. **Quantisation belongs on the measurement features, not the Q-values.** With
   $Q_a = w_{\text{out}}[a]\langle Z_a\rangle$ and four near-equal actions, quantising the
   output destroys the ranking.
4. **Test the "error compounds" hypothesis** (§5.3): apply noise to a fixed *fraction* of
   LunarLander's gates and see whether the survival threshold scales with the number of
   noisy steps rather than the total error. ~20 minutes on an existing checkpoint; it
   would turn the explanation from plausible to tested.
5. **LunarLander capacity levers**, in increasing cost: a 4×4 trainable mixing layer
   between measurements and actions (16 parameters, no extra noise); the plan's 8-qubit
   "one qubit per feature" fallback (removes the 8→4 bottleneck, but adds gates and
   therefore noise); more episodes per arm.

---

## 9. Reproduction

```bash
# FrozenLake Experiment C (about 25 min on CPU)
python analysis/noise_aware_sweep.py --env FrozenLake-v1 \
    --base depolarizing=0.01 --scales 0 0.25 0.5 1 2 --seeds 0 1 2 \
    --episodes 600 --eval-episodes 10 --out analysis_out/fl_sweep

# the per-state probe and the sweep figure
python analysis/policy_probe.py
python analysis/plot_noise_aware.py --summary analysis_out/fl_sweep/summary.csv

# LunarLander Experiment C (about 13 h: 3 clean arms then 3 noise-trained)
python analysis/noise_aware_sweep.py --env LunarLander-v3 \
    --base depolarizing=0.005 --scales 0 1 --seeds 0 1 2 \
    --episodes 1000 --eval-episodes 20 --train-every 1 \
    --conditions none depolarizing=0.0025 depolarizing=0.005 \
                 depolarizing=0.01 depolarizing=0.02 \
    --out analysis_out/lunarlander_sweep --resume

# was the value function diverging?
python analysis/lunarlander/inspect_weights.py \
    --checkpoints runs/checkpoints/LunarLander-v3_vqc_0
```

Raw data: `analysis_out/fl_sweep/` (FrozenLake), `analysis_out/lunarlander_sweep/`
(LunarLander), `analysis_out/p2_frozenlake/` (P2's runs). Every number above comes from
those CSVs; nothing is hand-entered.

---

## 10. References

* Wang et al. **QuantumNAT: Quantum Noise-Aware Training with Noise Injection,
  Quantization and Normalization.** DAC 2022. arXiv:2110.11331.
* P2's FrozenLake experiments: `docs/p2_frozenlake_results.md`.
* Working log and handover notes: `docs/HANDOVER.md`.
