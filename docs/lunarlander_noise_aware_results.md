# LunarLander under noise: the Experiment C attempt (31 Oct 2026)

**Result: negative.** Training the hybrid under depolarizing noise does not recover the
performance that noise destroys, because at this circuit size the agent barely learns the
task at all. Random play scores about **−150**; almost every cell below is worse than that.

## Setup

4-qubit / 4-layer hybrid (104 parameters), Double DQN, 1000 episodes, `train_every=1`,
3 seeds per arm, greedy evaluation over 20 episodes. Training noise is
`depolarizing=0.005` (the arm labelled "trained under noise"); the clean arm is
Experiment B. Cost: ~2.4 h per noise-trained seed, ~7–16 min per clean seed.

## Results (greedy return, 20 episodes; random ≈ −150)

| arm | seed | none | 0.0025 | **0.005** | 0.01 | 0.02 |
|---|---|---|---|---|---|---|
| clean | 0 | **−61** | −548 | −549 | −529 | −564 |
| clean | 1 | −431 | −405 | −397 | −496 | −464 |
| clean | 2 | −267 | −129 | −129 | −126 | −126 |
| **trained 0.005** | 0 | −241 | −300 | **−189** | −456 | −512 |
| **trained 0.005** | 1 | −348 | −638 | **−693** | −138 | −136 |
| **trained 0.005** | 2 | −164 | −238 | **−322** | −386 | −531 |
| **mean, clean arm** | | **−253** | −361 | **−358** | −384 | −385 |
| **mean, trained arm** | | −251 | −392 | **−401** | −326 | −393 |

At the matched noise the trained arm is **−401 against the clean arm's −358**: slightly
*worse*, and both are far below random play (−150). The only place the trained arm wins is
0.01 (−326 vs −384), which is not the level it was trained at — that is seed noise, not an
effect.


## What actually happened

1. **Only one of three clean seeds learned the task** (−61). The others ended at −431 and
   −267, i.e. below random. LunarLander remains a seed lottery at 104 parameters.
2. **That one good policy is destroyed by any noise at all**: −61 → −548 at p = 0.0025.
   The cliff is steeper than on FrozenLake, and it sits below every level tested here.
3. **Training under noise does not rebuild it.** The noise-trained arms land between
   −190 and −690, with no coherent dependence on the evaluation noise (seed 1 scores
   −693 at its own training level but −138 at twice that). That pattern is not a policy;
   it is a degenerate solution that happens to win or lose depending on which actions
   the noise distorts.
4. At p = 0.005 the readout keeps ~62% of its signal (72 channel applications of a
   per-gate factor 1 − 4p/3). A 104-parameter model cannot absorb that and still learn
   LunarLander from 1000 episodes.

## What this means

* **LunarLander cannot carry Experiment C at this scale.** This is a statement about the
  model/task/budget combination, not about quantum circuits: the same noise-injection
  method works on FrozenLake, where the clean agent genuinely solves the task and the
  noise-trained agent restores it (see `docs/noise_aware_findings.md` and
  `docs/p2_frozenlake_results.md`).
* **Report it as a boundary of applicability**, with the numbers above. It is a legitimate
  negative result and it is cheaper to defend than a hand-waved positive one.
* **If the team wants a LunarLander noise result**, the levers are: more capacity
  (8 qubits, one per feature — the plan's fallback), more episodes per arm, or a lower
  training-noise level (p ≤ 0.001) where the clean policy might survive long enough for
  an improvement to be measurable.

## Reproducing

```bash
python analysis/noise_aware_sweep.py --env LunarLander-v3 \
    --base depolarizing=0.005 --scales 0 1 --seeds 0 1 2 \
    --episodes 1000 --eval-episodes 20 --train-every 1 \
    --conditions none depolarizing=0.0025 depolarizing=0.005 depolarizing=0.01 depolarizing=0.02 \
    --out analysis_out/lunarlander_sweep --resume
```

Raw rows: `analysis_out/lunarlander_sweep/rows/*.json`; per-arm CSVs under `runs/`.
