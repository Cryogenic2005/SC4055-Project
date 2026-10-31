# LunarLander: why the hybrid was not learning

P4's task is LunarLander with a hybrid model (8→4 tanh encoder → 4-qubit VQC → 4 actions).
Through October it would not learn: 1000 episodes ended worse than a random policy, and
the failure looked like a capacity problem. It was not. Three bugs in the training setup
were hiding the circuit, and one of them is silent enough to be worth writing down.

**Headline:** the VQC reaches the solve threshold in individual episodes now (+211), and
its best greedy checkpoint went from −149 (random) to −110. It still collapses late in
training, and so does every classical baseline — that last part is a DQN problem, not a
quantum one.

## What was wrong

### 1. The readout learning rate made the value function diverge

`lr_output = 0.1` under Adam moves every `w_out` weight by ~0.1 *per step*. Over ~200k
steps they ran to **±50**, while `reward_scale = 0.01` puts the TD targets at O(±1). The
Q-function ended up 30–40× outside the range it was supposed to predict, so the bootstrap
target `r + γ·max_a Q(s')` was about +30 instead of about +1 — which pushes Q up again.
Classic deadly-triad divergence, and the policy that comes out of it is indistinguishable
from random.

The classical baselines never showed it because their weights train at `lr_mlp = 1e-3`
and their last layer initialises at 0.0035, so their Q-values start near zero. **The VQC
and the MLPs were never being trained under comparable settings.**

### 2. `tanh` on un-normalised observations

LunarLander's observations span ±1.5 (position) to ±π (angle) to ±5 (velocities). Fed
raw into the encoder's `tanh`, 19% of units sat pin-straight at ±1, so distinct states
collapsed onto the same four features. `encode_obs` now divides by the documented range
of each dimension (`OBS_SCALE`), before every model — including the MLPs, so the
comparison stays fair.

### 3. Exploration decayed before the Q-function settled

ε reached 0.01 by episode 400 and stayed there. From that point the agent collected data
with an almost-greedy policy built on a Q-function that had not converged, and never
recovered. Decay is now stretched to 800 episodes with a 0.05 floor.

## Effect of the three fixes

| quantity | before | after |
|---|---|---|
| encoder units saturated | 19 % | 0.1 % |
| max \|w_out\| at best checkpoint | 50 | 6 |
| Q range across random states | ±43 | ±1 |
| best greedy evaluation | −149 (= random) | −110 |
| best single training episode | ≈ −150 | +211 (over the 200 threshold) |
| greedy action distribution | 23/31/14/32 % | 14/11/25/49 % |

The action distribution is the clearest sign the policy became state-dependent.

## The late-training collapse, and the fix

After those three fixes the agent learned, but every arm still fell apart in the last
quarter: the VQC's greedy score went −110 (episode 750) → −425 by the end, and
`mlp_small` went −54 → −145. The weights showed the same divergence as before, only
slower — `|w_out|` drifting 6 → 15 and mean Q climbing to +4.2 against O(1) targets.

That is **maximisation bias**: `target = r + γ·max_a Q_target(s', a)` takes a maximum
over an estimate, so it is biased upward, and the bias feeds back through its own
bootstrap. The standard fix is Double DQN — the online network picks the next action,
the target network only scores it:

```python
next_action = argmax(q_online(s'))          # selection: unbiased w.r.t. the target
q_next      = q_target(s', next_action)     # evaluation
```

Added as `cfg.double_dqn` (on for LunarLander) plus global-norm gradient clipping
(`cfg.grad_clip = 1.0`). One seed, 1000 episodes, everything else unchanged:

| | before | after |
|---|---|---|
| greedy evaluation, 30 episodes, best checkpoint | −110 | **−14.6** |
| best checkpoint | episode 750 | **episode 999** |
| max \|w_out\| | 15 (drifting) | 2.7 (stable) |
| Q mean ± std | +4.2 ± 3.9 | −0.10 ± 0.48 |

−14.6 is a working policy (a crash scores −100 or worse; 0 is "landed safely, missed the
pad"; +200 is a good landing). And the best checkpoint is now the **last** one, so
training no longer needs to be stopped early to get a usable agent.

Training is still non-monotone — episode 750 was a genuine dip (−276 on the same
30-episode re-evaluation) — so keep selecting `best.pkl` on a validation eval, and use
enough evaluation episodes to make that selection meaningful (10 episodes had a standard
error of ~30 points here; 30 is a reasonable floor).

## What is still open

* **One seed.** The −14.6 figure is a single 1000-episode run. Run seeds 1–4 before
  quoting it; LunarLander's per-seed spread is large.
* **`eval_episodes`.** Checkpoint selection currently uses `cfg.eval_episodes`; for
  LunarLander that should be ≥ 20, or the "best" checkpoint is chosen on noise.
* **Capacity.** 104 parameters is small for this task. The 8-qubit fallback in the plan
  (one qubit per feature) is the natural next step if the seeded mean is not good enough.

## Reproducing

```bash
# train the hybrid with the fixed defaults
python -c "
from sc4055_project.qrl.config import default_config_for
from sc4055_project.tasks.lunarlander.train import train
train(default_config_for('LunarLander-v3', seed=0))"

# was the value function diverging?
python analysis/lunarlander/inspect_weights.py \
    --checkpoints runs/checkpoints/LunarLander-v3_vqc_0
```

## Consequence for the noise work

The noise-aware training experiment (`analysis/noise_aware_sweep.py`, results in
`docs/noise_aware_findings.md`) was run on FrozenLake *because* LunarLander had no
learning policy for noise to degrade. Now that there is one, the LunarLander version is
possible but expensive: density-matrix simulation makes a noisy episode ~11× slower, so
a 4-arm × 3-seed sweep is roughly 6 hours of CPU. Decide before M3.
