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

## What is still broken

Every architecture collapses late. The VQC's greedy score falls from −110 (episode 750)
to −425 by the end; `mlp_small` goes −54 → −145 over the same stretch. Inspecting the
weights shows the same divergence as before at a slower rate: `|w_out|` drifts 6 → 15
and mean Q climbs to +4.2 while the targets stay O(1). Lowering `lr_output` to 1e-3 does
not fix it (greedy −249 at episode 750, worse than 1e-2) — the readout then cannot adapt
inside 1000 episodes.

So 1e-2 is the local optimum, and what remains is ordinary DQN instability, shared by
every model:

* report `best.pkl` (chosen on a greedy validation eval, which `train.py` already does)
  rather than `final.pkl`, and state the selection rule in the report;
* hard target update every step (`target_update_every=1`, as the CartPole config does);
* gradient clipping;
* a larger replay buffer.

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
