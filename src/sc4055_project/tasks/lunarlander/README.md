# LunarLander-v3 task (owner: P4)

Hybrid quantum-classical agent on `LunarLander-v3`, plus the encoder-only ablation and
the classical MLP baselines. Implementation mirrors `tasks/frozenlake/`: same
`qrl.config.Config`, same noise module and the same CSV columns, so the runs drop
straight into `analysis/analyze.py`.

## Models

| `--model` | What | Params (4 layers) |
|---|---|---|
| `vqc` (default) | 8→4 tanh encoder → 4-qubit VQC → `<Z>` per action × trainable output weight | 104 |
| `encoder_only` | the same 8→4 encoder → trainable 4→4 linear layer (the plan's ablation) | 56 |
| `mlp_small` | 8→8→4 ReLU MLP, parameter-matched to the hybrid | 108 |
| `mlp_large` | 8→64→64→4 ReLU MLP (2×64) | 4996 |

The hybrid circuit is 4 qubits, 4 layers of RX/RY/RZ with a CZ ring, data re-uploading
on by default, and `Q_a = w_out[a] · <Z_a>`. Noise is inserted exactly as in the
FrozenLake agent (`qrl.noise.apply_noise` after every gate, readout flips before
measurement, `parameter-shift` when shots are on).

Threshold: mean return ≥ **200** over 100 episodes. Observation: 8 floats; actions: 4.

## Run it

```bash
pip install -e .            # or: uv sync   (installs gymnasium[box2d] + swig)
# Optional sanity check that Box2D works on this machine:
python -c "import gymnasium as gym; gym.make('LunarLander-v3').reset(seed=0)"

lunarlander train --seed 0                       # hybrid, Experiment A
lunarlander train --seed 0 --model encoder_only  # ablation
lunarlander train --seed 0 --model mlp_small     # parameter-matched baseline
lunarlander train --seed 0 --model mlp_large     # 2×64 baseline
lunarlander train --seed 0 --quick               # 6-episode smoke run

lunarlander evaluate --checkpoint runs/checkpoints/LunarLander-v3_vqc_0/best.pkl
lunarlander evaluate --checkpoint ... --conditions none depolarizing=0.01 preset=heron2026
```

`python -m sc4055_project.lunarlander ...` is equivalent if the console script is not on
your `PATH` yet (re-run `pip install -e .` after pulling changes to `pyproject.toml`).

## Outputs

```
runs/train/LunarLander-v3_<model>_<seed>[_<noise>].csv     # analysis-spec columns
runs/eval/LunarLander-v3_<model>_<seed>.csv                # Experiment B, 100 greedy eps/condition
runs/checkpoints/LunarLander-v3_<model>_<seed>/            # config.json, ep*.pkl, best.pkl, final.pkl
```

`config.json` records every hyperparameter, including `reward_scale`, so a checkpoint
reproduces its run.

## Two LunarLander-specific choices

1. **`reward_scale` 0.01** (set in `default_config_for`, override with
   `--reward-scale`). LunarLander pays ±100 on the terminal step, but the readout is a
   bounded `<Z>`, so an unscaled TD target would need an output weight two orders of
   magnitude above its initial value. Only the TD target is scaled; **the logged and
   evaluated return is always the raw Gymnasium return**.
2. **`w_out` starts at 1** (`agent.W_OUT_INIT`), so the initial Q-values are commensurate
   with the scaled rewards in [-1, 1]. If you pass `--reward-scale 1.0`, also raise
   `W_OUT_INIT` or expect the output weights to take a long time to reach the ±100 range
   that raw LunarLander rewards need. The classical baselines use their own init.

## Current status (seed 0, shipped defaults)

The plumbing is verified end to end: 20 tests, a quick CLI train/eval run, and two full
1000-episode runs. **None of them solve the task**, which is why the plan's go/no-go
exists. Concretely, `mlp_small` climbs from about -200 to a ~-68 rolling mean around
episode 600 and then destabilises; the hybrid does not improve on the same budget. The
hyperparameters in `default_config_for` are the plan's starting point, not a tuned
recipe: before M1, sweep `lr_output` / `target_update_every` / `gamma` and consider a
Double-DQN target. Do not read the default run as evidence about the quantum layer.

## Scope reminders from the plan

- P4 runs **Experiment B only** (train noise-free, evaluate under noise). Noise-aware
  training (`--noise ...`, Experiment C) is supported by the code for completeness but
  is not part of this task's deliverables.
- Noise enters only the quantum layer, so `encoder_only` / MLP evaluations repeat the
  same number under every condition. Use `--conditions none` for those.
- Box2D ships prebuilt wheels for Python 3.12 on Windows/macOS/Linux. If pip tries to
  build from source, `pip install swig` first; otherwise prefer WSL or Colab.
- Go/no-go on Sun 1 Nov: a clear learning curve plus an honest analysis beats an
  unreachable 200.
