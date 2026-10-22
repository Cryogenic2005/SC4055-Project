# SC4055 Group 1: How much hardware noise can a quantum RL agent tolerate?

Variational-quantum-circuit (VQC) DQN agents on four Gymnasium tasks of growing size
(FrozenLake 4 qubits, CartPole 4 qubits, Acrobot 6 qubits, LunarLander hybrid 8→4),
trained noise-free and under hardware-like noise, evaluated under a shared noise menu
(shots, coherent over-rotation, depolarizing, readout, Heron-2026 / NISQ-2022 presets).
One codebase, one protocol, one normalized score, so the four tasks sit on one plot.

Full plan: `docs/team_plan.md`. Run matrix: `docs/run_matrix.md`. CSV/figure contract: `docs/analysis_spec.md`.

## Setup (every laptop + Colab, day 1)

```bash
git clone https://github.com/Cryogenic2005/SC4055-Project.git
cd SC4055-Project
bash setup.sh            # creates .venv, installs pinned deps, runs the smoke test
```

Colab:

```python
!git clone https://github.com/Cryogenic2005/SC4055-Project.git
%cd SC4055-Project
!bash setup.sh --no-venv
```

Windows: use WSL or Colab (Box2D does not build natively).

The smoke test must print `ALL OK`. It checks the pinned versions, that the jitted
4-qubit QNode runs in well under a millisecond per batch (if not, you are on the slow
path), the density-matrix device, all four environments, and a 6-episode training run
that writes a CSV with the agreed columns.

**Do not** `pip install pennylane jax` unpinned: JAX 0.12 breaks PennyLane 0.45.

## Layout

```
qrl/                 shared package
  config.py          Config dataclass (env, model, circuit, noise, DQN hyperparameters)
  envs.py            make_env, per-task encoders, random baseline            (P4)
  noise.py           NoiseSpec, parse_noise, NOISE_MENU, presets, make_device, apply_noise  (P2)
  logging_utils.py   TrainLogger / EvalLogger with the analysis-spec columns
  agent.py           shared QAgent interface                                 (P1, stub)
  dqn.py             shared DQN loop                                         (P1, stub)
  evaluate.py        shared evaluation harness                               (P3, stub)
tasks/
  frozenlake/        P2: agent.py, train.py, evaluate.py (complete reference implementation)
  cartpole/          P1
  acrobot/           P3
  lunarlander/       P4
scripts/             smoke_test.py, random_baseline.py
tests/               pytest: noise contractions, agent shapes/jit/gradients, logger columns
analysis/            P3: analyze.py and figures
runs/                train/*.csv, eval/*.csv, random.csv, checkpoints/ (git-ignored)
docs/                plan, run matrix, analysis spec, project brief
```

## Running FrozenLake (P2 reference)

```bash
python tasks/frozenlake/train.py --seed 0                              # Experiment A (deterministic map, ~35 s, solves by ~ep 220)
python tasks/frozenlake/train.py --seed 0 --slippery                   # stochastic map (registered threshold 0.70)
python tasks/frozenlake/train.py --seed 0 --noise depolarizing=0.01    # Experiment C
python tasks/frozenlake/train.py --seed 0 --noise shots=100            # parameter-shift
python tasks/frozenlake/evaluate.py --checkpoint runs/checkpoints/FrozenLake-v1_vqc_0/final.pkl   # Experiment B, 17 conditions
python scripts/random_baseline.py FrozenLake-v1
```

Noise strings: `none`, `shots=N`, `coherent=σ`, `depolarizing=p`, `readout=p`, `preset=heron2026|nisq2022|stress`.

## Conventions that protect the correctness marks

- `depolarizing=p` is PennyLane's `DepolarizingChannel(p)` after every gate, per wire. A
  device's average gate error ε is converted with `gate_error_to_depol_p` (ε = 2p/3 for one qubit).
- Readout error is a symmetric `BitFlip(p)` on every measured wire right before
  measurement, so ⟨Z⟩ → (1−2p)⟨Z⟩. (PennyLane 0.45.1's `default.mixed(readout_prob=…)`
  is broken, see `qrl/noise.py`; a test guards the workaround.)
- Shots are never combined with backprop. A shots device switches to parameter-shift.
- Presets use exact expectation values; shot noise is its own axis in the sweep.
- "Solved" = mean of last 100 episodes ≥ `env.spec.reward_threshold` (CartPole 475).

## Known integration issues (fixed here)

| Issue | Fix |
|---|---|
| JAX 0.12 + PennyLane 0.45: `jax.core has no attribute is_concrete` | pin `jax[cpu]==0.10.2` |
| `gymnasium[box2d]` fails to build | `pip install swig` first (setup.sh does it) |
| `default.mixed(readout_prob=p)` → `TypeError: 'float' object is not iterable` | explicit `BitFlip` before measurement |
| Shots device inside `jax.vmap`/`jit` → `ConcretizationTypeError` | shots path loops in Python, no jit |
| FrozenLake MLP baseline sees the state index | MLPs get a 16-way one-hot (`envs.one_hot_obs`) |
| Q = w(⟨Z⟩+1)/2 (Skolik) cannot represent negative shaped rewards; training stalled at 0 | FrozenLake uses Q = w⟨Z⟩ |
| Late training collapse (train mean 1.0, greedy 0.0 at the end) | greedy eval every 25 episodes, `best.pkl` kept next to `final.pkl` |

## Team

Group 1: Le Nguyen Bao Huy, Yang Mingchen, Poobanchuen Panjapong, Wang Jianning.
Roles in `CONTRIBUTING.md`.
