# How we work in this repo

Four people, one codebase, one protocol. These rules keep the four results comparable.

## Ownership

| Person | Task folder | Shared component you own |
|---|---|---|
| P1 | `tasks/cartpole/` | `qrl/agent.py` (QAgent), `qrl/dqn.py` (DQN loop, optax setup) |
| P2 | `tasks/frozenlake/` | `qrl/noise.py`, presets, Qiskit cross-check, `tests/` |
| P3 | `tasks/acrobot/` | `qrl/evaluate.py`, `analysis/` (aggregation and plots) |
| P4 | `tasks/lunarlander/` | MLP baselines, `qrl/envs.py` wrappers incl. Box2D |

Edit your own folder and component freely. For anything in `qrl/` you do not own, open a PR and tag the owner.

## Branches and PRs

- `main` is always runnable: `pytest` and `scripts/smoke_test.py` pass (CI checks both).
- Work on `p<N>/<topic>` branches, e.g. `p2/shot-noise-training`. Small PRs, merged by the owner of the touched component.
- Never commit `runs/checkpoints/` or large CSVs. Results live in `runs/` locally and are aggregated by P3's scripts; share CSVs via the team drive, then commit only the final `results/` tables and figures used in the report.

## Non-negotiables from the plan

1. Pinned versions only (`requirements.txt`). Unpinned `pip install pennylane jax` breaks.
2. Fast path by default: `default.qubit` + `jax.jit` + `jax.vmap`. If a CartPole seed takes more than ~5 minutes you are on the slow path.
3. `default.mixed` only when channel noise is on; `apply_noise()` / `apply_readout_noise()` are the only places noise is inserted.
4. Never combine shots with backprop. Shots => `diff_method="parameter-shift"`.
5. Every logger writes the columns in `qrl/logging_utils.py` (= `docs/analysis_spec.md`). Do not rename columns.
6. Report "solved" against `env.spec.reward_threshold` (CartPole 475) and say so.
7. Experiment A is noise-free. Noise during training => `experiment="C"` (Config enforces this).

## Before you push

```bash
python -m pytest -q
python scripts/smoke_test.py
```
