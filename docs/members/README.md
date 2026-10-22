# Member guides

One page per person: what you own, what to deliver, by when, and how your work plugs into everyone else's.

| Guide | Task | Shared component | Report / video |
|---|---|---|---|
| [P1.md](P1.md) | CartPole-v1 (4 qubits) | `qrl/agent.py`, `qrl/dqn.py` | Method section, video intro |
| [P2.md](P2.md) | FrozenLake-v1 (4 qubits) | `qrl/noise.py`, presets, tests | Noise-model section, **report editor** |
| [P3.md](P3.md) | Acrobot-v1 (6 qubits) | `qrl/evaluate.py`, `analysis/` | Results figures, **video editor** |
| [P4.md](P4.md) | LunarLander-v3 (hybrid 8→4) | MLP baselines, `qrl/envs.py` | Related work, limitations, conclusion |

Write your name next to your P-number in `CONTRIBUTING.md` once roles are agreed.

## Shared milestones (all members)

| Date | Milestone |
|---|---|
| Fri 23 Oct | `bash setup.sh` prints `ALL OK` on your machine or Colab |
| Sat 24 – Sun 25 Oct | Your shared component works (see your guide) |
| Wed 28 Oct | **M1:** noise-free VQC trains on 1 seed for your task; protocol frozen |
| Sun 1 Nov | **M2:** Experiment A done (VQC + 2 MLPs × 5 seeds) |
| Wed 4 Nov | **M3:** Experiments B (noise sweep) and C (noise-aware training) done |
| Thu 5 Nov | Analysis meeting: agree 3–4 headline claims |
| Sun 8 Nov | **M4:** full draft report, slides, recorded video |
| Fri 13 Nov | Submit (PDF report + video and code links by email) |

## Rules every guide assumes

- Use `qrl.config.Config`, `qrl.noise.parse_noise` / `make_device` / `apply_noise`, and `qrl.logging_utils` loggers. CSV columns must not change.
- Seeds 0–4. "Solved" = mean of last 100 episodes ≥ `env.spec.reward_threshold`.
- Branch `p<N>/<topic>`, small PRs, `pytest` + `scripts/smoke_test.py` green before pushing.
- Never combine shots with backprop. `default.mixed` only when channel noise is on.
