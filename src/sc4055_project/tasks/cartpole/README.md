# cartpole task

Owner: **P1**

CartPole-v1 (4 qubits, arctan re-uploading, 5 layers). Shared component: core QAgent (qrl/agent.py) and DQN loop (qrl/dqn.py).

Task wrappers now live here:

- `train.py` – task-specific encoder (`arctan`) + call into shared `qrl.dqn.train_agent`
- `evaluate.py` – greedy eval under noise menu, writing `runs/eval/<run_name>.csv`

Entry point:

- `cartpole train --seed 0`
- `cartpole train --seed 0 --noise depolarizing=0.01`
- `cartpole evaluate --checkpoint runs/checkpoints/CartPole-v1_vqc_0/best.pkl`

These follow the same conventions as `tasks/frozenlake/` (CSV schema, checkpoint layout, best-checkpoint selection).
