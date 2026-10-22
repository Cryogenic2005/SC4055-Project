# lunarlander task

Owner: **P4**

LunarLander-v3 hybrid (8->4 classical encoder + 4-qubit VQC) plus the encoder-only ablation. Shared component: MLP baselines and env wrappers (qrl/envs.py).

Put your task-specific scripts here. Use `qrl.config.Config`, `qrl.noise` and `qrl.logging_utils` so your CSVs match the analysis spec. See `tasks/frozenlake/` for a complete worked example of the layout (train.py, evaluate.py, agent.py).
