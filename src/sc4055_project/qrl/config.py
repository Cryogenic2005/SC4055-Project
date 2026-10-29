"""Run configuration. Everything a run needs lives here so that a saved
config.json fully reproduces the run."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Gymnasium ids we support, with the plan's qubit budget and encoder.
TASKS: dict[str, dict[str, Any]] = {
    "FrozenLake-v1": dict(n_qubits=4, encoding="basis", n_actions=4, obs_dim=16),
    "CartPole-v1": dict(n_qubits=4, encoding="arctan", n_actions=2, obs_dim=4),
    "Acrobot-v1": dict(n_qubits=6, encoding="arctan", n_actions=3, obs_dim=6),
    # Hybrid: classical encoder 8 -> 4 feeding a 4-qubit VQC.
    "LunarLander-v3": dict(n_qubits=4, encoding="linear", n_actions=4, obs_dim=8),
}

# Thresholds we report against. CartPole: Gymnasium registers 475 (docs say 500).
REWARD_THRESHOLD = {
    "FrozenLake-v1": 0.70,
    "CartPole-v1": 475.0,
    "Acrobot-v1": -100.0,
    "LunarLander-v3": 200.0,
}

MODELS = ("vqc", "mlp_small", "mlp_large", "encoder_only")


@dataclass
class Config:
    # --- what ---
    env_id: str = "CartPole-v1"
    model: str = "vqc"              # vqc | mlp_small | mlp_large | encoder_only
    seed: int = 0
    experiment: str = "A"           # A = noise-free training, C = noise-aware training

    # --- circuit ---
    n_qubits: int | None = None     # default from TASKS
    n_layers: int = 5
    encoding: str | None = None     # basis | arctan | linear (default from TASKS)
    readout: str = "z"              # z: <Z_i> per action; zz: pairs <Z_i Z_j>
    data_reuploading: bool = True
    trainable_input_scaling: bool = True

    # --- noise during training (Experiment C); evaluation noise is passed separately ---
    # Examples: "none", "depolarizing=0.01", "coherent=0.1", "readout=0.02",
    #           "shots=1000", "preset=heron2026"
    noise: str = "none"

    # --- DQN ---
    n_episodes: int = 1000
    max_steps: int | None = None    # None -> env default
    gamma: float = 0.99
    batch_size: int = 32
    buffer_size: int = 10_000
    learning_starts: int = 500
    train_every: int = 1
    target_update_every: int = 50   # gradient steps
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 300
    epsilon_decay_rate: float | None = None   # if set: eps = max(end, start * rate**episode) (multiplicative)
    reward_shaping: bool = False    # FrozenLake only: train on +1 goal / -0.2 hole / -0.01 step (Chen et al.);
                                    # the LOGGED return is always the raw env return
    reward_scale: float = 1.0       # multiplier for the TD target only; the LOGGED return is always raw.
                                    # LunarLander: terminal rewards are +-100 but the readout is a bounded
                                    # <Z>, so its train.py defaults to 0.01. Other tasks keep 1.0.
    lr_circuit: float = 1e-3
    lr_input: float = 1e-3
    lr_output: float = 0.1
    lr_mlp: float = 1e-3

    # --- MLP sizes ---
    mlp_large_hidden: tuple[int, ...] = (64, 64)

    # --- bookkeeping ---
    run_dir: str = "runs"
    checkpoint_every: int = 25      # episodes; also runs a short greedy eval and updates best.pkl
    eval_episodes: int = 20         # greedy episodes per periodic eval (0 disables)
    log_every: int = 10             # print to stdout
    tag: str = ""                   # optional suffix for file names
    env_kwargs: dict = field(default_factory=dict)  # e.g. {"is_slippery": False}

    def __post_init__(self) -> None:
        if self.env_id not in TASKS:
            raise ValueError(f"Unknown env_id {self.env_id!r}; choose from {list(TASKS)}")
        if self.model not in MODELS:
            raise ValueError(f"Unknown model {self.model!r}; choose from {MODELS}")
        t = TASKS[self.env_id]
        if self.n_qubits is None:
            self.n_qubits = t["n_qubits"]
        if self.encoding is None:
            self.encoding = t["encoding"]
        if self.experiment == "A" and self.noise != "none":
            # Protect the protocol: A is always noise-free.
            self.experiment = "C"

    # convenience
    @property
    def n_actions(self) -> int:
        return TASKS[self.env_id]["n_actions"]

    @property
    def obs_dim(self) -> int:
        return TASKS[self.env_id]["obs_dim"]

    @property
    def threshold(self) -> float:
        return REWARD_THRESHOLD[self.env_id]

    @property
    def run_name(self) -> str:
        # Matches docs/analysis_spec: <task>_<model>_<seed>  (+ noise tag for C runs)
        name = f"{self.env_id}_{self.model}_{self.seed}"
        if self.noise != "none":
            name += "_" + self.noise.replace("=", "")
        if self.tag:
            name += "_" + self.tag
        return name

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        d = json.loads(Path(path).read_text())
        if isinstance(d.get("mlp_large_hidden"), list):
            d["mlp_large_hidden"] = tuple(d["mlp_large_hidden"])
        return cls(**d)

    @classmethod
    def from_overrides(cls, **kw: Any) -> "Config":
        return cls(**{k: v for k, v in kw.items() if v is not None})


def default_config_for(env_id: str, model: str = "vqc", **kw: Any) -> Config:
    """Per-task defaults chosen so that a 1-seed run finishes in minutes.
    These are starting points for the M1 hyperparameter sweep, not final values."""
    per_task: dict[str, dict[str, Any]] = {
        # 5 layers (68 params: 60 angles + 4 input scales + 4 output weights): 3 layers did not learn in 500 episodes in our tests; 5 layers reached
        # 100% greedy success on is_slippery=False with shaping + eps *= 0.99 per episode.
        # Verified 22 Oct: with these settings seeds reach ~0.95 train mean / 1.00 greedy on the
        # deterministic map in ~500 episodes (2 min). gamma 0.9 and no re-uploading follow Chen et al.
        "FrozenLake-v1": dict(n_layers=5, n_episodes=600, batch_size=32, learning_starts=32, gamma=0.9,
                              data_reuploading=False, target_update_every=50,
                              epsilon_decay_rate=0.99, reward_shaping=True, lr_circuit=1e-2, lr_output=0.1,
                              env_kwargs={"is_slippery": False}),  # Chen et al. map; --slippery for the stochastic one
        "CartPole-v1": dict(n_layers=5, n_episodes=1500, batch_size=16, learning_starts=1000,
                            epsilon_decay_episodes=500, target_update_every=1, lr_circuit=1e-3),
        "Acrobot-v1": dict(n_layers=3, n_episodes=600, batch_size=32, learning_starts=1000,
                           epsilon_decay_episodes=200),
        # lr_output was 0.1, which under Adam moves each readout weight by ~0.1 per
        # step: |w_out| ran to ~50 while the TD targets are O(1), so the Q-function
        # diverged by a factor of ~30 and the agent stayed at random. 1e-2 keeps
        # |w_out| ~3 and lets the VQC actually land (verified 25 Oct).
        # epsilon decay stretched to 800 episodes: at 400 the policy went greedy
        # before the Q-function had settled and never recovered.
        "LunarLander-v3": dict(n_layers=4, n_episodes=1000, batch_size=64, learning_starts=2000,
                               epsilon_decay_episodes=800, epsilon_end=0.05,
                               buffer_size=50_000, lr_output=1e-2,
                               reward_scale=0.01),  # terminal +-100 vs bounded <Z>; logged return stays raw
    }
    d = dict(per_task.get(env_id, {}))
    d.update(kw)
    return Config(env_id=env_id, model=model, **d)
