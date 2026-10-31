import csv
from pathlib import Path

import numpy as np

from sc4055_project.qrl.config import default_config_for
from sc4055_project.qrl.logging_utils import TRAIN_COLUMNS
from sc4055_project.tasks.cartpole.train import encode_obs, train


def test_encode_obs_arctan_shape_dtype():
    x = np.asarray([0.0, 1.0, -1.0, 10.0], dtype=np.float32)
    y = encode_obs(x)
    assert y.shape == (4,)
    assert y.dtype == np.float32
    assert np.allclose(y, np.arctan(x), atol=1e-7)


def test_tiny_training_run_writes_spec_columns(tmp_path: Path):
    cfg = default_config_for(
        "CartPole-v1",
        "vqc",
        n_episodes=2,
        max_steps=32,
        learning_starts=4,
        batch_size=4,
        eval_episodes=0,
        checkpoint_every=0,
        log_every=1,
        run_dir=str(tmp_path),
    )

    out = train(cfg)
    assert out["n_params"] > 0

    csvs = sorted(Path(tmp_path, "train").glob("*.csv"))
    assert len(csvs) == 1
    with open(csvs[0], newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 2
    assert list(rows[0].keys()) == TRAIN_COLUMNS
    assert rows[0]["task"] == "CartPole-v1"
    assert rows[0]["model"] == "vqc"
