import csv

from sc4055_project.qrl.logging_utils import (
    EVAL_COLUMNS,
    TRAIN_COLUMNS,
    EvalLogger,
    TrainLogger,
    write_random_baseline,
)


def test_train_logger_columns(tmp_path):
    p = tmp_path / "t.csv"
    with TrainLogger(
        p,
        task="FrozenLake-v1",
        model="vqc",
        experiment="A",
        train_noise="none",
        seed=0,
        n_params=28,
    ) as lg:
        lg.episode(0, 1.0)
        lg.episode(1, 0.0)
    with open(p) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == TRAIN_COLUMNS
    assert rows[1]["episode"] == "1" and rows[0]["n_params"] == "28"


def test_eval_logger_columns(tmp_path):
    p = tmp_path / "e.csv"
    with EvalLogger(
        p, task="FrozenLake-v1", model="vqc", experiment="B", train_noise="none", seed=0
    ) as lg:
        lg.episode("none", "0", 0, 1.0)
        lg.episode("depolarizing", "0.01", 0, 0.0)
    with open(p) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == EVAL_COLUMNS
    assert rows[1]["eval_noise_type"] == "depolarizing"


def test_random_upsert(tmp_path):
    p = tmp_path / "random.csv"
    write_random_baseline(p, "CartPole-v1", 22.0)
    write_random_baseline(p, "FrozenLake-v1", 0.01)
    write_random_baseline(p, "CartPole-v1", 23.0)
    with open(p) as f:
        rows = {r["task"]: float(r["r_random"]) for r in csv.DictReader(f)}
    assert rows == {"CartPole-v1": 23.0, "FrozenLake-v1": 0.01}
