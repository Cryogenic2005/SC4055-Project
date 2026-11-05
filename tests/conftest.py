from dataclasses import dataclass

import pytest


@dataclass
class TestConfig:
    model: str = "vqc"
    n_layers: int = 2
    n_qubits: int = 4
    n_actions: int = 4
    data_reuploading: bool = True
    seed: int = 0


@pytest.fixture
def config() -> TestConfig:
    return TestConfig()


@pytest.fixture
def config_N5() -> TestConfig:
    return TestConfig(n_layers=5)
