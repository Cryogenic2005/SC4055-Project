"""Textbook contractions that the noise module must reproduce (P2 unit tests)."""

import math

import pennylane as qml
import pytest

from sc4055_project.qrl.noise import (
    NOISE_MENU,
    PRESETS,
    NoiseSpec,
    apply_noise,
    apply_readout_noise,
    gate_error_to_depol_p,
    make_device,
    parse_noise,
)


def _z_after_prep(spec: NoiseSpec, angle: float = 0.0):
    """<Z> on |0> rotated by RX(angle), with the noise of `spec` after the gate."""
    dev = make_device(1, spec, seed=0)

    @qml.qnode(dev)
    def c():
        qml.RX(angle, wires=0)
        apply_noise(spec, [0])
        apply_readout_noise(spec, [0])
        return qml.expval(qml.PauliZ(0))

    return float(c())


def test_pennylane_readout_prob_bug_still_present():
    """If this test starts failing, PennyLane fixed default.mixed(readout_prob=...) and
    make_device() can go back to using it. See the docstring of make_device."""
    dev = qml.device("default.mixed", wires=1, readout_prob=0.1)

    @qml.qnode(dev)
    def c():
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(TypeError):
        c()


def test_parse_roundtrip():
    for s in NOISE_MENU:
        spec = parse_noise(s)
        assert spec.label == s, (s, spec.label)
    assert parse_noise("none").kind == "none"
    with pytest.raises(ValueError):
        parse_noise("banana=1")


def test_depolarizing_shrinks_z_towards_zero():
    # DepolarizingChannel(p): <Z> -> (1 - 4p/3) <Z>
    for p in (0.001, 0.01, 0.1):
        z = _z_after_prep(parse_noise(f"depolarizing={p}"))
        assert z == pytest.approx(1 - 4 * p / 3, abs=1e-9)


def test_readout_flip_scales_z_by_one_minus_two_p():
    for p in (0.01, 0.05):
        z = _z_after_prep(parse_noise(f"readout={p}"))
        assert z == pytest.approx(1 - 2 * p, abs=1e-9)


def test_noise_free_is_exact():
    assert _z_after_prep(parse_noise("none"), angle=math.pi / 3) == pytest.approx(
        math.cos(math.pi / 3), abs=1e-9
    )


def test_device_selection():
    assert make_device(2, parse_noise("none")).name == "default.qubit"
    assert make_device(2, parse_noise("coherent=0.1")).name == "default.qubit"
    assert make_device(2, parse_noise("shots=100")).shots.total_shots == 100
    assert make_device(2, parse_noise("depolarizing=0.01")).name == "default.mixed"
    assert make_device(2, parse_noise("readout=0.01")).name == "default.mixed"
    assert make_device(2, PRESETS["heron2026"]).name == "default.mixed"


def test_gate_error_conversion():
    # single qubit: eps = 2p/3
    assert gate_error_to_depol_p(2 / 3) == pytest.approx(1.0)
    assert gate_error_to_depol_p(3e-4) == pytest.approx(4.5e-4)


@pytest.mark.parametrize("n,eps", [(1, 3e-4), (1, 1e-2), (2, 3e-3), (2, 1e-2)])
def test_gate_error_conversion_matches_channel_fidelity(n, eps):
    """Per-wire DepolarizingChannel(p) on n wires must have average infidelity eps."""
    from functools import reduce
    from itertools import product

    import numpy as np

    p = gate_error_to_depol_p(eps, n)
    kraus = qml.DepolarizingChannel(p, wires=0).kraus_matrices()
    d = 2**n
    f_pro = sum(abs(np.trace(reduce(np.kron, ks))) ** 2 for ks in product(kraus, repeat=n)) / d**2
    assert 1 - (d * f_pro + 1) / (d + 1) == pytest.approx(eps, rel=1e-9)


def test_presets_run_and_stress_is_worse_than_heron():
    z = {k: _z_after_prep(v) for k, v in PRESETS.items()}
    assert all(0 < v < 1 for v in z.values()), z
    assert z["stress"] < z["heron2026"]
    # Heron-2026 gate errors are ~3x smaller than NISQ-2022 (readout is slightly larger).
    assert PRESETS["heron2026"].p1 < PRESETS["nisq2022"].p1
    assert PRESETS["heron2026"].p2 < PRESETS["nisq2022"].p2


def test_shots_device_returns_noisy_but_unbiased_estimate():
    spec = parse_noise("shots=1000")
    vals = [
        _z_after_prep(
            NoiseSpec(kind="shots", level=1000, shots=1000), angle=math.pi / 2
        )
        for _ in range(5)
    ]
    assert all(abs(v) < 0.2 for v in vals)  # true value 0, 1000 shots -> std ~0.03
