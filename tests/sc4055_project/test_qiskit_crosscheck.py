"""Qiskit cross-check of the noise conversion (skipped without the `qiskit` extra)."""

import pytest

pytest.importorskip("qiskit_aer")
pytest.importorskip("pennylane_qiskit")


def test_gate_error_conversion_agrees_with_qiskit():
    from sc4055_project.qiskit_crosscheck import check_conversion

    assert check_conversion()
