"""Cross-check the hand-written PennyLane noise model against Qiskit Aer (owner: P2).

    uv sync --extra qiskit        # or: pip install -e ".[qiskit]"
    qiskit-crosscheck [--checkpoint runs/checkpoints/FrozenLake-v1_vqc_0/best.pkl]

Three independent checks, each printed as PASS/FAIL:

1. Conversion. Qiskit's `average_gate_fidelity` of the channel we actually apply
   (per-wire DepolarizingChannel(p), p = gate_error_to_depol_p(eps, n)) must give back
   the device gate error eps, for the 1Q and 2Q Heron and NISQ numbers.
2. Circuit, Aer. The FrozenLake circuit with trained parameters is built in Qiskit with an
   Aer `NoiseModel` equivalent to `preset=heron2026` (depolarizing + thermal relaxation
   after every gate, symmetric readout error) and simulated as a density matrix. Its <Z_i>
   must match PennyLane `default.mixed` with `preset=heron2026` for all 16 states.
3. Circuit, importer. The same Aer `NoiseModel` is imported with `qml.from_qiskit_noise`
   and attached to a noise-free PennyLane QNode with `qml.add_noise`. The importer drops
   readout errors (it warns), so BitFlip readout is re-added by hand, as the plan says.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import (
    DensityMatrix,
    Kraus,
    SparsePauliOp,
    average_gate_fidelity,
)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)

from sc4055_project.qrl.agent import _build_circuit
from sc4055_project.qrl.noise import PRESETS, gate_error_to_depol_p, parse_noise
from sc4055_project.tasks.frozenlake.train import state_to_bits

TOL = 1e-5  # PennyLane may run in float32 (JAX x64 off): agreement is then ~1e-6
N_QUBITS = 4  # FrozenLake-v1 has 16 states, so 4 qubits
N_ACTIONS = 4  # FrozenLake-v1 has 4 actions, so 4 output qubits


def wire_error(spec):
    """Qiskit error for one wire after a gate: DepolarizingChannel(p) then thermal relaxation.
    PennyLane's DepolarizingChannel(p) equals Qiskit's depolarizing_error(4p/3)."""

    def err(p):
        e = depolarizing_error(4 * p / 3, 1)
        if spec.t1_us is not None:
            e = e.compose(
                thermal_relaxation_error(spec.t1_us, spec.t2_us, spec.gate_us)
            )
        return e

    return err(spec.p1), err(spec.p2)


def heron_noise_model(spec) -> NoiseModel:
    e1, e2w = wire_error(spec)
    nm = NoiseModel(basis_gates=["rx", "ry", "rz", "cz"])
    nm.add_all_qubit_quantum_error(e1, ["rx", "ry", "rz"])
    nm.add_all_qubit_quantum_error(
        e2w.tensor(e2w), ["cz"]
    )  # independent per-wire channel
    r = spec.readout_p
    nm.add_all_qubit_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]))
    return nm


def check_conversion() -> bool:
    ok = True
    for n, eps in [(1, 3e-4), (2, 3e-3), (1, 1e-3), (2, 1e-2)]:
        p = gate_error_to_depol_p(eps, n)
        chan = Kraus(qml.DepolarizingChannel(p, wires=0).kraus_matrices())
        for _ in range(n - 1):
            chan = chan.tensor(
                Kraus(qml.DepolarizingChannel(p, wires=0).kraus_matrices())
            )
        got = 1 - average_gate_fidelity(chan)
        good = abs(got - eps) < 1e-9
        ok &= good
        print(
            f"  {n}Q eps={eps:g}: p={p:.6g}, Qiskit average gate error={got:.6g}  {'PASS' if good else 'FAIL'}"
        )
    return ok


def qiskit_circuit(bits, theta, lam, n_layers) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
    for i in range(N_QUBITS):
        qc.rx(float(np.pi * bits[i] * lam[0][i]), i)
    for l in range(n_layers):
        for i in range(N_QUBITS):
            qc.rx(float(theta[l, i, 0]), i)
            qc.ry(float(theta[l, i, 1]), i)
            qc.rz(float(theta[l, i, 2]), i)
        for i in range(N_QUBITS):
            qc.cz(i, (i + 1) % N_QUBITS)
    qc.save_density_matrix()
    return qc


def aer_expz(qc, nm, readout_p) -> np.ndarray:
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    rho = DensityMatrix(
        sim
        .run(transpile(qc, sim, optimization_level=0))
        .result()
        .data()["density_matrix"]
    )
    z = np.array([
        rho.expectation_value(
            SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], N_QUBITS)
        ).real
        for i in range(N_QUBITS)
    ])
    return (
        1 - 2 * readout_p
    ) * z  # symmetric readout error on <Z>; checked with shots below


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--checkpoint", default="runs/checkpoints/FrozenLake-v1_vqc_0/best.pkl"
    )
    ap.add_argument(
        "--out", default="analysis_out/p2_frozenlake/qiskit_crosscheck.json"
    )
    a = ap.parse_args(argv)

    ckpt = Path(a.checkpoint)
    cfg = json.loads((ckpt.parent / "config.json").read_text())
    with open(ckpt, "rb") as fh:
        params = pickle.load(fh)
    theta, lam = np.asarray(params["theta"]), np.asarray(params["lam"])
    n_layers = cfg["n_layers"]
    spec = PRESETS["heron2026"]
    nm = heron_noise_model(spec)
    bits_all = state_to_bits(np.arange(16), N_QUBITS)

    print("1. Gate-error -> channel conversion (Qiskit average_gate_fidelity)")
    ok1 = check_conversion()

    print(
        "2. FrozenLake circuit, preset=heron2026: PennyLane default.mixed vs Qiskit Aer density matrix"
    )
    pl = _build_circuit(
        n_layers, N_QUBITS, N_ACTIONS, parse_noise("preset=heron2026"), reupload=False
    )
    pl_z = np.array([np.asarray(pl(b, theta, lam)) for b in bits_all])
    aer_z = np.array([
        aer_expz(qiskit_circuit(b, theta, lam, n_layers), nm, spec.readout_p)
        for b in bits_all
    ])
    clean = _build_circuit(
        n_layers, N_QUBITS, N_ACTIONS, parse_noise("none"), reupload=False
    )
    clean_z = np.array([np.asarray(clean(b, theta, lam)) for b in bits_all])
    d2 = float(np.max(np.abs(pl_z - aer_z)))
    ok2 = d2 < TOL
    print(
        f"  max |<Z>_PL - <Z>_Aer| over 16 states x 4 qubits = {d2:.2e}  {'PASS' if ok2 else 'FAIL'}"
        f"   (noise moves <Z> by up to {np.max(np.abs(pl_z - clean_z)):.3f})"
    )

    # Readout: confirm the analytic (1-2p) factor against Aer's sampled ReadoutError.
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    ro = NoiseModel()
    ro.add_all_qubit_readout_error(
        ReadoutError([
            [1 - spec.readout_p, spec.readout_p],
            [spec.readout_p, 1 - spec.readout_p],
        ])
    )
    counts = (
        AerSimulator(noise_model=ro, seed_simulator=0)
        .run(qc, shots=200_000)
        .result()
        .get_counts()
    )
    z_ro = (counts.get("0", 0) - counts.get("1", 0)) / 200_000
    ok_ro = abs(z_ro - (1 - 2 * spec.readout_p)) < 0.01
    print(
        f"  Aer sampled readout on |0>: <Z>={z_ro:.4f} vs 1-2p={1 - 2 * spec.readout_p:.4f}  {'PASS' if ok_ro else 'FAIL'}"
    )

    print(
        "3. qml.from_qiskit_noise import of the same Aer NoiseModel (+ readout re-added by hand)"
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pl_nm = qml.from_qiskit_noise(nm)
    dropped = [str(x.message) for x in w if "readout" in str(x.message).lower()]
    print(f"  importer warnings about readout: {len(dropped)}")
    dev = qml.device("default.mixed", wires=N_QUBITS)

    @qml.qnode(dev)
    def bare(bits):
        for i in range(N_QUBITS):
            qml.RX(np.pi * bits[i] * lam[0][i], wires=i)
        for l in range(n_layers):
            for i in range(N_QUBITS):
                qml.RX(theta[l, i, 0], wires=i)
                qml.RY(theta[l, i, 1], wires=i)
                qml.RZ(theta[l, i, 2], wires=i)
            for i in range(N_QUBITS):
                qml.CZ(wires=[i, (i + 1) % N_QUBITS])
        for i in range(N_QUBITS):
            qml.BitFlip(spec.readout_p, wires=i)  # readout re-added by hand
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    noisy = qml.add_noise(bare, pl_nm)
    imp_z = np.array([np.asarray(noisy(b)) for b in bits_all])
    d3 = float(np.max(np.abs(imp_z - pl_z)))
    ok3 = d3 < TOL
    print(f"  max |<Z>_imported - <Z>_preset| = {d3:.2e}  {'PASS' if ok3 else 'FAIL'}")

    ok = ok1 and ok2 and ok_ro and ok3
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "checkpoint": a.checkpoint,
                "conversion_pass": ok1,
                "aer_max_abs_diff": d2,
                "readout_sampled_z": z_ro,
                "importer_max_abs_diff": d3,
                "importer_readout_warnings": len(dropped),
                "all_pass": ok,
            },
            indent=2,
        )
    )
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
