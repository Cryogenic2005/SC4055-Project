"""Single entry point for every kind of noise used in the project.

Noise is described by a string so it can be passed on the command line and
logged verbatim:

    none
    shots=1000                 finite-shot estimation of expectation values
    coherent=0.1               Gaussian over-rotation, sigma in radians, resampled every call
    depolarizing=0.01          DepolarizingChannel(p) after every gate (default.mixed)
    readout=0.02               symmetric bit-flip on measurement (default.mixed readout_prob)
    preset=heron2026 | nisq2022 | stress

Conventions (write these in the report):
* `depolarizing=p` means PennyLane's `DepolarizingChannel(p)`: each Pauli error
  X, Y, Z occurs with probability p/3, applied to every wire a gate touches.
  For a 2-qubit gate we apply the single-qubit channel to both wires.
* A device's reported *average gate error* eps is NOT the same number as p.
  Use `gate_error_to_depol_p(eps, n_qubits)` when converting from calibration data
  (eps = 2p/3 for a single qubit, so p = 1.5 eps).
* Shots are only ever combined with parameter-shift gradients, never backprop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml


@dataclass(frozen=True)
class NoiseSpec:
    kind: str = "none"  # none | shots | coherent | depolarizing | readout | preset
    level: float = 0.0  # numeric level for the simple kinds
    name: str = ""  # preset name
    # Preset contents (all optional)
    p1: float = 0.0  # depolarizing p after each 1-qubit gate
    p2: float = 0.0  # depolarizing p after each 2-qubit gate (per wire)
    readout_p: float = 0.0
    amp_damp: float = 0.0  # amplitude damping gamma after each gate
    t1_us: float | None = None  # thermal relaxation (optional)
    t2_us: float | None = None
    gate_us: float = 0.05  # assumed gate duration for thermal relaxation
    shots: int | None = None
    coherent_sigma: float = 0.0

    # ----- derived -----
    @property
    def needs_mixed(self) -> bool:
        return self.kind in ("depolarizing", "readout") or (
            self.kind == "preset"
            and (
                self.p1 > 0
                or self.p2 > 0
                or self.readout_p > 0
                or self.amp_damp > 0
                or self.t1_us is not None
            )
        )

    @property
    def label(self) -> str:
        if self.kind == "none":
            return "none"
        if self.kind == "preset":
            return f"preset={self.name}"
        lvl = int(self.level) if self.kind == "shots" else self.level
        return f"{self.kind}={lvl}"

    # Columns for the eval CSV
    @property
    def eval_type(self) -> str:
        return self.kind

    @property
    def eval_level(self) -> str:
        if self.kind == "none":
            return "0"
        if self.kind == "preset":
            return self.name
        return str(int(self.level) if self.kind == "shots" else self.level)


def gate_error_to_depol_p(eps: float, n_qubits: int = 1) -> float:
    """Average gate infidelity eps -> PennyLane DepolarizingChannel parameter p.

    Fully depolarizing with probability lam gives r = lam (d-1)/d with d = 2**n.
    PennyLane's channel has lam = 4p/3 on one qubit, so for n=1: eps = 2p/3.
    For n=2 we keep applying the 1-qubit channel per wire and use the 1-qubit map
    as an approximation, which we state in the report."""
    d = 2**n_qubits
    lam = eps * d / (d - 1)
    return 0.75 * lam if n_qubits == 1 else 0.75 * lam / 2


# --- presets anchored to the plan's hardware table -----------------------------
# Heron r2/r3 (2024-26): 1Q ~3e-4, 2Q ~3e-3, readout ~1.5e-2, T1~150us, T2~100us, 1000 shots.
# Presets are simulated with exact expectation values (shots=None) so that they stay
# jit/vmap-able and so that shot noise remains its own axis in the sweep. The plan's
# "1000 shots" for Heron is covered by the separate `shots=1000` condition.
_HERON = dict(
    p1=gate_error_to_depol_p(3e-4),
    p2=gate_error_to_depol_p(3e-3, 2),
    readout_p=1.5e-2,
    t1_us=150.0,
    t2_us=100.0,
)
# Skolik et al. 2022 config (a): 1Q 1e-3, 2Q 1e-2, amplitude damping 3e-4, readout 1e-2.
_NISQ = dict(
    p1=gate_error_to_depol_p(1e-3),
    p2=gate_error_to_depol_p(1e-2, 2),
    readout_p=1e-2,
    amp_damp=3e-4,
)
PRESETS: dict[str, NoiseSpec] = {
    "heron2026": NoiseSpec(kind="preset", name="heron2026", **_HERON),  # pyright: ignore[reportArgumentType]
    "nisq2022": NoiseSpec(kind="preset", name="nisq2022", **_NISQ),  # pyright: ignore[reportArgumentType]
    "stress": NoiseSpec(
        kind="preset",
        name="stress",
        p1=min(10 * _HERON["p1"], 0.75),
        p2=min(10 * _HERON["p2"], 0.75),
        readout_p=min(10 * _HERON["readout_p"], 0.5),
        t1_us=_HERON["t1_us"] / 10,
        t2_us=_HERON["t2_us"] / 10,
    ),
}

# The sweep used by Experiment B (17 conditions incl. clean reference).
NOISE_MENU: list[str] = (
    ["none"]
    + [f"shots={n}" for n in (1000, 100, 10)]
    + [f"coherent={s}" for s in (0.05, 0.1, 0.2)]
    + [f"depolarizing={p}" for p in (0.001, 0.01, 0.05, 0.1)]
    + [f"readout={p}" for p in (0.01, 0.02, 0.05)]
    + ["preset=heron2026", "preset=nisq2022", "preset=stress"]
)


def parse_noise(s: str | None) -> NoiseSpec:
    if not s or s == "none":
        return NoiseSpec()
    if "=" not in s:
        raise ValueError(f"noise string {s!r} must look like kind=level")
    kind, val = s.split("=", 1)
    kind = kind.strip().lower()
    if kind == "preset":
        if val not in PRESETS:
            raise ValueError(f"unknown preset {val!r}; choose from {list(PRESETS)}")
        return PRESETS[val]
    level = float(val)
    if kind == "shots":
        return NoiseSpec(kind="shots", level=level, shots=int(level))
    if kind == "coherent":
        return NoiseSpec(kind="coherent", level=level, coherent_sigma=level)
    if kind == "depolarizing":
        return NoiseSpec(kind="depolarizing", level=level, p1=level, p2=level)
    if kind == "readout":
        return NoiseSpec(kind="readout", level=level, readout_p=level)
    raise ValueError(f"unknown noise kind {kind!r}")


def make_device(n_qubits: int, spec: NoiseSpec, seed: int | None = None):
    """default.qubit for noise-free / shots / coherent; default.mixed when channels are on.

    KNOWN BUG (PennyLane 0.45.1, verified 22 Oct 2026): `default.mixed(readout_prob=p)`
    raises `TypeError: 'float' object is not iterable` on any analytic measurement
    (default_mixed.py passes the float where qubit_mixed/measure.py expects a list of
    channel callables). We therefore do NOT use readout_prob; readout error is applied
    explicitly with `apply_readout_noise()` as a BitFlip(p) on every measured wire right
    before measurement, which is the same symmetric-flip channel for Z-basis readout.
    """
    shots = spec.shots if spec.kind in ("shots", "preset") else None
    if spec.needs_mixed:
        return qml.device("default.mixed", wires=n_qubits, shots=shots, seed=seed)
    return qml.device("default.qubit", wires=n_qubits, shots=shots, seed=seed)


def apply_readout_noise(spec: NoiseSpec, wires) -> None:
    """Symmetric readout flip. Call once, on the measured wires, just before `qml.expval`.
    <Z> -> (1 - 2p) <Z>."""
    if spec.readout_p > 0:
        wires = list(wires) if not isinstance(wires, int) else [wires]
        for w in wires:
            qml.BitFlip(spec.readout_p, wires=w)


def apply_noise(spec: NoiseSpec, wires) -> None:
    """Insert the channels that follow one gate acting on `wires`. Call inside the QNode
    right after each gate. No-op for statevector-only noise kinds."""
    if not spec.needs_mixed:
        return
    wires = list(wires) if not isinstance(wires, int) else [wires]
    p = spec.p1 if len(wires) == 1 else spec.p2
    for w in wires:
        if p > 0:
            qml.DepolarizingChannel(p, wires=w)
        if spec.amp_damp > 0:
            qml.AmplitudeDamping(spec.amp_damp, wires=w)
        if spec.t1_us is not None and spec.t2_us is not None:
            # pe = 0: relaxation towards |0>. Durations in the same unit (us).
            qml.ThermalRelaxationError(
                0.0, spec.t1_us, spec.t2_us, spec.gate_us, wires=w
            )


def perturb_angles(spec: NoiseSpec, rng: np.random.Generator, *arrays):
    """Coherent over-rotation: add N(0, sigma^2) to every rotation angle. Returns new arrays.
    Applied outside the QNode so the statevector fast path is unchanged."""
    sigma = spec.coherent_sigma
    if sigma <= 0:
        return arrays
    return tuple(
        a + rng.normal(0.0, sigma, size=np.shape(a)).astype(np.float32) for a in arrays
    )
