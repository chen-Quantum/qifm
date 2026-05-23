"""Illustrative-only: show how the continuous Hamiltonian evolution could
be approximated by a Trotterised circuit. Used in tests and in the course
notes; never required by the main pipeline.

If Qiskit is not installed, the functions return descriptive strings
instead of circuits.
"""

from __future__ import annotations

from typing import List, Sequence


def trotter_recipe(theta_per_term: Sequence[float], term_labels: Sequence[str],
                   T: float, n_steps: int) -> List[str]:
    """Return a list of step descriptions (textual) showing one Trotter
    decomposition of exp(-i T H) with H = sum_m theta_m B_m.
    """
    dt = float(T) / max(int(n_steps), 1)
    steps: List[str] = []
    for s in range(int(n_steps)):
        for t, lbl in zip(theta_per_term, term_labels):
            steps.append(f"step {s + 1}: exp(-i {dt * float(t):+.4f} {lbl})")
    return steps


def build_qiskit_circuit(n_qubits: int, theta_per_term: Sequence[float],
                          term_labels: Sequence[str], T: float = 1.0,
                          n_steps: int = 2):  # pragma: no cover
    """Return a Qiskit QuantumCircuit implementing one Trotter step per
    layer. Each label like 'Z_q', 'ZZ_qa_qb', 'X_q' is parsed and emitted
    as the corresponding Pauli rotation.
    """
    try:
        from qiskit import QuantumCircuit
    except Exception:
        raise RuntimeError("Qiskit is not installed; circuit_mapping skipped.")
    dt = float(T) / max(int(n_steps), 1)
    qc = QuantumCircuit(n_qubits)
    for _ in range(int(n_steps)):
        for t, lbl in zip(theta_per_term, term_labels):
            angle = float(t) * dt
            kind, _, rest = lbl.partition("_")
            qubits = [int(q) for q in rest.split("_") if q]
            if kind in ("Z", "P"):
                qc.rz(2 * angle, qubits[0] if qubits else 0)
            elif kind in ("X", "T"):
                qc.rx(2 * angle, qubits[0] if qubits else 0)
            elif kind == "ZZ":
                qc.rzz(2 * angle, qubits[0], qubits[1])
            elif kind == "XX":
                qc.rxx(2 * angle, qubits[0], qubits[1])
            else:
                # Skip generic potential terms (they are diagonal phases on
                # the chosen computational basis - emit an Rz on qubit 0).
                qc.rz(2 * angle, 0)
    return qc
