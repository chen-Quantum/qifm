"""Miscellaneous invariants."""

from __future__ import annotations


def test_circuit_mapping_text_recipe() -> None:
    from src.circuit_mapping import trotter_recipe
    steps = trotter_recipe([0.1, -0.2, 0.4], ["P_0", "P_1", "ZZ_0_1"], T=1.0, n_steps=2)
    assert len(steps) == 6
    for s in steps:
        assert "exp(" in s


def test_circuit_mapping_qiskit_optional() -> None:
    try:
        import qiskit  # noqa: F401
    except Exception:
        return  # Qiskit absent - skip silently per the brief.
    from src.circuit_mapping import build_qiskit_circuit
    qc = build_qiskit_circuit(3, [0.1, -0.2, 0.3, 0.5],
                                 ["Z_0", "Z_1", "ZZ_0_1", "XX_1_2"],
                                 T=1.0, n_steps=2)
    assert qc.num_qubits == 3
    assert qc.depth() >= 4
