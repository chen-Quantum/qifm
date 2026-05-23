"""Hermitian basis library for the learnable Hamiltonian H_theta.

The library is intentionally physics-flavoured (not a generic Pauli-string
basis) so the optimiser tunes interference geometry directly. Five
families:

    local_potential_terms(N)   -> N diagonal operators (one per site)
    hopping_terms(N)           -> N-1 nearest-neighbour real-hopping operators
    graph_laplacian_terms(N)   -> one operator: the line-graph Laplacian
    pauli_string_terms(n_q)    -> n_q + (n_q-1) Z, ZZ operators
    entangling_terms(N)        -> brick-wall XX + ZZ couplings (for power-of-2 N)

Each operator B_m is returned as a (N, N) complex Hermitian matrix.

The combined Hamiltonian is

    H_theta(x) = sum_m theta_m B_m + alpha V(x)

where V(x) is the diagonal data-potential built by `encodings.data_potential`.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def _is_hermitian(M: np.ndarray, tol: float = 1e-9) -> bool:
    return np.allclose(M, M.conj().T, atol=tol)


# -----------------------------------------------------------------------
# Basis families
# -----------------------------------------------------------------------

def local_potential_terms(N: int) -> List[np.ndarray]:
    """One diagonal operator per grid site. Each places a +1 on site j."""
    out: List[np.ndarray] = []
    for j in range(N):
        B = np.zeros((N, N), dtype=np.complex128)
        B[j, j] = 1.0
        out.append(B)
    return out


def hopping_terms(N: int) -> List[np.ndarray]:
    """N-1 nearest-neighbour hopping operators: |j><j+1| + |j+1><j|."""
    out: List[np.ndarray] = []
    for j in range(N - 1):
        B = np.zeros((N, N), dtype=np.complex128)
        B[j, j + 1] = 1.0
        B[j + 1, j] = 1.0
        out.append(B)
    return out


def graph_laplacian_terms(N: int) -> List[np.ndarray]:
    """One operator: the standard line-graph Laplacian, D - A."""
    A = np.zeros((N, N), dtype=np.complex128)
    for j in range(N - 1):
        A[j, j + 1] = 1.0
        A[j + 1, j] = 1.0
    D = np.diag(A.sum(axis=1))
    L = D - A
    return [L]


# -----------------------------------------------------------------------
# Pauli-string and brick-wall entanglers (for power-of-2 dimensions)
# -----------------------------------------------------------------------

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _kron_chain(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _single_pauli(n_q: int, qubit: int, P: np.ndarray) -> np.ndarray:
    ops = [(P if i == qubit else _I) for i in range(n_q)]
    return _kron_chain(ops)


def _two_pauli(n_q: int, q1: int, q2: int, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    ops = []
    for i in range(n_q):
        if i == q1:
            ops.append(P1)
        elif i == q2:
            ops.append(P2)
        else:
            ops.append(_I)
    return _kron_chain(ops)


def pauli_string_terms(n_q: int) -> List[np.ndarray]:
    """Z and nearest-neighbour ZZ Pauli strings on n_q qubits."""
    out: List[np.ndarray] = []
    for i in range(n_q):
        out.append(_single_pauli(n_q, i, _Z))
    for i in range(n_q - 1):
        out.append(_two_pauli(n_q, i, i + 1, _Z, _Z))
    return out


def entangling_terms(n_q: int) -> List[np.ndarray]:
    """Brick-wall XX + ZZ couplings between neighbours. Returns 2*(n_q-1)
    operators."""
    out: List[np.ndarray] = []
    for i in range(n_q - 1):
        out.append(_two_pauli(n_q, i, i + 1, _X, _X))
        out.append(_two_pauli(n_q, i, i + 1, _Z, _Z))
    return out


# -----------------------------------------------------------------------
# Library assembly
# -----------------------------------------------------------------------

def build_basis(N: int, families: Sequence[str]) -> Tuple[List[np.ndarray], List[str]]:
    """Construct the basis for a given Hilbert-space dimension N. For Pauli /
    entangling families we require N = 2**n_q.
    """
    basis: List[np.ndarray] = []
    labels: List[str] = []
    n_q = int(round(np.log2(N))) if (N & (N - 1) == 0 and N >= 2) else None
    for fam in families:
        if fam == "local_potential":
            ops = local_potential_terms(N)
            basis.extend(ops); labels.extend([f"V_{i}" for i in range(len(ops))])
        elif fam == "hopping":
            ops = hopping_terms(N)
            basis.extend(ops); labels.extend([f"T_{i}" for i in range(len(ops))])
        elif fam == "graph_laplacian":
            ops = graph_laplacian_terms(N)
            basis.extend(ops); labels.extend([f"L_lap"])
        elif fam == "pauli_string":
            if n_q is None:
                raise ValueError("pauli_string family requires N to be a power of 2.")
            ops = pauli_string_terms(n_q)
            basis.extend(ops); labels.extend([f"P_{i}" for i in range(len(ops))])
        elif fam == "entangling":
            if n_q is None:
                raise ValueError("entangling family requires N to be a power of 2.")
            ops = entangling_terms(n_q)
            basis.extend(ops); labels.extend([f"E_{i}" for i in range(len(ops))])
        else:
            raise ValueError(f"Unknown family {fam!r}")
    return basis, labels


def assemble_hamiltonian(theta: np.ndarray, basis: Sequence[np.ndarray],
                          V_x: np.ndarray | None = None,
                          alpha: float = 1.0) -> np.ndarray:
    """H = sum_m theta_m B_m + alpha * V_x (V_x optional)."""
    theta = np.asarray(theta, dtype=np.float64).flatten()
    if theta.size != len(basis):
        raise ValueError(f"theta has length {theta.size}, basis has length {len(basis)}")
    H = np.zeros_like(basis[0], dtype=np.complex128)
    for t, B in zip(theta, basis):
        H = H + t * B
    if V_x is not None:
        H = H + alpha * V_x
    # Hermitise to guard against numerical drift.
    return 0.5 * (H + H.conj().T)
