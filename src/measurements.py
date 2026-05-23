"""Measurement projectors and Born-rule probabilities.

Three projector families:

    fixed_partition_projectors(N, C)   - partitions the N basis states into
                                          C disjoint subspaces.
    trainable_observable_projectors    - Pi_c = M_c^dag M_c / Tr(...), where
                                          M_c is a learnable square block.
    class_subspace_projectors          - rank-r projectors onto random
                                          orthonormal subspaces.

Born probability for a class c on state psi:

    p_c = <psi | Pi_c | psi>

The trainable projectors are guaranteed positive semidefinite by
construction. We normalise so that sum_c p_c = 1 by stacking the M_c into
a tall isometry V and using Pi_c = V^dag |c><c| V.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def fixed_partition_projectors(N: int, C: int) -> List[np.ndarray]:
    """Partition the N computational-basis states into C contiguous blocks."""
    sizes = [N // C] * C
    for k in range(N % C):
        sizes[k] += 1
    out: List[np.ndarray] = []
    cursor = 0
    for s in sizes:
        P = np.zeros((N, N), dtype=np.complex128)
        for j in range(cursor, cursor + s):
            P[j, j] = 1.0
        cursor += s
        out.append(P)
    return out


def class_subspace_projectors(N: int, C: int, rank: int = 2,
                               seed: int = 0) -> List[np.ndarray]:
    """Random rank-r projectors onto orthonormal subspaces."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, C * rank)) + 1j * rng.standard_normal((N, C * rank))
    Q, _ = np.linalg.qr(A)
    out: List[np.ndarray] = []
    for c in range(C):
        V = Q[:, c * rank: (c + 1) * rank]
        out.append(V @ V.conj().T)
    return out


def stack_isometry(N: int, C: int, params: np.ndarray) -> np.ndarray:
    """Build a tall (CxN) isometry from a flat parameter vector.

    The parameter vector has 2 * C * N reals (real + imag); we reshape into
    a C x N complex matrix and QR-decompose to obtain an orthonormal frame.
    """
    if params.size != 2 * C * N:
        raise ValueError(f"params has size {params.size}, expected {2 * C * N}")
    real = params[: C * N].reshape(C, N)
    imag = params[C * N:].reshape(C, N)
    M = real + 1j * imag
    Q, _ = np.linalg.qr(M.T)
    V = Q.T[:C, :N]
    return V  # rows are orthonormal


def trainable_observable_projectors(N: int, C: int, params: np.ndarray
                                      ) -> List[np.ndarray]:
    V = stack_isometry(N, C, params)
    return [np.outer(V[c].conj(), V[c]).T for c in range(C)]


def born_probabilities(psi: np.ndarray, projectors: Sequence[np.ndarray]
                        ) -> np.ndarray:
    """p_c = <psi | Pi_c | psi>."""
    out = np.empty(len(projectors), dtype=np.float64)
    for c, Pi in enumerate(projectors):
        v = Pi @ psi
        out[c] = float(np.real(psi.conj() @ v))
    return np.clip(out, 0.0, None)
