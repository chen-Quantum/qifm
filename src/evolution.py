"""Unitary evolution |psi_T> = exp(-i T H) |psi_0>.

For small Hilbert-space dimension (we work with N <= 32) we use
scipy.linalg.expm directly. When the same H is reused across many
time-snapshots, we cache the eigendecomposition and propagate via the
diagonal exponential of eigenvalues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.linalg import expm


def evolve(H: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    U = expm(-1j * float(t) * H)
    return U @ psi0


@dataclass
class SpectralEvolver:
    """Cache the eigendecomposition of H. Apply exp(-i t H) in O(N^2)."""
    H: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    eigvecs_dag: np.ndarray

    @classmethod
    def from_H(cls, H: np.ndarray) -> "SpectralEvolver":
        # H is Hermitian by construction; use eigh.
        w, V = np.linalg.eigh(0.5 * (H + H.conj().T))
        return cls(H=H, eigvals=w, eigvecs=V, eigvecs_dag=V.conj().T)

    def evolve(self, psi0: np.ndarray, t: float) -> np.ndarray:
        coefs = self.eigvecs_dag @ psi0
        coefs = coefs * np.exp(-1j * float(t) * self.eigvals)
        return self.eigvecs @ coefs

    def snapshots(self, psi0: np.ndarray, times: Sequence[float]) -> np.ndarray:
        """Return an array of shape (K, N) with the evolved state at each t_k."""
        coefs0 = self.eigvecs_dag @ psi0
        out = np.empty((len(times), psi0.size), dtype=np.complex128)
        for k, t in enumerate(times):
            out[k] = self.eigvecs @ (coefs0 * np.exp(-1j * float(t) * self.eigvals))
        return out


def is_norm_preserving(psi0: np.ndarray, psiT: np.ndarray, atol: float = 1e-9) -> bool:
    return abs(float(np.linalg.norm(psiT)) - float(np.linalg.norm(psi0))) < atol
