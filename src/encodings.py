"""Classical-to-quantum input encodings.

Three encoders, all producing a normalised complex vector of length N = 2**n
qubits (or any chosen dimension):

    amplitude_encoding(x)  - pads / truncates x to length N, normalises.
    phase_encoding(x)      - 1/sqrt(N) sum_j exp(i pi (x . r_j)) |j> with a
                              fixed random projection r_j.
    hybrid_encoding(x)     - amplitude * exp(i pi (x . r_j)) per coordinate.

Each returns a (N,) complex array with unit L2 norm.
"""

from __future__ import annotations

import numpy as np

from . import SEED


def _pad_or_truncate(x: np.ndarray, N: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).flatten()
    if x.size == N:
        return x
    if x.size > N:
        return x[:N]
    out = np.zeros(N, dtype=np.float64)
    out[: x.size] = x
    return out


def amplitude_encoding(x: np.ndarray, N: int = 8) -> np.ndarray:
    v = _pad_or_truncate(x, N).astype(np.complex128)
    # Lift to strictly positive amplitudes via a soft offset, so all-zero
    # inputs do not produce undefined states.
    v = v + 1e-3
    nrm = float(np.linalg.norm(v))
    return v / max(nrm, 1e-12)


def phase_encoding(x: np.ndarray, N: int = 8, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = x.size
    R = rng.standard_normal((N, d))
    R = R / max(np.linalg.norm(R), 1e-12)
    phases = np.pi * (R @ np.asarray(x, dtype=np.float64).flatten())
    psi = np.exp(1j * phases) / np.sqrt(N)
    return psi


def hybrid_encoding(x: np.ndarray, N: int = 8, seed: int = SEED) -> np.ndarray:
    """Amplitude * phase. Amplitudes from amplitude_encoding; phases from
    phase_encoding."""
    amp = np.abs(amplitude_encoding(x, N))
    phs = np.angle(phase_encoding(x, N, seed=seed))
    psi = amp * np.exp(1j * phs)
    nrm = float(np.linalg.norm(psi))
    return psi / max(nrm, 1e-12)


def encode(x: np.ndarray, kind: str = "hybrid", N: int = 8,
            seed: int = SEED) -> np.ndarray:
    if kind == "amplitude":
        return amplitude_encoding(x, N=N)
    if kind == "phase":
        return phase_encoding(x, N=N, seed=seed)
    if kind == "hybrid":
        return hybrid_encoding(x, N=N, seed=seed)
    raise ValueError(f"Unknown encoding {kind!r}")


def encode_batch(X: np.ndarray, kind: str = "hybrid", N: int = 8,
                  seed: int = SEED) -> np.ndarray:
    """Encode a batch of shape (B, d) into (B, N) complex states."""
    out = np.empty((X.shape[0], N), dtype=np.complex128)
    for i in range(X.shape[0]):
        out[i] = encode(X[i], kind=kind, N=N, seed=seed)
    return out


def data_potential(x: np.ndarray, N: int = 8, seed: int = SEED) -> np.ndarray:
    """Build a diagonal Hermitian "potential" V_x of shape (N, N) that
    depends on the data point x.

    V_x = diag(R x) for a fixed random orthonormal projection R in R^{N x d}.
    """
    rng = np.random.default_rng(seed + 1)
    d = np.asarray(x).size
    R = rng.standard_normal((N, d))
    q, _ = np.linalg.qr(R)
    diag = q[:N, :d] @ np.asarray(x, dtype=np.float64).flatten()
    V = np.zeros((N, N), dtype=np.complex128)
    V[np.arange(N), np.arange(N)] = diag
    return V
