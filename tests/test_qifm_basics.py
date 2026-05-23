"""State normalisation, Hermiticity, unitarity, projector validity, forward shape."""

from __future__ import annotations

import numpy as np
import pytest

from src.encodings import data_potential, encode
from src.evolution import SpectralEvolver, evolve, is_norm_preserving
from src.hamiltonians import assemble_hamiltonian, build_basis
from src.measurements import (
    born_probabilities, fixed_partition_projectors,
    trainable_observable_projectors,
)


N = 8


def _random_theta(P: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(P) * 0.3


def test_state_normalisation() -> None:
    psi = encode(np.array([0.5, -0.3]), kind="hybrid", N=N, seed=0)
    assert pytest.approx(1.0, abs=1e-9) == float(np.linalg.norm(psi))


def test_hamiltonian_hermiticity() -> None:
    basis, _ = build_basis(N, families=("local_potential", "hopping", "graph_laplacian"))
    theta = _random_theta(len(basis), seed=1)
    V_x = data_potential(np.array([0.4, -0.2]), N=N, seed=0)
    H = assemble_hamiltonian(theta, basis, V_x=V_x, alpha=0.5)
    assert np.allclose(H, H.conj().T, atol=1e-9)


def test_evolution_preserves_norm() -> None:
    basis, _ = build_basis(N, families=("local_potential", "hopping"))
    theta = _random_theta(len(basis), seed=2)
    H = assemble_hamiltonian(theta, basis)
    psi0 = encode(np.array([0.2, 0.7]), kind="hybrid", N=N, seed=0)
    psiT = evolve(H, psi0, t=1.3)
    assert is_norm_preserving(psi0, psiT)


def test_spectral_evolver_matches_expm() -> None:
    basis, _ = build_basis(N, families=("local_potential", "hopping"))
    theta = _random_theta(len(basis), seed=3)
    H = assemble_hamiltonian(theta, basis)
    psi0 = encode(np.array([0.1, -0.5]), kind="hybrid", N=N, seed=0)
    a = evolve(H, psi0, t=1.0)
    b = SpectralEvolver.from_H(H).evolve(psi0, t=1.0)
    assert np.allclose(a, b, atol=1e-7)


def test_fixed_projectors_psd_and_sum_to_identity() -> None:
    Ps = fixed_partition_projectors(N, 2)
    total = sum(Ps)
    assert np.allclose(total, np.eye(N), atol=1e-12)
    for P in Ps:
        w = np.linalg.eigvalsh(P)
        assert w.min() >= -1e-12


def test_trainable_projectors_psd() -> None:
    rng = np.random.default_rng(7)
    C = 2
    params = rng.standard_normal(2 * C * N)
    Ps = trainable_observable_projectors(N, C, params)
    for P in Ps:
        w = np.linalg.eigvalsh(0.5 * (P + P.conj().T))
        assert w.min() >= -1e-9


def test_born_probabilities_sum_to_one_on_partition() -> None:
    psi = encode(np.array([0.3, -0.4]), kind="hybrid", N=N, seed=0)
    Ps = fixed_partition_projectors(N, 2)
    p = born_probabilities(psi, Ps)
    assert pytest.approx(1.0, abs=1e-9) == float(p.sum())
