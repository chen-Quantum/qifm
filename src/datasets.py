"""Synthetic + small datasets for the QIFM benchmark suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

from . import SEED


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    n_classes: int


def xor_2d(n: int = 100, noise: float = 0.05, seed: int = SEED) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)
    x = x + noise * rng.standard_normal(x.shape)
    return Dataset("xor_2d", x.astype(np.float32), y, 2)


def parity_4d(n: int = 200, seed: int = SEED) -> Dataset:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n, 4)).astype(np.float32)
    y = (bits.sum(axis=1).astype(int) % 2)
    return Dataset("parity_4d", bits, y, 2)


def two_moons(n: int = 100, noise: float = 0.15, seed: int = SEED) -> Dataset:
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return Dataset("two_moons", X.astype(np.float32), y.astype(int), 2)


def circles(n: int = 100, noise: float = 0.10, seed: int = SEED) -> Dataset:
    X, y = make_circles(n_samples=n, noise=noise, factor=0.4, random_state=seed)
    return Dataset("circles", X.astype(np.float32), y.astype(int), 2)


def spiral(n: int = 200, classes: int = 2, noise: float = 0.10, seed: int = SEED) -> Dataset:
    rng = np.random.default_rng(seed)
    n_per = n // classes
    X = np.zeros((classes * n_per, 2), dtype=np.float32)
    y = np.zeros(classes * n_per, dtype=int)
    for c in range(classes):
        t = np.linspace(0.0, 2.5, n_per) + 0.05 * rng.standard_normal(n_per)
        theta = t * np.pi + c * np.pi
        r = t
        X[c * n_per:(c + 1) * n_per, 0] = r * np.cos(theta)
        X[c * n_per:(c + 1) * n_per, 1] = r * np.sin(theta)
        y[c * n_per:(c + 1) * n_per] = c
    X = X + noise * rng.standard_normal(X.shape)
    return Dataset("spiral", X.astype(np.float32), y, classes)


def checkerboard(n: int = 200, k: int = 3, seed: int = SEED) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    cells = ((x * k).astype(int).sum(axis=1) % 2)
    return Dataset("checkerboard", x.astype(np.float32), cells.astype(int), 2)


def gaussian_mixture(n: int = 200, centers: int = 4, std: float = 0.30,
                     seed: int = SEED) -> Dataset:
    X, y = make_blobs(n_samples=n, centers=centers, cluster_std=std,
                       random_state=seed)
    # Convert to binary by grouping centers into 2 classes.
    y = (y % 2)
    return Dataset("gaussian_mixture", X.astype(np.float32), y.astype(int), 2)


def fourier_labels(n: int = 200, freq: int = 4, seed: int = SEED) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    score = np.sin(freq * np.pi * x[:, 0]) + np.cos(freq * np.pi * x[:, 1])
    y = (score > 0).astype(int)
    return Dataset("fourier_labels", x.astype(np.float32), y, 2)


def rotated_blobs(n: int = 200, angle: float = np.pi / 6, std: float = 0.30,
                  seed: int = SEED) -> Dataset:
    rng = np.random.default_rng(seed)
    X = np.concatenate([
        rng.standard_normal((n // 2, 2)) * std + np.array([1.0, 0.0]),
        rng.standard_normal((n // 2, 2)) * std + np.array([-1.0, 0.0]),
    ], axis=0)
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    X = X @ rot
    return Dataset("rotated_blobs", X.astype(np.float32), y, 2)


def sparse_signal(n: int = 200, dim: int = 8, k: int = 2,
                  seed: int = SEED) -> Dataset:
    """Inputs are sparse binary vectors; label depends on which two indices
    are active."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, dim), dtype=np.float32)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        idx = rng.choice(dim, size=k, replace=False)
        X[i, idx] = 1.0
        y[i] = int((idx.sum() % 2) == 0)
    return Dataset("sparse_signal", X, y, 2)


def split(ds: Dataset, n_train: int, n_test: int, seed: int) -> Tuple[np.ndarray,
                                                                       np.ndarray,
                                                                       np.ndarray,
                                                                       np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds.X))
    n_train = min(n_train, len(ds.X) - 1)
    n_test = min(n_test, len(ds.X) - n_train)
    tr = idx[:n_train]; te = idx[n_train:n_train + n_test]
    return ds.X[tr], ds.y[tr], ds.X[te], ds.y[te]


REGISTRY = {
    "xor_2d": xor_2d, "parity_4d": parity_4d, "two_moons": two_moons,
    "circles": circles, "spiral": spiral, "checkerboard": checkerboard,
    "gaussian_mixture": gaussian_mixture, "fourier_labels": fourier_labels,
    "rotated_blobs": rotated_blobs, "sparse_signal": sparse_signal,
}


def get_dataset(name: str, **kwargs) -> Dataset:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}")
    return REGISTRY[name](**kwargs)
