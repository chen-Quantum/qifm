"""QIFM model: forward pass, loss, training, prediction.

Forward pass on a single input x:

    psi_0 = encode(x)
    H     = sum_m theta_m B_m + alpha V(x)
    psi_k = exp(-i t_k H) psi_0,                k = 1..K
    p_c   = softmax_c( sum_k w_{kc} <psi_k | Pi_c | psi_k> )

Continuous learnable parameters:

    theta       : len(basis)            Hamiltonian coefficients
    w           : K x C                 multi-time readout weights
    proj_params : 2 * C * N             trainable observable matrix (if used)
    alpha       : scalar                strength of the data-dependent potential
    log_scales  : K                     soft positivity-preserving time scales

Training is done with scipy.optimize.minimize (L-BFGS-B) over the cross-
entropy loss with finite-difference gradients. Small parameter counts make
this tractable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from . import SEED
from .encodings import data_potential, encode_batch
from .evolution import SpectralEvolver
from .hamiltonians import assemble_hamiltonian, build_basis
from .measurements import (
    born_probabilities, class_subspace_projectors, fixed_partition_projectors,
    trainable_observable_projectors,
)


@dataclass
class QIFMConfig:
    N: int = 8                                          # Hilbert-space dimension
    families: Sequence[str] = ("local_potential", "hopping", "graph_laplacian")
    encoding: str = "hybrid"                            # amplitude / phase / hybrid
    n_times: int = 3                                    # multi-time snapshots
    t_max: float = 1.5
    use_data_potential: bool = True
    use_trainable_projectors: bool = True
    alpha: float = 0.6
    projector_rank: int = 2
    n_classes: int = 2
    seed: int = SEED
    max_iter: int = 80
    use_finite_diff: bool = True


@dataclass
class QIFM:
    config: QIFMConfig
    basis: List[np.ndarray] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    theta: Optional[np.ndarray] = None
    w: Optional[np.ndarray] = None
    proj_params: Optional[np.ndarray] = None
    alpha_param: float = 0.6
    times: Optional[np.ndarray] = None
    history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.basis, self.labels = build_basis(self.config.N, self.config.families)
        self.times = np.linspace(self.config.t_max / self.config.n_times,
                                  self.config.t_max, self.config.n_times)
        rng = np.random.default_rng(self.config.seed)
        self.theta = rng.standard_normal(len(self.basis)) * 0.3
        self.w = rng.standard_normal((self.config.n_times, self.config.n_classes)) * 0.3
        if self.config.use_trainable_projectors:
            self.proj_params = rng.standard_normal(2 * self.config.n_classes * self.config.N) * 0.3
        else:
            self.proj_params = None
        self.alpha_param = float(self.config.alpha)

    # -----------------------------------------------------------------
    # Parameter packing
    # -----------------------------------------------------------------

    def pack(self) -> np.ndarray:
        parts = [self.theta.flatten(), self.w.flatten(),
                  np.asarray([self.alpha_param], dtype=np.float64)]
        if self.proj_params is not None:
            parts.append(self.proj_params.flatten())
        return np.concatenate(parts)

    def unpack(self, vec: np.ndarray) -> None:
        n_theta = len(self.basis)
        n_w = self.config.n_times * self.config.n_classes
        cursor = 0
        self.theta = vec[cursor:cursor + n_theta]; cursor += n_theta
        self.w = vec[cursor:cursor + n_w].reshape(self.config.n_times,
                                                    self.config.n_classes)
        cursor += n_w
        self.alpha_param = float(vec[cursor]); cursor += 1
        if self.proj_params is not None:
            n_p = 2 * self.config.n_classes * self.config.N
            self.proj_params = vec[cursor:cursor + n_p]; cursor += n_p

    @property
    def n_parameters(self) -> int:
        return self.pack().size

    # -----------------------------------------------------------------
    # Forward / loss
    # -----------------------------------------------------------------

    def _projectors(self) -> List[np.ndarray]:
        if self.config.use_trainable_projectors and self.proj_params is not None:
            return trainable_observable_projectors(self.config.N, self.config.n_classes,
                                                     self.proj_params)
        return fixed_partition_projectors(self.config.N, self.config.n_classes)

    def forward_one(self, x: np.ndarray, projectors: List[np.ndarray] | None = None
                     ) -> np.ndarray:
        if projectors is None:
            projectors = self._projectors()
        psi0 = encode_batch(np.asarray(x).reshape(1, -1), kind=self.config.encoding,
                             N=self.config.N, seed=self.config.seed)[0]
        V_x = data_potential(x, N=self.config.N,
                              seed=self.config.seed) if self.config.use_data_potential else None
        H = assemble_hamiltonian(self.theta, self.basis, V_x=V_x, alpha=self.alpha_param)
        evolver = SpectralEvolver.from_H(H)
        snaps = evolver.snapshots(psi0, self.times)
        # Per-snapshot Born probabilities (K, C) -> weighted sum (C,)
        scores = np.zeros(self.config.n_classes, dtype=np.float64)
        for k, psi_k in enumerate(snaps):
            p = born_probabilities(psi_k, projectors)
            scores = scores + self.w[k] * p
        # Softmax for stability and class normalisation
        scores = scores - scores.max()
        e = np.exp(scores)
        return e / max(float(e.sum()), 1e-12)

    def forward(self, X: np.ndarray) -> np.ndarray:
        projectors = self._projectors()
        return np.stack([self.forward_one(X[i], projectors) for i in range(X.shape[0])],
                         axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        proba = self.forward(X)
        idx = np.arange(y.size)
        eps = 1e-9
        return float(-np.mean(np.log(np.clip(proba[idx, y], eps, 1.0))))

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> "QIFM":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.history = []

        def loss_fn(vec: np.ndarray) -> float:
            self.unpack(vec)
            L = self.loss(X, y)
            self.history.append(float(L))
            return L

        x0 = self.pack()
        opt = {
            "maxiter": int(self.config.max_iter),
            "disp": bool(verbose),
        }
        if self.config.use_finite_diff:
            # scipy's L-BFGS-B with default finite-difference gradients converges in far
            # fewer forward calls than Powell for parameter counts up to ~80.
            method = "L-BFGS-B"
            opt = dict(opt, maxfun=int(50 * x0.size))
        else:
            method = "L-BFGS-B"
        result = minimize(loss_fn, x0, method=method, options=opt)
        self.unpack(result.x)
        return self
