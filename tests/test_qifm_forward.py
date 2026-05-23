"""QIFM forward pass shape, training reduces loss, end-to-end tiny experiment."""

from __future__ import annotations

import numpy as np
import pytest

from src.qifm import QIFM, QIFMConfig
from src.datasets import xor_2d, split
from src.experiments import Experiment, run_experiment


def test_forward_pass_shape() -> None:
    model = QIFM(QIFMConfig(N=8, n_times=2, max_iter=5))
    X = np.array([[0.1, -0.2], [0.5, 0.4], [-0.3, 0.6]])
    proba = model.forward(X)
    assert proba.shape == (3, 2)
    # Each row sums to 1 (softmax).
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_training_reduces_loss() -> None:
    ds = xor_2d(n=60, noise=0.03, seed=0)
    X_tr, y_tr, _, _ = split(ds, 30, 20, seed=0)
    model = QIFM(QIFMConfig(N=8, n_times=2, max_iter=20, seed=0))
    L0 = model.loss(X_tr, y_tr)
    model.fit(X_tr, y_tr)
    L1 = model.loss(X_tr, y_tr)
    assert L1 < L0


def test_tiny_experiment_end_to_end() -> None:
    exp = Experiment(
        name="t_smoke", dataset="xor_2d", dataset_kwargs={"n": 40, "noise": 0.03},
        n_train=20, n_test=12, qifm=QIFMConfig(N=8, n_times=2, max_iter=15, seed=0),
        baselines=("logreg",), seeds=(0,),
    )
    out = run_experiment(exp, seeds=(0,))
    assert "per_seed" in out and len(out["per_seed"]) == 1
    s = out["per_seed"][0]
    assert 0.0 <= s["qifm_test_acc"] <= 1.0
    assert "logreg" in s["baselines"]
