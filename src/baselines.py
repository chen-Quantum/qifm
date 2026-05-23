"""Matched-budget classical baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def fit_logreg(X_train, y_train, X_test, y_test, seed: int = 0) -> Dict[str, float]:
    clf = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=500, random_state=int(seed)))
    clf.fit(X_train, y_train)
    return {
        "train_acc": float(clf.score(X_train, y_train)),
        "test_acc": float(clf.score(X_test, y_test)),
        "n_params": int(np.prod(clf[-1].coef_.shape) + clf[-1].intercept_.size),
    }


def fit_mlp(X_train, y_train, X_test, y_test, hidden: int = 4,
             seed: int = 0) -> Dict[str, float]:
    clf = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(int(hidden),), max_iter=400, random_state=int(seed)))
    clf.fit(X_train, y_train)
    mlp = clf[-1]
    n_params = int(sum(c.size for c in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))
    return {
        "train_acc": float(clf.score(X_train, y_train)),
        "test_acc": float(clf.score(X_test, y_test)),
        "n_params": n_params,
    }


def fit_rbf_svm(X_train, y_train, X_test, y_test, seed: int = 0) -> Dict[str, float]:
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale",
                                                 random_state=int(seed)))
    clf.fit(X_train, y_train)
    svc = clf[-1]
    n_params = int(svc.support_vectors_.size + svc.dual_coef_.size + 1)
    return {
        "train_acc": float(clf.score(X_train, y_train)),
        "test_acc": float(clf.score(X_test, y_test)),
        "n_params": n_params,
    }


def fit_rff(X_train, y_train, X_test, y_test, n_components: int = 32,
             seed: int = 0) -> Dict[str, float]:
    clf = make_pipeline(StandardScaler(),
                         RBFSampler(gamma=1.0, n_components=int(n_components),
                                     random_state=int(seed)),
                         LogisticRegression(max_iter=500, random_state=int(seed)))
    clf.fit(X_train, y_train)
    return {
        "train_acc": float(clf.score(X_train, y_train)),
        "test_acc": float(clf.score(X_test, y_test)),
        "n_params": int(np.prod(clf[-1].coef_.shape) + clf[-1].intercept_.size + 2 * n_components),
    }


REGISTRY = {
    "logreg": fit_logreg,
    "mlp": fit_mlp,
    "rbf_svm": fit_rbf_svm,
    "rff": fit_rff,
}
