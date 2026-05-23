"""Metrics + quantum_learning_signal."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    proba: np.ndarray | None = None) -> Dict[str, float]:
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if proba is not None:
        out["cross_entropy"] = float(log_loss(y_true, np.clip(proba, 1e-12, 1.0),
                                                labels=list(range(proba.shape[1]))))
    return out


def calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    p = proba[np.arange(y_true.size), 1] if proba.shape[1] == 2 else proba.max(axis=1)
    correct = (y_true == proba.argmax(axis=1)).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    for k in range(n_bins):
        m = (p >= edges[k]) & (p < edges[k + 1])
        if m.any():
            err += abs(p[m].mean() - correct[m].mean()) * float(m.mean())
    return float(err)


def learning_signal(qifm_metric: Sequence[float],
                     classical_by_method: Dict[str, Sequence[float]]
                     ) -> Dict[str, Any]:
    qifm = list(qifm_metric)
    if not classical_by_method:
        return {"signal_mean": float("nan"), "best_classical": None,
                "qifm_mean": float(np.mean(qifm)) if qifm else float("nan"),
                "best_classical_mean": float("nan"), "n": len(qifm)}
    method_means = {m: float(np.mean(v)) for m, v in classical_by_method.items()}
    best_method = max(method_means, key=method_means.get)
    best = list(classical_by_method[best_method])
    n = min(len(best), len(qifm))
    if n == 0:
        return {"signal_mean": float("nan"), "best_classical": best_method,
                "qifm_mean": float(np.mean(qifm)) if qifm else float("nan"),
                "best_classical_mean": method_means[best_method], "n": 0}
    diffs = np.asarray(qifm[:n]) - np.asarray(best[:n])
    return {
        "signal_mean": float(diffs.mean()),
        "signal_std": float(diffs.std(ddof=1)) if n > 1 else 0.0,
        "best_classical": best_method,
        "qifm_mean": float(np.mean(qifm)),
        "best_classical_mean": method_means[best_method],
        "n": int(n),
    }
