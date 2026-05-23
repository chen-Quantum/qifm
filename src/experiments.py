"""QIFM experiment registry and runner."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from . import SEED
from .baselines import REGISTRY as BASELINES
from .datasets import Dataset, get_dataset, split
from .metrics import compute_metrics, learning_signal
from .qifm import QIFM, QIFMConfig


QUICK_SEEDS = (0, 1)
FULL_SEEDS = (0, 1, 2)
DEFAULT_BASELINES = ("logreg", "mlp", "rbf_svm", "rff")


@dataclass
class Experiment:
    name: str
    dataset: str
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    n_train: int = 40
    n_test: int = 40
    qifm: QIFMConfig = field(default_factory=QIFMConfig)
    baselines: Sequence[str] = DEFAULT_BASELINES
    seeds: Sequence[int] = QUICK_SEEDS
    notes: str = ""


def run_experiment(exp: Experiment, seeds: Sequence[int]) -> Dict[str, Any]:
    per_seed: List[Dict[str, Any]] = []
    n_params_qifm = 0
    for s in seeds:
        ds = get_dataset(exp.dataset, **exp.dataset_kwargs, seed=int(s))
        X_tr, y_tr, X_te, y_te = split(ds, exp.n_train, exp.n_test, seed=int(s))
        cfg = QIFMConfig(**{**asdict(exp.qifm), "seed": int(s),
                              "n_classes": ds.n_classes})
        t0 = time.time()
        model = QIFM(cfg).fit(X_tr, y_tr)
        train_time = time.time() - t0
        proba = model.forward(X_te)
        pred = proba.argmax(axis=1)
        m = compute_metrics(y_te, pred, proba=proba)
        n_params_qifm = model.n_parameters
        seed_record: Dict[str, Any] = {
            "seed": int(s),
            "qifm_test_acc": m["accuracy"], "qifm_f1": m["f1_macro"],
            "qifm_cross_entropy": m["cross_entropy"],
            "qifm_train_time": train_time, "qifm_n_params": int(n_params_qifm),
            "baselines": {},
        }
        for b_name in exp.baselines:
            if b_name not in BASELINES:
                continue
            b = BASELINES[b_name](X_tr, y_tr, X_te, y_te, seed=int(s))
            seed_record["baselines"][b_name] = b
        per_seed.append(seed_record)
    return {
        "name": exp.name, "dataset": exp.dataset, "n_classes": ds.n_classes,
        "n_train": exp.n_train, "n_test": exp.n_test,
        "qifm_config": asdict(exp.qifm), "qifm_n_params": int(n_params_qifm),
        "notes": exp.notes, "seeds": list(seeds), "per_seed": per_seed,
    }


# -----------------------------------------------------------------------
# The 24 experiments
# -----------------------------------------------------------------------

def _qifm_default(**over) -> QIFMConfig:
    base = dict(N=8, families=("local_potential", "hopping", "graph_laplacian"),
                encoding="hybrid", n_times=3, t_max=1.4, use_data_potential=True,
                use_trainable_projectors=True, alpha=0.5, max_iter=80)
    base.update(over)
    return QIFMConfig(**base)


def all_experiments() -> List[Experiment]:
    exps: List[Experiment] = []
    # Core synthetic classification
    exps.append(Experiment("01_xor_2d_low_data", "xor_2d",
                            dict(n=80, noise=0.05),
                            n_train=20, n_test=40, qifm=_qifm_default()))
    exps.append(Experiment("02_xor_4d_parity", "parity_4d", dict(n=120),
                            n_train=40, n_test=40, qifm=_qifm_default(N=8)))
    exps.append(Experiment("03_two_moons_20", "two_moons", dict(n=80, noise=0.15),
                            n_train=20, n_test=40, qifm=_qifm_default()))
    exps.append(Experiment("04_two_moons_100", "two_moons", dict(n=160, noise=0.15),
                            n_train=80, n_test=40, qifm=_qifm_default()))
    exps.append(Experiment("05_circles", "circles", dict(n=120, noise=0.10),
                            n_train=40, n_test=40, qifm=_qifm_default()))
    exps.append(Experiment("06_spiral", "spiral", dict(n=160, classes=2),
                            n_train=80, n_test=40, qifm=_qifm_default()))
    exps.append(Experiment("07_checkerboard", "checkerboard", dict(n=160, k=3),
                            n_train=80, n_test=40, qifm=_qifm_default()))
    exps.append(Experiment("08_gaussian_mixture", "gaussian_mixture", dict(n=160, centers=4),
                            n_train=80, n_test=40, qifm=_qifm_default()))
    exps.append(Experiment("09_fourier_labels", "fourier_labels", dict(n=160, freq=4),
                            n_train=80, n_test=40, qifm=_qifm_default()))
    exps.append(Experiment("10_sparse_signal", "sparse_signal", dict(n=160, dim=8, k=2),
                            n_train=80, n_test=40, qifm=_qifm_default()))
    # Quantum-mechanism ablations
    exps.append(Experiment("11_qifm_phase_only", "two_moons", dict(n=120),
                            n_train=40, n_test=40, qifm=_qifm_default(encoding="phase"),
                            notes="Phase encoding only."))
    exps.append(Experiment("12_qifm_amp_only", "two_moons", dict(n=120),
                            n_train=40, n_test=40, qifm=_qifm_default(encoding="amplitude"),
                            notes="Amplitude encoding only."))
    exps.append(Experiment("13_qifm_hybrid_enc", "two_moons", dict(n=120),
                            n_train=40, n_test=40, qifm=_qifm_default(encoding="hybrid"),
                            notes="Hybrid amplitude-phase encoding."))
    exps.append(Experiment("14_qifm_no_hopping", "checkerboard", dict(n=160),
                            n_train=80, n_test=40,
                            qifm=_qifm_default(families=("local_potential", "graph_laplacian")),
                            notes="No hopping terms in basis."))
    exps.append(Experiment("15_qifm_no_entanglement", "two_moons", dict(n=120),
                            n_train=40, n_test=40,
                            qifm=_qifm_default(N=8,
                                                families=("local_potential", "hopping")),
                            notes="No graph Laplacian / entangling terms."))
    exps.append(Experiment("16_qifm_with_data_potential", "two_moons", dict(n=120),
                            n_train=40, n_test=40,
                            qifm=_qifm_default(use_data_potential=True),
                            notes="Data-modulated Hamiltonian."))
    exps.append(Experiment("17_qifm_fixed_projectors", "two_moons", dict(n=120),
                            n_train=40, n_test=40,
                            qifm=_qifm_default(use_trainable_projectors=False),
                            notes="Fixed basis-partition projectors."))
    exps.append(Experiment("18_qifm_trainable_observables", "two_moons", dict(n=120),
                            n_train=40, n_test=40,
                            qifm=_qifm_default(use_trainable_projectors=True),
                            notes="Trainable observable basis."))
    # Baselines and robustness
    exps.append(Experiment("19_qifm_vs_tiny_mlp", "two_moons", dict(n=120),
                            n_train=40, n_test=40,
                            qifm=_qifm_default(), baselines=("mlp",),
                            notes="QIFM vs a 4-hidden MLP."))
    exps.append(Experiment("20_qifm_vs_rbf_rff", "circles", dict(n=120),
                            n_train=40, n_test=40,
                            qifm=_qifm_default(), baselines=("rbf_svm", "rff"),
                            notes="QIFM vs RBF SVM and RFF."))
    exps.append(Experiment("21_noise_robustness", "two_moons", dict(n=120, noise=0.30),
                            n_train=40, n_test=40, qifm=_qifm_default()))
    exps.append(Experiment("22_low_data_sweep", "circles", dict(n=120),
                            n_train=16, n_test=40, qifm=_qifm_default()))
    exps.append(Experiment("23_param_count_sweep", "two_moons", dict(n=120),
                            n_train=40, n_test=40,
                            qifm=_qifm_default(N=4,
                                                families=("local_potential", "hopping")),
                            notes="N=4 small-parameter QIFM."))
    exps.append(Experiment("24_more_iterations", "spiral", dict(n=160),
                            n_train=80, n_test=40,
                            qifm=_qifm_default(max_iter=150),
                            notes="Longer training (max_iter=150)."))
    return exps


def long_table(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for exp in summary["results"]:
        per_seed = exp["per_seed"]
        if not per_seed:
            continue
        # QIFM row
        accs = [s["qifm_test_acc"] for s in per_seed]
        rows.append({
            "experiment": exp["name"], "method": "qifm",
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "n_params": int(per_seed[0]["qifm_n_params"]),
            "n": len(accs),
        })
        # Baseline rows
        baseline_accs: Dict[str, List[float]] = {}
        baseline_params: Dict[str, List[int]] = {}
        for s in per_seed:
            for b_name, entry in s["baselines"].items():
                baseline_accs.setdefault(b_name, []).append(entry["test_acc"])
                baseline_params.setdefault(b_name, []).append(entry["n_params"])
        for b_name, vals in baseline_accs.items():
            rows.append({
                "experiment": exp["name"], "method": f"classical_{b_name}",
                "mean_acc": float(np.mean(vals)),
                "std_acc": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_params": int(np.mean(baseline_params[b_name])),
                "n": len(vals),
            })
        # Signal row
        sig = learning_signal(accs, baseline_accs)
        rows.append({
            "experiment": exp["name"], "method": "LEARNING_SIGNAL",
            "best_classical": sig.get("best_classical"),
            "qifm_mean": sig.get("qifm_mean"),
            "best_classical_mean": sig.get("best_classical_mean"),
            "mean": sig.get("signal_mean"),
            "n": sig.get("n"),
        })
    return rows


def run_suite(experiments: Sequence[Experiment], seeds: Sequence[int],
               out_dir: Path,
               on_progress: Callable[[int, int, str], None] | None = None
               ) -> Dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for k, exp in enumerate(experiments):
        if on_progress:
            on_progress(k, len(experiments), exp.name)
        try:
            r = run_experiment(exp, seeds)
        except Exception as e:  # pragma: no cover
            r = {"name": exp.name, "error": str(e), "per_seed": []}
        results.append(r)
    return {"elapsed_seconds": time.time() - t0,
             "n_experiments": len(experiments),
             "seeds": list(seeds), "results": results}
