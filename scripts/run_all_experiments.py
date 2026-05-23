"""Run the full QIFM experiment suite."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "outputs" / "experiments"


def _write(summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "all_metrics.json").open("w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o:
                  float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    flat: List[Dict[str, Any]] = []
    for r in summary["results"]:
        if r.get("error"):
            flat.append({"experiment": r["name"], "method": "ERROR", "value": r["error"]})
            continue
        for s in r["per_seed"]:
            flat.append({
                "experiment": r["name"], "dataset": r["dataset"],
                "n_train": r["n_train"], "n_test": r["n_test"],
                "seed": s["seed"], "method": "qifm",
                "test_acc": s["qifm_test_acc"], "f1": s["qifm_f1"],
                "cross_entropy": s["qifm_cross_entropy"],
                "train_time": s["qifm_train_time"], "n_params": s["qifm_n_params"],
            })
            for b_name, entry in s["baselines"].items():
                flat.append({
                    "experiment": r["name"], "dataset": r["dataset"],
                    "n_train": r["n_train"], "n_test": r["n_test"],
                    "seed": s["seed"], "method": b_name,
                    "test_acc": entry["test_acc"], "n_params": entry["n_params"],
                })
    if flat:
        keys = list({k for row in flat for k in row.keys()})
        with (OUT_DIR / "all_metrics.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in flat:
                w.writerow(row)
    with (OUT_DIR / "bootstrap_summary.csv").open("w", newline="") as fh:
        keys = ["experiment", "method", "best_classical", "qifm_mean",
                 "best_classical_mean", "mean_acc", "std_acc", "mean", "n_params", "n"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)
    from src.experiments import QUICK_SEEDS, FULL_SEEDS, all_experiments, long_table, run_suite
    from src.visualize import per_experiment_plot
    seeds = QUICK_SEEDS if args.quick else FULL_SEEDS
    exps = all_experiments()
    if args.quick:
        for e in exps:
            e.qifm.max_iter = min(e.qifm.max_iter, 60)
            if e.qifm.N > 4:
                e.qifm.N = 4
            e.qifm.families = tuple(f for f in e.qifm.families
                                     if f in ("local_potential", "hopping", "graph_laplacian"))
            if not e.qifm.families:
                e.qifm.families = ("local_potential", "hopping")
            e.n_train = min(e.n_train, 32)
            e.n_test = min(e.n_test, 32)
    print(f"[QIFM] Running {len(exps)} experiments x {len(seeds)} seeds "
          f"({'quick' if args.quick else 'full'} mode)")
    t0 = time.time()

    def progress(k: int, total: int, name: str) -> None:
        dt = time.time() - t0
        print(f"  [{k + 1:>2}/{total}]  {name:34s}  elapsed {dt:6.1f}s")

    summary = run_suite(exps, seeds, OUT_DIR, on_progress=progress)
    rows = long_table(summary)
    _write(summary, rows)
    plots_dir = OUT_DIR / "per_experiment_plots"; plots_dir.mkdir(exist_ok=True)
    for exp in summary["results"]:
        if exp.get("per_seed"):
            per_experiment_plot(exp, plots_dir / f"{exp['name']}.png")
    print(f"\n[QIFM] finished in {summary['elapsed_seconds']:.1f}s")
    print(f"[QIFM] outputs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
