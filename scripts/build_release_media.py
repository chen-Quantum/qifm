"""Build all release media into outputs/release_media/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_EXP = ROOT / "outputs" / "experiments"
OUT_REL = ROOT / "outputs" / "release_media"


def _ensure_metrics() -> dict:
    p = OUT_EXP / "all_metrics.json"
    if not p.exists():
        from scripts.run_all_experiments import main as run_main
        run_main(["--quick"])
    with p.open() as fh:
        return json.load(fh)


def main() -> int:
    OUT_REL.mkdir(parents=True, exist_ok=True)
    summary = _ensure_metrics()
    from src.experiments import long_table
    from src.datasets import get_dataset, split
    from src.qifm import QIFM, QIFMConfig
    from src.hamiltonians import assemble_hamiltonian, build_basis
    from src.encodings import data_potential
    from src.evolution import SpectralEvolver
    from src.visualize import (
        ablation_summary, cinematic_summary, decision_boundary,
        hamiltonian_heatmap, interference_flow_gif,
        pipeline_overview, qifm_vs_baselines, readme_hero,
    )

    rows = long_table(summary)

    print("[QIFM media] 1/7  pipeline overview ...")
    pipeline_overview(OUT_REL / "pipeline_overview.png")

    # Train one QIFM on two_moons for the visuals.
    print("[QIFM media] 2/7  training a visual QIFM on two_moons ...")
    ds = get_dataset("two_moons", n=120, noise=0.15, seed=0)
    X_tr, y_tr, X_te, y_te = split(ds, 40, 40, seed=0)
    cfg = QIFMConfig(N=8, families=("local_potential", "hopping", "graph_laplacian"),
                      encoding="hybrid", n_times=3, t_max=1.4,
                      use_data_potential=True, use_trainable_projectors=True,
                      max_iter=80, seed=0)
    model = QIFM(cfg).fit(X_tr, y_tr)

    print("[QIFM media] 3/7  Hamiltonian heatmap ...")
    V_x = data_potential(X_tr[0], N=cfg.N, seed=cfg.seed)
    H = assemble_hamiltonian(model.theta, model.basis, V_x=V_x, alpha=model.alpha_param)
    hamiltonian_heatmap(H, OUT_REL / "hamiltonian_heatmap.png")

    print("[QIFM media] 4/7  decision boundary ...")
    decision_boundary(model, X_tr, y_tr, OUT_REL / "decision_boundary_qifm.png",
                       title="QIFM decision boundary on two-moons")

    print("[QIFM media] 5/7  interference flow GIF ...")
    from src.encodings import encode
    psi0 = encode(X_tr[0], kind=cfg.encoding, N=cfg.N, seed=cfg.seed)
    evolver = SpectralEvolver.from_H(H)
    snapshots = evolver.snapshots(psi0, np.linspace(0.0, cfg.t_max, 12))
    interference_flow_gif(snapshots, OUT_REL / "interference_flow.gif", fps=3)

    print("[QIFM media] 6/7  QIFM vs baselines + ablation summary ...")
    qifm_vs_baselines(rows, OUT_REL / "qifm_vs_baselines.png")
    ablation_summary(rows, OUT_REL / "ablation_summary.png")

    print("[QIFM media] 7/7  README hero + cinematic ...")
    readme_hero(rows,
                  OUT_REL / "decision_boundary_qifm.png",
                  OUT_REL / "hamiltonian_heatmap.png",
                  OUT_REL / "readme_hero.png")
    cinematic_summary(rows,
                       OUT_REL / "readme_hero.png",
                       OUT_REL / "decision_boundary_qifm.png",
                       snapshots, OUT_REL)

    print("\n[QIFM media] Done.  Files:")
    for p in sorted(OUT_REL.iterdir()):
        if p.is_file():
            print(f"  {p.name:42s}  {p.stat().st_size:>10d} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
