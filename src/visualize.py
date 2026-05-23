"""Release media + per-experiment plots for QIFM."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

COLOR = {
    "deep_blue": "#1F4E79", "blue": "#2E75B6", "orange": "#C65911",
    "green": "#548235", "purple": "#8064A2", "grey": "#595959",
    "panel": "#0E1B2C", "panel_light": "#1F2D44", "ink": "#F5F7FA",
    "amber": "#F2BD46", "red": "#C00000",
}


# -----------------------------------------------------------------------
# Pipeline overview
# -----------------------------------------------------------------------

def pipeline_overview(out_path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(16, 4.6), dpi=170)
    ax.set_xlim(0, 16); ax.set_ylim(0, 4.6); ax.set_axis_off()
    stages = [
        ("Input $x$",        "data sample",                                COLOR["deep_blue"]),
        ("Encoder",          "$|\\psi_0(x)\\rangle$\\ amp/phase/hybrid",   COLOR["blue"]),
        ("Hamiltonian",      "$H_\\theta(x)=\\sum_m\\theta_m B_m + \\alpha V_x$", COLOR["purple"]),
        ("Evolution",        "$|\\psi(t_k)\\rangle = e^{-i t_k H} |\\psi_0\\rangle$", COLOR["orange"]),
        ("Multi-time read",  "$K$ snapshots of $|\\psi(t_k)\\rangle$",     COLOR["orange"]),
        ("Trainable obs.",   "$\\Pi_c = M_c^\\dagger M_c$",                COLOR["green"]),
        ("Class prob.",      "$p(y\\!=\\!c|x) = \\sum_k w_{kc}\\langle\\Pi_c\\rangle$", COLOR["deep_blue"]),
    ]
    n = len(stages); x0 = 0.35; gap = 0.30
    box_w = (16 - 2 * x0 - gap * (n - 1)) / n
    y_c = 2.4; box_h = 2.05
    centers = []
    for i, (title, body, c) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        ax.add_patch(FancyBboxPatch((x, y_c - box_h / 2), box_w, box_h,
                                     boxstyle="round,pad=0.04,rounding_size=0.10",
                                     facecolor=c, edgecolor="none", alpha=0.94))
        ax.text(x + box_w / 2, y_c + 0.55, title, ha="center", va="center",
                 fontsize=11.5, weight="bold", color="white", linespacing=1.0)
        ax.text(x + box_w / 2, y_c - 0.35, body, ha="center", va="center",
                 fontsize=8.5, color="white", linespacing=1.2)
        centers.append(x + box_w / 2)
    for cx0, cx1 in zip(centers[:-1], centers[1:]):
        ax.annotate("", xy=(cx1 - box_w / 2 - 0.04, y_c),
                     xytext=(cx0 + box_w / 2 + 0.04, y_c),
                     arrowprops=dict(arrowstyle="-|>,head_length=0.28,head_width=0.16",
                                      color=COLOR["grey"], lw=1.8))
    ax.text(8.0, 4.2, "QIFM pipeline overview", ha="center",
            fontsize=17, weight="bold", color=COLOR["deep_blue"])
    ax.text(8.0, 0.28,
            "Continuous-time, data-modulated Hamiltonian dynamics  *  multi-time interference readout  *  no true quantum-advantage claim",
            ha="center", fontsize=9.5, color=COLOR["grey"], style="italic")
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)


# -----------------------------------------------------------------------
# Hamiltonian heatmap
# -----------------------------------------------------------------------

def hamiltonian_heatmap(H: np.ndarray, out_path: Path | str,
                         title: str = "Learned Hamiltonian (real part)") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    im0 = axes[0].imshow(np.real(H), cmap="RdBu_r", origin="lower")
    axes[0].set_title("Re($H_\\theta$)"); fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(np.imag(H), cmap="PRGn", origin="lower")
    axes[1].set_title("Im($H_\\theta$)"); fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(title, y=1.02, color=COLOR["deep_blue"], fontsize=13)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# Interference flow GIF (wavefunction probability density per snapshot)
# -----------------------------------------------------------------------

def interference_flow_gif(snapshots: np.ndarray, out_path: Path | str,
                            fps: int = 4) -> None:
    """snapshots: (K, N) complex array. Visualises |psi_k|^2 per snapshot."""
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"imageio missing: {exc}"); return
    K, N = snapshots.shape
    p = np.abs(snapshots) ** 2
    p = p / np.maximum(p.max(axis=1, keepdims=True), 1e-12)
    frames: List[np.ndarray] = []
    for k in range(K):
        fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor="white")
        ax.bar(np.arange(N), p[k], color=COLOR["deep_blue"], alpha=0.85)
        ax.set_ylim(0, 1.05); ax.set_xlim(-1, N)
        ax.set_xlabel("basis state j"); ax.set_ylabel("|psi_k|^2 / max")
        ax.set_title(f"snapshot {k + 1} / {K}", color=COLOR["deep_blue"])
        fig.tight_layout(); fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(rgba[..., :3].copy()); plt.close(fig)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)


# -----------------------------------------------------------------------
# Decision boundary for 2-D inputs
# -----------------------------------------------------------------------

def decision_boundary(model, X: np.ndarray, y: np.ndarray, out_path: Path | str,
                       title: str = "QIFM decision boundary", res: int = 28
                       ) -> None:
    if X.shape[1] != 2:
        return
    xmin, ymin = X.min(axis=0) - 0.3
    xmax, ymax = X.max(axis=0) + 0.3
    xs = np.linspace(xmin, xmax, res); ys = np.linspace(ymin, ymax, res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.flatten(), YY.flatten()], axis=-1)
    proba = model.forward(pts)
    Z = proba[:, 1].reshape(res, res) if proba.shape[1] == 2 else proba.argmax(axis=1).reshape(res, res)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.contourf(XX, YY, Z, levels=15, cmap="RdBu_r", alpha=0.85)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="black",
                linewidth=0.4, s=22)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="p(y=1|x)")
    ax.set_title(title, color=COLOR["deep_blue"])
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# QIFM vs baselines bar plot
# -----------------------------------------------------------------------

def qifm_vs_baselines(rows: List[Dict[str, Any]], out_path: Path | str,
                       title: str = "QIFM vs baselines (mean test accuracy)") -> None:
    by_exp: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if r["method"] in ("LEARNING_SIGNAL",):
            continue
        by_exp.setdefault(r["experiment"], {})[r["method"]] = r["mean_acc"]
    exps = sorted(by_exp.keys())
    methods = sorted({m for d in by_exp.values() for m in d.keys()})
    fig, ax = plt.subplots(figsize=(11, max(3.8, 0.32 * len(exps))))
    n_methods = len(methods)
    width = 0.8 / n_methods
    y = np.arange(len(exps))
    for k, m in enumerate(methods):
        vals = [by_exp[e].get(m, 0.0) for e in exps]
        color = COLOR["deep_blue"] if m == "qifm" else COLOR["orange" if k % 2 else "green"]
        ax.barh(y + k * width - 0.4 + width / 2, vals, height=width, label=m, color=color, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(exps, fontsize=8)
    ax.set_xlim(0, 1.05); ax.set_xlabel("test accuracy")
    ax.set_title(title, color=COLOR["deep_blue"])
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# Ablation summary
# -----------------------------------------------------------------------

def ablation_summary(rows: List[Dict[str, Any]], out_path: Path | str) -> None:
    ablation_exps = ["11_qifm_phase_only", "12_qifm_amp_only", "13_qifm_hybrid_enc",
                     "14_qifm_no_hopping", "15_qifm_no_entanglement",
                     "16_qifm_with_data_potential", "17_qifm_fixed_projectors",
                     "18_qifm_trainable_observables"]
    pairs = []
    for r in rows:
        if r["experiment"] in ablation_exps and r["method"] == "qifm":
            pairs.append((r["experiment"], r["mean_acc"]))
    if not pairs:
        fig = plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No ablation rows.", ha="center", va="center")
        plt.axis("off"); plt.savefig(out_path); plt.close(fig); return
    pairs.sort()
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.32 * len(pairs))))
    y = np.arange(len(pairs))
    ax.barh(y, [p[1] for p in pairs], color=COLOR["deep_blue"], alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels([p[0] for p in pairs], fontsize=9)
    ax.set_xlim(0, 1.05); ax.set_xlabel("QIFM mean test accuracy")
    ax.set_title("Ablation summary", color=COLOR["deep_blue"])
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# Per-experiment bar plot
# -----------------------------------------------------------------------

def per_experiment_plot(exp_summary: Dict[str, Any], out_path: Path | str) -> None:
    per_seed = exp_summary["per_seed"]
    if not per_seed:
        return
    methods: Dict[str, List[float]] = {"qifm": [s["qifm_test_acc"] for s in per_seed]}
    for s in per_seed:
        for b_name, entry in s["baselines"].items():
            methods.setdefault(b_name, []).append(entry["test_acc"])
    names = list(methods.keys())
    means = [float(np.mean(methods[m])) for m in names]
    stds = [float(np.std(methods[m])) for m in names]
    colors = [COLOR["deep_blue"] if m == "qifm" else COLOR["orange"] for m in names]
    fig, ax = plt.subplots(figsize=(7, max(2.6, 0.35 * len(names))))
    y = np.arange(len(names))
    ax.barh(y, means, xerr=stds, color=colors, alpha=0.85, capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.05); ax.set_xlabel("test accuracy")
    ax.set_title(exp_summary["name"], fontsize=10)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# README hero + cinematic
# -----------------------------------------------------------------------

def readme_hero(rows: List[Dict[str, Any]], decision_path: Path | str,
                 ham_path: Path | str, out_path: Path | str) -> None:
    fig = plt.figure(figsize=(13.5, 6.0), dpi=180, facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.1],
                           hspace=0.4, wspace=0.30,
                           left=0.05, right=0.98, top=0.80, bottom=0.08)
    fig.text(0.05, 0.93,
              "QIFM: Quantum Interference Field Machine",
              fontsize=20, weight="bold", color=COLOR["deep_blue"])
    fig.text(0.05, 0.89, "A neural-network alternative based on Hamiltonian dynamics and interference",
              fontsize=11.5, color=COLOR["grey"])
    fig.text(0.05, 0.86,
              "simulator-based prototype  *  no true quantum-advantage claim",
              fontsize=9.5, color=COLOR["grey"], style="italic")
    ax1 = fig.add_subplot(gs[0, 0])
    if Path(decision_path).exists():
        ax1.imshow(np.asarray(Image.open(decision_path).convert("RGB")))
    ax1.set_axis_off(); ax1.set_title("Decision boundary", color=COLOR["deep_blue"], pad=8)
    ax2 = fig.add_subplot(gs[0, 1])
    if Path(ham_path).exists():
        ax2.imshow(np.asarray(Image.open(ham_path).convert("RGB")))
    ax2.set_axis_off(); ax2.set_title("Learned Hamiltonian", color=COLOR["deep_blue"], pad=8)
    ax3 = fig.add_subplot(gs[0, 2])
    sig = [r for r in rows if r.get("method") == "LEARNING_SIGNAL" and r.get("mean") is not None]
    sig = sorted(sig, key=lambda r: -abs(r.get("mean") or 0))[:10]
    if sig:
        names = [r["experiment"][:14] for r in sig]
        means = [r["mean"] for r in sig]
        colors = [COLOR["green"] if m > 0 else COLOR["red"] for m in means]
        y = np.arange(len(sig))
        ax3.barh(y, means, color=colors, alpha=0.85)
        ax3.set_yticks(y); ax3.set_yticklabels(names, fontsize=8.5)
        ax3.axvline(0, color="black", lw=0.7)
        ax3.set_xlabel("qifm - best_classical", fontsize=9)
        ax3.set_title("Learning signal (top 10)", color=COLOR["deep_blue"], pad=8)
    else:
        ax3.set_axis_off(); ax3.text(0.5, 0.5, "(signal pending)", ha="center")
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)


def cinematic_summary(rows: List[Dict[str, Any]],
                       hero_path: Path | str,
                       decision_path: Path | str,
                       snapshots: np.ndarray,
                       out_dir: Path,
                       fps_mp4: int = 24, fps_gif: int = 6) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"imageio missing: {exc}"); return
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "cinematic_summary.mp4"
    gif = out_dir / "cinematic_summary.gif"
    writer = imageio.get_writer(mp4, fps=fps_mp4, codec="libx264", quality=9,
                                  macro_block_size=1)
    captured: List[np.ndarray] = []; counter = 0; every = 4

    def slide():
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=COLOR["panel"])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(COLOR["panel"])
        ax.set_xlim(0, 1280); ax.set_ylim(0, 720)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return fig, ax

    def to_arr(fig):
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        return rgba[..., :3].copy()

    def emit(frame):
        nonlocal counter
        writer.append_data(frame)
        if counter % every == 0:
            captured.append(frame)
        counter += 1

    try:
        for _ in range(fps_mp4 * 3):  # title slide
            fig, ax = slide()
            ax.add_patch(FancyBboxPatch((40, 260), 18, 360,
                                          boxstyle="round,pad=0",
                                          facecolor=COLOR["amber"], edgecolor="none"))
            ax.text(110, 580, "QIFM", color=COLOR["ink"], fontsize=72, weight="bold")
            ax.text(110, 510, "Quantum Interference Field Machine",
                     color=COLOR["ink"], fontsize=26)
            ax.text(110, 470, "a neural-network alternative",
                     color=COLOR["ink"], fontsize=22)
            ax.text(110, 400, "Continuous Hamiltonian dynamics  +  multi-time interference readout",
                     color=COLOR["amber"], fontsize=18)
            ax.text(110, 350, "Simulator prototype  *  no true quantum-advantage claim",
                     color=COLOR["grey"], fontsize=16)
            emit(to_arr(fig)); plt.close(fig)

        # Decision boundary panel (4s)
        if Path(decision_path).exists():
            img = np.asarray(Image.open(decision_path).convert("RGB"))
            for _ in range(fps_mp4 * 4):
                fig, ax = slide()
                ax.add_patch(FancyBboxPatch((40, 620), 18, 80,
                                              boxstyle="round,pad=0",
                                              facecolor=COLOR["amber"], edgecolor="none"))
                ax.text(80, 660, "Step 1 - learned decision boundary",
                         color=COLOR["ink"], fontsize=24, weight="bold")
                h_im, w_im = img.shape[:2]
                target_h = 480
                ratio = target_h / h_im
                target_w = int(w_im * ratio)
                ax.imshow(img, extent=(180, 180 + target_w, 80, 80 + target_h))
                ax.text(180 + target_w + 30, 480,
                         "QIFM partitions\\nthe plane via\\ninterference of\\nthe learned\\nHamiltonian.",
                         color=COLOR["ink"], fontsize=16)
                emit(to_arr(fig)); plt.close(fig)

        # Interference / snapshot animation (5s)
        if snapshots is not None and snapshots.size:
            K, N = snapshots.shape
            for k in range(fps_mp4 * 5):
                idx = int((k / (fps_mp4 * 5 - 1)) * (K - 1)) if K > 1 else 0
                fig, ax = slide()
                ax.add_patch(FancyBboxPatch((40, 620), 18, 80,
                                              boxstyle="round,pad=0",
                                              facecolor=COLOR["amber"], edgecolor="none"))
                ax.text(80, 660,
                         "Step 2 - multi-time interference readout",
                         color=COLOR["ink"], fontsize=22, weight="bold")
                p = np.abs(snapshots[idx]) ** 2
                p = p / max(p.max(), 1e-12)
                x_left = 250; x_right = 1080
                xs = np.linspace(x_left, x_right, N)
                bar_w = (x_right - x_left) / N * 0.7
                base_y = 120
                for j, hh in enumerate(p):
                    ax.add_patch(FancyBboxPatch((xs[j] - bar_w / 2, base_y),
                                                  bar_w, hh * 350,
                                                  boxstyle="round,pad=0",
                                                  facecolor=COLOR["amber"], edgecolor="none"))
                ax.text(80, 100, f"snapshot {idx + 1} / {K}",
                         color=COLOR["grey"], fontsize=14, family="monospace")
                emit(to_arr(fig)); plt.close(fig)

        # Learning signal closing (4s)
        for _ in range(fps_mp4 * 4):
            fig, ax = slide()
            ax.add_patch(FancyBboxPatch((40, 620), 18, 80,
                                          boxstyle="round,pad=0",
                                          facecolor=COLOR["amber"], edgecolor="none"))
            ax.text(80, 660, "Step 3 - matched-budget quantum learning signal",
                     color=COLOR["ink"], fontsize=22, weight="bold")
            sig = [r for r in rows if r.get("method") == "LEARNING_SIGNAL"
                   and r.get("mean") is not None]
            sig = sorted(sig, key=lambda r: r.get("mean", 0), reverse=True)[:10]
            n = len(sig)
            x0 = 280; y0 = 100; bar_w = 600; row_h = 40
            ax.plot([x0 + bar_w / 2, x0 + bar_w / 2], [y0 - 10, y0 + n * row_h],
                     color=COLOR["grey"], lw=0.6)
            scale = max(0.05, max((abs(r.get("mean", 0)) for r in sig), default=0.1))
            for i, r in enumerate(sig):
                m = r.get("mean", 0)
                color = COLOR["green"] if m > 0 else COLOR["red"]
                bx = x0 + bar_w / 2
                length = (m / scale) * (bar_w / 2)
                length = max(min(length, bar_w / 2 - 6), -(bar_w / 2 - 6))
                yy = y0 + (n - 1 - i) * row_h
                ax.add_patch(FancyBboxPatch((min(bx, bx + length), yy),
                                              abs(length), row_h * 0.7,
                                              boxstyle="round,pad=0",
                                              facecolor=color, edgecolor="none"))
                ax.text(x0 - 10, yy + row_h * 0.35, r["experiment"][:18],
                         ha="right", color=COLOR["ink"], fontsize=10)
                ax.text(x0 + bar_w + 25, yy + row_h * 0.35, f"{m:+.3f}",
                         ha="left", color=color, fontsize=11, weight="bold")
            emit(to_arr(fig)); plt.close(fig)

        for _ in range(fps_mp4 * 3):
            fig, ax = slide()
            ax.add_patch(FancyBboxPatch((40, 280), 18, 360,
                                          boxstyle="round,pad=0",
                                          facecolor=COLOR["amber"], edgecolor="none"))
            ax.text(110, 540, "Simulator prototype.",
                     color=COLOR["ink"], fontsize=34, weight="bold")
            ax.text(110, 490, "No true quantum-advantage claim.",
                     color=COLOR["amber"], fontsize=22)
            ax.text(110, 400, "Reproduce:", color=COLOR["grey"], fontsize=16)
            ax.text(110, 365, "    python scripts/run_all_experiments.py --quick",
                     color=COLOR["ink"], fontsize=16, family="monospace")
            ax.text(110, 335, "    python scripts/build_release_media.py",
                     color=COLOR["ink"], fontsize=16, family="monospace")
            ax.text(110, 305, "    pytest -q",
                     color=COLOR["ink"], fontsize=16, family="monospace")
            emit(to_arr(fig)); plt.close(fig)
    finally:
        writer.close()
    try:
        target_w = 640
        gif_frames: List[np.ndarray] = []
        for f in captured:
            img = Image.fromarray(f)
            ratio = target_w / img.width
            img = img.resize((target_w, int(img.height * ratio)), Image.BILINEAR)
            gif_frames.append(np.array(img))
        imageio.mimsave(gif, gif_frames, fps=fps_gif, loop=0)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"GIF write failed: {exc}")
