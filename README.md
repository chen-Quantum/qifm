# QIFM: Quantum Interference Field Machine

A simulator-based research prototype of a **continuous-time, complex-state
classifier** inspired by quantum dynamics. Instead of stacked affine layers and
pointwise nonlinearities, an input is encoded into a complex state that evolves
under a small trainable generator; class scores are read out from several
snapshots in time. It is benchmarked against classical baselines at matched data
and parameter budgets.

<p align="center">
  <img src="outputs/public_readme/hero.png" width="760">
</p>

> Simulator-based research prototype. No quantum-advantage claim.

## Core idea

- Encode an input $x$ into an initial complex state.
- Evolve it continuously in time under a small, **trainable generator**.
- Read out class scores from a few **time snapshots** via trainable projectors.
- Train all continuous parameters end-to-end by cross-entropy, and compare with
  classical baselines (logistic regression, a tiny MLP, RBF SVM, random Fourier
  features) at the same data and parameter budget.

## Mathematical sketch

Encode the input, then evolve the state continuously under a trainable generator
$H_\theta$:

```math
|\psi_0(x)\rangle = E_\theta(x),
\qquad
|\psi(t)\rangle = e^{-\,i\,t\,H_\theta}\,|\psi_0(x)\rangle.
```

Combine $K$ snapshot times $\{t_1,\dots,t_K\}$ into class scores and normalize:

```math
s_c(x) = \sum_{k=1}^{K} w_{k,c}\,
\langle \psi(t_k)\,|\,\Pi_c\,|\,\psi(t_k)\rangle,
\qquad
p(y=c\,|\,x) = \operatorname{softmax}_c\, s_c(x).
```

The forward map is norm-preserving (the evolution is unitary), so the model's
nonlinearity comes from the multi-snapshot read-out rather than from pointwise
activation functions.

## Selected visuals and results

<p align="center">
  <img src="outputs/public_readme/selected_result.png" width="520">
</p>

*Exploratory matched-budget comparison on toy classification tasks. Each point is
one task; the dashed line is parity. Results are mixed.*

<p align="center">
  <img src="outputs/public_readme/gallery.png" width="760">
</p>

*Selected toy datasets used for the exploration.*

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_all_experiments.py --quick
pytest -q
```

## Results status

- Exploratory, simulator-based, small scale (small state dimension).
- Results are **mixed** across the toy tasks; classical baselines match or exceed
  the prototype on a number of them.
- No quantum advantage is claimed. No state-of-the-art claim is made.

## Limitations

- NumPy / SciPy simulator only; small state dimension — classically tractable.
- Tiny optimizer (L-BFGS-B with finite-difference gradients); limited scaling.
- Synthetic, low-dimensional datasets.
- No hardware noise model; shot noise is ignored.
