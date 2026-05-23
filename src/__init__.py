"""QIFM: Quantum Interference Field Machine.

A simulator-based ML model that replaces the layers/neurons/activations
stack of a neural network with:

    input x  ->  initial quantum state |psi_0(x)>
              ->  continuous-time evolution under a learnable, data-modulated
                  Hamiltonian H_theta(x) = sum_m theta_m B_m + alpha V_x
              ->  multi-time readout via trainable observable projectors
                  p(y = c | x) ~ sum_k w_{kc} <psi(t_k) | Pi_c | psi(t_k)>

No claim of true quantum advantage is made. We report a quantum learning
signal: matched-budget performance vs classical baselines.
"""

SEED = 0x1C3F00D  # 29618189

__all__ = [
    "encodings", "hamiltonians", "evolution", "measurements",
    "qifm", "baselines", "datasets", "metrics", "experiments",
    "visualize", "circuit_mapping",
]
