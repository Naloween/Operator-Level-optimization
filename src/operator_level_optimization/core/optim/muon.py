"""
Muon: Orthogonalized Gradient Update.

Applies Newton-Schulz iteration to compute the approximate polar factor of the
Nesterov-momentum-updated gradient, then uses that as the update direction.

For a 2D weight matrix G, the polar factor U = NS(G) satisfies U ≈ Q where
G = Q S is the polar decomposition (Q orthogonal/semi-unitary, S PSD).
This removes the spectral distortion introduced by the gradient's singular values
and is equivalent to steepest descent in the spectral norm ball.

For 1D parameters (biases), we fall back to normalized gradient descent.
"""

import torch
from torch.optim.optimizer import Optimizer, required


def _newton_schulz(G: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    """Compute the approximate polar factor of G via Newton-Schulz iteration.

    Uses the quintic update:
        X <- a X + (b A + c A²) X,   A = X Xᵀ
    which converges cubically to the left polar factor of G (orthogonal factor
    in the polar decomposition G = Q S).

    For tall matrices (rows > cols), we transpose, run NS, then transpose back
    to always operate on a wide or square matrix.

    Args:
        G:     2D gradient tensor.
        steps: Number of NS iterations (5 is typically sufficient).
        eps:   Numerical floor added to the Frobenius norm before scaling.

    Returns:
        Approximate polar factor of G, same shape as G.
    """
    assert G.ndim == 2, f"NS requires 2D tensor, got shape {G.shape}"

    # Quintic NS coefficients (from Muon / gradient-release literature)
    a, b, c = 1.875, -1.25, 0.375

    # Scale so singular values start near 1
    X = G / (G.norm() + eps)

    # NS works best on wide matrices; transpose tall ones
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T  # now shape (cols, rows)

    for _ in range(steps):
        A = X @ X.T                          # A = X Xᵀ  (square, cols×cols)
        X = a * X + (b * A + c * (A @ A)) @ X

    if transposed:
        X = X.T

    return X


class Muon(Optimizer):
    r"""Muon optimizer: Nesterov momentum + Newton-Schulz orthogonalization.

    For each 2D parameter W:
      1. Maintain Nesterov momentum buffer m_t = β m_{t-1} + g_t.
      2. Form the Nesterov step  g_nes = g_t + β m_t.
      3. Compute U = NS(g_nes): the approximate polar factor.
      4. Update: W ← W - lr · U.

    For 1D parameters (biases), the polar factor of a vector is just the
    normalized vector, so we simply use g_nes / ‖g_nes‖₂.

    Arguments:
        params:           Parameters to optimize.
        lr (float):       Learning rate.
        momentum (float): Nesterov momentum coefficient β (default: 0.95).
        ns_iterations (int): Newton-Schulz iterations (default: 5).
        eps (float):      Numerical floor for scaling (default: 1e-8).
    """

    def __init__(
        self,
        params,
        lr: float = required,
        momentum: float = 0.95,
        ns_iterations: int = 5,
        eps: float = 1e-8,
        **kwargs,
    ):
        if lr is required:
            raise ValueError("lr is required")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(lr=lr, momentum=momentum, ns_iterations=ns_iterations, eps=eps)
        # Accept unknown kwargs for compatibility with sweep system
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_iterations"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Initialise momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]

                # Nesterov momentum:  m_t = β m_{t-1} + g_t
                buf.mul_(momentum).add_(g)
                # Nesterov step:  g_nes = g_t + β m_t
                g_nes = g.add(buf, alpha=momentum)

                if p.ndim == 2:
                    # Matrix: compute approximate polar factor via NS
                    update = _newton_schulz(g_nes, ns_steps, eps)
                else:
                    # Vector / scalar: polar factor is just the unit vector
                    norm = g_nes.norm()
                    update = g_nes / (norm + eps) if norm > eps else g_nes

                p.add_(update, alpha=-lr)

        return loss
