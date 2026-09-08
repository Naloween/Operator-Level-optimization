"""What a CReLU layer is, algebraically, and why it makes the nonlinear case tractable.

Context. "Why Deep Jacobian Spectra Separate" (Haas et al., ICML 2026) works in the
*fixed-gates* view: within one activation region the Jacobian is
`J = W_L D_{L-1} ... D_1 W_1` with the `D_l` constant, and its Prop 7.1 replaces the
balancing hypothesis with (depth scaling + spectral separation) to recover deep-linear-like
dynamics `s_k' = -L s_k^{2-2/L} <grad L, u_k v_k^T>`. What that setting cannot reach is the
genuinely nonlinear regime, where the gates depend on the input *and* move with the
weights.

For CReLU three exact facts change the picture. All are verified in
`tests/test_theory_crelu.py`, symbolically and against the model code.

**1. The gate is an isometry.** With `D(z) = [diag(1[z>0]) ; -diag(1[z<0])]` in
R^{2d x d}, `D(z)^T D(z) = diag(1[z>0] + 1[z<0]) = I` for every z. A ReLU gate is a
projection and destroys rank -- measured: rank J(x) falls to 1-4 of 6 at depth 8 -- whereas
a CReLU gate cannot. So *all* spectral collapse in a CReLU network is attributable to the
weights. This also removes the technical degeneracy the fixed-gates analysis has to work
around with conditioned `(r, p)`-gates: a CReLU gate is never singular, so the product
cannot be annihilated by a gate at depth.

**2. The layer is affine in the sign pattern.** Writing `W = [P | Q]`,

    M(z) = W D(z) = S + Delta . diag(sign(z)),      S = (P - Q)/2,  Delta = (P + Q)/2

exactly. Looks-linear initialization is `W = [O | -O]`, i.e. **`Delta = 0` exactly**, and
then `M(z) = O` for every z: the network is not approximately linear at initialization, it
*is* linear, and the fixed-gates picture holds with no error at all. `Delta` is therefore an
exact, per-layer, measurable coordinate for how nonlinear the layer has become.

**3. The nonlinearity is bounded uniformly over inputs.** `diag(sign(z))` is orthogonal, so
`Delta . diag(sign(z))` has the *same singular values as Delta for every input* -- only its
orientation is data-dependent. Weyl then gives

    |sigma_k(M(x)) - sigma_k(S)| <= ||Delta||_2      for all x.

Every per-sample Jacobian's layer spectrum sits within `||Delta||` of one shared linear
network's, uniformly in x. Spectral separation established for the linear part transfers to
every input with an explicit error, which is exactly what an argument in the fixed-gates
setting needs in order to survive the move to data-dependent gates.

Consequently the Jacobian expands as

    J(x) = S_L...S_1  +  sum_l A_l Delta_l diag(sign(z_l(x))) B_l  +  O(||Delta||^2)

whose zeroth order is an input-independent deep *linear* network -- the paper's setting with
`D = I`, so its theorems apply verbatim -- and whose first-order term carries the same
`A_l (.) B_l` context sandwich as the operator mismatch of Proposition 3.1, so the machinery
in `olo.diagnostics` already applies to it.

Whether this is a useful expansion is an empirical question, and the answer measured on
MNIST is yes, increasingly so with depth: after training to 93-94% test accuracy,
`||Delta||/||S||` averages 0.084 at depth 4, 0.037 at depth 32 and **0.0145 at depth 256**.
The deeper the network, the *less* each layer departs from linear.

What is not established here: that the `O(||Delta||^2)` remainder can be controlled
uniformly along a training trajectory. The expansion is exact at any instant, but
`sign(z_l(x))` is discontinuous in the weights, so a dynamical statement needs the rate of
region crossings -- which `olo.diagnostics.linearization.gate_flip_rate` measures, and which
no result here bounds.
"""
from __future__ import annotations

import torch


def split_layer(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(S, Delta)` for a CReLU weight `W = [P | Q]`: the linear part and the nonlinear one.

    `M(z) = W D(z) = S + Delta diag(sign(z))`, exactly.
    """
    if W.shape[1] % 2 != 0:
        raise ValueError(f"a CReLU layer has an even input width; got {tuple(W.shape)}")
    h = W.shape[1] // 2
    P, Q = W[:, :h], W[:, h:]
    return (P - Q) / 2, (P + Q) / 2


def nonlinearity(W: torch.Tensor) -> float:
    """`||Delta||_F / ||S||_F`: how far this layer has moved from linear. 0 at looks-linear."""
    S, D = split_layer(W.detach())
    denom = float(S.norm())
    return float(D.norm()) / denom if denom > 0 else float("inf")


def layer_profile(net) -> list[float]:
    """Per-layer nonlinearity for every gated layer of a CReLU network."""
    return [nonlinearity(W) for W in net.weights if W.shape[1] == 2 * net.width]


def spectral_bound(W: torch.Tensor) -> float:
    """`||Delta||_2`: the uniform-over-inputs bound on this layer's spectral deviation.

    Every input's layer map has singular values within this of the shared linear part's,
    because `diag(sign(z))` is orthogonal and so leaves `Delta`'s spectrum unchanged.
    """
    _, D = split_layer(W)
    return float(torch.linalg.matrix_norm(D.detach().to(torch.float64), ord=2))


def linear_part(net):
    """The input-independent deep linear network a CReLU net is a perturbation of.

    Its product is the zeroth order of the expansion, and it is exactly the object the
    fixed-gates theory is stated about.
    """
    out = []
    for W in net.weights:
        if W.shape[1] == 2 * net.width:
            out.append(split_layer(W)[0])
        else:
            out.append(W.detach())                 # first layer: no preceding gate
    return out


def linear_operator(net) -> torch.Tensor:
    """`S_L ... S_1`: the zeroth-order Jacobian, shared by every input."""
    S = linear_part(net)
    P = S[0]
    for k in range(1, len(S)):
        P = S[k] @ P
    return P
