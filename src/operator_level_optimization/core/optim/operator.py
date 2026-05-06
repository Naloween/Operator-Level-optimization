"""
Operator-level optimizer for factored weight matrices in MLPs.

For an L-layer MLP with weights W_0, ..., W_{L-1}, gradient descent on the
parameters does NOT correspond to gradient descent on the effective linear
operator P(x) = W_{L-1} D_{L-2}(x) ... D_0(x) W_0.  This optimizer corrects
that mismatch by projecting the desired operator-level update onto the parameter
manifold.

The projection problem for a batch of inputs {x^(b)} is:

    min_{dW_l}  (1/B) Σ_b ‖ Σ_l A_l(x^b) dW_l B_l(x^b) − dP*(x^b) ‖_F²
                + λ Σ_l ‖ dW_l ‖_F²

where dP*(x^b) = −G_P(x^b) is the desired operator-level update for sample b,
and A_l(x), B_l(x) are the left/right contexts (products of weights and
diagonal activation gates) computed during the forward pass.

Four approximation strategies are provided:

    'exact'          — Full coupled system solve via block Kronecker linear system.
                       Feasible only for very small networks (≲ 5K total params).

    'block_diagonal' — Decouple layers; solve each layer's Sylvester equation
                       (M_l dW N_l + λ dW = G_W_l) exactly via eigendecomposition
                       of the batch-averaged context covariances M_l, N_l.

    'forward_pass'   — Same Sylvester solve as block_diagonal, but context covariances
                       and gradient RHS are approximated using only forward-pass
                       quantities (Operator-FP, §4.5 of paper):
                           Â_l(x) = f(x) x_l^T / ‖x_l‖²
                           B̂_l(x) = h_{l-1} x^T / ‖x‖²
                       giving M_l^FP, N_l^FP, G_l^FP without layer-by-layer
                       context propagation.  Only grad_acts[L-1] = ∇_f L is needed
                       from the backward pass; for standard losses it can be supplied
                       analytically (see below) making the step truly forward-only.

    'kfac'           — Standard K-FAC: approximate M_l ≈ EMA(ΔzΔzᵀ) and
                       N_l ≈ EMA(hh^T), use the factored formula
                       dW ≈ (F_out + λI)^{-1} G_W (F_in + λI)^{-1}.
                       Equivalent to standard K-FAC (Martens & Grosse 2015);
                       exact as a Sylvester solve only when M_l and N_l commute.

    'secant'         — Secant-Exact: approximate each per-sample context by its
                       rank-1 secant (Â_l = f z_l^T/‖z_l‖², B̂_l = h_{l-1} x^T/‖x‖²),
                       then solve the resulting full coupled system exactly via the
                       Woodbury identity.  The rank-1 approximation collapses all
                       cross-layer coupling to a per-sample scalar γ^b, reducing the
                       coupled system to a B×B kernel solve.
                       Cost: O(B L n²  +  B² L n) vs O(B L n³) for block_diagonal.
                       Unlike block_diagonal, retains full cross-layer coupling.

    'cg'             — Conjugate gradient on the full coupled normal equations;
                       uses Φ and Φ* as matrix-free operators.  [TODO]

Forward-pass-only usage for 'forward_pass' mode (no loss.backward() required):

    # After model(x) with torch.no_grad(), supply ∇_f L manually:
    with torch.no_grad():
        logits = model(x)          # hooks capture activations
    loss = F.cross_entropy(logits, y)
    # analytically: ∇_f L = softmax(logits) - one_hot(y)  (cross-entropy)
    #               ∇_f L = (logits - y) / B               (MSE)
    optimizer.grad_acts[len(linear_layers) - 1] = grad_f
    optimizer.step()               # no backward() needed

Regularisation (λ):
    lam=<float>   Fixed Tikhonov λ at every layer (e.g. lam=1e-3). Simple and
                  predictable, but kills updates exponentially with depth for any
                  fixed λ > 0 once context covariances become small (§5.1 of paper).

    lam=None      Adaptive λ per layer: λ_ℓ = lam_alpha · (σ_max(M_ℓ) + σ_max(N_ℓ)).
                  Uses the top singular value rather than the median: in ReLU networks
                  more than half the neurons can be exactly dead, making median(spec)=0.
                  σ_max is always positive whenever any neuron fires in the batch.
                  Implements Proposition 5.2: restores O(1) update magnitude at all
                  depths without λ → 0.  Set lam_alpha (default 1.0) to tune scale.
                  This is the recommended mode for deep networks (L ≥ 4).

Usage:
    # Fixed λ (original behaviour):
    optimizer = OperatorLevelMLP(model.parameters(), lr=1e-3,
                                  approximation='block_diagonal', lam=1e-3)

    # Adaptive λ (depth-uniform, Prop. 5.2):
    optimizer = OperatorLevelMLP(model.parameters(), lr=1e-3,
                                  approximation='block_diagonal',
                                  lam=None, lam_alpha=1.0)

    # Adaptive λ + EMA smoothing of context matrices (stabilises σ_max at large lr):
    optimizer = OperatorLevelMLP(model.parameters(), lr=1e-3,
                                  approximation='block_diagonal',
                                  lam=None, lam_alpha=1.0, ema_decay=0.95)

    optimizer.attach_hooks(model)

    for x, y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
"""

import warnings

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Optional, List, Dict, Any, Tuple

# PyTorch warns when register_full_backward_hook fires for a module whose
# inputs don't require grad (only its weights do).  Our hook intentionally
# captures grad_output[0] = Δz_l, which is always populated, so the warning
# is spurious.
warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing",
    category=UserWarning,
)


# ---------------------------------------------------------------------------
# Newton-Schulz helper (shared with Muon)
# ---------------------------------------------------------------------------

def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-8) -> torch.Tensor:
    """Approximate polar factor of a 2D matrix G via quintic NS iteration."""
    assert G.ndim == 2
    a, b, c = 1.875, -1.25, 0.375
    X = G / (G.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.T
    return X


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _svd_psd(X: torch.Tensor):
    """SVD of a symmetric PSD matrix with ridge fallback for ill-conditioned inputs.

    LAPACK's dgesdd can fail (error code 256) for matrices with many repeated
    near-zero singular values (e.g. dead-neuron Gram matrices deep in a ReLU net
    after many ALS sweeps).  In that case we add a tiny ridge proportional to
    the mean diagonal and retry once.
    """
    # Primary path: fast device-local SVD.
    try:
        return torch.linalg.svd(X, full_matrices=False)
    except torch._C._LinAlgError:
        pass

    # Retry with progressively larger diagonal ridge.
    diag_scale = max(float(X.diagonal().abs().mean()), 1e-8)
    eye = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    for mult in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        try:
            X_reg = X + (diag_scale * mult) * eye
            return torch.linalg.svd(X_reg, full_matrices=False)
        except torch._C._LinAlgError:
            continue

    # Final fallback: symmetric eigensolve on CPU for pathological spectra.
    # For PSD matrices, singular values are eigenvalues and left/right vectors coincide.
    X_cpu = X.detach().to("cpu", dtype=torch.float64)
    eye_cpu = torch.eye(X_cpu.shape[0], device="cpu", dtype=torch.float64)
    for mult in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        try:
            X_reg_cpu = X_cpu + (diag_scale * mult) * eye_cpu
            evals, evecs = torch.linalg.eigh(X_reg_cpu)
            evals = evals.clamp(min=0.0)
            # Match SVD API ordering: descending singular values.
            order = torch.argsort(evals, descending=True)
            s = evals[order].to(dtype=X.dtype, device=X.device)
            U = evecs[:, order].to(dtype=X.dtype, device=X.device)
            Vh = U.T
            return U, s, Vh
        except torch._C._LinAlgError:
            continue

    raise RuntimeError(
        "_svd_psd failed after SVD and CPU-eigh fallbacks; matrix likely numerically unstable."
    )


# ---------------------------------------------------------------------------
# Sylvester solver
# ---------------------------------------------------------------------------

def _solve_sylvester(
    M: torch.Tensor,
    N: torch.Tensor,
    C: torch.Tensor,
    lam: Optional[float],
    lam_alpha: float = 1.0,
) -> torch.Tensor:
    """Solve the generalized Sylvester equation  M X N + λ X = C.

    Uses SVD of the symmetric PSD matrices M (k×k) and N (d×d):
        M = V Σ_M Vᵀ,  N = U Σ_N Uᵀ
    After the change of variables Y = Vᵀ X U, the equation becomes element-wise:
        Y[i,j] (Σ_M[i] Σ_N[j] + λ) = (Vᵀ C U)[i,j]

    Complexity: O(k³ + d³ + kd).  Much better than the O((kd)³) of the full
    vectorized solve.

    SVD (dgesdd) is used instead of eigh (dsyevd) for numerical robustness: at deep
    layers, ReLU dead neurons produce context matrices with many exactly-zero singular
    values.  LAPACK's symmetric eigensolver fails to converge for such matrices;
    the general SVD handles them correctly.

    Args:
        M:         (k, k) symmetric PSD matrix (e.g. batch-averaged AᵀA).
        N:         (d, d) symmetric PSD matrix (e.g. batch-averaged BBᵀ).
        C:         (k, d) right-hand side (e.g. standard gradient G_W).
        lam:       Tikhonov regularisation λ.  Pass None for adaptive λ.
        lam_alpha: Scaling coefficient α for adaptive λ (ignored if lam is not None).
                   Adaptive formula: λ = α · (σ_max(M) + σ_max(N)).
                   Uses the top singular value rather than the median because in ReLU
                   networks more than half the neurons can be dead (exact zero
                   activations), making median(spec) = 0 and hence λ = 0.  σ_max is
                   always positive whenever any neuron fires in the batch.

    Returns:
        X: (k, d) solution.
    """
    # Symmetric jitter: very ill-conditioned Gram matrices (long deep-linear
    # products, tiny λ) can make LAPACK's SVD fail even for PSD M, N.
    m_sz, n_sz = M.shape[0], N.shape[0]
    eps = float(torch.finfo(M.dtype).eps)
    mean_diag = max(
        float(M.diagonal().abs().mean()),
        float(N.diagonal().abs().mean()),
        1e-30,
    )
    scale = 1024.0 * eps * mean_diag
    M = M + scale * torch.eye(m_sz, device=M.device, dtype=M.dtype)
    N = N + scale * torch.eye(n_sz, device=N.device, dtype=N.dtype)

    # SVD of symmetric PSD matrices.  For symmetric PSD M: left singular vectors =
    # eigenvectors, singular values = eigenvalues.  Returned in descending order.
    V,  sv_M, _ = _svd_psd(M)   # M = V diag(sv_M) Vᵀ
    U,  sv_N, _ = _svd_psd(N)   # N = U diag(sv_N) Uᵀ

    if lam is None:
        # Adaptive λ: use the top singular value (= max eigenvalue) at this layer.
        # sv_M[0] and sv_N[0] are the maxima (descending SVD order).
        lam = lam_alpha * (sv_M[0].item() + sv_N[0].item())
        lam = max(lam, 1e-30)

    F = V.T @ C @ U                                              # (k, d)
    denom = sv_M.unsqueeze(1) * sv_N.unsqueeze(0) + lam         # (k, d)

    if lam == 0.0:
        # Truncated pseudoinverse: zero out modes whose eigenvalue product is
        # below a relative threshold.  Avoids 0/0 explosion for null-space
        # directions when no Tikhonov regularisation is used.
        sv_prod_max = sv_M[0].item() * sv_N[0].item()
        tol = max(sv_prod_max * 1e-6, 1e-30)
        mask = denom >= tol                                      # (k, d) bool
        Y = torch.where(mask, F / denom.clamp(min=tol), torch.zeros_like(F))
    else:
        denom = denom.clamp(min=1e-30)                           # numerical safety
        Y = F / denom                                            # (k, d)

    return V @ Y @ U.T


# ---------------------------------------------------------------------------
# ALS layer solves (batch Op-ALS with secant gates)
# ---------------------------------------------------------------------------

# Dense exact normal equations need O((d_k d_{k-1})²) memory; skip for huge layers.
_MAX_ALS_EXACT_FLAT_DIM = 8192


def _als_dW_from_mn(
    M_k: Optional[torch.Tensor],
    N_k: Optional[torch.Tensor],
    G_k: torch.Tensor,
    lam: Optional[float],
    lam_alpha: float,
) -> torch.Tensor:
    """Decoupled Sylvester solve using batch-averaged M, N (original Op-ALS)."""
    if M_k is None and N_k is None:
        lam_l = lam_alpha * 2.0 if lam is None else lam
        return G_k / (1.0 + lam_l)
    if M_k is None:
        U, sv_N, _ = _svd_psd(N_k)
        lam_l = max(lam_alpha * (1.0 + sv_N[0].item()), 1e-30) if lam is None else lam
        return (G_k @ U / (sv_N + lam_l).unsqueeze(0)) @ U.T
    if N_k is None:
        V, sv_M, _ = _svd_psd(M_k)
        lam_l = max(lam_alpha * (sv_M[0].item() + 1.0), 1e-30) if lam is None else lam
        return V @ ((V.T @ G_k) / (sv_M + lam_l).unsqueeze(1))
    return _solve_sylvester(M_k, N_k, G_k, lam, lam_alpha)


def _als_effective_lam(
    M_k: Optional[torch.Tensor],
    N_k: Optional[torch.Tensor],
    lam: Optional[float],
    lam_alpha: float,
) -> float:
    """Scalar λ for exact least-squares (reuse adaptive rule from decoupled path)."""
    if lam is not None:
        return float(lam)
    if M_k is None and N_k is None:
        return float(lam_alpha * 2.0)
    if M_k is None:
        sv_N = torch.linalg.svdvals(N_k)
        return float(max(lam_alpha * (1.0 + sv_N[0].item()), 1e-30))
    if N_k is None:
        sv_M = torch.linalg.svdvals(M_k)
        return float(max(lam_alpha * (sv_M[0].item() + 1.0), 1e-30))
    sv_M = torch.linalg.svdvals(M_k)
    sv_N = torch.linalg.svdvals(N_k)
    return float(max(lam_alpha * (sv_M[0].item() + sv_N[0].item()), 1e-30))


def _als_exact_ls_delta(
    A_k: torch.Tensor,
    B_curr: Optional[torch.Tensor],
    G_mean: torch.Tensor,
    k: int,
    L: int,
    D_out: int,
    lam_eff: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Minimize (1/B) Σ_b ‖A_k^b ΔW B_k^b − R^b‖_F^2 + λ‖ΔW‖_F^2 via normal equations."""
    device = A_k.device
    Bs = A_k.shape[0]
    d_k = A_k.shape[2]
    if B_curr is None:
        D_in = G_mean.shape[1]
        d_km1 = D_in
    else:
        d_km1 = B_curr.shape[1]
        D_in = B_curr.shape[2]

    H = torch.zeros((d_k * d_km1, d_k * d_km1), device=device, dtype=torch.float64)
    if k == 0:
        assert B_curr is None
        eye_D = torch.eye(D_in, device=device, dtype=torch.float64)
        for b in range(Bs):
            Ab = A_k[b].to(torch.float64)
            ATA = Ab.T @ Ab
            H.add_(torch.kron(eye_D, ATA), alpha=1.0 / Bs)
    elif k == L - 1:
        assert B_curr is not None
        eye_out = torch.eye(D_out, device=device, dtype=torch.float64)
        for b in range(Bs):
            Bb = B_curr[b].to(torch.float64)
            BBT = Bb @ Bb.T
            H.add_(torch.kron(BBT, eye_out), alpha=1.0 / Bs)
    else:
        assert B_curr is not None
        for b in range(Bs):
            Ab = A_k[b].to(torch.float64)
            Bb = B_curr[b].to(torch.float64)
            ATA = Ab.T @ Ab
            BBT = Bb @ Bb.T
            H.add_(torch.kron(BBT, ATA), alpha=1.0 / Bs)

    mean_diag = float(torch.diagonal(H).abs().mean().clamp(min=1e-30))
    jitter = 1024.0 * float(torch.finfo(torch.float64).eps) * mean_diag
    H.add_(torch.eye(H.shape[0], device=device, dtype=torch.float64), alpha=jitter)

    lam_eps = max(lam_eff, 1e-30)
    H_reg = H + lam_eps * torch.eye(H.shape[0], device=device, dtype=torch.float64)
    rhs = G_mean.T.reshape(-1).to(torch.float64).unsqueeze(1)
    try:
        x = torch.linalg.solve(H_reg, rhs).squeeze(1)
    except RuntimeError:
        # Singular / ill-conditioned Hessians (e.g. extreme lr, dead gates): lstsq fallback.
        x = torch.linalg.lstsq(H_reg, rhs, rcond=1e-10).solution.squeeze(1)
    dW = x.reshape(d_km1, d_k).T.to(dtype)
    return dW


def _als_per_sample_delta(
    A_k: torch.Tensor,
    B_curr: Optional[torch.Tensor],
    R: torch.Tensor,
    delta_anchor: torch.Tensor,
    k: int,
    L: int,
    D_out: int,
    lam: Optional[float],
    lam_alpha: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Solve decoupled Sylvester per sample, then batch-average ΔW."""
    Bs = A_k.shape[0]
    d_k = A_k.shape[2]
    if k == 0:
        d_in = R.shape[2]
        dW_sum = torch.zeros(d_k, d_in, device=A_k.device, dtype=dtype)
        for b in range(Bs):
            Ab = A_k[b]
            Rb = R[b]
            Mb = Ab.T @ Ab
            Gb = Ab.T @ Rb
            lam_b = _als_effective_lam(Mb, None, lam, lam_alpha)
            Gb_reg = Gb - lam_b * delta_anchor
            dW_sum.add_(_als_dW_from_mn(Mb, None, Gb_reg, lam_b, lam_alpha))
        return dW_sum / Bs
    if k == L - 1:
        assert B_curr is not None
        d_km1 = B_curr.shape[1]
        dW_sum = torch.zeros(D_out, d_km1, device=A_k.device, dtype=dtype)
        for b in range(Bs):
            Bb = B_curr[b]
            Rb = R[b]
            Nb = Bb @ Bb.T
            Gb = Rb @ Bb.T
            lam_b = _als_effective_lam(None, Nb, lam, lam_alpha)
            Gb_reg = Gb - lam_b * delta_anchor
            dW_sum.add_(_als_dW_from_mn(None, Nb, Gb_reg, lam_b, lam_alpha))
        return dW_sum / Bs

    assert B_curr is not None
    d_km1 = B_curr.shape[1]
    dW_sum = torch.zeros(d_k, d_km1, device=A_k.device, dtype=dtype)
    for b in range(Bs):
        Ab = A_k[b]
        Bb = B_curr[b]
        Rb = R[b]
        Mb = Ab.T @ Ab
        Nb = Bb @ Bb.T
        Gb = Ab.T @ Rb @ Bb.T
        lam_b = _als_effective_lam(Mb, Nb, lam, lam_alpha)
        Gb_reg = Gb - lam_b * delta_anchor
        dW_sum.add_(_als_dW_from_mn(Mb, Nb, Gb_reg, lam_b, lam_alpha))
    return dW_sum / Bs


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------

class OperatorLevelMLP(Optimizer):
    """Operator-level gradient projection optimizer for MLPs.

    Call `attach_hooks(model)` once before training to register the forward
    and backward hooks that capture pre-activations and their gradients.

    Args:
        params:        Model parameters.
        lr:            Learning rate η.
        momentum:      Nesterov momentum coefficient β₁ (default 0.9).
        lam:           Tikhonov regularisation λ (default 1e-4).
        approximation: One of 'exact', 'block_diagonal', 'kfac', 'cg'.
        kfac_decay:    EMA decay for K-FAC statistics (default 0.95).
        ns_steps:      Newton-Schulz iterations (used in 'exact' mode for
                       Nesterov step orthogonalisation; default 5).
        weight_decay:  L2 regularisation on weights (default 0.0).
        ema_decay:     EMA decay β for context matrices M_l and N_l (default 0.0 =
                       disabled).  When > 0, M̄_l ← β·M̄_l + (1-β)·M_l^{current}
                       and the smoothed M̄_l, N̄_l are used for both the Sylvester
                       solve and the adaptive λ computation.  Same approach as K-FAC
                       for Fisher factors.  Typical values: 0.9–0.99.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        lam: Optional[float] = 1e-4,
        lam_alpha: float = 1.0,
        approximation: str = "block_diagonal",
        kfac_decay: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        use_ns: bool = False,
        rank: Optional[int] = None,
        ema_decay: float = 0.0,
        rescale_lr: bool = False,
        rescale_variant: str = "max",
        n_sweeps: int = 50,
        als_reverse_sweep: bool = False,
        als_layer_solve: str = "mn",
        als_init_delta_orthogonal: bool = False,
        als_init_delta_scale: Optional[float] = None,
        als_gateperm_warmstart: bool = True,
        als_gateperm_rel_threshold: float = 0.5,
        als_gateperm_warmstart_no_batch_mean: bool = False,
        als_gateperm_warmstart_once: bool = True,
        als_ce_batch_scale: bool = False,
        als_lam_anchor_post_warmstart: bool = False,
        als_residual_threshold: Optional[float] = None,
        als_max_sweeps: Optional[int] = 50,
        als_early_stop_rel_tol: float = 0.0,
        als_early_stop_patience: int = 0,
        als_lam_regularize_delta: bool = False,
    ):
        """
        Args:
            lam:       Tikhonov regularisation λ.
                       • float > 0 — fixed λ used at every layer (default 1e-4).
                       • None      — adaptive λ per layer: λ_ℓ = lam_alpha ·
                                     (median(spec(M_ℓ)) + median(spec(N_ℓ))).
                                     Implements Proposition 5.2 of the paper:
                                     restores O(1) update magnitude at all depths
                                     without requiring λ → 0.
            lam_alpha: Scaling coefficient α for adaptive λ (ignored when lam is
                       a fixed float).  Typical range: 0.1–2.0 (default 1.0).
            rescale_lr: If True, rescale all per-layer updates by a global scalar
                       1/S before applying lr, where S is determined by
                       rescale_variant (see below).  Makes the effective step size
                       independent of gradient scale and architecture.
            rescale_variant: How to compute the rescaling denominator S.
                       • 'max'     — S = max_ℓ ||ΔW_ℓ||_F (all layers, current
                                     default).  Can be dominated by the output layer.
                       • 'sum'     — S = Σ_{ℓ<L} ||ΔW_ℓ||_F (hidden layers only,
                                     excludes output).  Avoids output-layer bias.
                       • 'cascade' — S = Σ_{k<L} (L-1-k) ||ΔW_k||_F (cascade-
                                     weighted sum; layer k penalised by the number
                                     of downstream gate layers it affects).
                                     Corresponds to σ→1 in Proposition 6.4 of the
                                     paper; 'sum' corresponds to σ→0.
            als_init_delta_orthogonal: For approximation='als', initialize each
                       layer's inner-sweep delta with a random orthogonal matrix
                       scaled by 1/L (L = depth) before the first ALS sweep.
                       Default False keeps the classic zero-delta start.
            als_init_delta_scale: Optional explicit scale for the orthogonal
                       delta init. If set, uses this value instead of 1/L.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lam is not None and not 0.0 <= lam:
            raise ValueError(f"Invalid lam: {lam}")
        if not 0.0 < lam_alpha:
            raise ValueError(f"Invalid lam_alpha: {lam_alpha}")
        if approximation not in ("exact", "block_diagonal", "local_bd", "forward_pass", "fp_ps", "fpc", "kfac", "operator_kfac", "cg", "whitened", "secant", "secant_r", "als"):
            raise ValueError(f"Unknown approximation: {approximation!r}")
        if rescale_variant not in ("max", "sum", "cascade"):
            raise ValueError(f"Unknown rescale_variant: {rescale_variant!r}. Choose 'max', 'sum', or 'cascade'.")
        if als_layer_solve not in ("mn", "exact", "per_sample"):
            raise ValueError(
                "als_layer_solve must be 'mn' (batch-averaged M,N Sylvester), "
                "'exact' (full normal equations per layer), or "
                f"'per_sample' (Sylvester per sample then mean); got {als_layer_solve!r}"
            )
        if als_init_delta_scale is not None and als_init_delta_scale < 0.0:
            raise ValueError(f"Invalid als_init_delta_scale: {als_init_delta_scale}")
        if als_gateperm_rel_threshold < 0.0:
            raise ValueError(f"Invalid als_gateperm_rel_threshold: {als_gateperm_rel_threshold}")
        if als_early_stop_rel_tol < 0.0:
            raise ValueError(f"Invalid als_early_stop_rel_tol: {als_early_stop_rel_tol}")
        if als_early_stop_patience < 0:
            raise ValueError(f"Invalid als_early_stop_patience: {als_early_stop_patience}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            lam=lam,
            lam_alpha=lam_alpha,
            approximation=approximation,
            kfac_decay=kfac_decay,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            use_ns=use_ns,
            rank=rank,
            ema_decay=ema_decay,
            rescale_lr=rescale_lr,
            rescale_variant=rescale_variant,
            n_sweeps=n_sweeps,
            als_reverse_sweep=als_reverse_sweep,
            als_layer_solve=als_layer_solve,
            als_init_delta_orthogonal=als_init_delta_orthogonal,
            als_init_delta_scale=als_init_delta_scale,
            als_gateperm_warmstart=als_gateperm_warmstart,
            als_gateperm_rel_threshold=als_gateperm_rel_threshold,
            als_gateperm_warmstart_no_batch_mean=als_gateperm_warmstart_no_batch_mean,
            als_gateperm_warmstart_once=als_gateperm_warmstart_once,
            als_ce_batch_scale=als_ce_batch_scale,
            als_lam_anchor_post_warmstart=als_lam_anchor_post_warmstart,
            als_residual_threshold=als_residual_threshold,
            als_max_sweeps=als_max_sweeps,
            als_early_stop_rel_tol=als_early_stop_rel_tol,
            als_early_stop_patience=als_early_stop_patience,
            als_lam_regularize_delta=als_lam_regularize_delta,
        )
        super().__init__(params, defaults)

        # Warn at most once per (layer_index, flat_dim) when exact→mn fallback triggers.
        self._als_exact_fallback_warned: set[tuple[int, int]] = set()

        # Buffers populated by hooks during forward/backward pass
        self.pre_acts: Dict[int, torch.Tensor] = {}   # h_{l-1}  (input to layer l)
        self.activations: Dict[int, torch.Tensor] = {}  # z_l     (output of layer l)
        self.grad_acts: Dict[int, torch.Tensor] = {}  # Δz_l    (grad w.r.t. z_l)
        self.hooks: List = []

        # Accumulation lists — populated instead of overwriting in each hook call.
        # For feedforward MLPs each list has exactly one entry (same as before).
        # For RNNs / weight-tied layers called T times per step, each list has T entries.
        # _consolidate_activations() merges these into the dicts above before step().
        self._pre_acts_list:  Dict[int, List[torch.Tensor]] = {}
        self._acts_list:      Dict[int, List[torch.Tensor]] = {}
        self._grad_acts_list: Dict[int, List[torch.Tensor]] = {}

        # Maps id(param_tensor) → hook index for weight-tied lookup in _step_local_bd.
        self._param_to_hook_idx: Dict[int, int] = {}
        self._als_prev_rel_jac: float = float("inf")
        self._als_last_warmstart_applied: bool = False
        self._als_warmstart_used_once: bool = False
        self._als_last_warmstart_dW_mean: float = 0.0
        self._als_last_warmstart_dW_max: float = 0.0
        self._als_last_solver_dW_mean: float = 0.0
        self._als_last_solver_dW_max: float = 0.0
        self._als_sweep_residuals: list[float] = []
        self._als_num_sweeps_used: int = 0

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def attach_hooks(self, model: nn.Module):
        """Register forward and backward hooks on all Linear layers in model.

        After this call every forward pass populates:
            self.pre_acts[l]    — h_{l-1}: input to layer l
            self.activations[l] — z_l: pre-activation output
        and every backward pass populates:
            self.grad_acts[l]   — Δz_l: gradient w.r.t. z_l

        For weight-tied / recurrent layers called T times per step, activations
        are accumulated in lists and merged by _consolidate_activations() at
        the start of step().  Single-call (feedforward) layers behave identically
        to the previous implementation.

        Call this once before training.  Re-calling removes old hooks first.
        """
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.pre_acts = {}
        self.activations = {}
        self.grad_acts = {}
        self._pre_acts_list  = {}
        self._acts_list      = {}
        self._grad_acts_list = {}
        self._param_to_hook_idx = {}

        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

        for idx, layer in enumerate(linear_layers):
            def make_fwd_hook(i):
                def fwd(module, inp, out):
                    # Skip during eval / no_grad passes — only accumulate during training.
                    if not module.training:
                        return
                    self._pre_acts_list.setdefault(i, []).append(inp[0].detach())
                    self._acts_list.setdefault(i, []).append(out.detach())
                return fwd

            def make_bwd_hook(i):
                def bwd(module, grad_input, grad_output):
                    if grad_output[0] is not None:
                        self._grad_acts_list.setdefault(i, []).append(
                            grad_output[0].detach()
                        )
                return bwd

            self.hooks.append(layer.register_forward_hook(make_fwd_hook(idx)))
            self.hooks.append(layer.register_full_backward_hook(make_bwd_hook(idx)))
            # Record weight → hook index for alignment in _step_local_bd
            self._param_to_hook_idx[id(layer.weight)] = idx

    def _consolidate_activations(self):
        """Merge accumulated hook lists into the pre_acts / activations / grad_acts dicts.

        For feedforward layers (one call per step), this is a no-op beyond moving the
        single-element list into the dict.  For recurrent layers (T calls per step),
        all T activation tensors are concatenated along dim 0, giving an effective
        batch of T*B samples that captures the temporal statistics.

        Multi-dimensional inputs (e.g. shape (B, T, d) from a Linear applied to a
        packed sequence) are flattened to (B*T, d) so covariance formulas stay 2-D.
        """
        def _merge(lst: List[torch.Tensor]) -> torch.Tensor:
            if len(lst) == 1:
                t = lst[0]
            else:
                t = torch.cat(lst, dim=0)
            # Flatten any leading extra dims: (B, T, d) → (B*T, d)
            if t.ndim > 2:
                t = t.reshape(-1, t.shape[-1])
            return t

        for i, lst in self._pre_acts_list.items():
            self.pre_acts[i] = _merge(lst)
        for i, lst in self._acts_list.items():
            self.activations[i] = _merge(lst)
        for i, lst in self._grad_acts_list.items():
            self.grad_acts[i] = _merge(lst)

        self._pre_acts_list.clear()
        self._acts_list.clear()
        self._grad_acts_list.clear()

    # ------------------------------------------------------------------
    # Context computation  (shared by exact and block_diagonal)
    # ------------------------------------------------------------------

    def _compute_contexts(
        self,
        params_list: List[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute per-sample left/right contexts A_l(x^b) and B_l(x^b).

        For 0-indexed layer l in a network with L layers (0 .. L-1):
            A_{L-1}(x) = I                                         (last layer)
            A_l(x)     = A_{l+1}(x) W_{l+1} diag(D_l(x))         (l < L-1)

            B_0(x)     = I                                         (first layer)
            B_l(x)     = diag(D_{l-1}(x)) W_{l-1} B_{l-1}(x)    (l > 0)

        where D_l(x) = diag(ReLU'(z_l)) is the activation gate at layer l
        (evaluated at the *current* forward-pass activations stored in
        self.activations).

        Returns:
            A_list: list of (batch, D_out, d_l_out) tensors.
            B_list: list of (batch, d_l_in, D_in) tensors.
        """
        L = len(params_list)
        A_list: List[Optional[torch.Tensor]] = [None] * L
        B_list: List[Optional[torch.Tensor]] = [None] * L

        # Activation gates D_l = σ(z_l) / z_l  (elementwise diagonal of the activation Jacobian).
        # Uses the ratio of the post-activation (= pre_acts[l+1], the input to the next layer)
        # to the pre-activation (= activations[l], the output of the linear layer):
        #   D_l = pre_acts[l+1] / activations[l]
        # This is exact for any pointwise activation and naturally handles special cases:
        #   • ReLU:   relu(z)/z = (z > 0)  — same as before
        #   • Linear: z/z = 1              — correct for networks with no activation
        #   • Other σ: σ(z)/z              — general secant approximation
        # Where |z_l| < eps the ratio is replaced by 1 (correct for linear layers at zero;
        # for dead ReLU neurons pre_acts[l+1]=0 and the ratio is 0, handled by the branch).
        # D_list[L-1] is a placeholder (the last layer's gate is never used in context
        # propagation since A_{L-1} = I by definition).
        D_list = []
        eps = 1e-8
        for l in range(L):
            z_l = self.activations[l].to(dtype)    # (batch, d_l_out) — linear-layer output
            if l + 1 < L:
                h_next = self.pre_acts[l + 1].to(dtype)  # post-activation = σ(z_l) or z_l
                D_l = torch.where(z_l.abs() > eps, h_next / z_l, torch.ones_like(z_l))
            else:
                D_l = torch.ones_like(z_l)         # unused placeholder for last layer
            D_list.append(D_l)

        # A_{L-1} = I  (shape: batch × D_out × D_out)
        D_out = params_list[-1].shape[0]
        A_list[-1] = (
            torch.eye(D_out, device=device, dtype=dtype)
            .unsqueeze(0).expand(batch_size, -1, -1)
        )
        # A_l = A_{l+1} W_{l+1} diag(D_l)   for l = L-2 down to 0
        for l in range(L - 2, -1, -1):
            W_next = params_list[l + 1]    # (d_{l+1}_out, d_{l+1}_in) = (d_{l+1}, d_l)
            D_l = D_list[l]                # (batch, d_l)
            # W_next @ diag(D_l): broadcast over batch
            # W_D[b, i, j] = W_next[i, j] * D_l[b, j]
            W_D = W_next.unsqueeze(0) * D_l.unsqueeze(1)  # (batch, d_{l+1}, d_l)
            A_list[l] = torch.bmm(A_list[l + 1], W_D)     # (batch, D_out, d_l)

        # B_0 = I: stored as None (avoid materialising a batch×D_in×D_in identity).
        # B_l for l≥1 has shape (batch, d_{l-1}, D_in) and is built by the recurrence
        #   B_l(x^b) = diag(D_{l-1}(x^b)) W_{l-1} B_{l-1}(x^b)
        # For l=1 the base case is B_0 = I, so B_1 = diag(D_0) W_0 directly.
        B_list[0] = None  # sentinel for identity; callers must handle k=0 specially
        if L > 1:
            D_prev = D_list[0]          # (batch, d_0_out)
            W_prev = params_list[0]     # (d_0_out, D_in)
            B_list[1] = D_prev.unsqueeze(-1) * W_prev.unsqueeze(0)  # (batch, d_0_out, D_in)
        # B_l = diag(D_{l-1}) W_{l-1} B_{l-1}   for l = 2 to L-1
        for l in range(2, L):
            W_prev = params_list[l - 1]    # (d_{l-1}_out, d_{l-1}_in)
            D_prev = D_list[l - 1]         # (batch, d_{l-1}_out)
            D_W = D_prev.unsqueeze(-1) * W_prev.unsqueeze(0)  # (batch, d_{l-1}_out, d_{l-1}_in)
            B_list[l] = torch.bmm(D_W, B_list[l - 1])         # (batch, d_{l-1}_out, D_in)

        return A_list, B_list

    # ------------------------------------------------------------------
    # Fast covariance computation (used by block_diagonal and operator_kfac)
    # ------------------------------------------------------------------

    def _compute_covariances(
        self,
        params_list: List[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """Compute per-layer covariances M_l and N_l without materialising full
        context tensors.

            M_l = (1/B) Σ_b A_l(x^b)^T A_l(x^b)   shape (d_l_out, d_l_out)
            N_l = (1/B) Σ_b B_l(x^b) B_l(x^b)^T   shape (d_l_in,  d_l_in)

        Two identity special cases are handled without any computation:
            M_{L-1} = I  (A_{L-1} = I by definition) → stored as None
            N_0      = I  (B_0     = I by definition) → stored as None

        For M_l: A_curr is propagated backwards one step at a time; only the
        current slice is kept in memory.  M_l is computed via a single reshaped
        matmul  A_2d^T @ A_2d / B  (one GEMM, no bmm+reduce).

        For N_l: instead of storing B_l (shape batch × d × D_in), we propagate
        per-sample outer products  Q_l^(b) = B_l^(b) B_l^(b)^T  (shape d × d)
        forward using the recurrence
            Q_1^(b) = DW_0^(b) (DW_0^(b))^T
            Q_l^(b) = DW_{l-1}^(b) Q_{l-1}^(b) (DW_{l-1}^(b))^T  for l >= 2
        where DW_{l-1}^(b) = diag(D_{l-1}^(b)) W_{l-1}.
        This eliminates the D_in dimension from all stored tensors for l >= 2.
        """
        L = len(params_list)
        M_list: List[Optional[torch.Tensor]] = [None] * L  # M_{L-1} stays None
        N_list: List[Optional[torch.Tensor]] = [None] * L  # N_0     stays None

        # Same gate formula as _compute_contexts: D_l = pre_acts[l+1] / activations[l].
        # See _compute_contexts docstring for the full rationale.
        eps = 1e-8
        D_list = []
        for l in range(L):
            z_l = self.activations[l].to(dtype)
            if l + 1 < L:
                h_next = self.pre_acts[l + 1].to(dtype)
                D_l = torch.where(z_l.abs() > eps, h_next / z_l, torch.ones_like(z_l))
            else:
                D_l = torch.ones_like(z_l)
            D_list.append(D_l)

        # ---- Left covariances M_l (propagate A_curr backwards) ----
        # A_{L-1} = I → M_{L-1} = I (special case, leave M_list[L-1] = None)
        D_out  = params_list[-1].shape[0]
        A_curr = (
            torch.eye(D_out, device=device, dtype=dtype)
            .unsqueeze(0).expand(batch_size, -1, -1)
        )  # (batch, D_out, D_out)

        for l in range(L - 2, -1, -1):
            W_next = params_list[l + 1]                              # (d_{l+1}, d_l)
            D_l    = D_list[l]                                       # (batch, d_l)
            W_D    = W_next.unsqueeze(0) * D_l.unsqueeze(1)         # (batch, d_{l+1}, d_l)
            A_curr = torch.bmm(A_curr, W_D)                         # (batch, D_out, d_l)
            # M_l via reshaped matmul: one GEMM instead of bmm + mean
            d_l   = A_curr.shape[2]
            A_2d  = A_curr.reshape(batch_size * D_out, d_l)
            M_list[l] = A_2d.T @ A_2d / batch_size                 # (d_l, d_l)

        # ---- Right covariances N_l (propagate Q_batch forwards) ----
        # N_0 = I (special case, leave N_list[0] = None)
        if L > 1:
            # l = 1: Q_1^(b) = DW_0^(b) (DW_0^(b))^T
            D_prev  = D_list[0]        # (batch, d_0_out)
            W_prev  = params_list[0]   # (d_0_out, D_in)
            DW      = D_prev.unsqueeze(-1) * W_prev.unsqueeze(0)  # (batch, d_0_out, D_in)
            Q_batch = torch.bmm(DW, DW.transpose(1, 2))           # (batch, d_0_out, d_0_out)
            N_list[1] = Q_batch.mean(0)

            for l in range(2, L):
                D_prev  = D_list[l - 1]       # (batch, n_{l-1})
                W_prev  = params_list[l - 1]  # (n_{l-1}, n_{l-2})
                DW      = D_prev.unsqueeze(-1) * W_prev.unsqueeze(0)     # (batch, n_{l-1}, n_{l-2})
                Q_batch = torch.bmm(torch.bmm(DW, Q_batch), DW.transpose(1, 2))  # (batch, n_{l-1}, n_{l-1})
                N_list[l] = Q_batch.mean(0)

        return M_list, N_list

    # ------------------------------------------------------------------
    # Nesterov momentum (applied to raw gradients before projection)
    # ------------------------------------------------------------------

    def _nesterov_grads(
        self,
        params_list: List[torch.Tensor],
        grad_list: List[torch.Tensor],
        beta: float,
    ) -> List[torch.Tensor]:
        """Apply Nesterov momentum to gradients and return modified grad list.

        m_t = β m_{t-1} + g_t
        g_nes = g_t + β m_t
        """
        nesterov = []
        for p, g in zip(params_list, grad_list):
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            buf.mul_(beta).add_(g)
            nesterov.append(g.add(buf, alpha=beta))
        return nesterov

    # ------------------------------------------------------------------
    # Step dispatcher
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Merge per-timestep hook lists into single tensors (no-op for feedforward).
        self._consolidate_activations()

        for group in self.param_groups:
            method = group["approximation"]
            if method == "exact":
                self._step_exact(group)
            elif method == "block_diagonal":
                self._step_block_diagonal(group)
            elif method == "local_bd":
                self._step_local_bd(group)
            elif method == "forward_pass":
                self._step_forward_pass(group)
            elif method == "fp_ps":
                self._step_fp_ps(group)
            elif method == "fpc":
                self._step_fpc(group)
            elif method == "kfac":
                self._step_kfac(group)
            elif method == "operator_kfac":
                self._step_operator_kfac(group)
            elif method == "cg":
                self._step_cg(group)
            elif method == "whitened":
                self._step_whitened(group)
            elif method == "secant":
                self._step_secant(group)
            elif method == "secant_r":
                self._step_secant_r(group)
            elif method == "als":
                self._step_als(group)
            else:
                raise ValueError(f"Unknown approximation: {method!r}")
            self._update_1d_params(group)

        return loss

    def _update_1d_params(self, group: Dict[str, Any]):
        """Normalized gradient descent with Nesterov momentum for 1D params (biases).

        Mirrors Muon's fallback: the polar factor of a vector is its unit vector,
        so the update is g_nes / ‖g_nes‖.  This ensures biases are updated on
        the same schedule as weight matrices regardless of approximation mode.
        """
        lr   = group["lr"]
        beta = group["momentum"]
        eps  = 1e-8
        for p in group["params"]:
            if p.grad is None or p.ndim != 1:
                continue
            g = p.grad.clone()
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(beta).add_(g)
            g_nes = g.add(buf, alpha=beta)
            norm = g_nes.norm()
            update = g_nes / (norm + eps) if norm > eps else g_nes
            p.add_(update, alpha=-lr)

    # ------------------------------------------------------------------
    # exact: full coupled Kronecker system
    # ------------------------------------------------------------------

    def _step_exact(self, group: Dict[str, Any]):
        """Full coupled system solve.

        Assembles the block Kronecker matrix A_sys (total_params × total_params)
        and solves A_sys vec(dW) = vec(G_W) via torch.linalg.solve.

        This is exact (up to batch averaging and the first-order linearisation
        of the nonlinearities) but only tractable for very small networks.

        Feasibility guideline: total_params ≲ 5K (A_sys would be 25M entries).
        For MNIST with a 784→64→10 MLP: total ≈ 51K → NOT feasible here.
        Use 'block_diagonal' or 'kfac' for anything MNIST-sized.
        """
        if not self.activations:
            raise RuntimeError(
                "attach_hooks(model) must be called before training when using "
                "approximation='exact'."
            )

        lr = group["lr"]
        lam = group["lam"]
        lam_alpha = group["lam_alpha"]
        beta = group["momentum"]
        wd = group["weight_decay"]

        params_list = [p for p in group["params"] if p.grad is not None and p.ndim == 2]
        if not params_list:
            return

        device = params_list[0].device
        dtype = params_list[0].dtype
        batch_size = next(iter(self.activations.values())).shape[0]

        # Raw gradients (+ weight decay)
        grad_list = [
            p.grad.clone().add_(p, alpha=wd) if wd else p.grad.clone()
            for p in params_list
        ]

        # Contexts
        A_list, B_list = self._compute_contexts(params_list, batch_size, device, dtype)

        # Assemble block Kronecker system
        shapes = [p.shape for p in params_list]
        n_per_layer = [s[0] * s[1] for s in shapes]
        total_params = sum(n_per_layer)

        A_sys = torch.zeros(total_params, total_params, device=device, dtype=dtype)
        b_sys = torch.zeros(total_params, device=device, dtype=dtype)

        offset_k = 0
        for k in range(len(params_list)):
            n_k = n_per_layer[k]
            b_sys[offset_k: offset_k + n_k] = grad_list[k].view(-1)

            offset_l = 0
            for l in range(len(params_list)):
                n_l = n_per_layer[l]

                At_Al = torch.bmm(
                    A_list[k].transpose(1, 2), A_list[l]
                )  # (batch, d_k_out, d_l_out)
                Bk_Blt = torch.bmm(
                    B_list[k], B_list[l].transpose(1, 2)
                )  # (batch, d_k_in,  d_l_in)

                kron_block = torch.zeros(n_k, n_l, device=device, dtype=dtype)
                for b in range(batch_size):
                    kron_block.add_(torch.kron(At_Al[b], Bk_Blt[b]))
                kron_block.div_(batch_size)

                if k == l:
                    if lam is None:
                        # Adaptive λ per layer: scale by spectral norm of each context.
                        # A_list[k]: (batch, d_out, d_out_k), B_list[k]: (batch, d_in_k, d_in)
                        sv_A = torch.linalg.svdvals(A_list[k].mean(0))
                        sv_B = torch.linalg.svdvals(B_list[k].mean(0))
                        lam_k = max(lam_alpha * (sv_A[0].item() + sv_B[0].item()), 1e-30)
                    else:
                        lam_k = lam
                    kron_block.add_(
                        torch.eye(n_k, device=device, dtype=dtype), alpha=lam_k
                    )

                A_sys[offset_k: offset_k + n_k, offset_l: offset_l + n_l] = kron_block
                offset_l += n_l

            offset_k += n_k

        # Solve: A_sys vec(dW) = vec(G_W)
        # With lam=0 the system may be singular; use lstsq (min-norm pseudoinverse).
        if lam == 0.0:
            dW_vec = torch.linalg.lstsq(
                A_sys, b_sys.unsqueeze(-1), rcond=1e-6,
            ).solution.squeeze(-1)
        else:
            dW_vec = torch.linalg.solve(A_sys, b_sys)

        # Collect raw updates, apply Nesterov momentum, then update parameters
        dW_raw_list = []
        offset = 0
        for p in params_list:
            n = p.numel()
            dW_raw_list.append(dW_vec[offset: offset + n].view_as(p).clone())
            offset += n

        dW_list = self._nesterov_grads(params_list, dW_raw_list, beta)

        if group["rescale_lr"]:
            dW_list = self._rescale_dW(dW_list, group.get("rescale_variant", "max"))

        for p, dW in zip(params_list, dW_list):
            p.add_(dW, alpha=-lr)

        self.activations.clear()
        self.pre_acts.clear()
        self.grad_acts.clear()

    # ------------------------------------------------------------------
    # Helper: cascade-aware rescaling (Section 6.3 of the paper)
    # ------------------------------------------------------------------

    @staticmethod
    def _rescale_dW(dW_list: List[torch.Tensor], variant: str) -> List[torch.Tensor]:
        """Divide all per-layer updates by scalar S determined by variant.

        'max'     — S = max_ℓ ||ΔW_ℓ|| (all layers).  Every layer moves by at
                    most lr per step; the largest-norm layer moves by exactly lr.
        'sum'     — Hidden layers (all but last) are divided by S_sum =
                    Σ_{k<L} ||ΔW_k||.  The output layer is normalised by its own
                    Frobenius norm (moves by exactly lr), preventing the output
                    gradient from dominating when hidden layers are small.
        'cascade' — Same as 'sum' but S_cas = Σ_{k<L} (L-1-k)||ΔW_k|| weights
                    each hidden layer by the number of downstream gate layers it
                    perturbs, as in Proposition 6.4 of the paper.

        For 'sum' and 'cascade' the output layer (w_L = 0 in the cascade bound)
        is excluded from S: it is treated with a unit-norm normalisation so that
        it cannot blow up when hidden-layer norms are small at initialisation.
        """
        L = len(dW_list)
        norms = [dW.norm().item() for dW in dW_list]

        if variant == "max":
            S = max(norms)
            if S <= 0:
                return dW_list
            return [dW / S for dW in dW_list]

        # sum / cascade: hidden layers only for S; output layer normalised separately.
        # Raw (un-normalised) forms — S grows with depth, so lr transfers meaning:
        # deeper networks naturally get smaller effective steps per unit lr.
        if variant == "sum":
            S_hidden = sum(norms[:-1])
        else:  # cascade: w_k = (L-1-k), penalises early layers by downstream gate count
            S_hidden = sum((L - 1 - k) * norms[k] for k in range(L - 1))

        if S_hidden <= 0:
            return dW_list

        hidden_scaled = [dW / S_hidden for dW in dW_list[:-1]]
        out_norm = norms[-1]
        out_scaled = dW_list[-1] / out_norm if out_norm > 0 else dW_list[-1]
        return hidden_scaled + [out_scaled]

    # ------------------------------------------------------------------
    # als: non-linearised EOPP via alternating least squares
    # ------------------------------------------------------------------

    def _step_als(self, group: Dict[str, Any]):
        """ALS step (MLP, secant gates): lr inside sweep, live contexts.

        Mirrors ``OperatorALS.step()`` but batched over samples with frozen
        activation gates (secant linearisation).

        Per-sample operator:  P^b = A_k^b W_k B_k^b.
        Target per sample:    P_target^b = P_init^b + lr * dP*^b.
        Residual:             R^b = P_target^b - P_curr^b.

        The lr is baked into the sweep target so that live contexts see the
        actual weight trajectory.  The sweep result is committed directly
        (no post-hoc rescaling), matching the deep linear OperatorALS.

        **Sweep order** (``als_reverse_sweep``): forward visits k = 0, …, L-1 with
        ``A_k`` fixed at sweep start and the left context ``B`` updated live — O(L)
        matmuls per sweep.  Reverse visits k = L-1, …, 0 with ``B_k`` fixed at
        sweep start and ``A_k`` updated live from the right — same asymptotic cost
        and symmetric Gauss–Seidel structure.
        """
        if not self.activations:
            raise RuntimeError(
                "attach_hooks(model) must be called before training when using "
                "approximation='als'."
            )
        if not self.grad_acts:
            raise RuntimeError(
                "approximation='als' requires loss.backward() before step()."
            )

        lr        = group["lr"]
        lam       = group["lam"]
        lam_alpha = group["lam_alpha"]
        n_sweeps  = group.get("n_sweeps", 50)
        residual_threshold = group.get("als_residual_threshold", None)
        max_sweeps = group.get("als_max_sweeps", None)
        early_stop_rel_tol = float(group.get("als_early_stop_rel_tol", 0.0))
        early_stop_patience = int(group.get("als_early_stop_patience", 0))
        # Sweep budgeting:
        # - without residual threshold: run n_sweeps, optionally capped by max_sweeps
        # - with residual threshold: legacy default cap=100 unless max_sweeps provided
        _sweep_limit = (max_sweeps if max_sweeps is not None else 100) if residual_threshold is not None else n_sweeps
        if max_sweeps is not None:
            _sweep_limit = min(int(_sweep_limit), int(max_sweeps))

        params_list = [p for p in group["params"] if p.ndim == 2]
        if not params_list:
            self.activations.clear()
            self.pre_acts.clear()
            self.grad_acts.clear()
            return

        L      = len(params_list)
        device = params_list[0].device
        dtype  = params_list[0].dtype
        Bs     = next(iter(self.activations.values())).shape[0]
        D_out  = params_list[-1].shape[0]

        delta = self.grad_acts[L - 1].to(dtype)            # (Bs, D_out)
        x_in  = self.pre_acts[0].to(dtype)                 # (Bs, D_in)

        # dP*^b = delta^b (x^b)^T   (per-sample operator gradient)
        dP_star = torch.einsum("bi,bj->bij", delta, x_in)  # (Bs, D_out, D_in)
        # Optional legacy scaling for mean-reduction CE: compensate 1/B in delta.
        if bool(group.get("als_ce_batch_scale", False)):
            dP_star = dP_star * float(Bs)

        # ---- Frozen activation gates (computed once) ----
        eps_gate = 1e-8
        D_list: List[torch.Tensor] = []
        for l in range(L):
            z_l = self.activations[l].to(dtype)
            if l + 1 < L:
                h_next = self.pre_acts[l + 1].to(dtype)
                D_l = torch.where(z_l.abs() > eps_gate, h_next / z_l,
                                  torch.ones_like(z_l))
            else:
                D_l = torch.ones_like(z_l)
            D_list.append(D_l)

        # Work on a detached copy of weights so ALS state (including optional
        # non-zero Delta init) is represented as Delta variables during sweeps,
        # then committed once at the end.
        W_work: List[torch.Tensor] = [p.data.clone() for p in params_list]
        W_base: List[torch.Tensor] = [w.clone() for w in W_work]
        solver_dW_accum: List[float] = [0.0 for _ in range(L)]

        def _operator_from_weights(Ws: List[torch.Tensor]) -> torch.Tensor:
            """Per-sample operator P(W) under frozen ReLU secant gates."""
            P = Ws[0].unsqueeze(0).expand(Bs, -1, -1)
            for l in range(1, L):
                W_D = Ws[l].unsqueeze(0) * D_list[l - 1].unsqueeze(1)
                P = torch.bmm(W_D, P)
            return P

        # ---- Build A_list from current working weights ----
        def _build_A() -> List[torch.Tensor]:
            A: List[Optional[torch.Tensor]] = [None] * L
            A[L - 1] = (torch.eye(D_out, device=device, dtype=dtype)
                        .unsqueeze(0).expand(Bs, -1, -1))
            for l in range(L - 2, -1, -1):
                W_D = W_work[l + 1].unsqueeze(0) * D_list[l].unsqueeze(1)
                A[l] = torch.bmm(A[l + 1], W_D)
            return A  # type: ignore[return-value]

        # P_init from base weights (before optional non-zero ΔW init).
        P_init = _operator_from_weights(W_work)              # (Bs, D_out, D_in)

        # P_target^b = P_init^b - lr * dP*^b  (gradient descent in operator space)
        P_target = P_init - lr * dP_star                     # (Bs, D_out, D_in)

        als_reverse = bool(group.get("als_reverse_sweep", False))
        layer_solve = group.get("als_layer_solve", "mn")
        init_delta_orth = bool(group.get("als_init_delta_orthogonal", False))
        init_delta_scale = group.get("als_init_delta_scale", None)
        gateperm_warmstart = bool(group.get("als_gateperm_warmstart", True))
        gateperm_rel_threshold = float(group.get("als_gateperm_rel_threshold", 0.5))
        gateperm_no_batch_mean = bool(
            group.get("als_gateperm_warmstart_no_batch_mean", False)
        )
        gateperm_once = bool(group.get("als_gateperm_warmstart_once", True))
        lam_anchor_post_warmstart = bool(group.get("als_lam_anchor_post_warmstart", False))
        if layer_solve not in ("mn", "exact", "per_sample"):
            raise ValueError(
                "als_layer_solve must be 'mn', 'exact', or 'per_sample'; "
                f"got {layer_solve!r}"
            )

        if init_delta_orth:
            delta_scale = float(init_delta_scale) if init_delta_scale is not None else (1.0 / float(L))
            for k in range(L):
                dW0 = torch.empty_like(W_work[k])
                nn.init.orthogonal_(dW0)
                W_work[k].add_(dW0, alpha=delta_scale)
        self._als_last_warmstart_applied = False
        self._als_last_warmstart_dW_mean = 0.0
        self._als_last_warmstart_dW_max = 0.0
        self._als_last_solver_dW_mean = 0.0
        self._als_last_solver_dW_max = 0.0
        self._als_sweep_residuals = []
        can_warmstart = gateperm_warmstart and float(self._als_prev_rel_jac) > gateperm_rel_threshold
        if gateperm_once and self._als_warmstart_used_once:
            can_warmstart = False
        if can_warmstart:
            # One-shot gate-permutation warmstart in weight space:
            # W_k <- mean_b O_k^b, where O_k^b routes active in->out coordinates.
            for k in range(L):
                d_out_k, d_in_k = int(W_work[k].shape[0]), int(W_work[k].shape[1])
                O_sum = torch.zeros_like(W_work[k])
                for b in range(Bs):
                    if k > 0:
                        g_in = D_list[k - 1][b]
                        active_in = [j for j in range(d_in_k) if float(g_in[j].item()) > 0.5]
                    else:
                        active_in = list(range(d_in_k))
                    if k < L - 1:
                        g_out = D_list[k][b]
                        active_out = [i for i in range(d_out_k) if float(g_out[i].item()) > 0.5]
                    else:
                        active_out = list(range(d_out_k))
                    m = min(len(active_in), len(active_out))
                    mapping = {active_in[i]: active_out[i] for i in range(m)}
                    used_out = set(mapping.values())
                    rem_src = [j for j in range(d_in_k) if j not in mapping]
                    rem_tgt = [i for i in range(d_out_k) if i not in used_out]
                    if not rem_tgt:
                        rem_tgt = list(range(d_out_k))
                    for t_idx, src in enumerate(rem_src):
                        mapping[src] = rem_tgt[t_idx % len(rem_tgt)]
                    O_b = torch.zeros_like(W_work[k])
                    for src in range(d_in_k):
                        O_b[mapping[src], src] = 1.0
                    O_sum.add_(O_b)
                if gateperm_no_batch_mean:
                    W_work[k].copy_(O_sum)
                else:
                    W_work[k].copy_(O_sum / float(max(Bs, 1)))
            self._als_last_warmstart_applied = True
            if gateperm_once:
                self._als_warmstart_used_once = True
            if lam_anchor_post_warmstart:
                # Re-anchor lambda at post-warmstart weights so warmstart itself
                # is not penalized by the centered regularization term.
                W_base = [w.clone() for w in W_work]
            with torch.no_grad():
                _ws_norms = [(W_work[k] - W_base[k]).norm().item() for k in range(L)]
                self._als_last_warmstart_dW_mean = float(sum(_ws_norms) / max(len(_ws_norms), 1))
                self._als_last_warmstart_dW_max = float(max(_ws_norms) if _ws_norms else 0.0)
        # Contexts for the first sweep reflect the actual sweep start state.
        A_list = _build_A()

        def _snapshot_B_prefixes() -> List[Optional[torch.Tensor]]:
            """B_frozen[k] = left context before layer k (None iff k == 0).

            O(L) total — same recurrence as the forward sweep's incremental ``B_curr``,
            but materialised for all k at once.  Used for reverse ALS with frozen B.
            """
            B_frozen: List[Optional[torch.Tensor]] = [None] * L
            B_acc: Optional[torch.Tensor] = None
            for k in range(L):
                B_frozen[k] = B_acc
                if k < L - 1:
                    D_W = D_list[k].unsqueeze(-1) * W_work[k].unsqueeze(0)
                    if B_acc is None:
                        B_acc = D_W
                    else:
                        B_acc = torch.bmm(D_W, B_acc)
            return B_frozen

        def _als_layer_delta(
            k: int,
            A_k: torch.Tensor,
            B_curr: Optional[torch.Tensor],
            R: torch.Tensor,
            cond_scale: float,
        ) -> torch.Tensor:
            # ---- M_k = (1/Bs) Σ_b A_k^T A_k ----
            M_k: Optional[torch.Tensor] = None
            if k < L - 1:
                d_k = A_k.shape[2]
                A_2d = A_k.reshape(Bs * D_out, d_k)
                M_k = A_2d.T @ A_2d / Bs

            # ---- N_k = (1/Bs) Σ_b B_k B_k^T ----
            N_k: Optional[torch.Tensor] = None
            if B_curr is not None:
                d_km1, D_in_b = B_curr.shape[1], B_curr.shape[2]
                B_2d = B_curr.permute(1, 0, 2).reshape(d_km1, Bs * D_in_b)
                N_k = B_2d @ B_2d.T / Bs

            def _condition_cov(C: Optional[torch.Tensor], scale: float) -> Optional[torch.Tensor]:
                if C is None or scale <= 0.0:
                    return C
                sigma_max = float(torch.linalg.matrix_norm(C, ord=2).item())
                if sigma_max <= 0.0 or not torch.isfinite(C).all():
                    return C
                return C + (scale * sigma_max) * torch.eye(
                    C.shape[0], device=C.device, dtype=C.dtype
                )

            M_k = _condition_cov(M_k, cond_scale)
            N_k = _condition_cov(N_k, cond_scale)

            # ---- G_k = (1/Bs) Σ_b A_k^T R^b B_k^T ----
            if B_curr is None:
                G_k = torch.bmm(A_k.transpose(1, 2), R).mean(0)
            elif k == L - 1:
                G_k = torch.bmm(R, B_curr.transpose(1, 2)).mean(0)
            else:
                AtR = torch.bmm(A_k.transpose(1, 2), R)
                G_k = torch.bmm(AtR, B_curr.transpose(1, 2)).mean(0)
            # Old behaviour: regularise ||ΔW||² from current position (delta_anchor=0).
            # New behaviour: regularise ||W_new - W_base||² (anchor fixed pre-warmstart).
            if bool(group.get("als_lam_regularize_delta", False)):
                delta_anchor = torch.zeros_like(W_work[k])
            else:
                delta_anchor = W_work[k] - W_base[k]
            lam_eff = _als_effective_lam(M_k, N_k, lam, lam_alpha)
            G_k_reg = G_k - lam_eff * delta_anchor

            if layer_solve == "mn":
                return _als_dW_from_mn(M_k, N_k, G_k_reg, lam_eff, lam_alpha)
            if layer_solve == "exact":
                d_flat = G_k.numel()
                if d_flat > _MAX_ALS_EXACT_FLAT_DIM:
                    key = (k, d_flat)
                    if key not in self._als_exact_fallback_warned:
                        self._als_exact_fallback_warned.add(key)
                        warnings.warn(
                            f"ALS als_layer_solve='exact': layer {k} flattened dim "
                            f"{d_flat} exceeds {_MAX_ALS_EXACT_FLAT_DIM}; using "
                            f"batch-averaged M,N (mn) for this layer.",
                            UserWarning,
                            stacklevel=1,
                        )
                    return _als_dW_from_mn(M_k, N_k, G_k_reg, lam_eff, lam_alpha)
                return _als_exact_ls_delta(
                    A_k, B_curr, G_k_reg, k, L, D_out, lam_eff, dtype,
                )
            return _als_per_sample_delta(
                A_k, B_curr, R, delta_anchor, k, L, D_out, lam, lam_alpha, dtype,
            )

        _sweep_idx = 0
        _best_sw = float("inf")
        _no_imp = 0
        while _sweep_idx < _sweep_limit:
            # Annealed context conditioning: 1.0 → 0.0 linearly over the sweep limit.
            # In adaptive mode (threshold set) always use 1.0 to keep conditioning stable.
            if residual_threshold is not None:
                cond_scale = 1.0
            else:
                cond_scale = 1.0 if _sweep_limit <= 1 else 1.0 - (float(_sweep_idx) / float(_sweep_limit - 1))
            if als_reverse:
                # Mirror of forward: there A is frozen at sweep start and B is built
                # live; here B is frozen at sweep start and A is extended live from the
                # right after each layer update.  Both are O(L) matmuls per sweep.
                B_frozen = _snapshot_B_prefixes()
                A_curr = (
                    torch.eye(D_out, device=device, dtype=dtype)
                    .unsqueeze(0)
                    .expand(Bs, -1, -1)
                )
                for k in range(L - 1, -1, -1):
                    A_k = A_curr
                    B_curr = B_frozen[k]

                    if B_curr is None:
                        P_curr = torch.bmm(
                            A_k,
                            W_work[0].unsqueeze(0).expand(Bs, -1, -1),
                        )
                    else:
                        WB = torch.einsum("ij,bjd->bid", W_work[k], B_curr)
                        P_curr = torch.bmm(A_k, WB)

                    R = P_target - P_curr
                    dW = _als_layer_delta(k, A_k, B_curr, R, cond_scale)
                    W_work[k].add_(dW)
                    solver_dW_accum[k] += float(dW.norm().item())

                    if k > 0:
                        W_D = (
                            W_work[k].unsqueeze(0) * D_list[k - 1].unsqueeze(1)
                        )
                        A_curr = torch.bmm(A_k, W_D)
            else:
                if _sweep_idx > 0:
                    A_list = _build_A()

                B_curr: Optional[torch.Tensor] = None

                for k in range(L):
                    A_k = A_list[k]

                    if B_curr is None:
                        P_curr = torch.bmm(
                            A_k,
                            W_work[0].unsqueeze(0).expand(Bs, -1, -1),
                        )
                    else:
                        WB = torch.einsum("ij,bjd->bid", W_work[k], B_curr)
                        P_curr = torch.bmm(A_k, WB)

                    R = P_target - P_curr
                    dW = _als_layer_delta(k, A_k, B_curr, R, cond_scale)
                    W_work[k].add_(dW)
                    solver_dW_accum[k] += float(dW.norm().item())

                    if k < L - 1:
                        D_W = D_list[k].unsqueeze(-1) * W_work[k].unsqueeze(0)
                        if B_curr is None:
                            B_curr = D_W
                        else:
                            B_curr = torch.bmm(D_W, B_curr)

            with torch.no_grad():
                _P_sw = _operator_from_weights(W_work)
                _n = (_P_sw - P_target).norm(dim=(1, 2))
                _d = P_target.norm(dim=(1, 2)).clamp_min(1e-30)
                _sw_res = float((_n / _d).mean().item())
                self._als_sweep_residuals.append(_sw_res)
            _sweep_idx += 1
            if residual_threshold is not None and _sw_res < residual_threshold:
                break
            if early_stop_patience > 0 and early_stop_rel_tol > 0.0:
                if _best_sw == float("inf"):
                    rel_imp = float("inf")
                else:
                    rel_imp = (_best_sw - _sw_res) / max(_best_sw, 1e-30)
                if _sw_res < _best_sw:
                    _best_sw = _sw_res
                if rel_imp < early_stop_rel_tol:
                    _no_imp += 1
                else:
                    _no_imp = 0
                if _no_imp >= early_stop_patience:
                    break

        self._als_num_sweeps_used = _sweep_idx
        self._als_last_solver_dW_mean = float(sum(solver_dW_accum) / max(len(solver_dW_accum), 1))
        self._als_last_solver_dW_max = float(max(solver_dW_accum) if solver_dW_accum else 0.0)

        for p, w_new in zip(params_list, W_work):
            p.data.copy_(w_new)
        with torch.no_grad():
            P_final = _operator_from_weights(W_work)
            num = (P_final - P_target).norm(dim=(1, 2))
            den = P_target.norm(dim=(1, 2)).clamp_min(1e-30)
            self._als_prev_rel_jac = float((num / den).mean().item())

        self.activations.clear()
        self.pre_acts.clear()
        self.grad_acts.clear()

    # ------------------------------------------------------------------
    # block_diagonal: per-layer Sylvester solve
    # ------------------------------------------------------------------

    def _step_block_diagonal(self, group: Dict[str, Any]):
        """Per-layer Sylvester equation solved via eigendecomposition.

        For each layer l, solves:
            M_l dW N_l + λ dW = G_{W_l}
        where
            M_l = (1/B) Σ_b A_l(x^b)ᵀ A_l(x^b)   (d_l_out × d_l_out)
            N_l = (1/B) Σ_b B_l(x^b) B_l(x^b)ᵀ   (d_l_in  × d_l_in)
            G_{W_l} = p.grad  (= standard backprop gradient, = A_l^T G_P B_l^T)

        This is exact for the batch at hand; it is NOT the K-FAC approximation
        (which would instead use EMA statistics and a factored inverse).
        """
        if not self.activations:
            raise RuntimeError(
                "attach_hooks(model) must be called before training when using "
                "approximation='block_diagonal'."
            )

        lr = group["lr"]
        lam = group["lam"]
        lam_alpha = group["lam_alpha"]
        beta = group["momentum"]
        wd = group["weight_decay"]
        use_ns = group["use_ns"]
        ns_steps = group["ns_steps"]
        rank = group["rank"]
        ema_decay = group.get("ema_decay", 0.0)

        params_list = [p for p in group["params"] if p.grad is not None and p.ndim == 2]
        if not params_list:
            return

        device = params_list[0].device
        dtype = params_list[0].dtype
        batch_size = next(iter(self.activations.values())).shape[0]

        grad_list = [
            p.grad.clone().add_(p, alpha=wd) if wd else p.grad.clone()
            for p in params_list
        ]

        M_list, N_list = self._compute_covariances(params_list, batch_size, device, dtype)

        # EMA smoothing of context matrices (stabilises adaptive λ across batches).
        # Only non-None entries are smoothed; None means identity (first/last layer).
        if ema_decay > 0.0:
            for l, p in enumerate(params_list):
                state = self.state[p]
                if M_list[l] is not None:
                    if "ema_M" not in state:
                        state["ema_M"] = M_list[l].clone()
                    else:
                        state["ema_M"].mul_(ema_decay).add_(M_list[l], alpha=1.0 - ema_decay)
                    M_list[l] = state["ema_M"]
                if N_list[l] is not None:
                    if "ema_N" not in state:
                        state["ema_N"] = N_list[l].clone()
                    else:
                        state["ema_N"].mul_(ema_decay).add_(N_list[l], alpha=1.0 - ema_decay)
                    N_list[l] = state["ema_N"]

        dW_raw_list = []
        for l, p in enumerate(params_list):
            M = M_list[l]   # None means identity (last layer)
            N = N_list[l]   # None means identity (first layer)
            G = grad_list[l]

            # Optional: rank-r truncation of gradient (before solve).
            # rank(G_l) <= D_out always; truncation filters operator directions.
            if rank is not None:
                U, s, Vt = torch.linalg.svd(G, full_matrices=False)
                r = min(rank, s.shape[0])
                G = (U[:, :r] * s[:r]) @ Vt[:r]

            if M is None and N is None:
                # Single layer network: both contexts are identity (eigenvalues = 1).
                # Adaptive: λ = α * (1 + 1) = 2α; fixed: λ as given.
                lam_l = lam_alpha * 2.0 if lam is None else lam
                dW_raw = G / (1.0 + lam_l)
            elif M is None:
                # Last layer: A=I → M eigenvalues are all 1; N has real eigenvalues.
                if lam is None:
                    sv_N = torch.linalg.svdvals(N)
                    lam_l = max(lam_alpha * (1.0 + sv_N[0].item()), 1e-30)
                else:
                    lam_l = lam
                dW_raw = torch.linalg.solve(
                    N.add(torch.eye(N.shape[0], device=device, dtype=dtype), alpha=lam_l),
                    G.T,
                ).T
            elif N is None:
                # First layer: B=I → N eigenvalues are all 1; M has real eigenvalues.
                if lam is None:
                    sv_M = torch.linalg.svdvals(M)
                    lam_l = max(lam_alpha * (sv_M[0].item() + 1.0), 1e-30)
                else:
                    lam_l = lam
                dW_raw = torch.linalg.solve(
                    M.add(torch.eye(M.shape[0], device=device, dtype=dtype), alpha=lam_l),
                    G,
                )
            else:
                dW_raw = _solve_sylvester(M, N, G, lam, lam_alpha)

            dW_raw_list.append(dW_raw)

        # Nesterov momentum on the Sylvester-solve outputs, then NS normalization.
        # Order: solve → momentum → NS → rescale → update.
        # NS is applied last so it normalizes the momentum-averaged natural gradient,
        # keeping the update scale-invariant (analogous to Muon) regardless of solve
        # amplification. The momentum buffer accumulates solve outputs; NS at the end
        # ensures the actual weight update is always well-scaled.
        dW_list = self._nesterov_grads(params_list, dW_raw_list, beta)
        if use_ns:
            dW_list = [_newton_schulz(dW, steps=ns_steps) for dW in dW_list]
        if group["rescale_lr"]:
            dW_list = self._rescale_dW(dW_list, group.get("rescale_variant", "max"))
        for p, dW in zip(params_list, dW_list):
            p.add_(dW, alpha=-lr)

        self.activations.clear()
        self.pre_acts.clear()
        self.grad_acts.clear()

    # ------------------------------------------------------------------
    # forward_pass: rank-1 context approximation, no context propagation
    # ------------------------------------------------------------------

    def _compute_fp_covariances(
        self,
        params_list: List[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], List[torch.Tensor]]:
        """Compute M_l^FP, N_l^FP, G_l^FP from forward-pass quantities only.

        Approximations (consistent rank-1 per sample, full-rank after batching):
            Â_l(x) = f(x) x_l^T / ‖x_l‖²   →  M_l^FP = E[‖f‖²/‖x_l‖⁴ · x_l x_lᵀ]
            B̂_l(x) = h_{l-1} x^T / ‖x‖²    →  N_l^FP = E[h_{l-1} h_{l-1}ᵀ / ‖x‖²]
            G_l^FP  = E[(f·∇_f L / ‖x_l‖²) · x_l h_{l-1}ᵀ]   (l < L-1)
            G_{L-1} = E[∇_f L · h_{L-2}ᵀ]                      (exact: A_{L-1} = I)

        Requires:
            activations[l]   — x_l = W_l h_{l-1}  (output of linear layer l)
            pre_acts[l]      — h_{l-1}             (input to linear layer l)
            activations[L-1] — f(x)                (network output = logits)
            grad_acts[L-1]   — ∇_f L               (output-level gradient only)

        Special cases (exact, no approximation):
            M[L-1] = None  (A_{L-1} = I exactly → M_{L-1} = I)
            N[0]   = None  (B_0     = I exactly → N_0     = I)

        Returns:
            M_list: list of (d_l, d_l) tensors or None.
            N_list: list of (d_{l-1}, d_{l-1}) tensors or None.
            G_list: list of (d_l, d_{l-1}) tensors.
        """
        L = len(params_list)
        M_list: List[Optional[torch.Tensor]] = [None] * L
        N_list: List[Optional[torch.Tensor]] = [None] * L
        G_list: List[torch.Tensor] = []

        f_x   = self.activations[L - 1].to(dtype)   # (B, D_out)
        del_f = self.grad_acts[L - 1].to(dtype)      # (B, D_out)
        x_in  = self.pre_acts[0].to(dtype)           # (B, D_in)

        x_in_norm_sq = (x_in * x_in).sum(1).clamp(min=1e-30)  # (B,)

        for l in range(L):
            x_l    = self.activations[l].to(dtype)   # (B, d_l)
            h_prev = self.pre_acts[l].to(dtype)       # (B, d_{l-1})

            xl_norm_sq = (x_l * x_l).sum(1).clamp(min=1e-30)   # (B,)
            f_norm_sq  = (f_x * f_x).sum(1)                     # (B,)

            if l == L - 1:
                # Last layer: A_{L-1} = I exactly, so δ_{L-1} = ∇_f L exactly.
                # Use the true gradient directly — no rank-1 projection needed.
                G_list.append(del_f.T @ h_prev / batch_size)
            else:
                # Intermediate layers: approximate δ_l ≈ Â_l^T ∇_f L
                #   = (z_l f^T / ‖z_l‖²)^T ∇_f L = (f · ∇_f L) / ‖z_l‖² · z_l
                scalar_G = (f_x * del_f).sum(1) / xl_norm_sq    # (B,)
                G_list.append((x_l * scalar_G[:, None]).T @ h_prev / batch_size)

            # M_l^FP = E[‖f‖²/‖x_l‖⁴ · x_l x_l^T]   (skip last: A_{L-1} = I)
            if l < L - 1:
                w_M = f_norm_sq / xl_norm_sq.pow(2)
                M_list[l] = (x_l * w_M[:, None]).T @ x_l / batch_size

            # N_l^FP = E[h_{l-1} h_{l-1}^T / ‖x_in‖²]   (skip first: B_0 = I)
            if l > 0:
                N_list[l] = (h_prev / x_in_norm_sq[:, None]).T @ h_prev / batch_size

        return M_list, N_list, G_list

    def _step_forward_pass(self, group: Dict[str, Any]):
        """Operator-FP: Sylvester solve with rank-1 forward-pass context approximation.

        Identical pipeline to block_diagonal with two changes:
          1. M_l, N_l, G_l come from _compute_fp_covariances (no context propagation).
          2. p.grad is not used; G_l^FP is the RHS of the Sylvester equation.

        For truly forward-pass-only operation (no loss.backward()):
          - Run model(x) under torch.no_grad() so hooks capture activations.
          - Compute ∇_f L analytically and assign to optimizer.grad_acts[L-1].
          - Call optimizer.step() directly; p.grad will be None for all layers.
        """
        if not self.activations:
            raise RuntimeError(
                "attach_hooks(model) must be called before training when using "
                "approximation='forward_pass'."
            )
        if not self.grad_acts:
            raise RuntimeError(
                "grad_acts[L-1] must be populated before optimizer.step(). "
                "Either call loss.backward() or assign ∇_f L manually to "
                "optimizer.grad_acts[L-1] for forward-pass-only operation."
            )

        lr       = group["lr"]
        lam      = group["lam"]
        lam_alpha = group["lam_alpha"]
        beta     = group["momentum"]
        use_ns   = group["use_ns"]
        ns_steps = group["ns_steps"]

        # Collect 2D params; p.grad may be None in forward-pass-only mode.
        params_list = [p for p in group["params"] if p.ndim == 2]
        if not params_list:
            return

        device     = params_list[0].device
        dtype      = params_list[0].dtype
        batch_size = next(iter(self.activations.values())).shape[0]

        M_list, N_list, G_list = self._compute_fp_covariances(
            params_list, batch_size, device, dtype
        )

        dW_raw_list = []
        for l, p in enumerate(params_list):
            M = M_list[l]
            N = N_list[l]
            G = G_list[l]

            if M is None and N is None:
                lam_l = lam_alpha * 2.0 if lam is None else lam
                dW_raw = G / (1.0 + lam_l)
            elif M is None:
                if lam is None:
                    sv_N = torch.linalg.svdvals(N)
                    lam_l = max(lam_alpha * (1.0 + sv_N[0].item()), 1e-30)
                else:
                    lam_l = lam
                dW_raw = torch.linalg.solve(
                    N.add(torch.eye(N.shape[0], device=device, dtype=dtype), alpha=lam_l),
                    G.T,
                ).T
            elif N is None:
                if lam is None:
                    sv_M = torch.linalg.svdvals(M)
                    lam_l = max(lam_alpha * (sv_M[0].item() + 1.0), 1e-30)
                else:
                    lam_l = lam
                dW_raw = torch.linalg.solve(
                    M.add(torch.eye(M.shape[0], device=device, dtype=dtype), alpha=lam_l),
                    G,
                )
            else:
                dW_raw = _solve_sylvester(M, N, G, lam, lam_alpha)

            dW_raw_list.append(dW_raw)

        dW_list = self._nesterov_grads(params_list, dW_raw_list, beta)
        if use_ns:
            dW_list = [_newton_schulz(dW, steps=ns_steps) for dW in dW_list]
        if group["rescale_lr"]:
            dW_list = self._rescale_dW(dW_list, group.get("rescale_variant", "max"))
        for p, dW in zip(params_list, dW_list):
            p.add_(dW, alpha=-lr)

        self.activations.clear()
        self.pre_acts.clear()
        self.grad_acts.clear()

    # ------------------------------------------------------------------
    # fp_ps: per-sample closed-form scalar solve, O(B·d) per layer
    # ------------------------------------------------------------------

    def _step_fp_ps(self, group: Dict[str, Any]):
        """Operator-FP per-sample: scalar closed-form Sylvester solution, no SVD.

        Exploits the rank-1 per-sample structure of FP context matrices to solve
        the Sylvester equation analytically per sample, then averages updates.

        Intermediate layers (l < L-1) — rank-1 M^b and N^b:
            c^b = [(f·∇L)/‖z_l‖²] / [‖f‖²‖h_{l-1}‖²/(‖z_l‖²‖x_in‖²) + λ]
            ΔW_l^b = c^b · z_l^b (h_{l-1}^b)ᵀ

        Last layer (l = L-1) — rank-1 N^b, exact gradient:
            a^b = ∇L^b / (‖h_{L-2}‖²/‖x_in‖² + λ)
            ΔW_{L-1}^b = a^b (h_{L-2}^b)ᵀ

        Batch average: ΔW_l = (1/B) Σ_b ΔW_l^b

        Complexity: O(B·d_l) per layer — same order as the forward pass.
        Requires fixed λ (no SVD → no σ_max for adaptive λ).
        """
        if not self.activations:
            raise RuntimeError(
                "attach_hooks(model) must be called before training when using "
                "approximation='fp_ps'."
            )
        if not self.grad_acts:
            raise RuntimeError(
                "grad_acts[L-1] must be set before step(). Assign ∇_f L to "
                "optimizer.grad_acts[L-1] before calling step()."
            )

        lr      = group["lr"]
        lam     = group["lam"]
        beta    = group["momentum"]
        use_ns  = group["use_ns"]
        ns_steps = group["ns_steps"]

        if lam is None:
            raise ValueError(
                "fp_ps requires a fixed lam (lam is None → adaptive λ not supported "
                "without per-sample σ_max; pass --lam explicitly)."
            )

        params_list = [p for p in group["params"] if p.ndim == 2]
        if not params_list:
            return

        device     = params_list[0].device
        dtype      = params_list[0].dtype
        L          = len(params_list)
        batch_size = next(iter(self.activations.values())).shape[0]

        f_x   = self.activations[L - 1].to(dtype)   # (B, D_out)
        del_f = self.grad_acts[L - 1].to(dtype)      # (B, D_out)
        x_in  = self.pre_acts[0].to(dtype)           # (B, D_in)

        x_in_norm_sq = (x_in * x_in).sum(1).clamp(min=1e-30)   # (B,)
        f_norm_sq    = (f_x * f_x).sum(1)                       # (B,)

        dW_raw_list = []
        for l, p in enumerate(params_list):
            x_l    = self.activations[l].to(dtype)   # (B, d_l)
            h_prev = self.pre_acts[l].to(dtype)       # (B, d_{l-1})

            h_norm_sq  = (h_prev * h_prev).sum(1).clamp(min=1e-30)  # (B,)
            xl_norm_sq = (x_l * x_l).sum(1).clamp(min=1e-30)        # (B,)

            if l == L - 1:
                # Last layer: A_{L-1} = I exactly; rank-1 N^b per sample.
                # a^b = ∇L^b / (‖h_{L-2}‖²/‖x_in‖² + λ)
                denom = h_norm_sq / x_in_norm_sq + lam              # (B,)
                dW_raw = (del_f / denom[:, None]).T @ h_prev / batch_size
            else:
                # Intermediate: rank-1 M^b and N^b.
                # μ_MN = ‖f‖²‖h‖² / (‖z_l‖²‖x_in‖²)
                # c^b  = scalar_G^b / (μ_MN^b + λ)
                scalar_G = (f_x * del_f).sum(1) / xl_norm_sq        # (B,)
                mu_mn    = f_norm_sq * h_norm_sq / (xl_norm_sq * x_in_norm_sq)  # (B,)
                c        = scalar_G / (mu_mn + lam)                  # (B,)
                dW_raw   = (x_l * c[:, None]).T @ h_prev / batch_size

            dW_raw_list.append(dW_raw)

        dW_list = self._nesterov_grads(params_list, dW_raw_list, beta)
        if use_ns:
            dW_list = [_newton_schulz(dW, steps=ns_steps) for dW in dW_list]
        if group["rescale_lr"]:
            dW_list = self._rescale_dW(dW_list, group.get("rescale_variant", "max"))
        for p, dW in zip(params_list, dW_list):
            p.add_(dW, alpha=-lr)

        self.activations.clear()
        self.pre_acts.clear()
        self.grad_acts.clear()

    # ------------------------------------------------------------------
    # fpc: exact gradient + rank-1 forward-pass curvature, O(d²) solve
    # ------------------------------------------------------------------

    def _step_fpc(self, group: Dict[str, Any]):
        """Operator-FPC: exact backprop gradient with rank-1 forward-pass curvature.

        Uses p.grad (exact gradient from loss.backward()) for direction, but replaces
        the full-rank M_l, N_l eigendecomposition with rank-1 context vectors computed
        from forward activations — giving a closed-form O(d²) Tikhonov solve, no SVD.

        Context construction (batch-averaged rank-1 approximation):
            u_l = E_b[‖f^b‖/‖x_l^b‖² · x_l^b]  (output direction, normalised)
            μ_l = E_b[‖f^b‖²/‖x_l^b‖²]           (output curvature scalar)
            v_l = E_b[h_{l-1}^b/‖x_in^b‖]          (input direction, normalised)
            ν_l = E_b[‖h_{l-1}^b‖²/‖x_in^b‖²]      (input curvature scalar)

        Closed-form rank-1 Sylvester solve (M+λI)ΔW(N+λI) = G, §8 eq. (fpc-solve):
            ΔW = (1/λ²)[G
                 − μ/(μ+λ)       u(u^T G)
                 − ν/(ν+λ)       (Gv)v^T
                 + μν/((μ+λ)(ν+λ)) u(u^T G v)v^T]

        Last layer (M_{L-1} = I exactly, one-sided solve):
            ΔW = (1/((1+λ)λ))[G − ν/(ν+λ) (Gv)v^T]

        Requires fixed lam (no SVD → no σ_max for adaptive λ).
        Training loop: use standard backward pass (loss.backward()), not fp_mode.
        Hooks must be attached for activations/pre_acts.
        """
        if not self.activations:
            raise RuntimeError(
                "attach_hooks(model) must be called before training when using "
                "approximation='fpc'."
            )

        lr       = group["lr"]
        lam      = group["lam"]
        beta     = group["momentum"]
        use_ns   = group["use_ns"]
        ns_steps = group["ns_steps"]

        if lam is None:
            raise ValueError(
                "fpc requires a fixed lam (adaptive λ not supported without SVD; "
                "pass --lam explicitly)."
            )

        params_list = [p for p in group["params"] if p.ndim == 2]
        if not params_list:
            return

        device     = params_list[0].device
        dtype      = params_list[0].dtype
        L          = len(params_list)
        batch_size = next(iter(self.activations.values())).shape[0]

        f_x  = self.activations[L - 1].to(dtype)  # (B, D_out) — network output
        x_in = self.pre_acts[0].to(dtype)          # (B, D_in)  — network input
        x_in_norm_sq = (x_in * x_in).sum(1).clamp(min=1e-30)  # (B,)

        dW_raw_list = []
        for l, p in enumerate(params_list):
            G      = p.grad.to(dtype)              # exact gradient (d_out, d_in)
            x_l    = self.activations[l].to(dtype) # (B, d_l)
            h_prev = self.pre_acts[l].to(dtype)    # (B, d_{l-1})

            xl_norm_sq = (x_l    * x_l   ).sum(1).clamp(min=1e-30)  # (B,)
            h_norm_sq  = (h_prev * h_prev).sum(1).clamp(min=1e-30)   # (B,)

            # --- Input-side rank-1 context (all layers) ---
            # v = normalised mean of h_prev / ‖x_in‖
            v_raw = (h_prev / x_in_norm_sq.sqrt().unsqueeze(1)).mean(0)  # (d_{l-1},)
            v_norm = v_raw.norm().clamp(min=1e-12)
            v = v_raw / v_norm                                             # unit (d_{l-1},)
            nu = (h_norm_sq / x_in_norm_sq).mean()                        # scalar

            if l == L - 1:
                # M_{L-1} = I exactly (A_{L-1} = I); one-sided solve:
                #   ΔW = (1/((1+λ)λ))[G − ν/(ν+λ) (Gv)v^T]
                Gv = G @ v                                  # (d_out,)
                dW_raw = (G - (nu / (nu + lam)) * torch.outer(Gv, v)) / ((1.0 + lam) * lam)
            else:
                # Rank-1 output context
                # u = normalised mean of ‖f‖/‖x_l‖² · x_l
                f_norm_sq = (f_x * f_x).sum(1)             # (B,)
                scale = (f_norm_sq / xl_norm_sq).sqrt()    # ‖f‖/‖x_l‖, (B,)  [= sqrt(μ^b)]
                u_raw = (x_l * scale.unsqueeze(1)).mean(0) # (d_l,)
                u_norm = u_raw.norm().clamp(min=1e-12)
                u = u_raw / u_norm                         # unit (d_l,)
                mu = (f_norm_sq / xl_norm_sq).mean()       # scalar E[‖f‖²/‖x_l‖²]

                # Closed-form rank-1 Sylvester solve
                uTG   = u @ G                              # (d_{l-1},)
                Gv    = G @ v                              # (d_l,)
                uTGv  = u @ Gv                             # scalar

                mu_c  = mu / (mu + lam)
                nu_c  = nu / (nu + lam)
                mn_c  = mu_c * nu_c / (1.0 - nu_c + 1e-30) * (1.0 - nu_c)
                # Simpler: mn_c = μν / ((μ+λ)(ν+λ))
                mn_c  = (mu * nu) / ((mu + lam) * (nu + lam))

                dW_raw = (
                    G
                    - mu_c * torch.outer(u, uTG)
                    - nu_c * torch.outer(Gv, v)
                    + mn_c * uTGv * torch.outer(u, v)
                ) / (lam * lam)

            dW_raw_list.append(dW_raw)

        dW_list = self._nesterov_grads(params_list, dW_raw_list, beta)
        if use_ns:
            dW_list = [_newton_schulz(dW, steps=ns_steps) for dW in dW_list]
        if group["rescale_lr"]:
            dW_list = self._rescale_dW(dW_list, group.get("rescale_variant", "max"))
        for p, dW in zip(params_list, dW_list):
            p.add_(dW, alpha=-lr)

        self.activations.clear()
        self.pre_acts.clear()
        self.grad_acts.clear()

    # ------------------------------------------------------------------
    # secant: rank-1 context approximation, exact coupled B×B kernel solve
    # ------------------------------------------------------------------

    def _step_secant(self, group: Dict[str, Any]):
        """Secant-Exact: solve the full coupled normal equations under rank-1
        context approximation via a B×B kernel system (Woodbury identity).

        Each per-sample context is approximated by its rank-1 secant:
            Â_l(x^b) = f(x^b) z_l(x^b)^T / ‖z_l(x^b)‖²
            B̂_l(x^b) = h_{l-1}(x^b) (x^b)^T / ‖x^b‖²

        Both satisfy the secant condition (exact action on the observed sample):
            Â_l z_l^b = f^b  and  B̂_l x^b = h_{l-1}^b.

        Under this substitution the cross-layer coupling collapses to a per-sample
        scalar γ^b = Σ_l z_l^{bT} ΔW_l h_{l-1}^b / ‖z_l^b‖², and the system
        operator S = (1/B) Σ_b w^b r̃^b ⊗ r̃^b  has rank ≤ B in parameter space.

        The Woodbury identity then gives the exact solution:
            ΔW_k = (1/λ) G_k  −  (1/λ) Σ_b α^b w^b r̃_k^b
        where r̃_k^b = z_k^b h_{k-1}^{bT} / ‖z_k^b‖²,  w^b = ‖f^b‖²/‖x^b‖²,
        and α ∈ R^B solves the B×B system
            (λ diag(1/w^b) + K) α = g,
        with kernel  K_{bb'} = Σ_l (z_l^{bT} z_l^{b'}) / (‖z_l^b‖² ‖z_l^{b'}‖²)
                                    · h_{l-1}^{bT} h_{l-1}^{b'}
        and RHS      g^b     = Σ_l z_l^{bT} G_l h_{l-1}^b / ‖z_l^b‖².

        Complexity per step:
            Kernel K : O(B² L n)
            RHS g    : O(B L n²)
            B×B solve: O(B³)          [negligible]
            Reconstruct ΔW: O(B L n²)
        Total: O(B L n²  +  B² L n),  vs O(B L n³) for block_diagonal.

        Requires: attach_hooks(model) and a standard forward+backward pass so
        that pre_acts, activations, grad_acts, and p.grad are all populated.
        """
        if not self.activations:
            raise RuntimeError(
                "attach_hooks(model) must be called before training when using "
                "approximation='secant'."
            )

        lr        = group["lr"]
        lam       = group["lam"]
        lam_alpha = group["lam_alpha"]
        beta      = group["momentum"]
        wd        = group["weight_decay"]
        use_ns    = group["use_ns"]
        ns_steps  = group["ns_steps"]

        if lam is None:
            # Adaptive λ: scale by the mean squared output norm over the batch,
            # divided by the mean squared input norm — same dimensional scaling
            # as the w^b weights, giving a geometry-aware regularisation.
            # Falls back to lam_alpha if norms are zero.
            pass  # computed below after norms are available

        params_list = [p for p in group["params"] if p.grad is not None and p.ndim == 2]
        if not params_list:
            return

        device     = params_list[0].device
        dtype      = params_list[0].dtype
        L          = len(params_list)
        batch_size = next(iter(self.activations.values())).shape[0]
        B          = batch_size

        # ------------------------------------------------------------------
        # Collect per-layer forward-pass quantities.
        # z_l^b  = activations[l]   : (B, n_l)   pre-activation output of layer l
        # h_{l-1}^b = pre_acts[l]   : (B, n_{l-1}) post-activation input to layer l
        # f^b       = activations[L-1]: (B, D_out) network output (last layer output)
        # x^b       = pre_acts[0]   : (B, D_in)  network input
        # ------------------------------------------------------------------
        eps = 1e-12

        Z  = [self.activations[l].to(dtype) for l in range(L)]   # (B, n_l)
        H  = [self.pre_acts[l].to(dtype)    for l in range(L)]   # (B, n_{l-1})
        f  = self.activations[L - 1].to(dtype)                    # (B, D_out)
        x  = self.pre_acts[0].to(dtype)                           # (B, D_in)

        # Squared norms (B,)
        z_norm_sq = [z.pow(2).sum(1).clamp(min=eps) for z in Z]   # ‖z_l^b‖²
        f_norm_sq = f.pow(2).sum(1)                                # ‖f^b‖²
        x_norm_sq = x.pow(2).sum(1).clamp(min=eps)                # ‖x^b‖²

        # Per-sample weights w^b = ‖f^b‖² / ‖x^b‖²  (B,)
        w = f_norm_sq / x_norm_sq                                  # (B,)

        # Adaptive λ: lam_alpha * mean(w^b) gives a scale matched to the
        # typical operator gain; clamp away from zero.
        if lam is None:
            lam = float(lam_alpha * w.mean().item())
            lam = max(lam, 1e-30)

        # Normalised pre-activations: z̃_l^b = z_l^b / ‖z_l^b‖²  (B, n_l)
        Z_tilde = [Z[l] / z_norm_sq[l].unsqueeze(1) for l in range(L)]

        # ------------------------------------------------------------------
        # Raw gradients (+ optional weight decay)
        # ------------------------------------------------------------------
        G = [
            p.grad.clone().add_(p, alpha=wd) if wd else p.grad.clone()
            for p in params_list
        ]  # list of (n_l, n_{l-1})

        # ------------------------------------------------------------------
        # Step 1 — Kernel  K ∈ R^{B×B}
        #   K_{bb'} = Σ_l  (z_l^{bT} z_l^{b'}) / (‖z_l^b‖² ‖z_l^{b'}‖²)
        #                 · h_{l-1}^{bT} h_{l-1}^{b'}
        #           = Σ_l  (Z̃_l @ Z_l^T) ⊙ (H_l @ H_l^T)
        # where Z̃_l = Z_l / ‖z_l^b‖² and the Hadamard product ⊙ is elementwise.
        # Each inner product matrix is (B, B); cost O(B² n) per layer.
        # ------------------------------------------------------------------
        K = torch.zeros(B, B, device=device, dtype=dtype)
        for l in range(L):
            # (B, B): entry [b, b'] = z̃_l^{bT} z_l^{b'}
            Kz = Z_tilde[l] @ Z[l].T          # (B, B)
            # (B, B): entry [b, b'] = h_{l-1}^{bT} h_{l-1}^{b'}
            Kh = H[l] @ H[l].T                # (B, B)
            K.add_(Kz * Kh)

        # ------------------------------------------------------------------
        # Step 2 — RHS  g ∈ R^B
        #   g^b = Σ_l  z̃_l^{bT} G_l h_{l-1}^b
        #       = Σ_l  sum_j [ (Z̃_l @ G_l @ H_l^T)[b, b] ]   (diagonal)
        # Efficient: compute Z̃_l @ G_l first (B, n_{l-1}), then element-wise
        # multiply with H_l and sum over the feature dimension.
        # Cost: O(B n²) per layer for the matmul, O(B n) for the reduction.
        # ------------------------------------------------------------------
        g_rhs = torch.zeros(B, device=device, dtype=dtype)
        for l in range(L):
            # Z̃_l @ G_l : (B, n_l) @ (n_l, n_{l-1}) → (B, n_{l-1})
            ZtG = Z_tilde[l] @ G[l]            # (B, n_{l-1})
            # element-wise product with H_l, then sum over n_{l-1}
            g_rhs.add_((ZtG * H[l]).sum(1))    # (B,)

        # ------------------------------------------------------------------
        # Step 3 — Solve  (λ diag(1/w^b) + K) α = g
        # The system matrix is symmetric PD (K ⪰ 0, diagonal shift > 0).
        # torch.linalg.solve uses LU; for a B×B matrix this is negligible cost.
        # ------------------------------------------------------------------
        diag_reg = lam / w.clamp(min=eps)                          # (B,)  λ/w^b
        A_sys = K + torch.diag(diag_reg)                           # (B, B)
        alpha = torch.linalg.solve(A_sys, g_rhs)                   # (B,)

        # ------------------------------------------------------------------
        # Step 4 — Reconstruct  ΔW_k = (1/λ) G_k − (1/λ) Σ_b α^b w^b r̃_k^b
        # r̃_k^b = z̃_k^b h_{k-1}^{bT}   so the correction is:
        #   correction_k = Σ_b (α^b w^b) z̃_k^b h_{k-1}^{bT}
        #                = Z̃_k^T diag(α ⊙ w) H_k
        # Shape: (n_k, B) @ (B, n_{k-1}) = (n_k, n_{k-1}).
        # Cost: O(B n²) per layer.
        # ------------------------------------------------------------------
        aw = alpha * w                                              # (B,)  α^b w^b

        dW_raw_list = []
        for l, p in enumerate(params_list):
            # correction: (n_l, B) @ (B, n_{l-1})
            correction = Z_tilde[l].T @ (aw.unsqueeze(1) * H[l])  # (n_l, n_{l-1})
            dW_raw = (G[l] - correction) / lam
            dW_raw_list.append(dW_raw)

        # ------------------------------------------------------------------
        # Nesterov momentum → optional NS → optional rescale → update
        # (same post-processing pipeline as block_diagonal and fpc)
        # ------------------------------------------------------------------
        dW_list = self._nesterov_grads(params_list, dW_raw_list, beta)
        if use_ns:
            dW_list = [_newton_schulz(dW, steps=ns_steps) for dW in dW_list]
        if group["rescale_lr"]:
            dW_list = self._rescale_dW(dW_list, group.get("rescale_variant", "max"))
        for p, dW in zip(params_list, dW_list):
            p.add_(dW, alpha=-lr)

        self.activations.clear()
        self.pre_acts.clear()
        self.grad_acts.clear()

    # ------------------------------------------------------------------
    # secant_r: block-diagonal rank-r via random JVPs
    # ------------------------------------------------------------------

    def _step_secant_r(self, group: Dict[str, Any]):
        """Block-diagonal rank-r: rank-r left context via random JVPs.

        Generalises block_diagonal by replacing the expensive full-rank left
        context Gram M_k (O(B n³ L) Q-recurrence) with a rank-r approximation
        built from r random JVPs through the cached gates (O(r B L n)).

        For each layer k, r fixed random directions Ω_k ∈ R^{n_k × r} (stored
        across steps for a consistent curvature subspace) are propagated forward
        through the gated upper network to give:

            P_k^b = A_k(x^b) Ω_k  ∈ R^{D_out × r},   b = 1,...,B

        using the SAME gate values D_l^b = (z_l^b > 0) from the original forward
        pass (not recomputed), giving a secant-consistent linearisation.

        The rank-r left Gram is:
            M̃_k = (1/B) Σ_b P_k^{bT} P_k^b  ∈ R^{r × r}        (cheap)

        so M̂_k = Ω_k M̃_k Ω_k^T is the rank-r approximation to the full M_k.
        The right Gram is unchanged:
            N̂_k = (1/B) H_k^T H_k  ∈ R^{n_{k-1} × n_{k-1}}

        Substituting M̂_k into the per-layer Sylvester equation and using the
        column-orthonormal Ω_k (Ω_k^T Ω_k = I_r), the ansatz
            ΔW_k = (1/λ) G_k − Ω_k F_k
        reduces to an r × n_{k-1} Sylvester equation for F_k:
            M̃_k F_k N̂_k + λ F_k = (1/λ) M̃_k (Ω_k^T G_k) N̂_k

        solved via _solve_sylvester with the r×r SVD (cost O(r³), negligible).

        Complexity per step:
            JVPs       : O(r B L n)   [r forward passes per layer per sample]
            M̃_k        : O(r² D_out B L)  [r×r Gram from P_k^b]
            N̂_k        : O(n² B L)   [standard activation Gram]
            Sylvester  : O(r³ + n³) per layer  [r³ negligible]
            Reconstruct: O(r n² L)   [Ω_k F_k per layer]
        Dominant: O(n² B L) — same order as the right-Gram in block_diagonal,
        but the left-context cost drops from O(B n³ L) to O(r B n L).
        """
        if not self.activations:
            raise RuntimeError(
                "attach_hooks(model) must be called before training when using "
                "approximation='secant_r'."
            )

        lr        = group["lr"]
        lam       = group["lam"]
        lam_alpha = group["lam_alpha"]
        beta      = group["momentum"]
        wd        = group["weight_decay"]
        use_ns    = group["use_ns"]
        ns_steps  = group["ns_steps"]
        r         = group["rank"] if group["rank"] is not None else 8

        params_list = [p for p in group["params"] if p.grad is not None and p.ndim == 2]
        if not params_list:
            return

        device = params_list[0].device
        dtype  = params_list[0].dtype
        L      = len(params_list)
        B      = next(iter(self.activations.values())).shape[0]

        # ------------------------------------------------------------------
        # Forward-pass quantities
        # Z[l]: (B, n_l)      pre-activation output (post-weight, pre-ReLU)
        # H[l]: (B, n_{l-1})  post-activation input to layer l
        # ------------------------------------------------------------------
        Z = [self.activations[l].to(dtype) for l in range(L)]
        H = [self.pre_acts[l].to(dtype)    for l in range(L)]

        # ReLU gates from the original forward pass — shared by ALL r projections
        Gates = [(Z[l] > 0).to(dtype) for l in range(L)]  # (B, n_l)

        # ------------------------------------------------------------------
        # Raw gradients
        # ------------------------------------------------------------------
        G = [
            p.grad.clone().add_(p, alpha=wd) if wd else p.grad.clone()
            for p in params_list
        ]

        # ------------------------------------------------------------------
        # Random orthonormal projection directions Ω[l] ∈ R^{n_l × r}
        # Initialised once via QR; stored in state for consistency across steps.
        # ------------------------------------------------------------------
        state = self.state[params_list[0]]
        # r_per_layer[l] = min(r, n_l): clips rank to layer width so that
        # Omega[l] can always be made orthonormal via QR (requires n_l >= r_l).
        # Without this clip, layers with n_l < r (e.g. the output layer when
        # r > D_out) get only unit-normalised — not orthonormal — columns,
        # breaking the Omega^T Omega = I assumption in the ansatz derivation.
        r_per_layer = [min(r, p.shape[0]) for p in params_list]
        need_init = (
            "secant_r_Omega" not in state
            or len(state["secant_r_Omega"]) != L
            or any(
                state["secant_r_Omega"][l].shape != (params_list[l].shape[0], r_per_layer[l])
                for l in range(L)
            )
        )
        if need_init:
            Omega_list = []
            for l in range(L):
                n_l = params_list[l].shape[0]
                r_l = r_per_layer[l]          # always <= n_l → QR gives orthonormal columns
                raw = torch.randn(n_l, r_l, device=device, dtype=dtype)
                Q, _ = torch.linalg.qr(raw)
                Omega_list.append(Q)
            state["secant_r_Omega"] = Omega_list
        Omega = state["secant_r_Omega"]

        # ------------------------------------------------------------------
        # JVPs: P[l] = A_l(x^b) Ω[l]  ∈ R^{B × D_out × r}
        #
        # Propagate each column ω_j of Ω[l] forward through layers l,...,L-2
        # (gate D_k^b then weight W_{k+1}).  Output layer has no gate.
        # All r columns are propagated together as a (B, n_l, r) tensor.
        # Gates are the SAME values as the original forward pass.
        # ------------------------------------------------------------------
        P = []
        for l in range(L):
            v = Omega[l].unsqueeze(0).expand(B, -1, -1).clone()  # (B, n_l, r)
            for k in range(l, L - 1):
                v = v * Gates[k].unsqueeze(2)                     # gate D_k^b
                v = torch.einsum("oi,bir->bor", params_list[k + 1], v)  # W_{k+1}
            P.append(v)   # (B, D_out, r)

        # ------------------------------------------------------------------
        # Per-layer block-diagonal rank-r Sylvester solve
        #
        # For each layer k:
        #   M̃_k = (1/B) Σ_b P_k^{bT} P_k^b  ∈ R^{r × r}
        #   N̂_k = (1/B) H_k^T H_k            ∈ R^{n_{k-1} × n_{k-1}}
        #   H̃_k = Ω_k^T G_k                   ∈ R^{r × n_{k-1}}
        #   RHS  = (1/λ) M̃_k H̃_k N̂_k
        #   F_k  = _solve_sylvester(M̃_k, N̂_k, RHS, λ)   [r × n_{k-1}]
        #   ΔW_k = (1/λ) G_k − Ω_k F_k
        # ------------------------------------------------------------------

        # Adaptive λ: use max singular value of M̃_k and N̂_k at first layer
        # (same spirit as block_diagonal adaptive λ).
        # We compute λ from layer 0 statistics and use it globally.
        if lam is None:
            # Compute M̃_0 and N̂_0 for scale estimation
            Pk0 = P[0]                                   # (B, D_out, r)
            Pk0_t = Pk0.permute(0, 2, 1)                 # (B, r, D_out)
            M_tilde_0 = torch.bmm(Pk0_t, Pk0).mean(0)   # (r, r)
            N_hat_0   = H[0].T @ H[0] / B               # (n_{-1}, n_{-1})
            sv_M = torch.linalg.svdvals(M_tilde_0)
            sv_N = torch.linalg.svdvals(N_hat_0)
            lam  = float(lam_alpha * (sv_M[0].item() + sv_N[0].item()))
            lam  = max(lam, 1e-30)

        dW_raw_list = []
        for l in range(L):
            Pk  = P[l]                                    # (B, D_out, r)
            Pk_t = Pk.permute(0, 2, 1)                   # (B, r, D_out)

            # M̃_l: (r, r)
            M_tilde = torch.bmm(Pk_t, Pk).mean(0)        # (r, r)

            # N̂_l: (n_{l-1}, n_{l-1})
            N_hat   = H[l].T @ H[l] / B                  # (n_{l-1}, n_{l-1})

            # Projected gradient: H̃_l = Ω_l^T G_l ∈ R^{r × n_{l-1}}
            H_tilde = Omega[l].T @ G[l]                  # (r, n_{l-1})

            # RHS of Sylvester for F_l
            RHS = (M_tilde @ H_tilde @ N_hat) / lam      # (r, n_{l-1})

            # Solve: M̃_l F_l N̂_l + λ F_l = RHS
            F = _solve_sylvester(M_tilde, N_hat, RHS, lam)  # (r, n_{l-1})

            dW_raw = G[l] / lam - Omega[l] @ F           # (n_l, n_{l-1})
            dW_raw_list.append(dW_raw)

        # ------------------------------------------------------------------
        # Nesterov → optional NS → optional rescale → update
        # ------------------------------------------------------------------
        dW_list = self._nesterov_grads(params_list, dW_raw_list, beta)
        if use_ns:
            dW_list = [_newton_schulz(dW, steps=ns_steps) for dW in dW_list]
        if group["rescale_lr"]:
            dW_list = self._rescale_dW(dW_list, group.get("rescale_variant", "max"))
        for p, dW in zip(params_list, dW_list):
            p.add_(dW, alpha=-lr)

        self.activations.clear()
        self.pre_acts.clear()
        self.grad_acts.clear()

    # ------------------------------------------------------------------
    # whitened: NS on batch-averaged left context A_l, B-side skipped
    # ------------------------------------------------------------------

    def _step_whitened(self, group: Dict[str, Any]):
        """Whitened Operator-BD: NS on batch-averaged left context matrices.

        Algorithm (whiten-after-averaging, B-side skipped):

          1. Propagate left context backward per sample, then average:
                 A_{L-1}^(b) = I
                 A_l^(b) = A_{l+1}^(b) W_{l+1} diag(D_l^(b))   (l < L-1)
                 A_l_avg = (1/B) sum_b A_l^(b)     shape: (D_out, d_l)
          2. Mixed gradient for each layer:
                 G_mix_l = (1/B) delta_y^T @ h_{l-1}   shape: (D_out, d_{l-1})
             where delta_y = grad_acts[L-1] and h_{l-1} = pre_acts[l].
          3. Whitened direction:
                 dW_l^raw = NS(A_l_avg)^T @ G_mix_l    shape: (d_l, d_{l-1})
             For l = L-1: A_{L-1} = I so dW^raw = G_mix (last layer degenerates).
          4. Nesterov momentum on {dW_l^raw}.
          5. NS on momentum output for depth-uniform scale.

        Cost: O(B * D_out * d_l) for context propagation per layer;
              O(D_out^2 * d_l) for NS on A_l_avg — cheap when D_out is small.
        Comparable to Muon overall.
        """
        if not self.grad_acts:
            raise RuntimeError(
                "attach_hooks(model) must be called before training when using "
                "approximation='whitened'."
            )

        lr = group["lr"]
        beta = group["momentum"]
        wd = group["weight_decay"]
        ns_steps = group["ns_steps"]

        params_list = [p for p in group["params"] if p.grad is not None and p.ndim == 2]
        if not params_list:
            return

        L = len(params_list)
        device = params_list[0].device
        dtype = params_list[0].dtype
        batch_size = self.grad_acts[L - 1].shape[0]

        # ---- Step 1: build A_l_avg by backward propagation ----
        # A_{L-1}(b) = I  for all b
        D_out = params_list[-1].shape[0]
        A_curr = (
            torch.eye(D_out, device=device, dtype=dtype)
            .unsqueeze(0).expand(batch_size, -1, -1)
        )
        A_avg_list: List[Optional[torch.Tensor]] = [None] * L
        for l in range(L - 2, -1, -1):
            W_next = params_list[l + 1]                          # (d_{l+1}, d_l)
            D_l = (self.activations[l] > 0).to(dtype)           # (batch, d_l)
            W_D = W_next.unsqueeze(0) * D_l.unsqueeze(1)        # (batch, d_{l+1}, d_l)
            A_curr = torch.bmm(A_curr, W_D)                     # (batch, D_out, d_l)
            A_avg_list[l] = A_curr.mean(0)                      # (D_out, d_l)

        # ---- Steps 2 & 3: mixed gradient + left-context whitening ----
        delta_y = self.grad_acts[L - 1]   # (batch, D_out)
        dW_raw_list = []
        for l, p in enumerate(params_list):
            h = self.pre_acts[l]                                 # (batch, d_{l-1})
            G_mix = delta_y.T @ h / batch_size                  # (D_out, d_{l-1})
            if wd:
                G_mix = G_mix.add_(p, alpha=wd)
            if l == L - 1:
                # A_{L-1} = I: NS(I)^T = I, no transformation
                dW_raw = G_mix
            else:
                Q_A = _newton_schulz(A_avg_list[l], steps=ns_steps)  # (D_out, d_l)
                dW_raw = Q_A.T @ G_mix      # (d_l, D_out) @ (D_out, d_{l-1}) = (d_l, d_{l-1})
            dW_raw_list.append(dW_raw)

        # ---- Steps 4 & 5: Nesterov momentum + NS ----
        dW_list = self._nesterov_grads(params_list, dW_raw_list, beta)
        for p, dW in zip(params_list, dW_list):
            dW_final = _newton_schulz(dW, steps=ns_steps)
            p.add_(dW_final, alpha=-lr)

        self.activations.clear()
        self.pre_acts.clear()
        self.grad_acts.clear()

    # ------------------------------------------------------------------
    # operator_kfac: exact batch contexts + factored formula
    # ------------------------------------------------------------------

    def _step_operator_kfac(self, group: Dict[str, Any]):
        """Operator-KFAC: exact batch context covariances with factored inverse.

        Computes M_l and N_l from the current batch (same as block_diagonal):
            M_l = (1/B) Σ_b A_l(x^b)ᵀ A_l(x^b)
            N_l = (1/B) Σ_b B_l(x^b) B_l(x^b)ᵀ

        Then applies the factored formula instead of the Sylvester solve:
            dW_l ≈ (M_l + λI)^{-1} G_{W_l} (N_l + λI)^{-1}

        Position in the approximation hierarchy:
            K-FAC  ⊂  Operator-KFAC  ⊂  Operator-BD
        K-FAC uses EMA Fisher statistics; Operator-KFAC uses exact batch
        context covariances; Operator-BD solves the Sylvester equation exactly.
        """
        if not self.activations:
            raise RuntimeError(
                "attach_hooks(model) must be called before training when using "
                "approximation='operator_kfac'."
            )

        lr = group["lr"]
        lam = group["lam"]
        lam_alpha = group["lam_alpha"]
        beta = group["momentum"]
        wd = group["weight_decay"]

        params_list = [p for p in group["params"] if p.grad is not None and p.ndim == 2]
        if not params_list:
            return

        device = params_list[0].device
        dtype = params_list[0].dtype
        batch_size = next(iter(self.activations.values())).shape[0]

        grad_list = [
            p.grad.clone().add_(p, alpha=wd) if wd else p.grad.clone()
            for p in params_list
        ]

        M_list, N_list = self._compute_covariances(params_list, batch_size, device, dtype)

        dW_raw_list = []
        for l, p in enumerate(params_list):
            M = M_list[l]
            N = N_list[l]
            G = grad_list[l]

            if M is None and N is None:
                lam_l = lam_alpha * 2.0 if lam is None else lam
                dW_raw = G / (1.0 + lam_l)
            elif M is None:
                if lam is None:
                    sv_N = torch.linalg.svdvals(N)
                    lam_l = max(lam_alpha * (1.0 + sv_N[0].item()), 1e-30)
                else:
                    lam_l = lam
                B_mat = N.add(torch.eye(N.shape[0], device=device, dtype=dtype), alpha=lam_l)
                dW_raw = torch.linalg.solve(B_mat, G.T).T
            elif N is None:
                if lam is None:
                    sv_M = torch.linalg.svdvals(M)
                    lam_l = max(lam_alpha * (sv_M[0].item() + 1.0), 1e-30)
                else:
                    lam_l = lam
                A_mat = M.add(torch.eye(M.shape[0], device=device, dtype=dtype), alpha=lam_l)
                dW_raw = torch.linalg.solve(A_mat, G)
            else:
                if lam is None:
                    sv_M = torch.linalg.svdvals(M)
                    sv_N = torch.linalg.svdvals(N)
                    lam_l = max(lam_alpha * (sv_M[0].item() + sv_N[0].item()), 1e-30)
                else:
                    lam_l = lam
                A_mat = M.add(torch.eye(M.shape[0], device=device, dtype=dtype), alpha=lam_l)
                B_mat = N.add(torch.eye(N.shape[0], device=device, dtype=dtype), alpha=lam_l)
                X      = torch.linalg.solve(A_mat, G)
                dW_raw = torch.linalg.solve(B_mat, X.T).T

            dW_raw_list.append(dW_raw)

        dW_list = self._nesterov_grads(params_list, dW_raw_list, beta)
        for p, dW in zip(params_list, dW_list):
            p.add_(dW, alpha=-lr)

        self.activations.clear()
        self.pre_acts.clear()
        self.grad_acts.clear()

    # ------------------------------------------------------------------
    # local_bd: per-layer Sylvester solve with LOCAL context covariances
    # ------------------------------------------------------------------

    def _step_local_bd(self, group: Dict[str, Any]):
        """Operator-BD with local per-layer covariances (no Jacobian chain propagation).

        For each layer l, builds context matrices directly from hook data:
            M_l = (1/N) Σ_n δ_{n,l} δ_{n,l}^T  where δ_{n,l} = grad_acts[l][n]
            N_l = (1/N) Σ_n h_{n,l} h_{n,l}^T  where h_{n,l} = pre_acts[l][n]
            N = effective sample count (B for feedforward, T*B for RNNs)

        then solves the exact Sylvester equation:
            M_l ΔW_l N_l + λ ΔW_l = G_{W_l}

        Unlike 'block_diagonal', this mode does NOT propagate contexts through the
        full network depth, making it:
          - Compatible with any activation function (tanh, sigmoid, LSTM gates)
          - Compatible with weight-tied / recurrent architectures (contexts are
            accumulated over T timesteps via _consolidate_activations)
          - Equivalent to K-FAC factors but with the exact Sylvester solve
            instead of the K-FAC factored approximation

        For feedforward ReLU MLPs, this is a slight approximation relative to
        'block_diagonal' (which uses the exact Jacobian chain).  For RNNs it
        is the correct formulation derived in §7.6 of the paper.
        """
        if not self.pre_acts:
            raise RuntimeError(
                "attach_hooks(model) must be called before training when using "
                "approximation='local_bd'."
            )

        lr = group["lr"]
        lam = group["lam"]
        lam_alpha = group["lam_alpha"]
        beta = group["momentum"]
        wd = group["weight_decay"]
        rescale_lr = group.get("rescale_lr", False)

        params_list = [p for p in group["params"] if p.grad is not None and p.ndim == 2]
        if not params_list:
            return

        device = params_list[0].device
        dtype = params_list[0].dtype

        grad_list = [
            p.grad.clone().to(dtype).add_(p, alpha=wd) if wd else p.grad.clone().to(dtype)
            for p in params_list
        ]

        dW_raw_list = []
        sylvester_mask = []  # True iff the layer used a full M+N Sylvester solve
        for l, p in enumerate(params_list):
            G = grad_list[l]

            # Look up hook index by param identity (handles non-Linear params
            # like embedding.weight that share the grad_acts namespace).
            hook_idx = self._param_to_hook_idx.get(id(p))
            h  = self.pre_acts.get(hook_idx)   # (N, d_in)   N = T*B for RNN
            dz = self.grad_acts.get(hook_idx)  # (N, d_out)

            # Build M and N from local covariances.
            # Skip (treat as identity) any matrix whose dimension exceeds the
            # threshold — e.g. the output projection to a large vocabulary
            # (30K × 30K is infeasible to invert).
            MAX_COV_DIM = 8192

            if dz is not None and dz.shape[-1] <= MAX_COV_DIM:
                dz = dz.to(dtype)
                N_eff = dz.shape[0]
                M = dz.T @ dz / N_eff   # (d_out, d_out)
            else:
                M = None

            if h is not None and h.shape[-1] <= MAX_COV_DIM:
                h = h.to(dtype)
                N_eff = h.shape[0]
                N_mat = h.T @ h / N_eff  # (d_in, d_in)
            else:
                N_mat = None

            # Solve Sylvester: M ΔW N + λ ΔW = G
            if M is None and N_mat is None:
                lam_l = lam_alpha * 2.0 if lam is None else lam
                dW_raw = G / (1.0 + lam_l)
            elif M is None:
                # M ≈ I (no output context): (N + λI) ΔW^T = G^T
                if lam is None:
                    sv = torch.linalg.svdvals(N_mat)
                    lam_l = max(lam_alpha * (1.0 + sv[0].item()), 1e-30)
                else:
                    lam_l = lam
                dW_raw = torch.linalg.solve(
                    N_mat.add(torch.eye(N_mat.shape[0], device=device, dtype=dtype),
                              alpha=lam_l),
                    G.T,
                ).T
            elif N_mat is None:
                # N ≈ I (no input context): (M + λI) ΔW = G
                if lam is None:
                    sv = torch.linalg.svdvals(M)
                    lam_l = max(lam_alpha * (sv[0].item() + 1.0), 1e-30)
                else:
                    lam_l = lam
                dW_raw = torch.linalg.solve(
                    M.add(torch.eye(M.shape[0], device=device, dtype=dtype),
                          alpha=lam_l),
                    G,
                )
            else:
                dW_raw = _solve_sylvester(M, N_mat, G, lam, lam_alpha)

            dW_raw_list.append(dW_raw)
            # Track whether this layer used a full Sylvester solve (both M, N available)
            # so that rescale_lr ignores fallback layers (e.g. huge embedding/output
            # matrices where M or N was skipped due to MAX_COV_DIM).
            sylvester_mask.append(M is not None and N_mat is not None)

        dW_list = self._nesterov_grads(params_list, dW_raw_list, beta)

        if rescale_lr:
            # Only consider layers that had a full Sylvester solve for the scale reference.
            # Layers falling back to G/(1+λ) (e.g. the embedding/output matrix when
            # vocab_size > MAX_COV_DIM) can have very large Frobenius norms that would
            # otherwise suppress all inner-layer updates to near-zero.
            solved_only = [dW for dW, ok in zip(dW_list, sylvester_mask) if ok]
            if not solved_only:
                solved_only = list(dW_list)
            variant = group.get("rescale_variant", "max")
            scaled = self._rescale_dW(solved_only, variant)
            # Re-merge: solved layers use rescaled updates, fallback layers keep original.
            it = iter(scaled)
            dW_list = [next(it) if ok else dW for dW, ok in zip(dW_list, sylvester_mask)]

        for p, dW in zip(params_list, dW_list):
            p.add_(dW, alpha=-lr)

        self.activations.clear()
        self.pre_acts.clear()
        self.grad_acts.clear()

    # ------------------------------------------------------------------
    # kfac: K-FAC with EMA statistics
    # ------------------------------------------------------------------

    def _step_kfac(self, group: Dict[str, Any]):
        """K-FAC-like update with exponential moving average statistics.

        Maintains running estimates:
            F_out_l ← κ F_out_l + (1-κ) (1/B) Δz_l Δz_lᵀ
            F_in_l  ← κ F_in_l  + (1-κ) (1/B) h_{l-1} h_{l-1}ᵀ

        and computes the approximate update:
            dW_l ≈ (F_out_l + λ I)^{-1} G_{W_l} (F_in_l + λ I)^{-1}

        This is an approximation to the Sylvester solve:
        it is exact when F_out and F_in commute with the batch covariances,
        which holds in the K-FAC factorisation assumption
            (F_out ⊗ F_in + λ I) ≈ (F_out + λ^{1/2} I) ⊗ (F_in + λ^{1/2} I).

        Note: requires both forward hooks (for h_{l-1}) and backward hooks
        (for Δz_l) to be attached via attach_hooks().
        """
        if not self.activations or not self.grad_acts:
            raise RuntimeError(
                "K-FAC requires both forward and backward hooks. "
                "Call attach_hooks(model) before training."
            )

        lr = group["lr"]
        lam = group["lam"]
        lam_alpha = group["lam_alpha"]
        beta = group["momentum"]
        kfac_decay = group["kfac_decay"]
        wd = group["weight_decay"]

        params_list = [p for p in group["params"] if p.grad is not None and p.ndim == 2]
        if not params_list:
            return

        device = params_list[0].device
        dtype = params_list[0].dtype

        grad_list = [
            p.grad.clone().add_(p, alpha=wd) if wd else p.grad.clone()
            for p in params_list
        ]

        dW_raw_list = []
        for l, p in enumerate(params_list):
            state = self.state[p]

            # h_{l-1}: input to layer l, shape (batch, d_l_in)
            h = self.pre_acts.get(l)
            # Δz_l: backward gradient w.r.t. z_l, shape (batch, d_l_out)
            dz = self.grad_acts.get(l)

            if h is None or dz is None:
                # Fall back to plain gradient step if hooks didn't fire
                dW_raw_list.append(grad_list[l])
                continue

            B = h.shape[0]

            # Batch covariances
            F_in_hat = (h.T @ h) / B            # (d_l_in,  d_l_in)
            F_out_hat = (dz.T @ dz) / B         # (d_l_out, d_l_out)

            # EMA update
            if "F_in" not in state:
                state["F_in"] = F_in_hat.clone()
                state["F_out"] = F_out_hat.clone()
            else:
                state["F_in"].mul_(kfac_decay).add_(F_in_hat, alpha=1.0 - kfac_decay)
                state["F_out"].mul_(kfac_decay).add_(F_out_hat, alpha=1.0 - kfac_decay)

            F_in = state["F_in"]
            F_out = state["F_out"]

            d_in = F_in.shape[0]
            d_out = F_out.shape[0]

            # Adaptive λ per layer (Prop 5.2): scale with max eigenvalue of each factor.
            if lam is None:
                sv_out = torch.linalg.svdvals(F_out)
                sv_in  = torch.linalg.svdvals(F_in)
                lam_l  = max(lam_alpha * (sv_out[0].item() + sv_in[0].item()), 1e-30)
            else:
                lam_l = lam

            A = F_out.add(torch.eye(d_out, device=device, dtype=dtype), alpha=lam_l)
            B_mat = F_in.add(torch.eye(d_in, device=device, dtype=dtype), alpha=lam_l)

            X      = torch.linalg.solve(A, grad_list[l])
            dW_raw = torch.linalg.solve(B_mat, X.T).T
            dW_raw_list.append(dW_raw)

        dW_list = self._nesterov_grads(params_list, dW_raw_list, beta)
        if group["use_ns"]:
            dW_list = [_newton_schulz(dW, steps=group["ns_steps"]) for dW in dW_list]
        for p, dW in zip(params_list, dW_list):
            p.add_(dW, alpha=-lr)

        self.activations.clear()
        self.pre_acts.clear()
        self.grad_acts.clear()

    # ------------------------------------------------------------------
    # cg: conjugate gradient (TODO)
    # ------------------------------------------------------------------

    def _step_cg(self, group: Dict[str, Any]):
        """CG on the full coupled normal equations.  [TODO: not yet implemented]

        Falls back to block_diagonal for now.
        """
        self._step_block_diagonal(group)


# ---------------------------------------------------------------------------
# OperatorALS — Alternating Least Squares for the Exact Operator Projection Problem
# ---------------------------------------------------------------------------

class OperatorALS:
    """Op-ALS: block coordinate descent on the Exact Operator Projection Problem (EOPP).

    For a deep linear network P = W_{L-1} ... W_0, this optimizer minimises

        min_{ΔW_k}  ‖(W_{L-1}+ΔW_{L-1}) · … · (W_0+ΔW_0) − P*‖_F²
                    + λ Σ_k ‖ΔW_k‖_F²

    by cycling through layers k = 0, …, L-1 and solving the self-consistent
    single-layer Sylvester equation at each step:

        M̃_k ΔW_k Ñ_k + λ ΔW_k = G̃_k

    with contexts M̃_k, Ñ_k, G̃_k recomputed from the *already-updated* weights at
    each inner step (self-consistent / live context update).

    Unlike Op-BD / Operator-Exact, which solve the Linearized OPP (LOPP) with
    frozen contexts, Op-ALS solves EOPP directly and can grow operator rank within
    a single sweep (see Proposition doubly-null in the paper).

    This implementation is specific to deep linear networks (no activation gates).
    The cost per sweep is L Sylvester solves — identical to Op-BD.

    Args:
        model:     nn.Module whose Linear layers form the deep linear chain.
        P_star:    Target operator tensor (d_out × d_in).
        lr:        Learning rate η applied to each layer update ΔW_k.
        lam:       Tikhonov regularisation λ (fixed float, same for all layers).
        n_sweeps:  Number of ALS inner sweeps per optimizer step (default 1).
    """

    needs_backward: bool = False  # signal to run_optimizer: skip loss.backward()

    def __init__(
        self,
        model: nn.Module,
        P_star: torch.Tensor,
        lr: float = 5e-2,
        lam: float = 0.1,
        n_sweeps: int = 1,
    ):
        self.layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        self.P_star = P_star.detach().clone()
        self.lr = lr
        self.lam = lam
        self.n_sweeps = n_sweeps

    # Compatibility shim so run_optimizer can call opt.zero_grad() blindly.
    def zero_grad(self) -> None:
        pass

    @torch.no_grad()
    def step(self) -> None:
        """Run n_sweeps ALS sweeps toward P_target = P_init + lr*(P* - P_init).

        lr is applied **inside** the sweep (via the target), so the live
        contexts see the actual weight trajectory.  The sweep result is
        committed directly — no post-hoc rescaling.
        """
        layers = self.layers
        orig_device = layers[0].weight.device
        dtype = layers[0].weight.dtype
        cpu = torch.device("cpu")

        P_star = self.P_star.to(device=cpu, dtype=dtype)
        P_init = self._compute_P()
        P_target = P_init + self.lr * (P_star - P_init)

        for _ in range(self.n_sweeps):
            self._als_sweep(P_target)

    def _compute_P(self) -> torch.Tensor:
        """Return current operator P = W_{L-1} @ … @ W_0 (on CPU)."""
        P = self.layers[0].weight.data.cpu().clone()
        for layer in self.layers[1:]:
            P = layer.weight.data.cpu() @ P
        return P

    @torch.no_grad()
    def rel_operator_error(self) -> float:
        """‖P(W) − P*‖_F / ‖P*‖_F with P on CPU (matches _als_sweep bookkeeping)."""
        P = self._compute_P()
        Ps = self.P_star.detach().cpu().to(dtype=P.dtype)
        return ((P - Ps).norm() / Ps.norm()).item()

    @torch.no_grad()
    def als_fixed_p_star_sweeps(self, n_sweeps: int) -> tuple[list[int], list[float]]:
        """Repeated full ALS sweeps with fixed target P_target = P* (no outer lr).

        Each sweep cycles all layers once (same as one call to ``step`` with lr=1
        and ``n_sweeps``=1, but without recomputing P_target from P_init).  This
        is block coordinate descent on the factorized objective toward the true
        end-to-end target operator.
        """
        layers = self.layers
        dtype = layers[0].weight.dtype
        cpu = torch.device("cpu")
        P_target = self.P_star.to(device=cpu, dtype=dtype)

        sweep_ids: list[int] = []
        errs: list[float] = []
        sweep_ids.append(0)
        errs.append(self.rel_operator_error())
        for s in range(1, n_sweeps + 1):
            self._als_sweep(P_target)
            sweep_ids.append(s)
            errs.append(self.rel_operator_error())
        return sweep_ids, errs

    def _als_sweep(self, P_target: torch.Tensor) -> None:
        """One ALS sweep: solve each layer toward P_target with live contexts.

        For each layer k = 0 … L-1:
          1. Compute live contexts A_k, B_k from current weights.
          2. Current operator: P_k = A_k W_k B_k.
          3. Residual: R_k = P_target - P_k.
          4. G_k = A_k^T R_k B_k^T.
          5. Solve  M_k dW_k N_k + lam dW_k = G_k.
          6. W_k += dW_k.
        """
        layers = self.layers
        L = len(layers)
        lam = self.lam

        device = torch.device("cpu")
        wire_dtype = layers[0].weight.dtype
        # Very small λ makes F/(σ_i σ_j + λ) large in fp32; deep products of W then
        # overflow.  Promote the sweep to float64 when λ is tiny (deep linear / ALS).
        work_dtype = (
            torch.float64
            if lam is not None and float(lam) <= 1e-3
            else wire_dtype
        )

        orig_device = layers[0].weight.device
        W_cpu = [layer.weight.data.to(device=device, dtype=work_dtype) for layer in layers]

        d_out = W_cpu[-1].shape[0]
        d_in  = W_cpu[0].shape[1]

        P_tgt = P_target.to(device=device, dtype=work_dtype)

        for k in range(L):
            A_k = torch.eye(d_out, device=device, dtype=work_dtype)
            for j in range(L - 1, k, -1):
                A_k = A_k @ W_cpu[j]

            B_k = torch.eye(d_in, device=device, dtype=work_dtype)
            for j in range(k):
                B_k = W_cpu[j] @ B_k

            P_curr = A_k @ W_cpu[k] @ B_k
            R_k = P_tgt - P_curr

            M_k = A_k.T @ A_k
            N_k = B_k @ B_k.T
            G_k = A_k.T @ R_k @ B_k.T

            dW = _solve_sylvester(M_k, N_k, G_k, lam)
            W_cpu[k].add_(dW, alpha=1.0)

        for layer, w_new in zip(layers, W_cpu):
            layer.weight.data.copy_(w_new.to(device=orig_device, dtype=wire_dtype))
