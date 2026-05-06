#!/usr/bin/env python3
"""
Fixed-Gates Linear Network (FGLN) depth diagnostics.

Model: square d×d chain with fixed diagonal binary gates:

    P(W) = W_{L-1} D_{L-2} W_{L-2} ... D_0 W_0

where D_i = diag(m_i) with m_i ~ Bernoulli(p) sampled once at init and then fixed.

Task: Gaussian regression with isotropic target operator:
    x ~ N(0, I_d)
    P* = U V^T   (Haar orthogonal; all singular values = 1)
    y = x P*^T

We test whether operator-ALS remains depth-robust when nonlinearity is removed but
fixed masking gates remain (a controlled proxy for ReLU gating without switching).

Usage:
    venv/bin/python labs/fgln/run.py --d 32 --depths 8 16 32 64 --n_steps 200 --p 0.5
    venv/bin/python labs/fgln/run.py --depths 32 64 --lr 1 --lam 1e-4 --n_sweeps 4 --init orth
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class FGLN(nn.Module):
    """Fixed-gates linear network: (Linear -> mask) repeated."""

    def __init__(self, d: int, depth: int, p_gate: float, seed: int):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if not (0.0 <= p_gate <= 1.0):
            raise ValueError("p_gate must be in [0, 1]")
        self.d = d
        self.depth = depth
        self.layers = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(depth)])

        # One gate after each layer except the last: masks[0] is D_0 (after W_0).
        g = torch.Generator().manual_seed(seed)
        masks = []
        for _ in range(max(depth - 1, 0)):
            m = (torch.rand(d, generator=g) < p_gate).to(torch.float32)
            masks.append(m)
        if masks:
            self.register_buffer("masks", torch.stack(masks, dim=0))  # (L-1, d)
        else:
            self.register_buffer("masks", torch.empty(0, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for k, layer in enumerate(self.layers):
            h = layer(h)
            if k < self.depth - 1:
                h = h * self.masks[k]
        return h


def compute_P_fgln(net: FGLN) -> torch.Tensor:
    """Compute end-to-end operator P(W) as a d×d matrix (detached)."""
    with torch.no_grad():
        W = [lay.weight for lay in net.layers]  # (d,d)
        if net.depth == 1:
            return W[0].detach().clone()

        d = W[0].shape[0]
        P = W[0]
        for k in range(net.depth - 1):
            Dk = torch.diag(net.masks[k].to(dtype=P.dtype, device=P.device))
            P = Dk @ P
            P = W[k + 1] @ P
        return P.detach().clone()


def _haar_orthogonal(d: int, rng: torch.Generator, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    a = torch.randn(d, d, generator=rng, dtype=torch.float64)
    u, _, vh = torch.linalg.svd(a)
    return (u @ vh).to(device=device, dtype=dtype)


def make_dataset(d: int, n: int, seed: int, rank_star: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=rng)
    U, _, Vh = torch.linalg.svd(torch.randn(d, d, generator=rng))
    if rank_star is None:
        P_star = U @ Vh
    else:
        r = int(rank_star)
        if not (1 <= r <= d):
            raise ValueError(f"rank_star must be in [1, {d}] or None; got {rank_star}")
        s = torch.zeros(d)
        s[:r] = 1.0
        P_star = U @ torch.diag(s) @ Vh
    Y = X @ P_star.T
    return X, Y, P_star


def init_weights(
    net: FGLN,
    init: str,
    seed: int,
    target_norm: float,
    *,
    rescale_mode: str = "all",
) -> None:
    """Init weights; optionally normalize ||P0||_F to target_norm.

    rescale_mode:
      - "all":  scale every layer by a common scalar (current default; can blow up
                intermediate contexts when masks shrink P0 strongly).
      - "last": scale only the last layer (keeps intermediate norms near the init).
      - "none": do not rescale.
    """
    g = torch.Generator().manual_seed(seed)
    d = net.d
    L = net.depth
    with torch.no_grad():
        if init == "xavier":
            for lay in net.layers:
                lay.weight.copy_(torch.randn(d, d, generator=g) / math.sqrt(float(d)))
        elif init == "orth":
            # Haar orthogonal per layer (like deep linear isotropic init, but masks break isotropy).
            dev = net.layers[0].weight.device
            dt = net.layers[0].weight.dtype
            for lay in net.layers:
                q = _haar_orthogonal(d, g, device=dev, dtype=dt)
                lay.weight.copy_(q)
        else:
            raise ValueError(f"Unknown init: {init!r}")

        if rescale_mode not in ("all", "last", "none"):
            raise ValueError(f"Unknown rescale_mode: {rescale_mode!r}")
        if rescale_mode != "none":
            P0 = compute_P_fgln(net)
            n0 = P0.norm().item()
            if n0 > 0 and target_norm > 0:
                if rescale_mode == "all":
                    scale = (target_norm / n0) ** (1.0 / max(L, 1))
                    for lay in net.layers:
                        lay.weight.mul_(scale)
                else:  # last
                    net.layers[-1].weight.mul_(target_norm / n0)


# -----------------------------------------------------------------------------
# Masked Operator-ALS (deep linear ALS with fixed diagonal gates)
# -----------------------------------------------------------------------------


@dataclass
class ALSConfig:
    lr: float
    lam: float
    n_sweeps: int
    als_reverse_sweep: bool


class MaskedOperatorALS:
    """ALS for FGLN: solves factorized masked operator objective toward P_target."""

    def __init__(
        self,
        model: FGLN,
        P_star: torch.Tensor,
        *,
        lr: float,
        lam: float,
        n_sweeps: int = 1,
        als_reverse_sweep: bool = True,
        debug_nan: bool = False,
        normalize_contexts: bool = False,
        damping: bool = False,
        damping_beta: float = 0.5,
        damping_min: float = 1e-6,
        damping_max_trials: int = 20,
        gauge_balance: bool = False,
        gauge_balance_passes: int = 1,
        adaptive_lambda_step: bool = False,
        adaptive_lambda_mode: str = "median",
        als_gateperm_warmstart: bool = True,
        als_gateperm_warmstart_once: bool = True,
        als_lam_anchor_post_warmstart: bool = False,
    ):
        self.model = model
        self.layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        self.P_star = P_star.detach().clone()
        self.lr = float(lr)
        self.lam = float(lam)
        self.n_sweeps = int(n_sweeps)
        self.als_reverse_sweep = bool(als_reverse_sweep)
        self.debug_nan = bool(debug_nan)
        self.normalize_contexts = bool(normalize_contexts)
        self.damping = bool(damping)
        self.damping_beta = float(damping_beta)
        self.damping_min = float(damping_min)
        self.damping_max_trials = int(damping_max_trials)
        self.gauge_balance = bool(gauge_balance)
        self.gauge_balance_passes = int(gauge_balance_passes)
        self.adaptive_lambda_step = bool(adaptive_lambda_step)
        self.adaptive_lambda_mode = str(adaptive_lambda_mode)
        self.als_gateperm_warmstart = bool(als_gateperm_warmstart)
        self.als_gateperm_warmstart_once = bool(als_gateperm_warmstart_once)
        self.als_lam_anchor_post_warmstart = bool(als_lam_anchor_post_warmstart)
        self._als_warmstart_used_once = False
        if self.adaptive_lambda_mode not in ("median", "max"):
            raise ValueError("adaptive_lambda_mode must be 'median' or 'max'")

        self._S0: Optional[float] = None

        if not (0.0 < self.damping_beta < 1.0):
            raise ValueError("damping_beta must be in (0,1)")
        if not (0.0 < self.damping_min <= 1.0):
            raise ValueError("damping_min must be in (0,1]")
        if self.damping_max_trials < 1:
            raise ValueError("damping_max_trials must be >= 1")
        if self.gauge_balance_passes < 0:
            raise ValueError("gauge_balance_passes must be >= 0")

        if self.n_sweeps < 1:
            raise ValueError("n_sweeps must be >= 1")

    def _compute_P(self) -> torch.Tensor:
        return compute_P_fgln(self.model).cpu()

    @torch.no_grad()
    def step(self, P_tgt_override: Optional[torch.Tensor] = None) -> None:
        # Target mixing like OperatorALS: P_target = P_init + lr*(P*-P_init)
        # IMPORTANT: build P_init/P_target in float64 to avoid catastrophic
        # cancellation when lr is small and contexts are huge (masked products).
        cpu = torch.device("cpu")
        P_init = self._compute_P().to(device=cpu, dtype=torch.float64)
        if P_tgt_override is None:
            P_star = self.P_star.to(device=cpu, dtype=torch.float64)
            P_target = P_init + float(self.lr) * (P_star - P_init)
        else:
            P_target = P_tgt_override.to(device=cpu, dtype=torch.float64)
        lam_step = self.lam
        if self.adaptive_lambda_step:
            lam_step = self._compute_adaptive_lambda_step()
        layers = self.layers
        W_ref = [lay.weight.data.to(device=cpu, dtype=torch.float64).clone() for lay in layers]
        if self.als_gateperm_warmstart and not (self.als_gateperm_warmstart_once and self._als_warmstart_used_once):
            self._apply_gateperm_warmstart()
            self._als_warmstart_used_once = True
            if self.als_lam_anchor_post_warmstart:
                W_ref = [lay.weight.data.to(device=cpu, dtype=torch.float64).clone() for lay in layers]
        for _ in range(self.n_sweeps):
            self._als_sweep(P_target, lam=lam_step, W_ref=W_ref)

    def _als_sweep(self, P_target: torch.Tensor, *, lam: float, W_ref: list[torch.Tensor]) -> None:
        layers = self.layers
        L = len(layers)
        d_out = layers[-1].weight.shape[0]
        d_in = layers[0].weight.shape[1]
        lam = float(lam)

        device = torch.device("cpu")
        # Always run the masked-context ALS sweep in float64.
        # With fixed masks, intermediate context products can grow extremely large
        # with depth even when the end-to-end operator is well-scaled; float32
        # overflows and obscures whether the method is conceptually failing.
        work_dtype = torch.float64

        W = [lay.weight.data.to(device=device, dtype=work_dtype) for lay in layers]  # (d,d)
        masks = self.model.masks.to(device=device, dtype=work_dtype)  # (L-1,d)

        P_tgt = P_target.to(device=device, dtype=work_dtype)

        # Helper: build A_k = W_{L-1} D_{L-2} ... W_{k+1} D_k  (or I if k=L-1)
        def build_A(k: int) -> torch.Tensor:
            A = torch.eye(d_out, device=device, dtype=work_dtype)
            for j in range(L - 1, k, -1):
                A = A @ W[j]
                if j - 1 >= 0 and (j - 1) < L - 1:
                    # After W_{j-1} is D_{j-1}, which sits between W_j and W_{j-1}
                    # In A_k product, when we just multiplied W_j, the next factor is D_{j-1}.
                    D = torch.diag(masks[j - 1])
                    A = A @ D
            return A

        # Helper: build B_k = D_{k-1} W_{k-1} ... D_0 W_0  (or I if k=0)
        def build_B(k: int) -> torch.Tensor:
            B = torch.eye(d_in, device=device, dtype=work_dtype)
            for j in range(k):
                # apply W_j then if j <= k-2 apply D_j (since gate after W_j)
                B = W[j] @ B
                if j < L - 1 and j < k - 1:
                    D = torch.diag(masks[j])
                    B = D @ B
            return B

        layer_order = range(L - 1, -1, -1) if self.als_reverse_sweep else range(L)
        for k in layer_order:
            A_k = build_A(k)
            B_k = build_B(k)

            P_curr = A_k @ W[k] @ B_k
            R = P_tgt - P_curr

            M = A_k.T @ A_k
            N = B_k @ B_k.T
            G = A_k.T @ R @ B_k.T

            if not (torch.isfinite(M).all() and torch.isfinite(N).all() and torch.isfinite(G).all()):
                if self.debug_nan:
                    def _summ(x: torch.Tensor) -> str:
                        return f"shape={tuple(x.shape)} dtype={x.dtype} maxabs={x.abs().max().item():.3e} finite={bool(torch.isfinite(x).all())}"
                    msg = (
                        f"Non-finite context at layer k={k}:\n"
                        f"  A_k: {_summ(A_k)}\n"
                        f"  B_k: {_summ(B_k)}\n"
                        f"  W_k: {_summ(W[k])}\n"
                        f"  M:   {_summ(M)}\n"
                        f"  N:   {_summ(N)}\n"
                        f"  G:   {_summ(G)}\n"
                    )
                    print(msg)
                raise FloatingPointError("Non-finite context encountered (M/N/G) during ALS sweep.")

            # Optional context normalization (exactly equivalent equation).
            # Original:   M dW N + lam dW = G
            # Divide by (m*n): (M/m) dW (N/n) + (lam/(m*n)) dW = G/(m*n)
            # This yields the exact same minimizer dW, but improves numeric scale.
            if self.normalize_contexts:
                m = torch.linalg.svdvals(M).max()
                n = torch.linalg.svdvals(N).max()
                # Avoid dividing by 0; if either is 0, equation is already lam-dominated.
                mn = (m * n).clamp(min=torch.finfo(work_dtype).tiny)
                M_use = M / m.clamp(min=torch.finfo(work_dtype).tiny)
                N_use = N / n.clamp(min=torch.finfo(work_dtype).tiny)
                G_use = G / mn
                lam_use = lam / float(mn)
            else:
                M_use, N_use, G_use, lam_use = M, N, G, lam

            delta_anchor = W[k] - W_ref[k]
            G_use = G_use - lam_use * delta_anchor
            dW = _solve_sylvester(M_use, N_use, G_use, lam_use)

            if not self.damping:
                W[k].add_(dW)
            else:
                # Damped update via backtracking on the *same* local objective:
                #   f(Wk) = ||A Wk B - P_tgt||_F^2 + lam ||Wk - Wk_old||_F^2
                W_old = W[k].clone()

                W_ref_k = W_ref[k]

                def f_local(Wk: torch.Tensor) -> torch.Tensor:
                    Rk = P_tgt - (A_k @ Wk @ B_k)
                    return (Rk * Rk).sum() + lam * ((Wk - W_ref_k) * (Wk - W_ref_k)).sum()

                f0 = f_local(W_old)
                if not torch.isfinite(f0):
                    raise FloatingPointError("Non-finite local objective before update.")

                gamma = 1.0
                accepted = False
                for _ in range(self.damping_max_trials):
                    W_try = W_old + gamma * dW
                    f1 = f_local(W_try)
                    if torch.isfinite(f1) and f1 <= f0:
                        W[k].copy_(W_try)
                        accepted = True
                        break
                    gamma *= self.damping_beta
                    if gamma < self.damping_min:
                        break

                if not accepted:
                    # If we can't find a decreasing step, skip this layer update.
                    W[k].copy_(W_old)

        if self.gauge_balance and L >= 2 and self.gauge_balance_passes > 0:
            # Scalar gauge balancing: for each adjacent pair (W_k, W_{k+1}),
            # apply W_k <- c W_k, W_{k+1} <- (1/c) W_{k+1}.
            # This leaves every local product W_{k+1} D_k W_k exactly unchanged
            # (scalar commutes with D_k), hence preserves the represented operator P(W)
            # and does not change the ALS objective value; it only improves internal
            # conditioning by preventing norm blow-up in intermediate products.
            tiny = torch.finfo(work_dtype).tiny

            def _balance_pair(k: int) -> None:
                n0 = torch.linalg.norm(W[k])
                n1 = torch.linalg.norm(W[k + 1])
                if not (torch.isfinite(n0) and torch.isfinite(n1)):
                    return
                if n0 <= tiny or n1 <= tiny:
                    return
                c = torch.sqrt(n1 / n0)
                if not torch.isfinite(c) or c <= 0:
                    return
                W[k].mul_(c)
                W[k + 1].div_(c)

            for _ in range(self.gauge_balance_passes):
                for k in range(L - 1):
                    _balance_pair(k)
                for k in range(L - 2, -1, -1):
                    _balance_pair(k)

        # Commit back to model
        for lay, w_new in zip(layers, W):
            lay.weight.data.copy_(w_new.to(device=lay.weight.device, dtype=lay.weight.dtype))

    @torch.no_grad()
    def _apply_gateperm_warmstart(self) -> None:
        """Initialize W with gate-subspace orthogonal routing (FGLN single-mask case)."""
        L = len(self.layers)
        if L < 1:
            return
        masks = self.model.masks.to(device=torch.device("cpu"))
        for k, lay in enumerate(self.layers):
            w = lay.weight.data.to(device=torch.device("cpu"), dtype=torch.float64)
            d_out_k, d_in_k = int(w.shape[0]), int(w.shape[1])
            if k > 0:
                g_in = masks[k - 1]
                active_in = [j for j in range(d_in_k) if float(g_in[j].item()) > 0.5]
            else:
                active_in = list(range(d_in_k))
            if k < L - 1:
                g_out = masks[k]
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
            O = torch.zeros_like(w)
            for src in range(d_in_k):
                O[mapping[src], src] = 1.0
            lay.weight.data.copy_(O.to(device=lay.weight.device, dtype=lay.weight.dtype))

    def _compute_adaptive_lambda_step(self) -> float:
        """
        Option A (per outer optimizer step):
          lambda_step = lambda_base * S(t)/S(0)
          S(t) = median_k( sigma_max(M_k) * sigma_max(N_k) )

        Here M_k = A_k^T A_k and N_k = B_k B_k^T in the same context
        convention as `_als_sweep`. Since M_k, N_k are PSD, sigma_max is their
        largest eigenvalue.
        """
        layers = self.layers
        L = len(layers)
        d_out = layers[-1].weight.shape[0]
        d_in = layers[0].weight.shape[1]
        device = torch.device("cpu")
        work_dtype = torch.float64

        W = [lay.weight.data.to(device=device, dtype=work_dtype) for lay in layers]  # (d,d)
        masks = self.model.masks.to(device=device, dtype=work_dtype)  # (L-1,d)

        # Prebuild diagonal gate matrices D_k = diag(masks[k]) for k in [0, L-2]
        D = [torch.diag(masks[k]) for k in range(L - 1)]

        # A_k = W_{L-1} D_{L-2} ... W_{k+1} D_k  (and A_{L-1}=I)
        # Recurrence: A_{L-1}=I and A_k = A_{k+1} @ W_{k+1} @ D_k.
        A_list: list[torch.Tensor] = [torch.eye(d_out, device=device, dtype=work_dtype) for _ in range(L)]
        A_list[L - 1] = torch.eye(d_out, device=device, dtype=work_dtype)
        for k in range(L - 2, -1, -1):
            A_list[k] = A_list[k + 1] @ W[k + 1] @ D[k]

        # B_k = W_{k-1} D_{k-2} ... D_0 W_0  (and B_0=I) matching `_als_sweep`.
        # Recurrence: B_0=I; for k>=2, B_k = W_{k-1} @ D_{k-2} @ B_{k-1}; for k=1, B_1=W_0.
        B_list: list[torch.Tensor] = [torch.eye(d_in, device=device, dtype=work_dtype) for _ in range(L)]
        B_list[0] = torch.eye(d_in, device=device, dtype=work_dtype)
        if L >= 2:
            B_list[1] = W[0]
        for k in range(2, L):
            B_list[k] = W[k - 1] @ D[k - 2] @ B_list[k - 1]

        prods: list[float] = []
        for k in range(L):
            A_k = A_list[k]
            B_k = B_list[k]
            M = A_k.T @ A_k
            N = B_k @ B_k.T

            # Largest eigenvalue of PSD matrix.
            sM = float(torch.linalg.eigvalsh(M).max().item())
            sN = float(torch.linalg.eigvalsh(N).max().item())
            prods.append(sM * sN)

        if self.adaptive_lambda_mode == "median":
            S_t = float(torch.tensor(prods, device=device).median().item())
        else:
            S_t = float(max(prods))

        if self._S0 is None:
            self._S0 = S_t
        S0 = max(self._S0, 1e-300)  # avoid div-by-zero
        return float(self.lam) * (S_t / S0)


def _solve_sylvester(M: torch.Tensor, N: torch.Tensor, G: torch.Tensor, lam: float) -> torch.Tensor:
    # SVD-based solve for M X N + lam X = G
    # M = U diag(s) U^T, N = V diag(t) V^T
    def _svd_psd(A: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
        # Add a tiny ridge on the diagonal to avoid svd convergence failures
        # on nearly-singular PSD matrices from masked products.
        if not torch.isfinite(A).all():
            raise FloatingPointError("Non-finite matrix passed to SVD.")
        if ridge > 0:
            A = A + ridge * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        U, s, Vh = torch.linalg.svd(A)
        return U, s

    # Heuristic ridge: proportional to trace / dim (scale-aware), but tiny.
    eps = 1e-12 if M.dtype == torch.float64 else 1e-8
    ridge_M = eps * (M.trace().abs() / max(M.shape[0], 1)).clamp(min=1.0)
    ridge_N = eps * (N.trace().abs() / max(N.shape[0], 1)).clamp(min=1.0)

    try:
        U, s = _svd_psd(M, float(ridge_M))
        V, t = _svd_psd(N, float(ridge_N))
    except torch._C._LinAlgError:
        # Fallback: increase ridge aggressively
        U, s = _svd_psd(M, float(ridge_M) * 1e4 + eps)
        V, t = _svd_psd(N, float(ridge_N) * 1e4 + eps)
    UtGV = U.T @ G @ V
    denom = s.unsqueeze(1) * t.unsqueeze(0) + lam
    X_hat = UtGV / denom
    return U @ X_hat @ V.T


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


def run_one_depth(args: argparse.Namespace, depth: int) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, Y, P_star = make_dataset(args.d, args.n_samples, args.data_seed, rank_star=args.rank_star)
    X, Y, P_star = X.to(device), Y.to(device), P_star.to(device)

    net0 = FGLN(args.d, depth=depth, p_gate=args.p, seed=args.gate_seed).to(device)
    target_norm = P_star.norm().item()
    init_weights(
        net0,
        init=args.init,
        seed=args.init_seed,
        target_norm=target_norm,
        rescale_mode=args.rescale_mode,
    )

    if args.diagnose:
        diag_dir = Path(args.out_dir) / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        write_context_diagnostics(
            net0,
            depth=depth,
            p_gate=args.p,
            rank_star=args.rank_star,
            tag=f"d{args.d}_L{depth}_p{args.p:g}_init{args.init}_rescale{args.rescale_mode}",
            out_dir=diag_dir,
            also_after_one_step=args.diagnose_after_step,
            als_kwargs=dict(
                lr=args.lr,
                lam=args.lam,
                n_sweeps=args.n_sweeps,
                als_reverse_sweep=args.als_reverse_sweep,
                als_gateperm_warmstart=args.als_gateperm_warmstart,
                als_gateperm_warmstart_once=args.als_gateperm_warmstart_once,
                als_lam_anchor_post_warmstart=args.als_lam_anchor_post_warmstart,
                normalize_contexts=args.normalize_contexts,
                damping=args.damping,
                damping_beta=args.damping_beta,
                damping_min=args.damping_min,
                damping_max_trials=args.damping_max_trials,
                gauge_balance=args.gauge_balance,
                gauge_balance_passes=args.gauge_balance_passes,
                adaptive_lambda_step=args.adaptive_lambda_step,
                adaptive_lambda_mode=args.adaptive_lambda_mode,
            ),
            P_star=P_star,
        )
        if args.diagnose_only:
            # Return a trivial curve to keep the rest of the script consistent.
            return np.asarray([0]), np.asarray([np.nan])

    # Train with masked ALS
    opt = MaskedOperatorALS(
        net0, P_star=P_star, lr=args.lr, lam=args.lam, n_sweeps=args.n_sweeps,
        als_reverse_sweep=args.als_reverse_sweep,
        als_gateperm_warmstart=args.als_gateperm_warmstart,
        als_gateperm_warmstart_once=args.als_gateperm_warmstart_once,
        als_lam_anchor_post_warmstart=args.als_lam_anchor_post_warmstart,
        debug_nan=args.debug_nan,
        normalize_contexts=args.normalize_contexts,
        damping=args.damping,
        damping_beta=args.damping_beta,
        damping_min=args.damping_min,
        damping_max_trials=args.damping_max_trials,
        gauge_balance=args.gauge_balance,
        gauge_balance_passes=args.gauge_balance_passes,
        adaptive_lambda_step=args.adaptive_lambda_step,
        adaptive_lambda_mode=args.adaptive_lambda_mode,
    )

    P_star_norm = P_star.norm().item()
    steps, errs = [0], []
    P0 = compute_P_fgln(net0)
    errs.append((P0.cpu() - P_star.cpu()).norm().item() / P_star_norm)

    for step in range(1, args.n_steps + 1):
        try:
            opt.step()
        except (FloatingPointError, torch._C._LinAlgError) as e:
            # Record failure and stop this depth cleanly.
            print(f"  [warn] step={step}: numerical failure: {type(e).__name__}: {e}")
            steps.append(step)
            errs.append(float("nan"))
            break

        P = compute_P_fgln(net0)
        if not torch.isfinite(P).all():
            print(f"  [warn] step={step}: non-finite P(W); stopping.")
            steps.append(step)
            errs.append(float("nan"))
            break

        errs.append((P.cpu() - P_star.cpu()).norm().item() / P_star_norm)
        steps.append(step)

    return np.asarray(steps), np.asarray(errs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FGLN depth diagnostics (masked deep linear operator-ALS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--depths", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--p", type=float, default=0.5, help="Bernoulli gate prob for D_l diagonals")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--rank_star", type=int, default=None,
                   help="If set, use a rank-r isotropic target operator P* "
                        "(top r singular values = 1, rest 0) instead of full-rank orthogonal.")
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=1e-4)
    p.add_argument("--n_sweeps", type=int, default=1)
    p.add_argument(
        "--als_reverse_sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ALS layer updates in reverse order (L-1 ... 0).",
    )
    p.add_argument("--init", choices=["xavier", "orth"], default="orth")
    p.add_argument("--rescale_mode", choices=["all", "last", "none"], default="all",
                   help="How to rescale weights to match ||P0|| to ||P*||.")
    p.add_argument("--init_seed", type=int, default=0)
    p.add_argument("--gate_seed", type=int, default=0)
    p.add_argument("--data_seed", type=int, default=1)
    p.add_argument("--out_dir", default="outputs/fgln")
    p.add_argument("--debug_nan", action="store_true", default=False,
                   help="Print diagnostic stats when non-finite values occur.")
    p.add_argument("--normalize_contexts", action="store_true", default=False,
                   help="Normalize each layer's (M,N,G,lam) by sigma_max(M)*sigma_max(N) "
                        "in an exactly-equivalent way to improve numerics.")
    p.add_argument("--damping", action="store_true", default=False,
                   help="Backtracking damping per layer on the same local ALS objective.")
    p.add_argument("--damping_beta", type=float, default=0.5)
    p.add_argument("--damping_min", type=float, default=1e-6)
    p.add_argument("--damping_max_trials", type=int, default=20)
    p.add_argument("--gauge_balance", action="store_true", default=False,
                   help="Apply scalar gauge balancing between adjacent layers after each sweep.")
    p.add_argument("--gauge_balance_passes", type=int, default=1,
                   help="Number of forward+backward balancing passes per sweep.")
    p.add_argument("--diagnose", action="store_true", default=False,
                   help="Write per-layer context spectrum diagnostics.")
    p.add_argument("--diagnose_after_step", action="store_true", default=False,
                   help="Also write diagnostics after one optimizer step.")
    p.add_argument("--diagnose_only", action="store_true", default=False,
                   help="If set, only write diagnostics (no training loop).")
    p.add_argument("--adaptive_lambda_step", action="store_true", default=False,
                   help="Option A: adaptive lambda per outer optimizer step via context conditioning.")
    p.add_argument("--adaptive_lambda_mode", choices=["median", "max"], default="median",
                   help="How to aggregate per-layer sigma_max(M_k)*sigma_max(N_k) into S(t).")
    p.add_argument("--als_gateperm_warmstart", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--als_gateperm_warmstart_once", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--als_lam_anchor_post_warmstart", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def _sv_minmax_psd(A: torch.Tensor) -> tuple[float, float]:
    sv = torch.linalg.svdvals(A)
    smax = float(sv.max().item())
    smin = float(sv.min().item())
    return smin, smax


def context_stats_for_net(net: FGLN) -> list[dict]:
    """Compute per-layer context statistics (float64) for current weights."""
    layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
    L = len(layers)
    d = layers[0].weight.shape[0]
    device = torch.device("cpu")
    W = [lay.weight.detach().to(device=device, dtype=torch.float64) for lay in layers]
    masks = net.masks.detach().to(device=device, dtype=torch.float64)

    def build_A(k: int) -> torch.Tensor:
        A = torch.eye(d, device=device, dtype=torch.float64)
        for j in range(L - 1, k, -1):
            A = A @ W[j]
            if j - 1 >= 0 and (j - 1) < L - 1:
                A = A @ torch.diag(masks[j - 1])
        return A

    def build_B(k: int) -> torch.Tensor:
        B = torch.eye(d, device=device, dtype=torch.float64)
        for j in range(k):
            B = W[j] @ B
            if j < L - 1 and j < k - 1:
                B = torch.diag(masks[j]) @ B
        return B

    rows: list[dict] = []
    for k in range(L):
        A_k = build_A(k)
        B_k = build_B(k)
        M = A_k.T @ A_k
        N = B_k @ B_k.T
        sminM, smaxM = _sv_minmax_psd(M)
        sminN, smaxN = _sv_minmax_psd(N)
        condM = (smaxM / max(sminM, 1e-300))
        condN = (smaxN / max(sminN, 1e-300))
        rows.append(dict(
            k=k,
            A_maxabs=float(A_k.abs().max().item()),
            B_maxabs=float(B_k.abs().max().item()),
            M_smin=sminM,
            M_smax=smaxM,
            N_smin=sminN,
            N_smax=smaxN,
            M_cond=condM,
            N_cond=condN,
            log10_M_smax=float(np.log10(max(smaxM, 1e-300))),
            log10_N_smax=float(np.log10(max(smaxN, 1e-300))),
            log10_M_cond=float(np.log10(max(condM, 1e-300))),
            log10_N_cond=float(np.log10(max(condN, 1e-300))),
        ))
    return rows


def write_context_diagnostics(
    net: FGLN,
    *,
    depth: int,
    p_gate: float,
    rank_star: Optional[int],
    tag: str,
    out_dir: Path,
    also_after_one_step: bool,
    als_kwargs: dict,
    P_star: torch.Tensor,
) -> None:
    def _write(rows: list[dict], suffix: str) -> None:
        out = out_dir / f"context_diag_{tag}{suffix}.csv"
        cols = list(rows[0].keys()) if rows else []
        lines = [",".join(cols)]
        for r in rows:
            lines.append(",".join(str(r[c]) for c in cols))
        out.write_text("\n".join(lines) + "\n")
        print(f"Saved diagnostics: {out}")

    # Mask summary
    if depth > 1:
        active = net.masks.sum(dim=1).cpu().numpy()
        msum = dict(
            depth=depth,
            p=p_gate,
            rank_star=rank_star if rank_star is not None else "",
            min_active=int(active.min()),
            max_active=int(active.max()),
            mean_active=float(active.mean()),
        )
    else:
        msum = dict(depth=depth, p=p_gate, rank_star=rank_star if rank_star is not None else "",
                    min_active="", max_active="", mean_active="")
    (out_dir / f"mask_summary_{tag}.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in msum.items()) + "\n"
    )

    rows0 = context_stats_for_net(net)
    _write(rows0, "_init")

    if also_after_one_step:
        # one optimizer step (uses the same P_star passed)
        opt = MaskedOperatorALS(net, P_star=P_star, debug_nan=False, **als_kwargs)
        opt.step()
        rows1 = context_stats_for_net(net)
        _write(rows1, "_after1")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for depth in args.depths:
        print(
            f"Running FGLN: d={args.d} depth={depth} p={args.p} lr={args.lr} "
            f"lam={args.lam} sweeps={args.n_sweeps} als_reverse={int(args.als_reverse_sweep)} init={args.init}"
        )
        steps, errs = run_one_depth(args, depth)
        results[depth] = (steps, errs)
        print(f"  final rel err = {errs[-1]:.6g}")

    if args.diagnose_only:
        # Diagnostics-only mode: we wrote the CSVs already, do not try to plot.
        return

    # Plot convergence curves
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for depth, (steps, errs) in sorted(results.items(), key=lambda kv: kv[0]):
        ax.plot(steps, errs, linewidth=1.8, label=f"L={depth}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ALS outer steps")
    ax.set_ylabel(r"$\|P-P^*\|_F / \|P^*\|_F$")
    ax.set_title(
        f"FGLN masked operator-ALS (d={args.d}, p={args.p}, lr={args.lr}, lam={args.lam}, "
        f"sweeps={args.n_sweeps}, reverse={int(args.als_reverse_sweep)}, init={args.init})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()

    tag = (
        f"d{args.d}_p{args.p:g}_lr{args.lr:g}_lam{args.lam:g}"
        f"_sw{args.n_sweeps}_rev{int(args.als_reverse_sweep)}_{args.init}"
    )
    if args.adaptive_lambda_step:
        tag += f"_adapt{args.adaptive_lambda_mode}"
    out_png = out_dir / f"fgln_convergence_{tag}.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

    # Save CSV
    out_csv = out_dir / f"fgln_convergence_{tag}.csv"
    lines = ["depth,step,rel_err"]
    for depth, (steps, errs) in sorted(results.items(), key=lambda kv: kv[0]):
        for s, e in zip(steps.tolist(), errs.tolist()):
            lines.append(f"{depth},{int(s)},{e:.12g}")
    out_csv.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

