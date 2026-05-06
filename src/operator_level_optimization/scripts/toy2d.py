#!/usr/bin/env python3
"""
Toy 2D visualization: one-step update directions in operator space.

Deep linear network  W_L...W_1 = P ∈ R^{1×2}  on a linear regression task.
Each optimizer takes one step from P₀; the resulting ΔP is drawn as an arrow
in (p₁, p₂) space.  Loss contours are overlaid (quadratic, exact).

Usage
-----
    # default: depth=2, hidden=2, Gaussian init
    venv/bin/python labs/toy_2d_linear/run.py

    # specify starting point P₀ = (0.5, -0.3)
    venv/bin/python labs/toy_2d_linear/run.py --p0 0.5 -0.3

    # larger hidden state
    venv/bin/python labs/toy_2d_linear/run.py --hidden 16 --depth 3

See --help for all options.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from operator_level_optimization.core.optim import Muon, KFAC, Shampoo, SOAP


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

def make_dataset(
    n: int, noise: float, seed: int,
    feature_scales: tuple[float, float] = (1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X, y) for a random linear regression task.

    X ∈ R^{N×2}, y = X w_true + ε.  w_true is fixed by seed.
    feature_scales: multiply each feature column by these factors, inducing
    anisotropy in the loss Hessian (useful for visualising GD zigzag).
    """
    rng = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 2, generator=rng)
    scale = torch.tensor(feature_scales, dtype=X.dtype)
    X = X * scale
    w_true = torch.randn(2, generator=rng)
    y = X @ w_true + noise * torch.randn(n, generator=rng)
    return X, y


def ols_solution(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """P* = w*ᵀ ∈ R^{1×2}, OLS solution."""
    w = torch.linalg.lstsq(X, y.unsqueeze(1)).solution  # (2, 1)
    return w.T  # (1, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Network
# ══════════════════════════════════════════════════════════════════════════════

class DeepLinear(nn.Module):
    """Pure linear chain: input_dim=2, output_dim=1, no biases.

    depth = number of weight matrices L.
      L=1  →  W ∈ R^{1×2}
      L=2  →  W₂(1×h) · W₁(h×2)
      L≥3  →  W_L(1×h) · W_{L-1}(h×h) · ... · W₁(h×2)
    """

    def __init__(self, depth: int, hidden: int):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be ≥ 1")
        layers: list[nn.Linear] = []
        if depth == 1:
            layers.append(nn.Linear(2, 1, bias=False))
        else:
            layers.append(nn.Linear(2, hidden, bias=False))
            for _ in range(depth - 2):
                layers.append(nn.Linear(hidden, hidden, bias=False))
            layers.append(nn.Linear(hidden, 1, bias=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def compute_P(net: DeepLinear) -> torch.Tensor:
    """Return P = W_L...W₁ ∈ R^{1×2} (detached copy)."""
    with torch.no_grad():
        P = net.layers[0].weight
        for layer in net.layers[1:]:
            P = layer.weight @ P
    return P.detach().clone()


def _als_block_delta(A: torch.Tensor, B: torch.Tensor, R: torch.Tensor, lam: float) -> torch.Tensor:
    """Solve min ||A dW B - R||_F^2 + lam ||dW||_F^2 exactly."""
    A = A.contiguous()
    B = B.contiguous()
    R = R.contiguous()
    K = torch.kron(B.T.contiguous(), A)
    rhs = R.reshape(-1, 1)
    lhs = K.T @ K
    if lam > 0.0:
        lhs = lhs + lam * torch.eye(lhs.shape[0], dtype=lhs.dtype, device=lhs.device)
    sol = torch.linalg.pinv(lhs) @ (K.T @ rhs)
    return sol.reshape(A.shape[1], B.shape[0])


# ══════════════════════════════════════════════════════════════════════════════
# Initialization
# ══════════════════════════════════════════════════════════════════════════════

def init_gaussian(net: DeepLinear, seed: int) -> None:
    """Standard Gaussian init: W_l ~ N(0, 1/fan_in)."""
    rng = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in net.layers:
            fan_in = layer.weight.shape[1]
            layer.weight.copy_(
                torch.randn(layer.weight.shape, generator=rng) / fan_in ** 0.5
            )


def init_identity_like(net: DeepLinear) -> None:
    """Identity-like initialization across a rectangular deep linear chain."""
    with torch.no_grad():
        for layer in net.layers:
            w = torch.zeros_like(layer.weight)
            d = min(w.shape[0], w.shape[1])
            if d > 0:
                idx = torch.arange(d, device=w.device)
                w[idx, idx] = 1.0
            layer.weight.copy_(w)


def init_haar(net: DeepLinear, seed: int) -> None:
    """Haar (random orthogonal) init: each layer is a random isometry."""
    rng = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in net.layers:
            h_out, h_in = layer.weight.shape
            n = max(h_out, h_in)
            G = torch.randn(n, n, generator=rng)
            Q, _ = torch.linalg.qr(G)
            layer.weight.copy_(Q[:h_out, :h_in])


def init_to_p0(net: DeepLinear, p0: torch.Tensor) -> None:
    """Balanced init such that W_L...W₁ = p0 exactly.

    Each layer is scaled by ‖p0‖^{1/L}.  All action flows through the first
    hidden unit; middle layers are scaled identity matrices.

    Layout (depth=L, hidden=h):
      W₁ ∈ R^{h×2}   : first row = s · (p0/‖p0‖), rest zeros
      Wₗ ∈ R^{h×h}   : s · Iₕ  (l = 2..L-1)
      W_L ∈ R^{1×h}  : [s, 0, ..., 0]
    """
    v = p0.float().squeeze()  # (2,)
    norm = v.norm().item()
    s = (norm + 1e-30) ** (1.0 / len(net.layers))
    unit = v / (norm + 1e-30)

    with torch.no_grad():
        if len(net.layers) == 1:
            net.layers[0].weight.copy_(v.unsqueeze(0))
            return

        h = net.layers[0].weight.shape[0]

        w1 = torch.zeros_like(net.layers[0].weight)  # (h, 2)
        w1[0] = s * unit
        net.layers[0].weight.copy_(w1)

        for l in range(1, len(net.layers) - 1):
            wm = torch.zeros_like(net.layers[l].weight)  # (h, h)
            wm.fill_diagonal_(s)
            net.layers[l].weight.copy_(wm)

        wL = torch.zeros_like(net.layers[-1].weight)  # (1, h)
        wL[0, 0] = s
        net.layers[-1].weight.copy_(wL)


# ══════════════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════════════

def mse(net: DeepLinear, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_col = y.unsqueeze(-1) if y.ndim == 1 else y
    return 0.5 * ((net(X) - y_col) ** 2).mean()


class ToyOperatorALS:
    """Deep-linear ALS-exact matching deep_linear_compare implementation."""

    needs_backward = False

    def __init__(self, model: DeepLinear, X: torch.Tensor, y: torch.Tensor, lr: float, lam: float, n_sweeps: int):
        self.model = model
        self.X = X.detach()
        self.y = y.detach()
        self.lr = lr
        self.lam = lam
        self.n_sweeps = n_sweeps

    def zero_grad(self) -> None:
        pass

    @torch.no_grad()
    def step(self) -> None:
        _als_shared_target_step(
            self.model,
            self.X,
            self.y,
            lr=self.lr,
            lam=self.lam,
            n_sweeps=self.n_sweeps,
            reverse=True,
            warmstart_identity=False,
        )


def _als_shared_target_step(
    net: DeepLinear,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    lr: float,
    lam: float,
    n_sweeps: int,
    reverse: bool = True,
    warmstart_identity: bool = True,
) -> None:
    """ALS step for rectangular deep-linear chain with shared operator target."""
    with torch.no_grad():
        W = [l.weight.data.clone() for l in net.layers]
        P = compute_P(net)
        pred = (X @ P.T).squeeze(-1)
        grad_p = (2.0 / X.shape[0]) * (pred - y).unsqueeze(1) * X
        grad_p = grad_p.mean(dim=0, keepdim=True)
        P_tgt = P - lr * grad_p

        if warmstart_identity:
            for k in range(len(W)):
                I = torch.zeros_like(W[k])
                m = min(W[k].shape[0], W[k].shape[1])
                idx = torch.arange(m, device=W[k].device)
                I[idx, idx] = 1.0
                W[k] = I

        order = list(range(len(W) - 1, -1, -1)) if reverse else list(range(len(W)))
        for _ in range(n_sweeps):
            for k in order:
                if k == len(W) - 1:
                    A = torch.eye(W[k].shape[0], dtype=W[k].dtype, device=W[k].device)
                else:
                    A = W[-1]
                    for i in range(len(W) - 2, k, -1):
                        A = A @ W[i]
                if k == 0:
                    B = torch.eye(W[k].shape[1], dtype=W[k].dtype, device=W[k].device)
                else:
                    B = W[k - 1]
                    for i in range(k - 2, -1, -1):
                        B = B @ W[i]
                R = P_tgt - (A @ W[k] @ B)
                dW = _als_block_delta(A, B, R, lam)
                W[k] = W[k] + dW

        for layer, w_new in zip(net.layers, W):
            layer.weight.data.copy_(w_new)


# ══════════════════════════════════════════════════════════════════════════════
# One-step updates
# ══════════════════════════════════════════════════════════════════════════════

def _one_step(
    net_ref: DeepLinear,
    X: torch.Tensor,
    y: torch.Tensor,
    make_opt,          # callable: nn.Module → Optimizer
) -> torch.Tensor:
    """Clone net, run one optimizer step, return ΔP = P_new − P₀."""
    net = copy.deepcopy(net_ref)
    P0 = compute_P(net_ref)

    opt = make_opt(net)
    if hasattr(opt, "attach_hooks"):
        opt.attach_hooks(net)

    opt.zero_grad()
    loss_val = mse(net, X, y)
    if not getattr(opt, "needs_backward", False):
        loss_val.backward()
    opt.step()

    return (compute_P(net) - P0).detach()


def compute_updates(
    net: DeepLinear,
    X: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    """Return {name: ΔP} for every enabled optimizer.
    """
    lam = args.lam

    # Build ordered map: key (for --optimizers filter) → (display name, factory)
    candidates: list[tuple[str, str, object]] = [
        ("heavyball", "Heavy Ball", lambda m: torch.optim.SGD(
            m.parameters(), lr=args.lr_hb, momentum=args.hb_momentum
        )),
        ("adam",     "Adam",     lambda m: torch.optim.Adam(
                                     m.parameters(), lr=args.lr_adam,
                                     betas=(args.adam_b1, args.adam_b2))),
        ("muon",     "Muon",     lambda m: Muon(
                                     m.parameters(), lr=args.lr_muon,
                                     momentum=args.momentum)),
        ("kfac",    "K-FAC",    lambda m: KFAC(m.parameters(), lr=args.lr_kfac)),
        ("shampoo", "Shampoo",  lambda m: Shampoo(m.parameters(), lr=args.lr_shampoo)),
        ("soap",    "SOAP",     lambda m: SOAP(m.parameters(), lr=args.lr_soap)),
        ("als_exact", "ALS-Exact", None),
    ]

    enabled = set(args.optimizers)
    updates: dict[str, torch.Tensor] = {}
    for key, name, factory in candidates:
        if key not in enabled:
            continue
        print(f"  computing step: {name} ...", end=" ", flush=True)
        try:
            if key == "als_exact":
                updates[name] = _one_step(
                    net,
                    X,
                    y,
                    lambda m: ToyOperatorALS(
                        m, X, y, lr=args.lr_als, lam=lam, n_sweeps=args.als_sweeps
                    ),
                )
            else:
                updates[name] = _one_step(net, X, y, factory)
            print("ok")
        except Exception as exc:
            print(f"FAILED ({exc})")
    return updates


# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════

_COLORS = {
    "Heavy Ball": "#888888",
    "Adam":    "#e07b00",
    "Muon":     "#2ca02c",
    "K-FAC":    "#1f77b4",
    "Shampoo":  "#17becf",
    "SOAP":     "#8c564b",
    "ALS-Exact": "#9467bd",
}

_PLOT_OPT_ORDER = ["Heavy Ball", "Adam", "Muon", "K-FAC", "Shampoo", "SOAP", "ALS-Exact"]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _loss_grid(
    p1g: np.ndarray,
    p2g: np.ndarray,
    X: torch.Tensor,
    y: torch.Tensor,
) -> np.ndarray:
    """Evaluate L(P) = 0.5/N ‖X Pᵀ − y‖² on a meshgrid (exact, analytical)."""
    A = X[:, 0].numpy()
    B = X[:, 1].numpy()
    yt = y.numpy()
    N = len(yt)
    sAA = (A * A).sum() / N
    sBB = (B * B).sum() / N
    sAB = (A * B).sum() / N
    sAy = (A * yt).sum() / N
    sBy = (B * yt).sum() / N
    syy = (yt * yt).sum() / N
    return 0.5 * (
        sAA * p1g**2
        + 2 * sAB * p1g * p2g
        + sBB * p2g**2
        - 2 * sAy * p1g
        - 2 * sBy * p2g
        + syy
    )


def make_figure(
    net: DeepLinear,
    P0: torch.Tensor,
    Pstar: torch.Tensor,
    updates: dict[str, torch.Tensor],
    X: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
    out_path: str,
) -> None:
    p0 = P0.squeeze().numpy()
    ps = Pstar.squeeze().numpy()
    dist_to_pstar = float(np.linalg.norm(ps - p0))

    # ── view: center midway between P₀ and P*, span fits both with margin ─────
    center = (p0 + ps) / 2.0
    # Half-distance from center to either endpoint, plus 30% margin for arrows
    half_dist = dist_to_pstar / 2.0
    span = half_dist * 1.6  # 60% extra room around the P₀–P* segment

    # In non-normalized mode, also ensure real arrows fit in view
    raw_arrow_ends = np.stack([p0 + dP.squeeze().numpy() for dP in updates.values()]) \
        if updates else p0[None]
    arrow_spread = float(np.abs(raw_arrow_ends - center).max()) + 1e-6
    if not args.normalize_arrows:
        span = max(span, arrow_spread * 1.2)

    res = args.grid_res
    p1_vals = np.linspace(center[0] - span, center[0] + span, res)
    p2_vals = np.linspace(center[1] - span, center[1] + span, res)
    P1g, P2g = np.meshgrid(p1_vals, p2_vals)
    Z = _loss_grid(P1g, P2g, X, y)

    # ── figure: wider to accommodate external legend ──────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Contours
    z_min, z_max = Z.min(), Z.max()
    levels = np.linspace(z_min, z_min + (z_max - z_min) * 0.9, args.contour_levels)
    cf = ax.contourf(P1g, P2g, Z, levels=levels, cmap="Blues", alpha=0.45)
    ax.contour(P1g, P2g, Z, levels=levels, colors="white", linewidths=0.4, alpha=0.6)
    cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"$\mathcal{L}(P)$", fontsize=11)

    # P₀ and P* — always both in frame by construction
    ax.scatter(*p0, s=90,  marker="o", color="black",  zorder=10)
    ax.scatter(*ps, s=250, marker="*", color="gold",   zorder=10,
               edgecolors="black", linewidths=0.8)
    ax.annotate(r"$P_0$", xy=p0, xytext=(p0[0] + span*0.05, p0[1] + span*0.05),
                fontsize=10, color="black")
    ax.annotate(r"$P^*$", xy=ps, xytext=(ps[0] + span*0.05, ps[1] + span*0.05),
                fontsize=10, color="#b8860b")

    # Update arrows (optionally normalized to equal display length)
    arrowprops_base = dict(arrowstyle="->", lw=2.2, mutation_scale=16)
    raw_dps = {name: dP.squeeze().numpy() for name, dP in updates.items()}
    if args.normalize_arrows:
        # Arrows scaled to 45% of half_dist so they're clearly visible but don't crowd P*
        display_len = half_dist * 0.85
        dps = {
            name: dp / (np.linalg.norm(dp) + 1e-30) * display_len
            for name, dp in raw_dps.items()
        }
    else:
        dps = raw_dps

    legend_handles = []
    draw_names = [n for n in _PLOT_OPT_ORDER if n in dps] + [n for n in dps.keys() if n not in _PLOT_OPT_ORDER]
    base_lw = 4.6
    lw_decay = 0.45
    min_lw = 1.6
    for idx, name in enumerate(draw_names):
        dp = dps[name]
        color = _COLORS.get(name, "black")
        lw = max(min_lw, base_lw - idx * lw_decay)
        ax.annotate(
            "", xy=p0 + dp, xytext=p0,
            arrowprops={**arrowprops_base, "color": color, "lw": lw},
        )
        mag = float(np.linalg.norm(raw_dps[name]))
        handle, = ax.plot([], [], color=color, linewidth=2,
                          label=f"{name}  ($|\\Delta P|={mag:.2e}$)")
        legend_handles.append(handle)

    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_xlabel(r"$p_1$", fontsize=12)
    ax.set_ylabel(r"$p_2$", fontsize=12)
    depth = len(net.layers)
    hidden = net.layers[0].weight.shape[0] if depth > 1 else "—"
    if not args.no_title_one_step:
        ax.set_title(
            f"One-step update directions in operator space\n"
            f"$L={depth}$ layers, $h={hidden}$, $N={args.n_samples}$, "
            f"$\\lambda={args.lam}$",
            fontsize=11,
        )
    # Legend outside the axes (below), so it never overlaps content
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        fontsize=9,
        framealpha=0.9,
        handlelength=1.8,
    )
    ax.set_aspect("equal")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-step trajectory
# ══════════════════════════════════════════════════════════════════════════════

def compute_trajectory(
    net_ref: DeepLinear,
    X: torch.Tensor,
    y: torch.Tensor,
    make_opt,
    n_steps: int,
) -> np.ndarray:
    """Run optimizer for n_steps from a clone of net_ref; return trajectory as
    (n_steps+1, 2) array of P values in operator space."""
    net = copy.deepcopy(net_ref)
    opt = make_opt(net)
    if hasattr(opt, "attach_hooks"):
        opt.attach_hooks(net)

    traj = [compute_P(net).squeeze().numpy().copy()]
    for _ in range(n_steps):
        opt.zero_grad()
        loss_t = mse(net, X, y)
        if not getattr(opt, "needs_backward", False):
            loss_t.backward()
        opt.step()
        traj.append(compute_P(net).squeeze().numpy().copy())
    return np.array(traj)  # (n_steps+1, 2)


def compute_trajectories(
    net: DeepLinear,
    X: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    lam = args.lam
    candidates: list[tuple[str, str, object]] = [
        ("heavyball", "Heavy Ball", lambda m: torch.optim.SGD(
            m.parameters(), lr=args.lr_hb, momentum=args.hb_momentum
        )),
        ("adam",     "Adam",     lambda m: torch.optim.Adam(
                                     m.parameters(), lr=args.lr_adam,
                                     betas=(args.adam_b1, args.adam_b2))),
        ("muon",     "Muon",     lambda m: Muon(
                                     m.parameters(), lr=args.lr_muon,
                                     momentum=args.momentum)),
        ("kfac",    "K-FAC",    lambda m: KFAC(m.parameters(), lr=args.lr_kfac)),
        ("shampoo", "Shampoo",  lambda m: Shampoo(m.parameters(), lr=args.lr_shampoo)),
        ("soap",    "SOAP",     lambda m: SOAP(m.parameters(), lr=args.lr_soap)),
        ("als_exact", "ALS-Exact", None),
    ]

    enabled = set(args.optimizers)
    trajs: dict[str, np.ndarray] = {}
    for key, name, factory in candidates:
        if key not in enabled:
            continue
        print(f"  trajectory: {name} ...", end=" ", flush=True)
        try:
            if key == "als_exact":
                t = compute_trajectory(
                    net,
                    X,
                    y,
                    lambda m: ToyOperatorALS(
                        m, X, y, lr=args.lr_als, lam=lam, n_sweeps=args.als_sweeps
                    ),
                    args.n_steps,
                )
            else:
                t = compute_trajectory(net, X, y, factory, args.n_steps)
            if np.isnan(t).any():
                first_nan = int(np.argmax(np.isnan(t).any(axis=1)))
                print(f"DIVERGED at step {first_nan} — skipping")
            else:
                trajs[name] = t
                fp = t[-1]
                print(f"ok  (final P=[{fp[0]:.4f},{fp[1]:.4f}])")
        except Exception as exc:
            print(f"FAILED ({exc})")
    return trajs


def save_updates_data(
    path: str,
    P0: torch.Tensor,
    Pstar: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    updates: dict[str, torch.Tensor],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, np.ndarray] = {
        "P0": P0.detach().cpu().numpy(),
        "Pstar": Pstar.detach().cpu().numpy(),
        "X": X.detach().cpu().numpy(),
        "y": y.detach().cpu().numpy(),
    }
    for name, dP in updates.items():
        data[f"update_{_slug(name)}"] = dP.detach().cpu().numpy()
    np.savez(out, **data)


def save_trajectories_data(
    path: str,
    P0: torch.Tensor,
    Pstar: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    trajs: dict[str, np.ndarray],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, np.ndarray] = {
        "P0": P0.detach().cpu().numpy(),
        "Pstar": Pstar.detach().cpu().numpy(),
        "X": X.detach().cpu().numpy(),
        "y": y.detach().cpu().numpy(),
    }
    for name, t in trajs.items():
        data[f"traj_{_slug(name)}"] = t
    np.savez(out, **data)


def _draw_trajectory_panel(
    ax,
    net: DeepLinear,
    P0: torch.Tensor,
    Pstar: torch.Tensor,
    trajs: dict[str, np.ndarray],
    X: torch.Tensor,
    y: torch.Tensor,
    n_steps: int,
    contour_levels: int,
    grid_res: int,
    paper: bool = False,
    show_ylabel: bool = True,
    show_legend: bool = False,
) -> list:
    """Draw one trajectory panel into ax.  Returns legend handles."""
    fs = 13 if paper else 11
    p0 = P0.squeeze().numpy()
    ps = Pstar.squeeze().numpy()

    # Compute span only from points that stay within 3× P0-P* distance of P*.
    # Diverged trajectories (K-FAC at L=32) are excluded and clipped by axes limits.
    ref_dist = float(np.linalg.norm(ps - p0)) + 1e-6
    candidate_pts = [p0, ps]
    for t in trajs.values():
        candidate_pts.append(t[0])  # always include start
        close = t[np.linalg.norm(t - ps, axis=1) < 3.0 * ref_dist]
        if len(close):
            candidate_pts.extend(close.tolist())
    all_pts = np.array(candidate_pts)
    center = (all_pts.max(0) + all_pts.min(0)) / 2.0
    span = float(np.abs(all_pts - center).max()) * 1.2 + 1e-3

    p1v = np.linspace(center[0] - span, center[0] + span, grid_res)
    p2v = np.linspace(center[1] - span, center[1] + span, grid_res)
    P1g, P2g = np.meshgrid(p1v, p2v)
    Z = _loss_grid(P1g, P2g, X, y)

    z_min, z_max = Z.min(), Z.max()
    levels = np.linspace(z_min, z_min + (z_max - z_min) * 0.9, contour_levels)
    cf = ax.contourf(P1g, P2g, Z, levels=levels, cmap="Blues", alpha=0.40)
    ax.contour(P1g, P2g, Z, levels=levels, colors="white", linewidths=0.4, alpha=0.5)

    legend_handles = []
    stride = max(1, n_steps // 80)
    draw_names = [n for n in _PLOT_OPT_ORDER if n in trajs] + [n for n in trajs.keys() if n not in _PLOT_OPT_ORDER]
    for idx, name in enumerate(draw_names):
        traj = trajs[name]
        color = _COLORS.get(name, "black")
        xs, ys = traj[:, 0], traj[:, 1]
        z = 3 + idx
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.85, zorder=z)
        dot_idx = np.arange(0, len(traj), stride)
        ax.scatter(xs[dot_idx], ys[dot_idx], s=10, color=color, zorder=z + 0.1, alpha=0.75)
        ax.scatter(xs[-1], ys[-1], s=60, color=color, zorder=z + 0.2,
                   edgecolors="white", linewidths=0.8)
        handle, = ax.plot([], [], color=color, linewidth=2, label=name)
        legend_handles.append(handle)

    ax.scatter(*p0, s=100, marker="o", color="black", zorder=10)
    ax.scatter(*ps, s=280, marker="*", color="gold", zorder=10,
               edgecolors="black", linewidths=0.8)
    ax.annotate(r"$P_0$", xy=p0, xytext=(p0[0] + span*0.04, p0[1] + span*0.04),
                fontsize=fs - 1, color="black")
    ax.annotate(r"$P^*$", xy=ps, xytext=(ps[0] + span*0.04, ps[1] + span*0.04),
                fontsize=fs - 1, color="#b8860b")

    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_xlabel(r"$p_1$", fontsize=fs)
    if show_ylabel:
        ax.set_ylabel(r"$p_2$", fontsize=fs)
    ax.tick_params(labelsize=fs - 2)
    ax.set_aspect("equal")

    depth = len(net.layers)
    if paper:
        ax.set_title(f"$L = {depth}$", fontsize=fs + 1)
    else:
        hidden = net.layers[0].weight.shape[0] if depth > 1 else "—"
        ax.set_title(
            f"Training trajectories in operator space\n"
            f"$L={depth}$, $h={hidden}$, $N=100$, $\\lambda=0.1$, {n_steps} steps",
            fontsize=fs - 1,
        )

    return legend_handles, cf


def make_trajectory_figure(
    net: DeepLinear,
    P0: torch.Tensor,
    Pstar: torch.Tensor,
    trajs: dict[str, np.ndarray],
    X: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    legend_handles, cf = _draw_trajectory_panel(
        ax, net, P0, Pstar, trajs, X, y,
        args.n_steps, args.contour_levels, args.grid_res,
        paper=False,
    )
    cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"$\mathcal{L}(P)$", fontsize=11)
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9,
              framealpha=0.9, handlelength=1.8)
    if args.no_title_trajectory:
        ax.set_title("")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Trajectory figure saved: {out}")
    plt.close(fig)


def make_paper_figure(
    panels: list[tuple],   # list of (net, P0, Pstar, trajs, X, y)
    args: argparse.Namespace,
    out_path: str,
) -> None:
    """Two-panel paper figure: one column per depth."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.2),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    all_handles = None
    for i, (ax, (net, P0, Pstar, trajs, X, y)) in enumerate(zip(axes, panels)):
        handles, cf = _draw_trajectory_panel(
            ax, net, P0, Pstar, trajs, X, y,
            args.n_steps, args.contour_levels, args.grid_res,
            paper=True,
            show_ylabel=(i == 0),
        )
        if all_handles is None:
            all_handles = handles

    # Shared legend below both panels
    fig.legend(handles=all_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=len(all_handles),
               fontsize=12, framealpha=0.9, handlelength=2.0)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Paper figure saved: {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark mode: long-run MSE + spectrum snapshots
# ══════════════════════════════════════════════════════════════════════════════

def _optimizer_factories(args: argparse.Namespace, X: torch.Tensor, y: torch.Tensor):
    lam = args.lam

    def op_als_exact():
        return lambda m: ToyOperatorALS(m, X, y, lr=args.lr_als, lam=lam, n_sweeps=args.als_sweeps)

    return [
        ("adam", "Adam", lambda m: torch.optim.Adam(m.parameters(), lr=args.lr_adam, betas=(args.adam_b1, args.adam_b2))),
        ("muon", "Muon", lambda m: Muon(m.parameters(), lr=args.lr_muon, momentum=args.momentum)),
        ("kfac", "K-FAC", lambda m: KFAC(m.parameters(), lr=args.lr_kfac)),
        ("shampoo", "Shampoo", lambda m: Shampoo(m.parameters(), lr=args.lr_shampoo)),
        ("soap", "SOAP", lambda m: SOAP(m.parameters(), lr=args.lr_soap)),
        ("als_exact", "ALS-Exact", op_als_exact()),
    ]


def _train_curve(
    net_ref: DeepLinear,
    X: torch.Tensor,
    y: torch.Tensor,
    make_opt,
    max_steps: int,
    early_patience: int,
    early_min_delta: float,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, bool]:
    net = copy.deepcopy(net_ref)
    opt = make_opt(net)
    if hasattr(opt, "attach_hooks"):
        opt.attach_hooks(net)

    losses = []
    best = float("inf")
    bad = 0
    P_init = compute_P(net)
    diverged = False
    for _ in range(max_steps):
        opt.zero_grad()
        loss_t = mse(net, X, y)
        if not getattr(opt, "needs_backward", False):
            loss_t.backward()
        opt.step()
        loss_val = float(mse(net, X, y).detach().cpu())
        if not np.isfinite(loss_val):
            diverged = True
            break
        losses.append(loss_val)

        if loss_val < best - early_min_delta:
            best = loss_val
            bad = 0
        else:
            bad += 1
        if bad >= early_patience:
            break

    return np.asarray(losses, dtype=np.float64), P_init, compute_P(net), diverged


def _plot_mse_curves(curves: dict[str, np.ndarray], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, vals in curves.items():
        xs = np.arange(1, len(vals) + 1)
        vals_clamped = np.minimum(vals, 1.0)
        ax.plot(xs, vals_clamped, label=name, color=_COLORS.get(name, None), linewidth=1.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE (clamped at 1.0)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_spectrum_snapshots(
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for name, (s0, s1) in spectra.items():
        x0 = np.arange(1, len(s0) + 1)
        x1 = np.arange(1, len(s1) + 1)
        axes[0].plot(x0, s0, marker="o", linewidth=1.5, markersize=3, label=name, color=_COLORS.get(name, None))
        axes[1].plot(x1, s1, marker="o", linewidth=1.5, markersize=3, label=name, color=_COLORS.get(name, None))

    axes[0].set_title("Init operator spectrum")
    axes[1].set_title("End operator spectrum")
    axes[0].set_xlabel("Singular index")
    axes[1].set_xlabel("Singular index")
    axes[0].set_ylabel("Singular value")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _run_base_dir(args: argparse.Namespace) -> Path:
    run_name = (
        f"depth{args.depth}_hidden{args.hidden}_steps{args.long_steps}_"
        f"seed{args.data_seed}_noise{args.noise:g}"
    )
    return Path(args.run_dir) / run_name


def _save_optimizer_run(path: Path, loss_curve: np.ndarray, s0: np.ndarray, s1: np.ndarray, diverged: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        loss_curve=loss_curve,
        spectrum_init=s0,
        spectrum_end=s1,
        diverged=np.asarray([1 if diverged else 0], dtype=np.int32),
    )


def _load_optimizer_run(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    data = np.load(path)
    return (
        data["loss_curve"],
        data["spectrum_init"],
        data["spectrum_end"],
        bool(int(data["diverged"][0])),
    )


def _render_long_benchmark_from_cache(args: argparse.Namespace, init_mode: str) -> None:
    base_dir = _run_base_dir(args)
    init_dir = base_dir / init_mode
    curves: dict[str, np.ndarray] = {}
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    display_names = {
        "adam": "Adam",
        "muon": "Muon",
        "kfac": "K-FAC",
        "shampoo": "Shampoo",
        "soap": "SOAP",
        "als_exact": "ALS-Exact",
        "heavyball": "Heavy Ball",
    }
    for key in args.optimizers:
        name = display_names[key]
        cache_path = init_dir / f"{name.replace(' ', '_').lower()}.npz"
        if not cache_path.exists():
            continue
        loss_curve, s0, s1, _ = _load_optimizer_run(cache_path)
        curves[name] = loss_curve
        spectra[name] = (s0, s1)

    out_dir = base_dir / "figures"
    if curves:
        _plot_mse_curves(
            curves,
            out_dir / f"mse_linear_{init_mode}_L{args.depth}.png",
            title=f"MSE vs step ({init_mode} init, L={args.depth})",
        )
    if spectra:
        _plot_spectrum_snapshots(
            spectra,
            out_dir / f"spectrum_init_end_{init_mode}_L{args.depth}.png",
            title=f"Operator spectrum snapshots ({init_mode} init, L={args.depth})",
        )


def run_long_benchmark(args: argparse.Namespace, init_mode: str, X: torch.Tensor, y: torch.Tensor) -> None:
    net = DeepLinear(args.depth, args.hidden)
    if init_mode == "xavier":
        init_gaussian(net, args.init_seed)
    elif init_mode == "identity":
        init_identity_like(net)
    elif init_mode == "haar":
        init_haar(net, args.init_seed)
    else:
        raise ValueError(f"Unknown init mode: {init_mode}")

    base_dir = _run_base_dir(args)
    init_dir = base_dir / init_mode
    meta_path = base_dir / "metadata.json"
    init_dir.mkdir(parents=True, exist_ok=True)
    if not meta_path.exists():
        meta = {
            "depth": args.depth,
            "hidden": args.hidden,
            "long_steps": args.long_steps,
            "early_patience": args.early_patience,
            "early_min_delta": args.early_min_delta,
            "data_seed": args.data_seed,
            "noise": args.noise,
            "n_samples": args.n_samples,
            "optimizers": args.optimizers,
            "benchmark_inits": args.benchmark_inits,
            "lrs": {
                "adam": args.lr_adam,
                "muon": args.lr_muon,
                "kfac": args.lr_kfac,
                "shampoo": args.lr_shampoo,
                "soap": args.lr_soap,
                "als_exact": args.lr_als,
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    enabled = set(args.optimizers)
    for key, name, factory in _optimizer_factories(args, X, y):
        if key not in enabled:
            continue
        cache_path = init_dir / f"{name.replace(' ', '_').lower()}.npz"
        if cache_path.exists() and not args.force_recompute:
            print(f"[{init_mode}] using cached {name}")
            continue
        print(f"[{init_mode}] training {name} ...", end=" ", flush=True)
        try:
            loss_curve, P_init, P_end, diverged = _train_curve(
                net,
                X,
                y,
                factory,
                max_steps=args.long_steps,
                early_patience=args.early_patience,
                early_min_delta=args.early_min_delta,
            )
            if len(loss_curve) == 0:
                print("diverged immediately")
                continue
            s0 = torch.linalg.svdvals(P_init).detach().cpu().numpy()
            if torch.isfinite(P_end).all():
                s1 = torch.linalg.svdvals(P_end).detach().cpu().numpy()
            else:
                s1 = np.full_like(s0, np.nan)
            _save_optimizer_run(cache_path, loss_curve, s0, s1, diverged)
            tag = " (diverged)" if diverged else ""
            print(f"done ({len(loss_curve)} steps, final mse={loss_curve[-1]:.4e}){tag}")
        except Exception as exc:
            print(f"FAILED ({exc})")
    _render_long_benchmark_from_cache(args, init_mode)

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

_ALL_OPTS = ["heavyball", "adam", "muon", "kfac", "shampoo", "soap", "als_exact"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2D toy: one-step update directions for deep linear networks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Architecture
    g = p.add_argument_group("architecture")
    g.add_argument("--depth",  type=int, default=2,
                   help="Number of weight matrices L (depth=2 → 1 hidden layer)")
    g.add_argument("--hidden", type=int, default=2,
                   help="Hidden dimension h (ignored when depth=1)")

    # Dataset
    g = p.add_argument_group("dataset")
    g.add_argument("--n_samples",     type=int,   default=100)
    g.add_argument("--noise",         type=float, default=0.1)
    g.add_argument("--data_seed",     type=int,   default=0)
    g.add_argument("--feature_scales", type=float, nargs=2, default=(1.0, 1.0),
                   metavar=("S1", "S2"),
                   help="Scale factors for each input feature column. "
                        "Use e.g. --feature_scales 4.0 1.0 for a 16× Hessian anisotropy.")

    # Initialization
    g = p.add_argument_group("initialization")
    g.add_argument("--p0",       type=float, nargs=2, default=None, metavar=("P1", "P2"),
                   help="Fix starting point P₀=(p1,p2). Default: Gaussian random weights.")
    g.add_argument("--init_seed", type=int, default=0,
                   help="Seed for Gaussian weight init (ignored when --p0 is set)")
    g.add_argument("--init_mode", type=str, default="gaussian",
                   choices=["gaussian", "identity", "haar"],
                   help="Weight initialization mode when --p0 is not provided.")

    # Per-optimizer learning rates
    g = p.add_argument_group("learning rates")
    g.add_argument("--lr_hb",    type=float, default=1e-3)
    g.add_argument("--lr_adam", type=float, default=1e-3)
    g.add_argument("--lr_muon",  type=float, default=1e-3)
    g.add_argument("--lr_kfac",  type=float, default=5e-5)
    g.add_argument("--lr_shampoo", type=float, default=1e-3)
    g.add_argument("--lr_soap", type=float, default=1e-3)
    g.add_argument("--lr_als",   type=float, default=1.0)

    # Shared optimizer hyperparameters
    g = p.add_argument_group("optimizer hyperparameters")
    g.add_argument("--lam",        type=float, default=1e-4,
                   help="Tikhonov λ for ALS-Exact optimizer")
    g.add_argument("--momentum",   type=float, default=0.0,
                   help="Momentum for Muon")
    g.add_argument("--hb_momentum", type=float, default=0.9)
    g.add_argument("--adam_b1",   type=float, default=0.9)
    g.add_argument("--adam_b2",   type=float, default=0.999)
    g.add_argument("--als_sweeps", type=int, default=10)

    # Which optimizers to include
    g = p.add_argument_group("optimizer selection")
    g.add_argument("--optimizers", nargs="+", default=_ALL_OPTS,
                   choices=_ALL_OPTS, metavar="OPT",
                   help=f"Optimizers to include. Choices: {_ALL_OPTS}")

    # Trajectory mode
    g = p.add_argument_group("trajectory")
    g.add_argument("--trajectory", action="store_true",
                   help="Generate a training-trajectory figure instead of the one-step figure")
    g.add_argument("--n_steps",    type=int, default=200,
                   help="Number of optimizer steps for trajectory mode")
    g.add_argument("--out_traj", default=None,
                   help="Output path for trajectory figure (default: <out_dir>/trajectory.png)")
    g.add_argument("--paper_depths", type=int, nargs="+", default=None,
                   metavar="L",
                   help="Generate a single multi-panel paper figure with one panel per depth. "
                        "E.g. --paper_depths 2 32  --out_traj paper_traj.pdf")
    g.add_argument("--no_paper_share_p0", action="store_true",
                   help="In --paper_depths mode, do not force a shared P0 across depths.")
    g.add_argument("--no_title_trajectory", action="store_true",
                   help="Do not render title text on trajectory figure.")
    g.add_argument("--save_data", default=None,
                   help="Optional .npz path to save raw data used for figure generation.")

    g = p.add_argument_group("long benchmark")
    g.add_argument("--long_benchmark", action="store_true",
                   help="Run long training benchmark and export MSE/spectrum plots.")
    g.add_argument("--long_steps", type=int, default=10000,
                   help="Maximum training steps for long benchmark.")
    g.add_argument("--early_patience", type=int, default=400,
                   help="Early stopping patience in steps.")
    g.add_argument("--early_min_delta", type=float, default=1e-10,
                   help="Minimum MSE improvement to reset early stopping counter.")
    g.add_argument("--benchmark_inits", nargs="+", default=["xavier", "identity"],
                   choices=["xavier", "identity"], metavar="INIT",
                   help="Initialization schemes to benchmark.")
    g.add_argument("--run_dir", default="runs/toy_2d_linear",
                   help="Base directory where expensive benchmark data is cached.")
    g.add_argument("--force_recompute", action="store_true",
                   help="Recompute optimizer traces even if cached data exists.")
    g.add_argument("--render_only", action="store_true",
                   help="Only render figures from cached run data (no training).")

    # Figure
    g = p.add_argument_group("figure")
    g.add_argument("--contour_levels", type=int,   default=20)
    g.add_argument("--grid_res",       type=int,   default=300,
                   help="Contour grid resolution (pixels per axis)")
    g.add_argument("--normalize_arrows", action=argparse.BooleanOptionalAction, default=True,
                   help="Normalize all one-step arrows to equal display length (direction-only view).")
    g.add_argument("--out", default="outputs/toy2d/figure.pdf",
                   help="Output path (.pdf or .png)")
    g.add_argument("--no_title_one_step", action="store_true",
                   help="Do not render title text on one-step figure.")

    args = p.parse_args()

    return args


def main() -> None:
    args = parse_args()

    print(f"Dataset: N={args.n_samples}, noise={args.noise}, seed={args.data_seed}")
    X, y = make_dataset(args.n_samples, args.noise, args.data_seed,
                        tuple(args.feature_scales))
    Pstar = ols_solution(X, y)
    print(f"P* = [{Pstar[0,0]:.4f}, {Pstar[0,1]:.4f}]")

    net = DeepLinear(args.depth, args.hidden)

    if args.p0 is not None:
        p0_tensor = torch.tensor([args.p0])  # (1, 2)
        print(f"P0 (fixed) = [{args.p0[0]:.4f}, {args.p0[1]:.4f}]")
        init_to_p0(net, p0_tensor)
    else:
        if args.init_mode == "identity":
            init_identity_like(net)
            P0 = compute_P(net)
            print(f"P0 (Identity-like init) = [{P0[0,0]:.4f}, {P0[0,1]:.4f}]")
        elif args.init_mode == "haar":
            init_haar(net, args.init_seed)
            P0 = compute_P(net)
            print(f"P0 (Haar init) = [{P0[0,0]:.4f}, {P0[0,1]:.4f}]")
        else:
            init_gaussian(net, args.init_seed)
            P0 = compute_P(net)
            print(f"P0 (Gaussian init) = [{P0[0,0]:.4f}, {P0[0,1]:.4f}]")

    P0 = compute_P(net)

    if args.long_benchmark:
        base_dir = _run_base_dir(args)
        print(
            f"Running long benchmark: depth={args.depth}, hidden={args.hidden}, "
            f"steps<= {args.long_steps}, optimizers={args.optimizers}, inits={args.benchmark_inits}"
        )
        print(f"Run cache directory: {base_dir}")
        for init_mode in args.benchmark_inits:
            if args.render_only:
                _render_long_benchmark_from_cache(args, init_mode)
            else:
                run_long_benchmark(args, init_mode, X, y)
    elif args.paper_depths is not None:
        # Multi-panel paper figure: run trajectories for each depth from same P0
        panels = []
        share_p0 = not args.no_paper_share_p0
        p0_shared = P0.clone()
        for depth in args.paper_depths:
            print(f"\n--- L={depth} ---")
            net_d = DeepLinear(depth, args.hidden)
            if args.p0 is not None:
                init_to_p0(net_d, torch.tensor([args.p0]))
            elif args.init_mode == "identity":
                init_identity_like(net_d)
            elif args.init_mode == "haar":
                init_haar(net_d, args.init_seed)
            elif share_p0:
                init_to_p0(net_d, p0_shared)
            else:
                init_gaussian(net_d, args.init_seed)
            P0_d = compute_P(net_d)
            print(f"Computing trajectories ({args.n_steps} steps) ...")
            trajs_d = compute_trajectories(net_d, X, y, args)
            panels.append((net_d, P0_d, Pstar, trajs_d, X, y))
        out_traj = args.out_traj or str(Path(args.out).parent / "paper_trajectory.pdf")
        print("\nRendering paper figure ...")
        make_paper_figure(panels, args, out_traj)
    elif args.trajectory:
        print(f"Computing trajectories ({args.n_steps} steps, optimizers: {args.optimizers}) ...")
        trajs = compute_trajectories(net, X, y, args)
        if args.save_data:
            save_trajectories_data(args.save_data, P0, Pstar, X, y, trajs)
        out_traj = args.out_traj or str(Path(args.out).parent / "trajectory.png")
        print("Rendering trajectory figure ...")
        make_trajectory_figure(net, P0, Pstar, trajs, X, y, args, out_traj)
    else:
        print(f"Computing one-step updates (optimizers: {args.optimizers}) ...")
        updates = compute_updates(net, X, y, args)
        if args.save_data:
            save_updates_data(args.save_data, P0, Pstar, X, y, updates)
        print("Rendering figure ...")
        make_figure(net, P0, Pstar, updates, X, y, args, args.out)


if __name__ == "__main__":
    main()
