"""Switch the CReLU nonlinearity off exactly, and see what the low-rank bias does.

Theorem 17: at a looks-linear configuration with a linear teacher and squared error, the
per-sample operator gradient is *even* in `(x, y) -> (-x, -y)` while the gate sign matrix
is *odd*, so a batch closed under negation makes `mean_b[R_b E_b]` vanish term by term.
`Delta` never leaves zero and the CReLU network stays a deep linear network forever.

Two things follow, and they point in opposite directions, which is the point of running it:

* the reduction becomes exact -- under symmetrization every deep-linear result applies to
  the CReLU network with zero error rather than under a hypothesis;
* the bias does **not** go away, because the drive `D = -L s^phi (g_j - g_k)` lives in the
  linear network too. Symmetrizing changes which theory applies, not what the spectrum does.

The `mu` knob makes the input law sign-asymmetric (`x ~ N(mu*1, I)`), which is what a real
dataset of non-negative pixels looks like and what makes the raw arm's seed systematic
rather than a `B^{-1/2}` fluctuation.

Depths are compared **at equal operator growth**, not at equal step count, because
Corollary 14.2 says growth is the clock; comparing at a fixed step budget would report the
learning-rate schedule instead of the depth.

Writes `runs/theory/symmetrize.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from olo.models.crelu_mlp import CReLUMLP
from olo.tasks.base import mse
from olo.theory.crelu import split_layer
from olo.theory.instability import (
    amplification,
    log_velocity_exponent,
    scale_separation_law,
    separation,
)
from olo.theory.nonlinear import decompose


@torch.no_grad()
def delta_max(net) -> float:
    """`max_l ||Delta_l||_inf`: exactly zero iff the network is still looks-linear."""
    return max(float(split_layer(W)[1].abs().max()) for W in net.weights[1:])


def effective_rank(s: np.ndarray) -> float:
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def operator_gradient(net, X, Y):
    """`dL/dJ` per sample, by differentiating the loss written in terms of the operator."""
    P = net.operator(X).detach().requires_grad_(True)
    pred = torch.einsum("bij,bj->bi", P.expand(X.shape[0], *P.shape[1:]), X)
    (G,) = torch.autograd.grad(mse(pred, Y), P)
    return G.detach() * (X.shape[0] if not net.input_independent else 1)


def run(arm: str, depth: int, width: int, mu: float, batch: int, lr_c: float,
        target_growth: float, max_steps: int, eval_every: int, seed: int) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    P_star = torch.linalg.qr(
        torch.randn(width, width, generator=g, dtype=torch.float64))[0] * 2.0

    net = CReLUMLP(d_in=width, d_out=width, width=width, depth=depth).double()
    net.initialize("looks_linear", seed=seed)
    opt = torch.optim.SGD(net.parameters(), lr=lr_c / depth)
    gg = torch.Generator().manual_seed(seed + 7)

    rec, s0 = [], None
    for step in range(max_steps + 1):
        X = torch.randn(batch, width, generator=gg, dtype=torch.float64) + mu
        if arm == "symmetrized":
            X = torch.cat([X, -X])       # (-x, -y) is a valid sample of a linear teacher
        Y = X @ P_star.T

        if step % eval_every == 0:
            with torch.no_grad():
                s = torch.linalg.svdvals(net.operator(X[:1]))[0].numpy()
            sbar = float(np.exp(np.log(np.clip(s, 1e-300, None)).mean()))
            row = {"step": step, "sbar": sbar, "sep": separation(s),
                   "eff_rank": effective_rank(s), "delta": delta_max(net),
                   "loss": float(mse(net(X), Y).detach()),
                   "spectrum": [float(v) for v in s]}
            if separation(s) > 1e-9:
                t = decompose(net, X[:1], operator_gradient(net, X, Y))
                a = amplification(t.s.numpy(), t.total.numpy())
                row |= {"rate": a.rate, "uniform": a.uniform, "spread": a.spread,
                        "r2": a.r2, "psi": a.psi}
            rec.append(row)
            if s0 is None:
                s0 = s.copy()
            if sbar >= target_growth or not np.isfinite(sbar):
                break

        opt.zero_grad(set_to_none=True)
        mse(net(X), Y).backward()
        opt.step()

    last = rec[-1]
    # Feedback against drive, in the currency of Lemma 12: r' = b r + (e_j - e_k), so the
    # feedback contributes |b| * r to the separation velocity and the drive contributes the
    # regression residual. Averaged over the snapshots where the rate is identifiable.
    fb = [abs(r["rate"]) * r["sep"] for r in rec if np.isfinite(r.get("rate", np.nan))]
    dr = [r["spread"] for r in rec if np.isfinite(r.get("spread", np.nan))]
    return {
        "arm": arm, "depth": depth, "width": width, "mu": mu, "batch": batch,
        "lr": lr_c / depth, "seed": seed, "phi": log_velocity_exponent(depth),
        "steps_to_growth": last["step"], "sbar_final": last["sbar"],
        "sep_initial": rec[0]["sep"], "sep_final": last["sep"],
        "eff_rank_final": last["eff_rank"], "delta_final": last["delta"],
        "law_prediction": scale_separation_law(s0, np.array(last["spectrum"]), depth),
        "feedback_mean": float(np.mean(fb)) if fb else float("nan"),
        "drive_mean": float(np.mean(dr)) if dr else float("nan"),
        "feedback_share": (float(np.mean(fb) / (np.mean(fb) + np.mean(dr)))
                           if fb and dr else float("nan")),
        "trace": rec,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["raw", "symmetrized"])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--mu", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr-c", type=float, default=2e-3)
    ap.add_argument("--target-growth", type=float, default=1.8)
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/theory/symmetrize.json")
    a = ap.parse_args()

    out, t0 = [], time.time()
    for arm in a.arms:
        for mu in a.mu:
            for depth in a.depths:
                for seed in a.seeds:
                    r = run(arm, depth, a.width, mu, a.batch, a.lr_c, a.target_growth,
                            a.max_steps, a.eval_every, seed)
                    out.append(r)
                    print(f"{arm:<12} L={depth:<4} mu={mu:<4g} "
                          f"steps={r['steps_to_growth']:<7} sbar={r['sbar_final']:.3f} "
                          f"sep={r['sep_final']:.4f} eff_rank={r['eff_rank_final']:.3f} "
                          f"delta={r['delta_final']:.2e} "
                          f"fb_share={r['feedback_share']:.2f} "
                          f"({time.time()-t0:.0f}s)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "runs": out}))
    print(f"wrote {a.out}  ({len(out)} runs, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
