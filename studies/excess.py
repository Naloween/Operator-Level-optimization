"""Is the unbalanced excess `K - L` big enough to be worth a theorem? And is the target reachable?

Checks §4.5 and §4.6 of `theory/13-paper-plan.md`, both on the depth-128 checkpoints already in
`runs/theory/grid_ckpt/` -- no new training.

§4.5. `c_k = sum_l ||A_l^T u_k||^2 ||B_l v_k||^2` is the exact mode gain, and `K_k := c_k
s_k^{2/L-2}`. Arora-Cohen-Hazan Thm 1 gives `K_k = L` exactly, under balancedness. Lemma B of
`theory/08` gives, with NO hypotheses and `g := prod_l ||W_l||_2`,

    c_k >= L s_k^2 g^{-2/L},   equivalently   K_k >= L (s_k/g)^{2/L}   <=  L

which is strictly WEAKER than `K_k >= L`: that stronger form needs the aligned diagonal picture
(Lemma 18 of `theory/05`) and is expected to fail off-alignment. Both are checked separately
below, because which of the two survives decides what can be claimed. The question this answers is whether `K_k - L` is a real quantity
on real trajectories or a distinction without a difference -- if it is negligible, the unbalanced
extension is not worth claiming. We correlate it against the unbalancedness magnitude
`max_l ||W_{l+1}^T W_{l+1} - W_l W_l^T||_F` (Cohen's Assumption 1).

§4.6. The step actually requested is `Delta_* = -eta G`. The first-order reachable set is the
range of `M(dW) = sum_l A_l dW_l B_l`, so the best residual ANY first-order method can attain is
`||(I - Pi_range(M)) Delta_*||`. If that floor is near 1 the objective is unsolvable there and no
solver rescues it; if it is near 0 the gap is the solver's fault. Computed by LSQR on `M` applied
matrix-free, never forming `M^T M`.

Writes `runs/theory/excess.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def load(path: Path):
    d = torch.load(path, weights_only=False)
    cfg = d["config"]
    if cfg["arch"] != "deep_linear":
        return None
    keys = [k for k in d["model"] if k.startswith("layers.")]
    keys.sort(key=lambda k: int(k.split(".")[1]))
    Ws = [d["model"][k].to(torch.float64).numpy() for k in keys]
    return cfg, Ws, d["stop"], d["step"]


def contexts(Ws):
    L = len(Ws)
    B = [np.eye(Ws[0].shape[1])]
    for l in range(L - 1):
        B.append(Ws[l] @ B[-1])
    A = [np.eye(Ws[-1].shape[0])]
    for l in range(L - 1, 0, -1):
        A.append(A[-1] @ Ws[l])
    return A[::-1], B


def gains(Ws, rel_floor=1e-12):
    """c_k and K_k for every direction with a usable singular value.

    The floor is RELATIVE to s_max. At depth 128 the tail of the spectrum sits at s/s_max ~ 1e-16,
    where several directions share an identical value -- the signature of a numerically zero
    cluster whose singular vectors are arbitrary. Including them produces spurious violations of
    Lemma B, which is a theorem and cannot actually fail.
    """
    J = Ws[0]
    for W in Ws[1:]:
        J = W @ J
    if not np.all(np.isfinite(J)):
        return None
    U, s, Vt = np.linalg.svd(J)
    A, B = contexts(Ws)
    L = len(Ws)
    floor = rel_floor * float(s.max()) if s.max() > 0 else 0.0
    out = []
    for k in range(len(s)):
        if s[k] <= floor:
            continue
        c = sum(float(np.sum((A[l].T @ U[:, k]) ** 2)) * float(np.sum((B[l] @ Vt[k]) ** 2))
                for l in range(L))
        with np.errstate(over="ignore", under="ignore"):
            K = c * s[k] ** (2.0 / L - 2.0)
        if np.isfinite(K) and K > 0:
            out.append((float(s[k]), float(c), float(K)))
    return out


def spec_product(Ws):
    """g = prod_l ||W_l||_2, the constant in Lemma B."""
    lg = sum(float(np.log(np.linalg.norm(W, 2))) for W in Ws)
    return float(np.exp(lg)), lg


def imbalance(Ws):
    return max(float(np.linalg.norm(Ws[l + 1].T @ Ws[l + 1] - Ws[l] @ Ws[l].T))
               for l in range(len(Ws) - 1))


def reach_floor(Ws, Delta, iters=400):
    """||(I - Pi_range(M)) Delta|| / ||Delta||, by LSQR on M applied matrix-free.

    LSQR is applied to M itself rather than the normal equations: forming M^T M squares the
    condition number, which at depth 128 is exactly the regime under test.
    """
    A, B = contexts(Ws)
    L = len(Ws)
    shapes = [W.shape for W in Ws]

    def mv(dWs):
        return sum(A[l] @ dWs[l] @ B[l] for l in range(L))

    def rmv(H):
        return [A[l].T @ H @ B[l].T for l in range(L)]

    # LSQR (Paige-Saunders), no explicit matrix
    u = Delta.copy()
    beta = float(np.linalg.norm(u))
    if beta == 0:
        return 0.0
    u = u / beta
    v = rmv(u)
    alpha = float(np.sqrt(sum(float(np.sum(t * t)) for t in v)))
    if alpha == 0:
        return 1.0
    v = [t / alpha for t in v]
    w = [t.copy() for t in v]
    x = [np.zeros(s) for s in shapes]
    phibar, rhobar = beta, alpha
    for _ in range(iters):
        u = mv(v) - alpha * u
        beta = float(np.linalg.norm(u))
        if beta > 0:
            u = u / beta
        nv = rmv(u)
        v = [p_ - beta * q for p_, q in zip(nv, v)]
        alpha = float(np.sqrt(sum(float(np.sum(t * t)) for t in v)))
        if alpha > 0:
            v = [t / alpha for t in v]
        rho_ = float(np.hypot(rhobar, beta))
        if rho_ == 0:
            break
        cs, sn = rhobar / rho_, beta / rho_
        theta = sn * alpha
        rhobar = -cs * alpha
        phi, phibar = cs * phibar, sn * phibar
        x = [xi + (phi / rho_) * wi for xi, wi in zip(x, w)]
        w = [vi - (theta / rho_) * wi for vi, wi in zip(v, w)]
        if phibar <= 1e-13 * float(np.linalg.norm(Delta)):
            break
    return float(np.linalg.norm(mv(x) - Delta) / np.linalg.norm(Delta))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default="runs/theory/grid_ckpt")
    ap.add_argument("--lsqr-iters", type=int, default=400)
    ap.add_argument("--out", default="runs/theory/excess.json")
    a = ap.parse_args()

    rows, t0 = [], time.time()
    for p in sorted(Path(a.ckpt_dir).glob("deep_linear__*.pt")):
        got = load(p)
        if got is None:
            continue
        cfg, Ws, stop, step = got
        g = gains(Ws)
        if not g:
            print(f"  skip {p.name}: no usable directions", flush=True)
            continue
        s, c, K = (np.array([x[i] for x in g]) for i in range(3))
        L = len(Ws)
        _, lg = spec_product(Ws)
        # Lemma B floor, in logs so depth 128 does not overflow
        floor_B = L * np.exp((2.0 / L) * (np.log(s) - lg))
        tight_B = K / floor_B
        J = Ws[0]
        for W in Ws[1:]:
            J = W @ J
        Delta = -J / max(float(np.linalg.norm(J)), 1e-300)   # a unit operator-step direction
        row = {
            "name": p.stem, **{k: cfg[k] for k in ("init", "task", "seed", "depth", "width")},
            "stop": stop, "step": step, "L": L, "n_dirs": len(g),
            "K_min": float(K.min()), "K_med": float(np.median(K)), "K_max": float(K.max()),
            "excess_med": float(np.median(K) / L),
            "viol_strong": int((K < L * (1 - 1e-9)).sum()),        # vs the aligned form K >= L
            "viol_lemmaB": int((tight_B < 1 - 1e-9).sum()),        # vs Lemma B's actual floor
            "tightB_min": float(tight_B.min()), "tightB_med": float(np.median(tight_B)),
            "imbalance": imbalance(Ws),
            "s_max": float(s.max()), "s_min": float(s.min()),
            "reach_floor": reach_floor(Ws, Delta, a.lsqr_iters),
        }
        rows.append(row)
        print(f"{p.stem:<42} K/L med={row['excess_med']:8.3f} "
              f"[{row['K_min']/L:.3f},{row['K_max']/L:9.3f}]  viol(K>=L)={row['viol_strong']:<3} "
              f"viol(LemB)={row['viol_lemmaB']:<3} tightB_min={row['tightB_min']:.3f}  "
              f"imb={row['imbalance']:.2e}  floor={row['reach_floor']:.2e}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}  ({len(rows)} checkpoints, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
