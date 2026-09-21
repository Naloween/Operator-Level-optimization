"""E8: operator-space trajectories, rebuilt from the previous submission's figure.

The plot is in OPERATOR space, which is the whole point and which a weight-space plot cannot do.
The model is a deep linear chain W_L...W_1 = P in R^{1x2} on a two-feature regression, so however
many layers and hidden units the weight space has, the operator has exactly two components and its
trajectory can be drawn exactly. The loss is quadratic in P,

    L(P) = (1/N) || X P^T - y ||^2,

so the contours are exact ellipses, not a sampled surface, and P* is the OLS solution.

Why weight space was the wrong plane. A model whose WEIGHT space is two-dimensional (a scalar
two-layer chain) has a rank-one step map M = [w2, w1], and then M^+ is a multiple of M^T: gradient
descent and the minimal reference are collinear, measured at cos = 1.0000000000, and the figure
cannot show a direction difference at all. With L layers into a 2-dimensional operator, M has rank
two and the two separate. The previous submission's figure worked for exactly this reason.

Each optimiser descends the SAME quadratic in P. What differs is only how the factorisation
converts a weight step into an operator step, so every separation visible here is attributable to
the parameterisation rather than to the objective -- which is the cleanest possible statement of
what this paper measures.

    gd, adam    the usual steps on the weights, with the operator change they happen to induce
    ref         the minimum-norm weight update realising the ideal operator step -eta*dL/dP,
                so its operator trajectory follows the operator-space gradient by construction
    als         alternating exact minimisation, kept because it is what the previous figure used
                and because it is NOT a minimal realisation -- the contrast is the point

Writes runs/theory/fig_traj2d.png and runs/theory/traj2d.json.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def dataset(n, seed, scales=(1.0, 0.35)):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2)) * np.asarray(scales)
    w = rng.standard_normal(2)
    return X, X @ w


def init_weights(L, h, seed, init):
    """Shapes: (h,2), (h,h) x (L-2), (1,h). P = W_L...W_1 is 1x2."""
    rng = np.random.default_rng(seed)
    dims = [(h, 2)] + [(h, h)] * (L - 2) + [(1, h)] if L >= 2 else [(1, 2)]
    Ws = []
    for (m, k) in dims:
        if init == "orthogonal":
            a = rng.standard_normal((max(m, k), min(m, k)))
            q, r = np.linalg.qr(a)
            q = q * np.sign(np.diag(r))
            W = q if m >= k else q.T
        else:                                        # xavier
            W = rng.standard_normal((m, k)) / k ** 0.5
        Ws.append(W)
    return Ws


def operator(Ws):
    P = Ws[0]
    for W in Ws[1:]:
        P = W @ P
    return P                                          # (1, 2)


def contexts(Ws):
    """A_l (1 x m_l) and B_l (k_l x 2) with P = A_l W_l B_l."""
    L = len(Ws)
    B = [np.eye(Ws[0].shape[1])]
    for l in range(L - 1):
        B.append(Ws[l] @ B[-1])
    A = [np.eye(Ws[-1].shape[0])]
    for l in range(L - 1, 0, -1):
        A.append(A[-1] @ Ws[l])
    return A[::-1], B


def step_map(Ws):
    """M as an explicit 2 x n_params matrix: vec(dW) -> vec(dP)."""
    A, B = contexts(Ws)
    blocks = [np.kron(B[l].T, A[l]) for l in range(len(Ws))]   # vec is row-major via kron(B^T, A)
    return np.concatenate(blocks, axis=1)


def unpack(v, Ws):
    out, off = [], 0
    for W in Ws:
        q = W.size
        out.append(v[off:off + q].reshape(W.shape))
        off += q
    return out


def op_grad(P, X, y):
    """dL/dP for L = (1/N)||X P^T - y||^2, shape (1,2)."""
    r = X @ P.ravel() - y
    return (2.0 / X.shape[0]) * (r @ X).reshape(1, 2)


def take(arm, Ws, X, y, lr, eta, state, lam=1e-2):
    P = operator(Ws)
    G = op_grad(P, X, y)
    M = step_map(Ws)
    gW = M.T @ G.ravel()                              # = dL/dW, the backprop gradient
    if arm == "gd":
        d = -lr * gW
    elif arm == "adam":
        b1, b2, e = 0.9, 0.999, 1e-8
        state["m"] = b1 * state["m"] + (1 - b1) * gW
        state["v"] = b2 * state["v"] + (1 - b2) * gW * gW
        state["t"] += 1
        mh = state["m"] / (1 - b1 ** state["t"])
        vh = state["v"] / (1 - b2 ** state["t"])
        d = -lr * mh / (np.sqrt(vh) + e)
    elif arm == "ref":
        d = np.linalg.pinv(M) @ (-eta * G.ravel())    # minimum-norm realisation of the ideal step
    elif arm == "als":
        # one damped sweep of alternating exact minimisation over the layers
        d = np.zeros(M.shape[1])
        A, B = contexts(Ws)
        off, tgt = 0, (-eta * G)
        for l, W in enumerate(Ws):
            a, b = A[l], B[l]
            Ml = np.kron(b.T, a)
            # Ridge-damped: an undamped per-layer exact solve is a greedy sweep that over-corrects
            # and diverges here (||P-P*||=154 at L=2). Damping is what the previous submission's
            # ALS used too; without it the arm shows an implementation artefact, not the method.
            H = Ml.T @ Ml + lam * np.eye(Ml.shape[1])
            dl = np.linalg.solve(H, Ml.T @ tgt.ravel())
            d[off:off + W.size] = dl
            tgt = tgt - (Ml @ dl).reshape(1, 2)
            off += W.size
    else:
        raise ValueError(arm)
    return [W + dw for W, dw in zip(Ws, unpack(d, Ws))]


def run(arm, L, h, seed, init, X, y, lr, eta, steps, lam=1e-2):
    """Returns (operator path, per-step alignment with the operator gradient).

    The alignment is cos(realised dP, -dL/dP): +1 means the step moved the operator exactly where
    the loss asked, NEGATIVE means it moved the operator the wrong way. Note this is a fidelity
    measure, NOT a quality one -- the operator gradient is steepest descent, which on an
    anisotropic loss is itself an indirect route, so a method can be less aligned and still reach
    the optimum sooner. That is why Adam can beat the reference here while scoring lower.
    """
    Ws = init_weights(L, h, seed, init)
    state = {"m": 0.0, "v": 0.0, "t": 0}
    path = [operator(Ws).ravel().copy()]
    align = []
    for _ in range(steps):
        P0 = operator(Ws)
        G = op_grad(P0, X, y)
        Ws = take(arm, Ws, X, y, lr, eta, state, lam)
        P = operator(Ws).ravel()
        if not np.all(np.isfinite(P)) or np.abs(P).max() > 1e3:
            break
        dP = P - P0.ravel()
        ideal = -G.ravel()
        nd, ni = np.linalg.norm(dP), np.linalg.norm(ideal)
        align.append(float(dP @ ideal / (nd * ni)) if nd > 1e-300 and ni > 1e-300 else np.nan)
        path.append(P.copy())
    return np.array(path), np.array(align)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 32])
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="xavier", choices=["xavier", "orthogonal"])
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--adam-lr", type=float, default=0.006)
    ap.add_argument("--eta", type=float, default=0.06)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--out", default="runs/theory/fig_traj2d.png")
    a = ap.parse_args()

    X, y = dataset(a.n, a.seed)
    Pstar = np.linalg.lstsq(X, y, rcond=None)[0]
    arms = ["gd", "adam", "als", "ref"]
    rates = {"gd": a.lr, "adam": a.adam_lr, "als": a.lr, "ref": a.lr}
    data, align = {}, {}
    for L in a.depths:
        for arm in arms:
            pth, al = run(arm, L, a.hidden, a.seed, a.init, X, y, rates[arm], a.eta, a.steps)
            data[f"L{L}_{arm}"] = pth.tolist()
            align[f"L{L}_{arm}"] = al.tolist()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"gd": ("#4c78a8", "-"), "adam": ("#59a14f", "-"),
             "als": ("#b07aa1", "-"), "ref": ("#e15759", "-")}
    fig, axes = plt.subplots(1, len(a.depths), figsize=(5.4 * len(a.depths), 4.9), squeeze=False)
    def view(L):
        """Limits per panel, from the arms that converge. A diverging arm must be allowed to run
        off-frame rather than rescale the panel until the others are an indistinguishable blob."""
        pts = [Pstar.reshape(1, 2)]
        for arm in arms:
            q = np.array(data[f"L{L}_{arm}"])
            if not len(q):
                continue
            d0 = np.linalg.norm(q[0] - Pstar) + 1e-9
            if np.linalg.norm(q[-1] - Pstar) <= 3 * d0:      # converging or at least not fleeing
                pts.append(q)
        z = np.concatenate(pts)
        lo, hi = z.min(0), z.max(0)
        pad = 0.30 * (hi - lo + 1e-9) + 0.05
        return lo - pad, hi + pad

    for ax, L in zip(axes[0], a.depths):
        lo, hi = view(L)
        g1 = np.linspace(lo[0], hi[0], 300)
        g2 = np.linspace(lo[1], hi[1], 300)
        G1, G2 = np.meshgrid(g1, g2)
        R = (X[:, 0][:, None, None] * G1 + X[:, 1][:, None, None] * G2) - y[:, None, None]
        Z = (R ** 2).mean(0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.contourf(G1, G2, Z, levels=28, cmap="Blues_r", alpha=0.55, zorder=0)
        for arm in arms:
            p = np.array(data[f"L{L}_{arm}"])
            col, ls = style[arm]
            ax.plot(p[:, 0], p[:, 1], color=col, ls=ls, lw=2.1, zorder=3,
                    label={"ref": "reference (minimal)"}.get(arm, arm))
            ax.plot(p[-1, 0], p[-1, 1], "o", color=col, ms=5, zorder=4)
        p0 = np.array(data[f"L{L}_gd"])[0]
        ax.plot(*p0, "o", color="k", ms=8, zorder=5)
        ax.annotate("$P_0$", p0, textcoords="offset points", xytext=(8, 6), fontsize=11)
        ax.plot(*Pstar, "*", color="gold", ms=18, mec="0.3", zorder=5)
        ax.set_title(f"$L={L}$"); ax.set_xlabel("$p_1$")
    axes[0][0].set_ylabel("$p_2$")
    axes[0][0].legend(loc="best", frameon=True, fontsize=9)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=160)
    Path(a.out).with_suffix(".json").write_text(json.dumps(
        {"args": vars(a), "Pstar": Pstar.tolist(), "paths": data, "align": align}, indent=1))
    print(f"wrote {a.out}")
    for L in a.depths:
        fin = {arm: float(np.linalg.norm(np.array(data[f'L{L}_{arm}'])[-1] - Pstar)) for arm in arms}
        print(f"  L={L:>3} final ||P-P*||: " + "  ".join(f"{k} {v:.4f}" for k, v in fin.items()))
        # A run that diverged contributes an alignment statistic computed from a handful of
        # blown-up steps: measured, gd at L=8 seed 3 kept 2 of 400 steps and reported "100%
        # anti-aligned", which is one step of a diverging trajectory, not a property of gd.
        div = {k: len(data[f'L{L}_{k}']) < 0.9 * a.steps for k in arms}
        if any(div.values()):
            print(f"       DIVERGED (excluded from alignment): " +
                  ", ".join(k for k in arms if div[k]))
        print(f"       alignment cos(dP, -dL/dP):  " + "  ".join(
            f"{k} mean {np.nanmean(np.array(align[f'L{L}_{k}'])):+.3f} "
            f"min {np.nanmin(np.array(align[f'L{L}_{k}'])):+.3f} "
            f"neg {100*np.nanmean(np.array(align[f'L{L}_{k}'])<0):.0f}%"
            for k in arms if not div[k]))


if __name__ == "__main__":
    main()
