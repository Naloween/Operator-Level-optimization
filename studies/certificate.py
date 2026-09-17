"""What does the UNBIASED weight trajectory look like, and is ALS's realisation the minimal one?

Checks §4.2 and §4.4 of `theory/13-paper-plan.md`.

The endpoint check (§4.3) established that the two arms select different solutions, but that much
is known (Gunasekar et al. 2017; Arora, Cohen, Hu & Luo 2019; Li, Luo & Lyu 2021), and in the
deep linear case the unbiased arm's *operator* path is plain convex GD, which is understood
completely. So the only object here that is not already in the literature is the **weight path
that realises it**. This script characterises it.

Problem (P), the minimal realisation of one prescribed operator step:

    min  sum_l ||dW_l||_F^2    s.t.   prod_l (W_l + dW_l) = P*

Lemma (KKT): at a regular optimum there is a SINGLE multiplier Lambda with

    dW_l = A'_l^T Lambda B'_l^T     for every l,

`A'_l`, `B'_l` being the contexts at the *new* weights. This is a one-line consequence of
stationarity of the Lagrangian, so verifying it numerically tests the implementation, not the
mathematics. What is worth measuring is the certificate applied to solvers that do not have the
form built in:

    rho(dW) := min_Lambda sum_l ||dW_l - A'_l^T Lambda B'_l^T||^2 / sum_l ||dW_l||^2

`rho = 0` iff the candidate step is stationary for (P). We report it for

    minnorm  the joint augmented-Lagrangian solution of (P)   -- the reference answer
    dicho    the recursive dichotomy                           -- split the chain, solve each half
    als      one exact ALS sweep onto the same target P*       -- the submitted method
    gd       the coordinate gradient step                      -- does not target P* at all
    ngd      pseudo-inverse Fisher / linearised min-norm       -- the first-order counterfactual

`minnorm` is NOT the dichotomy: it attacks (P) jointly over all layers at once, which is why its
own `rho` is ~0 by construction (any stationary point of that Lagrangian has the KKT form). It is
the reference the others are measured against. `dicho` splits the chain at its midpoint, solves
the two-factor problem `min ||Y'-Y||^2 + ||X'-X||^2 s.t. Y'X' = Z` exactly, and recurses. Each
split being solved exactly is deliberate: it isolates the error due to the greedy SURROGATE
(`||Z_hi - W_hi||^2 + ||Z_lo - W_lo||^2` stands in for the true cost-to-go `V`) from any error
due to solving the splits badly. From a zero base point the two agree -- there `V` is the
Schatten-2/L quasi-norm and the recursion is split-consistent -- so any gap measured here is
exactly the surrogate error at a general base point.

and compare `||dW||`, which is what (P) actually minimises.

§4.4 is the degeneracy check: at `W = 0` every derivative of order `< L` vanishes, so the
linearised map `M(dW) = sum_l A_l dW_l B_l` is identically zero and `ngd` must return exactly 0
while the exact solve does not.

Writes `runs/theory/certificate.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------------------------------- helpers


def product(Ws):
    J = Ws[0]
    for W in Ws[1:]:
        J = W @ J
    return J


def contexts(Ws):
    """A[l] is everything left of W_l, B[l] everything right, so J = A[l] W_l B[l]."""
    L = len(Ws)
    B = [np.eye(Ws[0].shape[1])]
    for l in range(L - 1):
        B.append(Ws[l] @ B[-1])
    A = [np.eye(Ws[-1].shape[0])]
    for l in range(L - 1, 0, -1):
        A.append(A[-1] @ Ws[l])
    return A[::-1], B


def _pack(dWs):
    return np.concatenate([d.ravel() for d in dWs])


def _unpack(v, shapes):
    out, i = [], 0
    for sh in shapes:
        n = sh[0] * sh[1]
        out.append(v[i:i + n].reshape(sh))
        i += n
    return out


# --------------------------------------------------------------------------- the certificate


def transfer_matrix(A, B, n_out, n_in):
    """Dense matrix of Lambda -> (A_l^T Lambda B_l^T)_l, stacked. Small d only."""
    cols = []
    for i in range(n_out):
        for j in range(n_in):
            E = np.zeros((n_out, n_in))
            E[i, j] = 1.0
            cols.append(_pack([A[l].T @ E @ B[l].T for l in range(len(A))]))
    return np.stack(cols, axis=1)


def rho(dWs, Ws_new):
    """KKT residual: how far the step is from having a single shared multiplier."""
    A, B = contexts(Ws_new)
    n_out, n_in = Ws_new[-1].shape[0], Ws_new[0].shape[1]
    Phi = transfer_matrix(A, B, n_out, n_in)
    y = _pack(dWs)
    ny = float(y @ y)
    if ny <= 0:
        return 0.0
    res = y - Phi @ np.linalg.lstsq(Phi, y, rcond=None)[0]
    return float(res @ res) / ny


# --------------------------------------------------------------------------- solvers


def balanced_chain(Pstar, L, shapes):
    """Closed-form solution of (P) from a ZERO base point: the balanced SVD chain.

    Needed as an initialiser, not a convenience. The augmented Lagrangian is solved by L-BFGS,
    a first-order method, so at a degenerate base point it is pinned by the same homogeneity
    argument as everything else (plan §3.3) and returns 0. The escape must be constructive.
    """
    U, S, Vt = np.linalg.svd(Pstar)
    r = min(shapes[0][1], shapes[-1][0], len(S))
    Sr = np.diag(S[:r] ** (1.0 / L))
    Ws = []
    for l in range(L):
        left = U[:, :r] if l == L - 1 else np.eye(shapes[l][0])[:, :r]
        right = Vt[:r, :] if l == 0 else np.eye(shapes[l][1])[:r, :]
        Ws.append(left @ Sr @ right)
    return Ws


def _al(Ws, Pstar, x0, mu0=1e6, outer=60, tol=1e-13):
    """(P) by augmented Lagrangian; the inner objective uses analytic gradients."""
    shapes = [W.shape for W in Ws]
    Lam = np.zeros_like(Pstar)
    mu, x = mu0, x0.copy()

    def fg(v):
        dWs = _unpack(v, shapes)
        New = [W + d for W, d in zip(Ws, dWs)]
        C = product(New) - Pstar
        f = sum(float(np.sum(d * d)) for d in dWs) - float(np.sum(Lam * C)) + 0.5 * mu * float(np.sum(C * C))
        A, B = contexts(New)
        S = mu * C - Lam
        g = [2 * dWs[l] + A[l].T @ S @ B[l].T for l in range(len(Ws))]
        return f, _pack(g)

    for _ in range(outer):
        r = minimize(fg, x, jac=True, method="L-BFGS-B",
                     options={"maxiter": 3000, "ftol": 1e-18, "gtol": 1e-14})
        x = r.x
        dWs = _unpack(x, shapes)
        C = product([W + d for W, d in zip(Ws, dWs)]) - Pstar
        cn = float(np.linalg.norm(C))
        if cn < tol * max(float(np.linalg.norm(Pstar)), 1e-300):
            break
        Lam = Lam - mu * C
        mu *= 2.0
    return _unpack(x, shapes), cn


def solve_minnorm(Ws, Pstar, **kw):
    """Constructive initialiser + polish, keeping whichever branch is feasible and smaller."""
    shapes = [W.shape for W in Ws]
    n = sum(s[0] * s[1] for s in shapes)
    chain = _pack([c - W for c, W in zip(balanced_chain(Pstar, len(Ws), shapes), Ws)])
    ref = max(float(np.linalg.norm(Pstar)), 1e-300)

    # The raw chain is feasible by construction, so it is always an admissible candidate. Without
    # it, a penalty solver started at a degenerate base point can slide back to dW = 0 -- where
    # the gradient is exactly zero for L >= 3 -- and never leave.
    cands = [(_unpack(chain, shapes),
              float(np.linalg.norm(product([W + d for W, d in zip(Ws, _unpack(chain, shapes))]) - Pstar)))]
    for x0 in (np.zeros(n), chain):
        cands.append(_al(Ws, Pstar, x0, **kw))

    best = None
    for dW, cn in cands:
        nrm = sum(float(np.sum(d * d)) for d in dW)
        key = (0 if cn < 1e-8 * ref else 1, nrm)
        if best is None or key < best[0]:
            best = (key, dW, cn)
    return best[1], best[2]


def balanced_split(Z):
    """Y0 X0 = Z with ||Y0||^2 + ||X0||^2 minimal: the two-factor case of the chain."""
    U, S, Vt = np.linalg.svd(Z)
    r = min(U.shape[1], Vt.shape[0], len(S))
    R = np.diag(np.sqrt(S[:r]))
    return U[:, :r] @ R, R @ Vt[:r, :]


def split2(Y, X, Z, mu0=1e6, outer=60, tol=1e-13):
    """min ||Y'-Y||^2 + ||X'-X||^2 s.t. Y'X' = Z, by augmented Lagrangian from two starts."""
    ny, nx = Y.shape, X.shape

    def run(dY0, dX0):
        Lam = np.zeros_like(Z)
        mu = mu0
        v = np.concatenate([dY0.ravel(), dX0.ravel()])

        def fg(v):
            dY = v[:ny[0] * ny[1]].reshape(ny)
            dX = v[ny[0] * ny[1]:].reshape(nx)
            Yp, Xp = Y + dY, X + dX
            C = Yp @ Xp - Z
            S = mu * C - Lam
            f = (float(np.sum(dY * dY)) + float(np.sum(dX * dX))
                 - float(np.sum(Lam * C)) + 0.5 * mu * float(np.sum(C * C)))
            return f, np.concatenate([(2 * dY + S @ Xp.T).ravel(), (2 * dX + Yp.T @ S).ravel()])

        cn = np.inf
        for _ in range(outer):
            v = minimize(fg, v, jac=True, method="L-BFGS-B",
                         options={"maxiter": 3000, "ftol": 1e-18, "gtol": 1e-14}).x
            dY = v[:ny[0] * ny[1]].reshape(ny)
            dX = v[ny[0] * ny[1]:].reshape(nx)
            C = (Y + dY) @ (X + dX) - Z
            cn = float(np.linalg.norm(C))
            if cn < tol * max(float(np.linalg.norm(Z)), 1e-300):
                break
            Lam = Lam - mu * C
            mu *= 2.0
        return dY, dX, cn

    Y0, X0 = balanced_split(Z)
    best = None
    ref = max(float(np.linalg.norm(Z)), 1e-300)
    for dY0, dX0 in ((np.zeros(ny), np.zeros(nx)), (Y0 - Y, X0 - X)):
        dY, dX, cn = run(dY0, dX0)
        key = (0 if cn < 1e-8 * ref else 1,
               float(np.sum(dY * dY)) + float(np.sum(dX * dX)))
        if best is None or key < best[0]:
            best = (key, Y + dY, X + dX)
    return best[1], best[2]


def solve_dichotomy(Ws, Pstar, **kw):
    """Split the chain at its midpoint, solve the two-factor problem, recurse into each half."""
    L = len(Ws)
    New = [None] * L

    def rec(a, b, Z):
        if a == b:
            New[a] = Z
            return
        m = (a + b) // 2
        X = product(Ws[a:m + 1])        # W_m ... W_a
        Y = product(Ws[m + 1:b + 1])    # W_b ... W_{m+1}
        Yp, Xp = split2(Y, X, Z, **kw)
        rec(m + 1, b, Yp)
        rec(a, m, Xp)

    rec(0, L - 1, Pstar)
    res = float(np.linalg.norm(product(New) - Pstar))
    return [n - W for n, W in zip(New, Ws)], res


def solve_als(Ws, Pstar, sweeps=200, lam=0.0, tol=1e-14):
    """The submitted method: exact per-layer least squares, reverse Gauss-Seidel sweeps."""
    New = [W.copy() for W in Ws]
    L = len(Ws)
    prev = np.inf
    for _ in range(sweeps):
        for l in range(L - 1, -1, -1):
            A, B = contexts(New)
            Al, Bl = A[l], B[l]
            if lam > 0:
                Gl = Al.T @ Al + lam * np.eye(Al.shape[1])
                Hl = Bl @ Bl.T + lam * np.eye(Bl.shape[0])
                New[l] = np.linalg.solve(Gl, Al.T @ Pstar @ Bl.T) @ np.linalg.inv(Hl)
            else:
                New[l] = np.linalg.pinv(Al) @ Pstar @ np.linalg.pinv(Bl)
        r = float(np.linalg.norm(product(New) - Pstar))
        if abs(prev - r) <= tol * max(r, 1e-300):
            break
        prev = r
    return [n - W for n, W in zip(New, Ws)], r


def solve_ngd(Ws, Pstar):
    """Linearised min-norm = pseudo-inverse-Fisher natural gradient. Pinned wherever M == 0."""
    A, B = contexts(Ws)
    n_out, n_in = Ws[-1].shape[0], Ws[0].shape[1]
    shapes = [W.shape for W in Ws]
    # dense matrix of M: (dW_l) -> sum_l A_l dW_l B_l
    cols = []
    for l in range(len(Ws)):
        for i in range(shapes[l][0]):
            for j in range(shapes[l][1]):
                E = np.zeros(shapes[l])
                E[i, j] = 1.0
                cols.append((A[l] @ E @ B[l]).ravel())
    M = np.stack(cols, axis=1)
    target = (Pstar - product(Ws)).ravel()
    x = np.linalg.pinv(M) @ target                      # min-norm least-squares solution
    return _unpack(x, shapes), float(np.linalg.norm(M @ x - target))


# --------------------------------------------------------------------------- experiments


def one_step(d, L, seed, eta, init, rng) -> dict:
    if init == "zero":
        Ws = [np.zeros((d, d)) for _ in range(L)]
    elif init == "tiny":
        eps = 1e-3 ** (1.0 / L)
        Ws = [eps * np.linalg.qr(rng.standard_normal((d, d)))[0] for _ in range(L)]
    else:
        Ws = [rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(L)]

    Atgt = rng.standard_normal((d, d)) / np.sqrt(d)
    J = product(Ws)
    G = J - Atgt                                       # dL/dJ for 1/2||J - A||^2, isotropic inputs
    Pstar = J - eta * G

    out = {"d": d, "L": L, "seed": seed, "init": init, "eta": eta,
           "J_norm": float(np.linalg.norm(J)), "target_move": float(eta * np.linalg.norm(G))}

    cands = {}
    dW, res = solve_minnorm(Ws, Pstar)
    cands["minnorm"] = (dW, res)
    dW, res = solve_dichotomy(Ws, Pstar)
    cands["dicho"] = (dW, res)
    dW, res = solve_als(Ws, Pstar)
    cands["als"] = (dW, res)
    A, B = contexts(Ws)
    dW = [-eta * A[l].T @ G @ B[l].T for l in range(L)]
    cands["gd"] = (dW, float(np.linalg.norm(product([W + d for W, d in zip(Ws, dW)]) - Pstar)))
    dW, res = solve_ngd(Ws, Pstar)
    cands["ngd"] = (dW, res)

    for name, (dW, _lin) in cands.items():
        New = [W + d for W, d in zip(Ws, dW)]
        res = float(np.linalg.norm(product(New) - Pstar))   # the residual that actually matters
        out[name] = {
            "dW_norm": float(np.sqrt(sum(float(np.sum(d * d)) for d in dW))),
            "residual": res,
            "rel_residual": res / max(float(np.linalg.norm(Pstar)), 1e-300),
            "rho": rho(dW, New),
            "per_layer": [float(np.linalg.norm(d)) for d in dW],
        }
    mn = out["minnorm"]["dW_norm"]
    for name in cands:
        out[name]["excess"] = out[name]["dW_norm"] / mn if mn > 0 else float("nan")
    return out


def trajectory(d, L, seed, eta, steps, rng) -> dict:
    """Run the unbiased arm in WEIGHT space and watch what the realisation does to the network."""
    eps = 1e-2 ** (1.0 / L)
    Ws = [eps * np.linalg.qr(rng.standard_normal((d, d)))[0] for _ in range(L)]
    Atgt = rng.standard_normal((d, d)) / np.sqrt(d)
    rec, path = [], 0.0
    for step in range(steps):
        J = product(Ws)
        G = J - Atgt
        Pstar = J - eta * G
        dW, res = solve_minnorm(Ws, Pstar)
        dcW, dcres = solve_dichotomy(Ws, Pstar)
        A, B = contexts(Ws)
        gdW = [-eta * A[l].T @ G @ B[l].T for l in range(L)]
        nmn = float(np.sqrt(sum(float(np.sum(x * x)) for x in dW)))
        ndc = float(np.sqrt(sum(float(np.sum(x * x)) for x in dcW)))
        cos_dc = sum(float(np.sum(a * b)) for a, b in zip(dW, dcW)) / max(nmn * ndc, 1e-300)
        ngd_ = float(np.sqrt(sum(float(np.sum(x * x)) for x in gdW)))
        cos = (sum(float(np.sum(a * b)) for a, b in zip(dW, gdW)) / max(nmn * ngd_, 1e-300))
        New = [W + x for W, x in zip(Ws, dW)]
        imb = max(float(np.linalg.norm(New[l + 1].T @ New[l + 1] - New[l] @ New[l].T))
                  for l in range(L - 1)) if L > 1 else 0.0
        path += nmn
        rec.append({
            "step": step, "loss": 0.5 * float(np.sum(G * G)),
            "res": res, "dW_minnorm": nmn, "dW_gd": ngd_, "cos_step": cos,
            "imbalance": imb, "rho": rho(dW, New),
            "dW_dicho": ndc, "dicho_excess": ndc / max(nmn, 1e-300),
            "dicho_res": dcres, "cos_dicho": cos_dc,
            "rho_dicho": rho(dcW, [W + x for W, x in zip(Ws, dcW)]),
            "layer_share": [float(np.linalg.norm(x)) / max(nmn, 1e-300) for x in dW],
            "path_len": path,
        })
        Ws = New
    return {"d": d, "L": L, "seed": seed, "eta": eta, "trace": rec}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--inits", nargs="+", default=["xavier", "tiny", "zero"])
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--traj-steps", type=int, default=40)
    ap.add_argument("--out", default="runs/theory/certificate.json")
    a = ap.parse_args()

    t0, steps_out, traj_out = time.time(), [], []
    for init in a.inits:
        for L in a.depths:
            for s in a.seeds:
                rng = np.random.default_rng(1000 * s + L)
                r = one_step(a.d, L, s, a.eta, init, rng)
                steps_out.append(r)
                print(f"[step] {init:<7} L={L} s={s} | " + "  ".join(
                    f"{n}: |dW|={r[n]['dW_norm']:.3e} x{r[n]['excess']:.2f} "
                    f"res={r[n]['rel_residual']:.1e} rho={r[n]['rho']:.1e}"
                    for n in ("minnorm", "dicho", "als", "ngd", "gd")), flush=True)

    for L in a.depths:
        rng = np.random.default_rng(7 + L)
        t = trajectory(a.d, L, 0, a.eta, a.traj_steps, rng)
        traj_out.append(t)
        last = t["trace"][-1]
        print(f"[traj] L={L} steps={a.traj_steps} | final loss={last['loss']:.3e} "
              f"imbalance={last['imbalance']:.3e} cos(step,gd)={last['cos_step']:+.4f} "
              f"rho={last['rho']:.1e} path={last['path_len']:.3f} | "
              f"dicho x{last['dicho_excess']:.3f} cos={last['cos_dicho']:+.4f} "
              f"rho={last['rho_dicho']:.1e} res={last['dicho_res']:.1e}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "steps": steps_out,
                                       "trajectories": traj_out}, indent=1))
    print(f"wrote {a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
