"""The previous submission's comparators, as functional step rules.

`exp.step_for` works on a plain list of weight tensors and returns the displacement `dW`, so a
`torch.optim.Optimizer` subclass does not fit: it wants leaf `nn.Parameter`s with `.grad` and it
mutates them in place. These are functional transcriptions of `src/olo/optim/baselines/*.py`,
kept deliberately close to those files -- same constants, same update order, same eigensolver
choice -- so that "the baselines from the previous paper" means the same rules and not merely the
same names. `tests_baselines.py` checks each against its `olo` original on random inputs.

Every rule takes `mutate`: when False it must compute the step WITHOUT touching persistent state.
The alignment probe scores a hypothetical step at every probe point and would otherwise advance
each optimiser's moments by one extra step per probe, and -- worse -- seed them in the probe's
dtype. That bug already cost a debugging session once (see `exp.step_for`), so the scoring path
is part of every rule's signature here rather than an afterthought.
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------------- shared numerics
def svd_psd(X):
    """Eigendecomposition of a symmetric PSD matrix as (vectors, values), descending.

    Via the general SVD rather than `eigh`: deep rectifier networks produce Grams with many
    repeated near-zero singular values, on which LAPACK's symmetric solver fails to converge
    while `gesdd` succeeds. An orthogonal or identity initialisation -- the control this project
    leans on -- is exactly the repeated-eigenvalue case, so a baseline built on `eigh` would
    crash precisely in the well-conditioned regime.
    """
    if not torch.isfinite(X).all():
        # A diverging learning rate poisons the accumulated statistics before the weights
        # themselves go non-finite, so the first thing to fail is this decomposition. Say so
        # plainly rather than exhausting three fallbacks on a matrix full of inf.
        raise ValueError("svd_psd: non-finite input (the optimiser has diverged)")
    try:
        U, s, _ = torch.linalg.svd(X)
        return U, s
    except torch._C._LinAlgError:
        pass
    scale = max(float(X.diagonal().abs().mean()), 1e-30)
    eye = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    for mult in (1e-6, 1e-4, 1e-2):
        try:
            U, s, _ = torch.linalg.svd(X + (scale * mult) * eye)
            return U, s
        except torch._C._LinAlgError:
            continue
    U, s, _ = torch.linalg.svd(X.cpu().double())
    return U.to(X.device, X.dtype), s.to(X.device, X.dtype)


def _inv_pow_spd(m, power, eps):
    eye = torch.eye(m.shape[0], device=m.device, dtype=m.dtype)
    Q, ev = svd_psd(m + eps * eye)
    return (Q * ev.clamp(min=eps).pow(power).unsqueeze(0)) @ Q.T


def newton_schulz(G, steps=5, eps=1e-7):
    """Approximate polar factor of a 2D matrix by the quintic Newton-Schulz iteration.

    Coefficients are Muon's published ones, (3.4445, -4.7750, 2.0315), tuned by its author so
    that five iterations suffice. The transcription this file started from used the *classical*
    quintic coefficients (1.875, -1.25, 0.375); those converge to the same polar factor in the
    limit but are far from it after five steps, so the two produce materially different updates
    at the step budget Muon actually uses. `tests_baselines.py` quantifies the gap.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transposed = X.shape[0] > X.shape[1]          # the iteration is stable on wide matrices
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if transposed else X


def _buffers(state, key, ref, mutate):
    """Persistent buffers when training, detached copies when scoring."""
    if key not in state:
        fresh = [torch.zeros_like(w) for w in ref]
        if not mutate:
            return fresh
        state[key] = fresh
    return state[key] if mutate else [b.clone().to(ref[i].dtype) for i, b in enumerate(state[key])]


# --------------------------------------------------------------------------- the rules
def heavyball(g, state, hyper, mutate=True):
    # Tolerates a bare learning rate: `step_for` passes `(lr,)` when the sweep supplies a scalar,
    # and unpacking a 1-tuple into two names raised on every call -- which `traj_grid` was
    # swallowing in a bare except, and which would have failed every heavyball cell of the queued
    # sweep with nothing but a "failed" status to show for it.
    lr = hyper[0] if isinstance(hyper, (list, tuple)) else hyper
    momentum = hyper[1] if isinstance(hyper, (list, tuple)) and len(hyper) > 1 else 0.9
    buf = _buffers(state, "hb", g, mutate)
    out = []
    for i, gi in enumerate(g):
        b = buf[i].mul_(momentum).add_(gi) if mutate else buf[i] * momentum + gi
        out.append(-lr * b)
    return out


def muon(g, state, hyper, mutate=True):
    """Momentum, then replace the update by its polar factor: all singular values equal.

    One-sided by construction -- it normalises the update's own spectrum and carries nothing
    about the contexts that decide how a layer update moves the composed operator. The framework
    therefore groups it with the elementwise methods, and predicts it will not restore a
    collapsed operator spectrum despite updates that are perfectly conditioned in weight space.
    """
    lr, momentum = hyper[0], (hyper[1] if len(hyper) > 1 else 0.95)
    nesterov, ns_steps, eps = True, 5, 1e-7
    buf = _buffers(state, "muon", g, mutate)
    out = []
    for i, gi in enumerate(g):
        # Official form: buf.lerp_(g, 1 - beta) then update = g.lerp_(buf, beta). The earlier
        # transcription used buf = beta*buf + g, which is exactly 1/(1-beta) times this and so
        # gives an identical direction once Newton-Schulz normalises -- verified to cos = 1.0 --
        # but the published form is kept so the code reads as the algorithm it claims to be.
        b = buf[i].lerp_(gi, 1 - momentum) if mutate else buf[i] + (gi - buf[i]) * (1 - momentum)
        d = gi + (b - gi) * momentum if nesterov else b
        if d.ndim != 2:
            out.append(-lr * d / (d.norm() + eps))
            continue
        u = newton_schulz(d, ns_steps, eps)
        out.append(-lr * max(1.0, d.shape[0] / d.shape[1]) ** 0.5 * u)
    return out


def shampoo(g, state, hyper, mutate=True, graft=True):
    """Kronecker-factored preconditioning from accumulated gradient outer products.

    Two-sided, which by the framework's account is the property that matters: the update carries
    a left and a right factor and so can act on the operator's row and column geometry. But the
    factors come from past gradients, not the current contexts, so they coincide with the
    reference's bases only when both contexts are near isometries.

    Matched to `google-research/scalable_shampoo/pytorch/shampoo.py`: statistics accumulate as a
    plain sum (`beta2 = 1.0`), the ridge is `matrix_eps = 1e-12`, and the Shampoo direction is
    **grafted** to SGD's step magnitude -- take Shampoo's direction, take SGD's length. Grafting
    is not in the 2018 paper; it is in every implementation anyone benchmarks, and without it the
    step scale is set entirely by the preconditioner, which makes a shared learning-rate grid
    meaningless against the other arms. The earlier transcription had beta = 0.9, eps = 1e-4 and
    no grafting.
    """
    lr = hyper[0]
    beta, eps = (hyper[1] if len(hyper) > 1 else 1.0), 1e-12
    L = state.get("sh_L"), state.get("sh_R")
    if state.get("sh_L") is None:
        Ls = [torch.zeros(w.shape[1], w.shape[1], device=w.device, dtype=w.dtype) for w in g]
        Rs = [torch.zeros(w.shape[0], w.shape[0], device=w.device, dtype=w.dtype) for w in g]
        if mutate:
            state["sh_L"], state["sh_R"] = Ls, Rs
    else:
        Ls, Rs = state["sh_L"], state["sh_R"]
        if not mutate:
            Ls = [x.clone().to(g[i].dtype) for i, x in enumerate(Ls)]
            Rs = [x.clone().to(g[i].dtype) for i, x in enumerate(Rs)]
    out = []
    for i, gi in enumerate(g):
        if gi.ndim != 2:
            out.append(-lr * gi)
            continue
        w2 = 1.0 if beta == 1.0 else 1.0 - beta      # beta2 = 1 is a plain sum, as in the paper
        Ls[i].mul_(beta).add_(gi.T @ gi, alpha=w2)
        Rs[i].mul_(beta).add_(gi @ gi.T, alpha=w2)
        upd = _inv_pow_spd(Rs[i], -0.25, eps) @ gi @ _inv_pow_spd(Ls[i], -0.25, eps)
        if graft:                                     # SGD grafting: Shampoo direction, SGD length
            upd = upd * (gi.norm() / (upd.norm() + 1e-16))
        out.append(-lr * upd)
    return out


def soap(g, state, hyper, step, mutate=True):
    """Adam run inside Shampoo's eigenbasis: the preconditioner picks a basis, Adam scales in it."""
    lr = hyper[0]
    b1, b2 = 0.95, 0.95                  # official SOAP defaults; the transcription had Adam's
    sb, eps = (hyper[1] if len(hyper) > 1 else b2), 1e-8
    precond_every = 10                   # official `precondition_frequency`; was 1 (every step)
    if state.get("so_m") is None:
        m = [torch.zeros_like(w) for w in g]
        v = [torch.zeros_like(w) for w in g]
        GL = [torch.zeros(w.shape[0], w.shape[0], device=w.device, dtype=w.dtype) for w in g]
        GR = [torch.zeros(w.shape[1], w.shape[1], device=w.device, dtype=w.dtype) for w in g]
        if mutate:
            state["so_m"], state["so_v"], state["so_L"], state["so_R"] = m, v, GL, GR
    else:
        m, v, GL, GR = state["so_m"], state["so_v"], state["so_L"], state["so_R"]
        if not mutate:
            m = [x.clone().to(g[i].dtype) for i, x in enumerate(m)]
            v = [x.clone().to(g[i].dtype) for i, x in enumerate(v)]
            GL = [x.clone().to(g[i].dtype) for i, x in enumerate(GL)]
            GR = [x.clone().to(g[i].dtype) for i, x in enumerate(GR)]
    # The eigenbasis is held between refreshes, as in the official implementation. Recomputing it
    # every step is not more faithful, it is a different algorithm -- and it costs ~10x, which is
    # what made SOAP the second most expensive arm in the sweep.
    # Held as two flat tensor lists rather than a list of (ql, qr) pairs: `exp._ckpt_state`
    # persists lists of tensors, and a list of tuples silently fails its type check, so a cached
    # basis stored that way would vanish on every resume without anything reporting it.
    QL, QR = state.get("so_QL"), state.get("so_QR")
    refresh = QL is None or (step % precond_every == 0)
    t = step + 1
    newQL, newQR = ([], []) if refresh else (None, None)
    out = []
    for i, gi in enumerate(g):
        if gi.ndim == 2:
            GL[i].mul_(sb).add_(gi @ gi.T, alpha=1.0 - sb)
            GR[i].mul_(sb).add_(gi.T @ gi, alpha=1.0 - sb)
            if refresh:
                eyeL = torch.eye(GL[i].shape[0], device=gi.device, dtype=gi.dtype)
                eyeR = torch.eye(GR[i].shape[0], device=gi.device, dtype=gi.dtype)
                ql, _ = svd_psd(GL[i] + eps * eyeL)
                qr, _ = svd_psd(GR[i] + eps * eyeR)
                newQL.append(ql); newQR.append(qr)
            else:
                ql, qr = QL[i].to(gi.dtype), QR[i].to(gi.dtype)
            gt = ql.T @ gi @ qr
        else:
            ql = qr = None
            gt = gi
            if refresh:
                newQL.append(None); newQR.append(None)
        m[i].mul_(b1).add_(gt, alpha=1.0 - b1)
        v[i].mul_(b2).add_(gt.square(), alpha=1.0 - b2)
        upd = m[i] / v[i].sqrt().add(eps)
        if ql is not None:
            upd = ql @ upd @ qr.T
        out.append(-(lr * (1.0 - b2 ** t) ** 0.5 / (1.0 - b1 ** t)) * upd)
    if mutate and refresh:
        state["so_QL"], state["so_QR"] = newQL, newQR
    return out


def kfac(Ws, X, R, g, state, hyper, arch, mutate=True):
    """K-FAC with its Kronecker factors read off the chain, not estimated in parameter space.

    Layer k is preconditioned by the inverses of two covariances,
        A_k = E[h_{k-1} h_{k-1}^T]  (layer inputs),   G_k = E[d_k d_k^T]  (pre-activation grads),
        dW_k = -lr (G_k + lam I)^{-1} grad_k (A_k + lam I)^{-1}.
    Both are available exactly here: h_{k-1}(x) = B_k(x) x and d_k(x) = A_k(x)^T r(x), where A, B
    are the same contexts the reference step uses. This is the comparator the framework makes a
    falsifiable claim about -- in the isotropic deep-linear case K-FAC diagonalises in the
    reference's own bases and differs only in its denominator, (s_A^2 + lam)(s_B^2 + lam) against
    s_A^2 s_B^2 + lam, so it is predicted to be the closest baseline and to depart from the
    projection exactly in the high-damping regime used for stability at depth.
    """
    import gpu
    lr = hyper[0]
    damping = hyper[1] if len(hyper) > 1 else 1e-2
    decay, eps = 0.95, 1e-8
    _, gs = gpu.forward(Ws, X, arch)
    A, B = gpu.contexts(Ws, gs, X, arch)
    n = X.shape[0]
    Acov = state.get("kf_A")
    Gcov = state.get("kf_G")
    fresh = Acov is None
    if fresh:
        Acov, Gcov = [None] * len(Ws), [None] * len(Ws)
    elif not mutate:
        Acov = [None if a is None else a.clone().to(X.dtype) for a in Acov]
        Gcov = [None if a is None else a.clone().to(X.dtype) for a in Gcov]
    out = []
    for k in range(len(Ws)):
        h = torch.einsum("nij,nj->ni", B[k], X)            # layer input, per sample
        d = torch.einsum("nji,nj->ni", A[k], R)            # pre-activation gradient, per sample
        Ak = (h.T @ h) / n
        Gk = (d.T @ d) / n
        Acov[k] = Ak.clone() if Acov[k] is None else Acov[k].mul_(decay).add_(Ak, alpha=1 - decay)
        Gcov[k] = Gk.clone() if Gcov[k] is None else Gcov[k].mul_(decay).add_(Gk, alpha=1 - decay)
        eA = torch.eye(Acov[k].shape[0], device=X.device, dtype=X.dtype)
        eG = torch.eye(Gcov[k].shape[0], device=X.device, dtype=X.dtype)
        # Factored Tikhonov damping (Martens & Grosse, sec. 6.3): adding lam*I to A (x) G while
        # preserving the Kronecker structure means splitting it as (A + pi sqrt(lam) I) and
        # (G + sqrt(lam)/pi I) with pi chosen from the factors' average eigenvalues. Adding the
        # same lam to both -- which is what this file did first, and pi = 1 -- damps whichever
        # factor is smaller far harder than the other, and in a deep net their scales differ by
        # orders of magnitude.
        trA = float(Acov[k].diagonal().sum()) / Acov[k].shape[0]
        trG = float(Gcov[k].diagonal().sum()) / Gcov[k].shape[0]
        pi = (max(trA, 1e-30) / max(trG, 1e-30)) ** 0.5
        rl = (damping + eps) ** 0.5
        # Both factors are SINGULAR BY CONSTRUCTION here: A_k = h^T h / n and G_k = d^T d / n are
        # built from a batch of n = 100 samples at width 128, so their rank is at most 100 < 128.
        # The solve is therefore well posed only because of the damping, and the pi split can take
        # that away: pi multiplies one factor's damping and divides the other's, so a large pi
        # leaves G with rl/pi ~ 0 and torch.linalg.solve raises on a singular matrix. That is a
        # numerical failure of this implementation, not divergence of K-FAC, and it is exactly what
        # it looked like -- crashes seconds into a run rather than blow-ups late in one.
        # Two guards, neither of which changes the method where it was already working: clamp the
        # split so it cannot become pathological, and floor each factor's damping relative to that
        # factor's own scale so positive-definiteness is guaranteed whatever pi does.
        pi = min(max(pi, 1e-4), 1e4)
        dA = max(rl * pi, 1e-6 * max(trA, 1e-30))
        dG = max(rl / pi, 1e-6 * max(trG, 1e-30))
        for _ in range(6):                       # escalate if it still fails, rather than crash
            try:
                nat = torch.cholesky_solve(
                    (torch.cholesky_solve(g[k], torch.linalg.cholesky(Gcov[k] + dG * eG))
                     ).T.contiguous(),
                    torch.linalg.cholesky(Acov[k] + dA * eA)).T.contiguous()
                break
            except Exception:
                dA, dG = dA * 10, dG * 10
        else:
            nat = g[k]                            # fully damped: fall back to the gradient step
        out.append(-lr * nat)
    if mutate and fresh:
        state["kf_A"], state["kf_G"] = Acov, Gcov
    return out
