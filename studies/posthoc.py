"""Post-hoc measurements computed from saved snapshots -- no retraining.

This is what the snapshot storage in `exp.py` exists for: a measurement invented after the runs
finished can still be evaluated at the points where it matters. Adds three quantities.

1. LOCAL COMPLEXITY, after Humayun et al., "Deep Networks Always Grok and Here is Why" (2402.15555).
   They measure the density of linear regions near a point and report that regions migrate away
   from the training samples during training. A ReLU network's linear region is exactly the set of
   inputs sharing a gate pattern, so region density around x is measured by how readily the pattern
   changes under a small perturbation:

       LC(x, eps) = E_delta [ fraction of gates differing between x and x + eps*delta ],  delta ~ unit sphere

   Falling LC means the partition boundaries have moved away from the data. Note this is the
   density of the partition in SPACE, where `churn` in exp.py is its motion in TIME; they answer
   different questions and can move independently.

2. HAMMING RESTRICTED TO LIVE UNITS. `exp.gate_stats` computes pattern diversity over all gates,
   which is diluted when many units are globally dead: a dead unit contributes an identical zero to
   every input and so drives the mean Hamming distance down without saying anything about
   input-dependence. Measured here over units active for at least one input, which is the quantity
   the "is it more input-dependent" question actually asks.

3. LAYERWISE REPRESENTATIONS. The hidden activations at each layer, projected to two dimensions,
   to see how class structure is built up under each optimiser.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu


def _patterns(Ws, X, arch):
    _, gs = gpu.forward(Ws, X, arch)
    if arch == "crelu":
        on = [g > 0 for g in gs]
    else:
        on = [g > 0.5 for g in gs]
    return torch.cat([o.reshape(o.shape[0], -1) for o in on], dim=1)


def local_complexity(Ws, X, arch, eps_list=(0.01, 0.03, 0.1, 0.3), n_dir=8, seed=0):
    """Fraction of gates that flip under a perturbation of size eps, averaged over directions.

    Reported twice. `lc_<eps>` is over ALL gates; `lcl_<eps>` is over gates belonging to units that
    are alive, i.e. active for at least one unperturbed input. The distinction is not cosmetic: a
    globally dead unit can never flip, so any optimiser that kills units drives the all-gates
    number down for a reason that has nothing to do with where the partition boundaries lie.
    Measured on ReLU under Adam, which kills ~59% of units, the two disagree sharply, and only the
    live-restricted number answers the question Humayun et al. pose.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = _patterns(Ws, X, arch)
    live = base.any(0)
    out = {}
    for eps in eps_list:
        acc, accl = [], []
        for _ in range(n_dir):
            d = torch.randn(X.shape, generator=g).to(X.device, X.dtype)
            d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
            p = _patterns(Ws, X + eps * d, arch)
            diff = (p != base)
            acc.append(float(diff.float().mean()))
            accl.append(float(diff[:, live].float().mean()) if int(live.sum()) else float("nan"))
        out[f"lc_{eps:g}"] = float(np.mean(acc))
        out[f"lcl_{eps:g}"] = float(np.mean(accl))
    return out


def hamming_live(Ws, X, arch):
    """Pattern diversity over units that are alive, i.e. active for at least one input."""
    p = _patterns(Ws, X, arch)
    live = p.any(0)
    if int(live.sum()) == 0:
        return dict(hamming_live=float("nan"), live_frac=0.0)
    q = p[:, live]
    n = q.shape[0] // 2
    if n == 0:
        return dict(hamming_live=float("nan"), live_frac=float(live.float().mean()))
    return dict(hamming_live=float((q[:n] != q[n:2 * n]).float().mean()),
                live_frac=float(live.float().mean()))


def layer_reps(Ws, X, arch):
    """Hidden activations at every layer, as numpy, for projection."""
    reps = []
    h = X
    L = len(Ws)
    if arch == "crelu":
        z = X @ Ws[0].T
        for l in range(1, L):
            reps.append(z.detach().cpu().numpy().copy())
            c = torch.cat([z.clamp(min=0), (-z).clamp(min=0)], dim=1)
            z = c @ Ws[l].T
        reps.append(z.detach().cpu().numpy().copy())
        return reps
    for l in range(L - 1):
        z = h @ Ws[l].T
        h = z * (z > 0).to(z.dtype) if arch != "leaky" else z * torch.where(z > 0, 1.0, 0.1)
        reps.append(h.detach().cpu().numpy().copy())
    reps.append((h @ Ws[-1].T).detach().cpu().numpy().copy())
    return reps


def pca2(A):
    A = A - A.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return A @ Vt[:2].T, (S[:2] ** 2).sum() / max((S ** 2).sum(), 1e-30)


def snapshots(run_dir: Path):
    out = []
    for p in sorted(run_dir.glob("snap_*.pt"), key=lambda q: int(q.stem.split("_")[1])):
        out.append((int(p.stem.split("_")[1]), p))
    return out


def compute(run_dir: Path, cfg: dict, X, dev, force=False, X_div=None, R_fn=None):
    """Compute the post-hoc metrics at every snapshot of one run; cache to posthoc.json.

    `X_div` is a SEPARATE, larger probe set for operator diversity, because that statistic is
    bounded above by the number of probe inputs and saturates when the set is small: measured on a
    trained ReLU network, the diversity sat at 89% of its ceiling with 32 probes and the endpoint
    to initialisation ratio drifted from 0.89 to 0.66 as the probe count went from 32 to 512. The
    gate statistics are means rather than ranks, so they do not saturate and keep the smaller set.
    """
    dest = run_dir / "posthoc.json"
    if dest.exists() and not force:
        return json.loads(dest.read_text())
    Xd = X if X_div is None else X_div
    rows = []
    for step, path in snapshots(run_dir):
        Ws = [w.to(dev) for w in torch.load(path, map_location=dev, weights_only=False)["Ws"]]
        rec = {"step": step, "n_div": int(Xd.shape[0]), "n_gate": int(X.shape[0])}
        rec.update(local_complexity(Ws, X, cfg["arch"]))
        rec.update(hamming_live(Ws, X, cfg["arch"]))
        rec.update(operator_diversity(Ws, Xd, cfg["arch"]))
        if R_fn is not None:
            rec.update(decode_operator_step(Ws, X, R_fn(Ws), cfg["arch"]))
        rows.append(rec)
    dest.write_text(json.dumps(rows, indent=1))
    return rows


def operator_diversity(Ws, X, arch):
    """How much the operator VARIES ACROSS INPUTS, as opposed to its rank at any one input.

    `exp.operator_stats` reports the participation ratio of P(x) at a single input, averaged over
    inputs: that is pointwise complexity. It cannot distinguish a network whose operator is simple
    and the SAME everywhere (a nearly linear function) from one whose operator is simple at each
    input but DIFFERENT at each input (a richly piecewise-linear function). The two are opposite
    claims about what the network computes, and the literature on Adam versus SGD turns on exactly
    that distinction.

    Stack the operators as rows, D = [vec P(x_1); ...; vec P(x_n)], and measure

        div        participation ratio of the singular values of D           (total spread)
        div_c      the same for D with its column mean removed               (spread about the
                   mean operator -- the actual input-dependence)
        shared     ||1 mean(D)||_F^2 / ||D||_F^2, the fraction of the operator common to all
                   inputs; 1 means input-independent (a deep linear network), 0 means every input
                   gets an unrelated operator
    """
    W = [w.double() for w in Ws]
    Xd = X.double()
    _, gs = gpu.forward(W, Xd, arch)
    A, B = gpu.contexts(W, gs, Xd, arch)
    P = A[0] @ W[0] @ B[0]                                  # (n, d_out, d_in)
    D = P.reshape(P.shape[0], -1)
    def pr(M):
        s = torch.linalg.svdvals(M)
        t = (s ** 2).sum()
        return float((s.sum() ** 2) / t) if t > 0 else float("nan")
    mean = D.mean(0, keepdim=True)
    tot = float((D ** 2).sum())
    return dict(op_div=pr(D), op_div_c=pr(D - mean),
                op_shared=float((mean ** 2).sum() * D.shape[0] / tot) if tot > 0 else float("nan"))


def decode_operator_step(Ws, X, R, arch, eta=0.3, k_ideal=2000):
    """If Adam's operator step is not aligned with what the loss asked, what IS it aligned with?

    `cos_op` says only that M(dW_adam) is nearly orthogonal to the request. That is a negative
    statement. This asks the positive one by scoring several candidate weight steps -- all computed
    from the SAME weights -- against the same set of reference directions in operator space:

        candidates      grad    -eta * dL/dW                 (gradient descent)
                        sign    -eta * sign(dL/dW)           (the sign-normalised step, which is
                                what Adam approaches when the second moment tracks the first:
                                m/sqrt(v) -> sign(g))
                        ref     M^+ Dstar                    (the reference)

        directions      Dstar   what the loss asked of the operator
                        P       the operator itself, i.e. pure rescaling of what is already there

    Reporting cos(M(candidate), Dstar) and cos(M(candidate), P) separates "moves the operator where
    the loss asked" from "inflates or shrinks the operator it already has", which are the two
    things a step can do that the alignment number alone cannot tell apart.
    """
    W = [w.double() for w in Ws]
    Xd, Rd = X.double(), R.double()
    _, gs = gpu.forward(W, Xd, arch)
    A, B = gpu.contexts(W, gs, Xd, arch)
    shapes = [tuple(w.shape) for w in W]
    M = gpu.StepMap(A, B, shapes)
    G = Rd.unsqueeze(2) * Xd.unsqueeze(1)
    Dstar = (-eta * G).reshape(-1)
    P = (A[0] @ W[0] @ B[0]).reshape(-1)
    ref = gpu.solve_min_norm(M, Dstar, iters=k_ideal, lam_rel=1e-7, stall=1e-10)

    gW = torch.cat([g.reshape(-1) for g in gpu.coord_grad(W, Xd, Rd / Xd.shape[0], arch)])
    cands = {"grad": -eta * gW, "sign": -eta * torch.sign(gW), "ref": ref}

    def cos(a, b):
        na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")

    out = {}
    for name, d in cands.items():
        h = M.mv(d)
        out[f"{name}_vs_Dstar"] = cos(h, Dstar)
        out[f"{name}_vs_P"] = cos(h, P)
        # how much of the step's operator effect is pure rescaling of the existing operator
        nP = torch.linalg.vector_norm(P)
        proj = (h @ P) / (nP ** 2) if nP > 0 else 0.0
        nh = torch.linalg.vector_norm(h)
        out[f"{name}_rescale_frac"] = float((proj * nP / nh) ** 2) if nh > 0 else float("nan")
    out["grad_vs_sign"] = cos(M.mv(cands["grad"]), M.mv(cands["sign"]))
    out["gradW_vs_signW"] = cos(cands["grad"], cands["sign"])
    return out
