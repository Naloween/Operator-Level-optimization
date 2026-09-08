"""Prescribe the force on the operator; measure what the spectrum does about it.

Any real task confounds the question. A teacher-student loss supplies an operator gradient
`G = (J - P*) Sigma_hat` whose anisotropy comes from the target *and* from the empirical
input covariance -- with n = 256 samples in d = 16 the latter alone has condition number
~2.8, which measured larger than the effect under test. So the drive is removed rather
than hoped away: the operator gradient is **prescribed**,

    G_b = -c * U_b diag(s_b^p) V_b^T          (a surrogate loss `<G_b, J(x_b)>`),

which back-propagates to exactly the weight gradients gradient flow would produce under
that operator force, and nothing else. Then `g_k = -c s_k^p` holds by construction and the
theory's task exponent `p` is a dial rather than a measurement:

    p =  0   push every mode equally      -> psi = 1 - 2/L        (separation grows)
    p =  1   push proportionally to size  -> psi = 2 - 2/L        (grows faster)
    p = -1   push hardest on small modes  -> psi = -2/L < 0       (separation *shrinks*)

with `psi = d log r / d log sbar` the predicted exponent. Signs and magnitudes are all
predicted in advance, at every depth, with no free parameter. `c < 0` reverses the force so
the operator shrinks, which the theory says must *un*-separate the spectrum.

The seed is set exactly: multiplying every layer by `diag(exp(a/L))` leaves an identity or
looks-linear network perfectly balanced while giving the operator the spectrum `exp(a)`.
So `r(0)` is a dial too, and the prediction that it enters *multiplicatively* -- that an
exact isometry is a fixed point of the bias at any depth -- is testable over decades.

Deep linear is the case the theory is stated for and should hold exactly. CReLU at
identity initialization starts exactly linear but does not stay so: the two blocks of
`W = [P | Q]` receive different gradients, so `Delta = (P+Q)/2` leaves zero on the first
step and the network generates its own nonlinearity. How far the law survives that is the
question this study exists to answer.

Writes `runs/theory/forcing.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from olo.models.crelu_mlp import CReLUMLP
from olo.models.deep_linear import DeepLinear
from olo.theory.crelu import split_layer
from olo.theory.instability import (
    amplification,
    amplification_exponent,
    predict_spectrum,
    scale_separation_law,
    separation,
)
from olo.theory.nonlinear import decompose

MODELS = {"crelu_mlp": CReLUMLP, "deep_linear": DeepLinear}


def build(model: str, depth: int, width: int, init: str, r0: float, seed: int):
    """A net whose operator has log-spectrum `linspace(-r0/2, r0/2)`, exactly balanced."""
    net = MODELS[model](d_in=width, d_out=width, width=width, depth=depth).double()
    net.initialize(init, seed=seed)
    if r0 != 0:
        a = torch.linspace(-0.5, 0.5, width, dtype=torch.float64) * r0
        with torch.no_grad():
            d = torch.diag(torch.exp(a / depth))
            for W in net.weights:
                W.copy_(d @ W)
    return net


def forcing(J: torch.Tensor, p: float, c: float) -> torch.Tensor:
    """`G = -c U diag(s^p) V^T`: the operator gradient with task exponent exactly `p`."""
    U, S, Vh = torch.linalg.svd(J)
    k = S.shape[-1]
    return -c * (U[..., :, :k] * S.clamp_min(1e-300).pow(p).unsqueeze(-2)) @ Vh[..., :k, :]


def effective_rank(s: np.ndarray) -> float:
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def nonlinearity(net) -> float:
    """`max_l ||Delta_l|| / ||S_l||`: how far the CReLU net has left the linear regime."""
    out = 0.0
    for W in net.weights[1:]:
        S, D = split_layer(W)
        n = float(D.norm() / S.norm().clamp_min(1e-300))
        out = max(out, n)
    return out


def measure(net, X: torch.Tensor, p: float, c: float, sample: int = 0) -> dict:
    with torch.no_grad():
        J = net.operator(X)
        G = forcing(J, p, c)
    Xd = X[:1] if net.input_independent else X
    terms = decompose(net, Xd, G, sample=sample)
    s = terms.s.numpy()
    amp = amplification(s, terms.total.numpy(), terms.self_term.numpy(),
                        terms.cross_term.numpy())
    return {
        "spectrum": [float(v) for v in s],
        "sep": separation(s),
        "eff_rank": effective_rank(s),
        "rate": amp.rate, "rate_self": amp.rate_self, "rate_cross": amp.rate_cross,
        "uniform": amp.uniform, "psi": amp.psi, "r2": amp.r2,
        "cross_norm": float(np.linalg.norm(terms.cross_term.numpy())),
        "self_norm": float(np.linalg.norm(terms.self_term.numpy())),
        "delta": nonlinearity(net) if not net.input_independent else 0.0,
    }


def run(model: str, depth: int, init: str, r0: float, p: float, c: float, steps: int,
        tau: float, width: int, force_batch: int, seed: int, eval_every: int,
        diag_batch: int, s_cap: float) -> dict:
    torch.manual_seed(seed)
    net = build(model, depth, width, init, r0, seed)
    gen = torch.Generator().manual_seed(seed + 17)
    # Two batches, deliberately separate. The force is averaged over `force_batch`
    # inputs, which is what sets the size of the gate-frequency fluctuation that seeds
    # the bias; `diag_batch` only decides how many inputs the O(B^2) cross-term
    # decomposition sums over, and must not be tied to it.
    X = torch.randn(force_batch, width, generator=gen, dtype=torch.float64)
    Xd = X[:min(diag_batch, force_batch)]

    # Step in the natural clock. A `p >= 0` force is superlinear in `s` and reaches
    # infinity in finite time, which would leave two usable diagnostics before the cap.
    # Rescaling the force by `1/(L sbar^psi)` each step makes every step advance
    # `log sbar` by about `tau` instead. The factor is mode-independent, so it multiplies
    # every `s_k'` alike: a reparameterization of time, leaving the trajectory in
    # spectrum space -- which is all the law speaks about -- untouched.
    opt = torch.optim.SGD(net.parameters(), lr=1.0)
    psi = amplification_exponent(depth, p)
    rec, stop = [], None
    for step in range(steps + 1):
        if step % eval_every == 0:
            m = measure(net, Xd, p, c)
            m["step"] = step
            rec.append(m)
            top = max(m["spectrum"])
            if not np.isfinite(top) or top > s_cap or top < 1.0 / s_cap:
                stop = "scale_cap"
                break
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        J = net.operator(X)
        with torch.no_grad():
            # A `p = 1` force is superlinear in `s` and reaches infinity in finite time,
            # so the scale is checked every step, not every eval: an overflow between
            # diagnostics would leave a NaN spectrum rather than a stopped run.
            if not torch.isfinite(J.detach()).all():
                stop = "diverged"      # checked before the SVD, which throws on non-finite
                break
            sv = torch.linalg.svdvals(J.detach())
            sbar = float(torch.exp(torch.log(sv.clamp_min(1e-300)).mean()))
            if not np.isfinite(sbar) or sbar > s_cap or sbar < 1.0 / s_cap:
                stop = "scale_cap"
                break
            G = forcing(J.detach(), p, c * tau / (depth * sbar**psi))
        # Averaged over the batch, not summed: `decompose` divides the cross-input sum by
        # B, so a summed surrogate would step B times further than the velocity it reports.
        ((G * J).sum() / J.shape[0]).backward()
        opt.step()

    return {"model": model, "depth": depth, "init": init, "r0": r0, "p": p, "c": c,
            "tau": tau, "seed": seed, "width": width, "stop": stop,
            "force_batch": force_batch, "diag_batch": Xd.shape[0],
            "psi_pred": amplification_exponent(depth, p), "trace": rec}


def summarize(rec: dict) -> dict:
    """Measured `psi = d log r / d log sbar` against the prediction, plus the law's error."""
    tr = rec["trace"]
    L = rec["depth"]
    s0 = np.array(tr[0]["spectrum"])
    sbar = lambda s: float(np.exp(np.log(np.clip(s, 1e-300, None)).mean()))
    xs, ys, errs = [], [], []
    for m in tr:
        s = np.array(m["spectrum"])
        if m["sep"] <= 0 or tr[0]["sep"] <= 0 or sbar(s) <= 0:
            continue
        xs.append(np.log(sbar(s) / sbar(s0)))
        ys.append(np.log(m["sep"] / tr[0]["sep"]))
        pred = predict_spectrum(s0, s, L, rec["p"])
        errs.append(float(np.abs(np.log(pred / s)).max()))
    out = {"psi_pred": rec["psi_pred"], "n_points": len(xs)}
    if len(xs) >= 3 and np.ptp(xs) > 1e-6:
        out["psi_measured"] = float(np.polyfit(xs, ys, 1)[0])
        out["growth"] = float(np.exp(max(np.abs(xs))))
    out["law_max_log_err"] = float(max(errs)) if errs else float("nan")
    out["sep_ratio"] = tr[-1]["sep"] / tr[0]["sep"] if tr[0]["sep"] > 0 else float("nan")
    out["sep_ratio_pred"] = scale_separation_law(
        s0, np.array(tr[-1]["spectrum"]), L, rec["p"])
    out["delta_final"] = tr[-1]["delta"]
    out["cross_share_final"] = (tr[-1]["rate_cross"] / tr[-1]["rate"]
                                if tr[-1]["rate"] else float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--p", type=float, nargs="+", default=[-1.0, 0.0, 1.0])
    ap.add_argument("--r0", type=float, nargs="+", default=[1e-4])
    ap.add_argument("--c", type=float, nargs="+", default=[1.0])
    ap.add_argument("--models", nargs="+", default=["deep_linear", "crelu_mlp"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--force-batch", type=int, nargs="+", default=[64],
                    help="inputs the force is averaged over: sets the seeding fluctuation")
    ap.add_argument("--diag-batch", type=int, default=8)
    ap.add_argument("--s-cap", type=float, default=1e4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/theory/forcing.json")
    a = ap.parse_args()

    jobs = [(m, d, p, r, c, b, s) for m in a.models for d in a.depths for p in a.p
            for r in a.r0 for c in a.c for b in a.force_batch for s in a.seeds]
    out, t0 = [], time.time()
    for i, (m, d, p, r, c, b, s) in enumerate(jobs):
        rec = run(m, d, "identity", r, p, c, a.steps, a.tau, a.width, b, s,
                  a.eval_every, a.diag_batch, a.s_cap)
        rec["summary"] = summarize(rec)
        out.append(rec)
        sm = rec["summary"]
        print(f"[{i+1}/{len(jobs)}] {m:<11} L={d:<4} p={p:<5g} c={c:<5g} r0={r:<7g} B={b:<5} "
              f"psi {sm.get('psi_measured', float('nan')):>7.3f} vs {sm['psi_pred']:>6.3f} "
              f"sep {rec['trace'][0]['sep']:.2g}->{rec['trace'][-1]['sep']:.3g} "
              f"delta={sm['delta_final']:.2e} "
              f"({rec['stop'] or 'ok'}, {time.time()-t0:.0f}s)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "runs": out}))
    print(f"wrote {a.out}  ({len(out)} runs, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
