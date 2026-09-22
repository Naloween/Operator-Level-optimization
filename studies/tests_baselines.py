"""Check the functional comparators against the `olo` originals they were transcribed from.

Two properties per rule:

  * **Agreement.** Several steps of the functional rule reproduce the same weights as the
    `torch.optim` original driven with the same gradients. "The baselines from the previous
    paper" has to mean the same update, not the same name, or every cross-paper comparison is
    meaningless.
  * **Scoring purity.** Calling with `mutate=False` must not advance any buffer. The alignment
    probe scores a hypothetical step at every probe point; if scoring mutated state, each probe
    would silently age the optimiser by one step, and the damage would show up as a slow drift
    that no single measurement looks wrong enough to catch.

Run: `python tests_baselines.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import baselines

TOL = 1e-9


def _olo_reference(name, shapes, grads_per_step, lr, extra):
    """Drive the olo torch.optim original with a fixed gradient sequence; return final weights."""
    from olo.optim.baselines.muon import _Muon
    from olo.optim.baselines.shampoo import _Shampoo
    from olo.optim.baselines.soap import _SOAP
    ctor = {"muon": _Muon, "shampoo": _Shampoo, "soap": _SOAP}[name]
    ps = [torch.nn.Parameter(torch.zeros(s, dtype=torch.float64)) for s in shapes]
    opt = ctor(ps, lr=lr, **extra)
    for gs in grads_per_step:
        for p, gi in zip(ps, gs):
            p.grad = gi.clone()
        opt.step()
    return [p.detach().clone() for p in ps]


def _functional(name, shapes, grads_per_step, lr, extra_hyper):
    W = [torch.zeros(s, dtype=torch.float64) for s in shapes]
    state = {}
    for t, gs in enumerate(grads_per_step):
        if name == "muon":
            d = baselines.muon(list(gs), state, (lr,) + extra_hyper, True)
        elif name == "shampoo":
            d = baselines.shampoo(list(gs), state, (lr,) + extra_hyper, True)
        else:
            d = baselines.soap(list(gs), state, (lr,) + extra_hyper, t, True)
        W = [w + di for w, di in zip(W, d)]
    return W


def test_agreement():
    torch.manual_seed(0)
    shapes = [(6, 4), (5, 6), (3, 5)]
    grads = [[torch.randn(s, dtype=torch.float64) for s in shapes] for _ in range(4)]
    cases = [
        ("muon", 0.05, (0.95,), dict(momentum=0.95, nesterov=True, ns_steps=5, eps=1e-8)),
        ("shampoo", 0.05, (0.9,), dict(beta=0.9, momentum=0.0, eps=1e-4)),
        ("soap", 0.05, (0.95,), dict(betas=(0.9, 0.999), shampoo_beta=0.95, eps=1e-8)),
    ]
    for name, lr, hyp, extra in cases:
        want = _olo_reference(name, shapes, grads, lr, extra)
        got = _functional(name, shapes, grads, lr, hyp)
        err = max(float((a - b).abs().max()) for a, b in zip(want, got))
        status = "ok " if err < TOL else "FAIL"
        print(f"  [{status}] {name:<8} max|functional - olo| = {err:.3e}")
        assert err < TOL, f"{name} disagrees with the olo original by {err:.3e}"


def test_scoring_does_not_mutate():
    torch.manual_seed(1)
    shapes = [(6, 4), (5, 6)]
    g = [torch.randn(s, dtype=torch.float64) for s in shapes]
    for name, fn in [("heavyball", lambda st, m: baselines.heavyball(g, st, (0.1, 0.9), m)),
                     ("muon", lambda st, m: baselines.muon(g, st, (0.1, 0.95), m)),
                     ("shampoo", lambda st, m: baselines.shampoo(g, st, (0.1, 0.9), m)),
                     ("soap", lambda st, m: baselines.soap(g, st, (0.1, 0.95), 0, m))]:
        st = {}
        fn(st, True)                         # one real step, so buffers exist and are non-zero
        before = {k: [x.clone() for x in v] for k, v in st.items() if isinstance(v, list)}
        a = fn(st, False)
        b = fn(st, False)                    # scoring twice must give the same answer
        after = {k: [x.clone() for x in v] for k, v in st.items() if isinstance(v, list)}
        moved = max((float((x - y).abs().max()) for k in before
                     for x, y in zip(before[k], after[k])), default=0.0)
        repeat = max(float((x - y).abs().max()) for x, y in zip(a, b))
        ok = moved == 0.0 and repeat == 0.0
        print(f"  [{'ok ' if ok else 'FAIL'}] {name:<10} state drift {moved:.1e}, "
              f"score repeatability {repeat:.1e}")
        assert ok, f"{name} mutates state while scoring"


def test_kfac_shapes():
    """K-FAC runs on the real chain and produces a step of the right shape for every layer."""
    import exp, gpu
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for arch in ("relu", "crelu"):
        Ws = gpu.init_net(8, 16, 3, 4, arch, 0, dev)
        X = torch.randn(32, 8, device=dev)
        out, _ = gpu.forward(Ws, X, arch)
        R = torch.randn_like(out)
        g = gpu.coord_grad(Ws, X, R / X.shape[0], arch)
        st = {}
        d = baselines.kfac(Ws, X, R / X.shape[0], g, st, (0.1, 1e-2), arch, True)
        assert len(d) == len(Ws)
        for di, w in zip(d, Ws):
            assert di.shape == w.shape, f"{arch}: {di.shape} != {w.shape}"
            assert torch.isfinite(di).all(), f"{arch}: non-finite K-FAC step"
        before = [x.clone() for x in st["kf_A"]]
        baselines.kfac(Ws, X, R / X.shape[0], g, st, (0.1, 1e-2), arch, False)
        drift = max(float((a - b).abs().max()) for a, b in zip(before, st["kf_A"]))
        print(f"  [{'ok ' if drift == 0 else 'FAIL'}] kfac/{arch:<6} shapes match, "
              f"covariance drift while scoring {drift:.1e}")
        assert drift == 0.0


if __name__ == "__main__":
    print("agreement with the olo originals:")
    test_agreement()
    print("scoring purity:")
    test_scoring_does_not_mutate()
    print("kfac on the real chain:")
    test_kfac_shapes()
    print("all baseline tests passed")
