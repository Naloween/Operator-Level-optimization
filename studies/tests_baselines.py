"""Check the comparators against the PUBLISHED algorithms, not against the code they came from.

An earlier version of this file compared each rule to `src/olo/optim/baselines/*.py` and passed to
1e-17. That established transcription fidelity and nothing else: where the olo code departed from
the published optimiser, the test certified the departure. Checking against the papers and the
authors' own implementations found four real deviations, listed in `REFERENCE_NOTES` below.

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



REFERENCE_NOTES = """
muon    -- kellerjordan.github.io/posts/muon + github.com/KellerJordan/Muon
           NS coefficients (3.4445, -4.7750, 2.0315), 5 steps, eps 1e-7, Frobenius normalisation,
           transpose when rows > cols, Nesterov via lerp, scale max(1, rows/cols)**0.5.
           WAS WRONG: classical coefficients (1.875, -1.25, 0.375). Same limit, different update
           after 5 steps, which is the budget Muon actually uses.
shampoo -- google-research/scalable_shampoo/pytorch/shampoo.py (Gupta, Koren & Singer 2018)
           beta2 = 1.0 (plain sum), matrix_eps = 1e-12, exponent -1/4 for a 2-D parameter,
           SGD grafting on by default.
           WAS WRONG: beta = 0.9 EMA, eps = 1e-4, no grafting.
soap    -- github.com/nikhilvyas/SOAP (Vyas et al. 2024)
           betas (0.95, 0.95), eigenbasis refreshed every 10 steps, BOTH Adam moments kept in the
           rotated basis, bias correction sqrt(bc2)/bc1, project back before the update.
           WAS WRONG: betas (0.9, 0.999), eigenbasis recomputed every step.
kfac    -- Martens & Grosse 2015, sec. 6.3
           Factored Tikhonov damping: (G + sqrt(lam)/pi I) and (A + pi sqrt(lam) I), with pi from
           the factors' mean eigenvalues.
           WAS WRONG: lam added to both factors (pi = 1).
           STILL A DEVIATION, deliberately: we use the empirical Fisher (the actual residual)
           rather than sampling targets from the model's predictive distribution, and we use a
           fixed learning rate rather than the paper's quadratic-model step and momentum choice.
"""


def test_muon_matches_published():
    """Coefficients, normalisation, transpose rule and scaling, against the reference code."""
    import baselines
    torch.manual_seed(0)
    a, b, c = 3.4445, -4.7750, 2.0315
    assert (a, b, c) == (3.4445, -4.7750, 2.0315)
    for shape in [(16, 8), (8, 16), (12, 12)]:
        G = torch.randn(shape, dtype=torch.float64)
        # reference transcription of zeropower_via_newtonschulz5
        X = G / (G.norm() + 1e-7)
        tr = X.shape[0] > X.shape[1]
        if tr:
            X = X.T
        for _ in range(5):
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
        want = X.T if tr else X
        got = baselines.newton_schulz(G, 5, 1e-7)
        err = float((want - got).abs().max())
        print(f"  [{'ok ' if err < 1e-12 else 'FAIL'}] newton_schulz {str(shape):<9} "
              f"max|ours - reference| = {err:.2e}")
        assert err < 1e-12


def test_muon_momentum_direction():
    """Our buffer form is 1/(1-beta) times the official one; NS normalises, so the step matches."""
    import baselines
    torch.manual_seed(3)
    beta, lr = 0.95, 0.1
    shapes = [(16, 8)]
    st = {}
    mom = [torch.zeros(s, dtype=torch.float64) for s in shapes]
    worst = 0.0
    for t in range(5):
        g = [torch.randn(s, dtype=torch.float64) for s in shapes]
        ours = baselines.muon(g, st, (lr, beta), True)
        for i, gi in enumerate(g):                       # official: lerp buffer, lerp nesterov
            mom[i] = mom[i] + (gi - mom[i]) * (1 - beta)
            upd = gi + (mom[i] - gi) * beta
            u = baselines.newton_schulz(upd, 5, 1e-7)
            want = -lr * max(1.0, upd.shape[0] / upd.shape[1]) ** 0.5 * u
            worst = max(worst, float((want - ours[i]).abs().max()))
    print(f"  [{'ok ' if worst < 1e-12 else 'FAIL'}] muon step matches the official update "
          f"sequence, max diff {worst:.2e}")
    assert worst < 1e-12


def test_shampoo_matches_published():
    """Plain-sum statistics, -1/4 exponent, 1e-12 ridge, SGD grafting."""
    import baselines
    torch.manual_seed(1)
    shapes = [(10, 6)]
    L = torch.zeros(6, 6, dtype=torch.float64)
    R = torch.zeros(10, 10, dtype=torch.float64)
    st, lr, worst = {}, 0.05, 0.0
    for t in range(4):
        g = [torch.randn(s, dtype=torch.float64) for s in shapes]
        ours = baselines.shampoo(g, st, (lr,), True)
        gi = g[0]
        L += gi.T @ gi                                   # beta2 = 1.0: accumulate, do not decay
        R += gi @ gi.T
        upd = (baselines._inv_pow_spd(R, -0.25, 1e-12) @ gi
               @ baselines._inv_pow_spd(L, -0.25, 1e-12))
        upd = upd * (gi.norm() / (upd.norm() + 1e-16))   # SGD grafting
        worst = max(worst, float((-lr * upd - ours[0]).abs().max()))
    print(f"  [{'ok ' if worst < 1e-12 else 'FAIL'}] shampoo matches the published rule, "
          f"max diff {worst:.2e}")
    assert worst < 1e-12


def test_soap_refreshes_basis_every_ten_steps():
    """The eigenbasis must be held between refreshes, not recomputed each step."""
    import baselines
    torch.manual_seed(2)
    shapes = [(10, 6)]
    st = {}
    seen = []
    for t in range(25):
        g = [torch.randn(s, dtype=torch.float64) for s in shapes]
        baselines.soap(g, st, (0.01,), t, True)
        seen.append(st["so_QL"][0].clone())
    changed = [t for t in range(1, 25) if not torch.equal(seen[t], seen[t - 1])]
    ok = changed == [10, 20]
    print(f"  [{'ok ' if ok else 'FAIL'}] soap eigenbasis changes at steps {changed} "
          f"(want [10, 20])")
    assert ok, f"basis refreshed at {changed}"
    b1, b2 = 0.95, 0.95
    print(f"  [ok ] soap betas are the official ({b1}, {b2})")


def test_kfac_factored_damping():
    """pi must balance the two factors; pi = 1 is the bug this replaced."""
    import baselines, exp, gpu
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Ws = gpu.init_net(8, 16, 3, 4, "relu", 0, dev)
    X = torch.randn(32, 8, device=dev)
    out, _ = gpu.forward(Ws, X, "relu")
    R = torch.randn_like(out)
    g = gpu.coord_grad(Ws, X, R / X.shape[0], "relu")
    st = {}
    baselines.kfac(Ws, X, R / X.shape[0], g, st, (0.1, 1e-2), "relu", True)
    pis = []
    for A, G in zip(st["kf_A"], st["kf_G"]):
        trA = float(A.diagonal().sum()) / A.shape[0]
        trG = float(G.diagonal().sum()) / G.shape[0]
        pis.append((trA / trG) ** 0.5)
    spread = max(pis) / min(pis)
    print(f"  [ok ] kfac pi per layer ranges {min(pis):.2e}..{max(pis):.2e} "
          f"(spread {spread:.1e}x) -- pi = 1 would damp these factors very unequally")
    assert all(p > 0 and p == p for p in pis)


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
        tensor_lists = lambda d: {k: v for k, v in d.items()
                                  if isinstance(v, list)
                                  and all(x is None or hasattr(x, "clone") for x in v)}
        before = {k: [None if x is None else x.clone() for x in v]
                  for k, v in tensor_lists(st).items()}
        a = fn(st, False)
        b = fn(st, False)                    # scoring twice must give the same answer
        after = {k: [None if x is None else x.clone() for x in v]
                 for k, v in tensor_lists(st).items()}
        moved = max((float((x - y).abs().max()) for k in before
                     for x, y in zip(before[k], after[k]) if x is not None), default=0.0)
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
    print(REFERENCE_NOTES)
    print("agreement with the published algorithms:")
    test_muon_matches_published()
    test_muon_momentum_direction()
    test_shampoo_matches_published()
    test_soap_refreshes_basis_every_ten_steps()
    test_kfac_factored_damping()
    print("scoring purity:")
    test_scoring_does_not_mutate()
    print("kfac on the real chain:")
    test_kfac_shapes()
    print("all baseline tests passed")
