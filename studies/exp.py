"""Experiment runner with config-keyed, resumable, re-analysable storage.

Every run is identified by the SHA-1 of its canonical configuration, never by a filename, so a run
is skipped because its *configuration* is already complete rather than because a name matched. The
layout is

    runs/exp/<experiment>/<run_id>/
        config.json     the exact configuration, canonical JSON (the identity of the run)
        meta.json       status, git commit, device, torch version, wall clock, error if any
        metrics.jsonl   one JSON object per probe, appended -- the full trajectory
        ckpt.pt         weights + optimiser moments + step, for resuming
        snap_<step>.pt  periodic weight snapshots, so any quantity can be recomputed later
                        without rerunning training

`metrics.jsonl` and the snapshots are what make later analysis possible without recomputation: the
per-probe record carries the loss and accuracy curves, the operator rank, and the full gate
statistics, and the snapshots let a new measurement be evaluated after the fact at the points where
it matters.

Measured per probe, all defined in the paper draft:
    loss/accuracy    on the train, validation and test splits
    pr               participation ratio of P(x), averaged over inputs with P(x) != 0
    dead_inputs      fraction of probe inputs with P(x) = 0 exactly
    dead_units       fraction of hidden units inactive for every probe input
    density          mean fraction of open gates
    hamming          mean normalised Hamming distance between two inputs' gate patterns,
                     i.e. how much of the behaviour is input-dependent at all
    churn            fraction of gates that flipped since the previous probe
    cos_op/cos_w     (optional, expensive) alignment of a step with the reference, in operator
                     and in weight space respectively; recorded for this arm and, from the same
                     weights, for gradient descent and Adam, so the three are comparable
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import gpu

ROOT = Path(__file__).resolve().parent.parent / "runs" / "exp"


# --------------------------------------------------------------------------- identity & storage
def canon(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def run_id(cfg: dict) -> str:
    return hashlib.sha1(canon(cfg).encode()).hexdigest()[:16]


def run_dir(experiment: str, cfg: dict) -> Path:
    return ROOT / experiment / run_id(cfg)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()[:12]
    except Exception:
        return "unknown"


def is_complete(d: Path, cfg: dict) -> bool:
    """Complete means: this exact configuration finished. Name matching is never used."""
    mp, cp = d / "meta.json", d / "config.json"
    if not (mp.exists() and cp.exists()):
        return False
    try:
        if canon(json.loads(cp.read_text())) != canon(cfg):
            return False
        return json.loads(mp.read_text()).get("status") == "complete"
    except Exception:
        return False


# --------------------------------------------------------------------------- task & losses
CIFAR_ROOT = "/home/naloween/Documents/datasets"


def cifar10(n_train, n_val, n_test, seed, dev):
    """CIFAR-10 flattened to 3072-dim vectors, standardised, with a held-out validation split.

    Standardising subtracts a constant, which is a change of input and not of the model, so
    f(x) = P(x) x is untouched: homogeneity is a property of the bias-free network, not the data.
    """
    from torchvision.datasets import CIFAR10
    import numpy as _np
    tr = CIFAR10(CIFAR_ROOT, train=True, download=False)
    te = CIFAR10(CIFAR_ROOT, train=False, download=False)
    Xtr_all = torch.tensor(_np.asarray(tr.data), dtype=torch.float32).reshape(len(tr.data), -1) / 255.0
    Ytr_all = torch.tensor(_np.asarray(tr.targets), dtype=torch.long)
    Xte_all = torch.tensor(_np.asarray(te.data), dtype=torch.float32).reshape(len(te.data), -1) / 255.0
    Yte_all = torch.tensor(_np.asarray(te.targets), dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    ptr = torch.randperm(Xtr_all.shape[0], generator=g)
    pte = torch.randperm(Xte_all.shape[0], generator=g)
    itr, iva = ptr[:n_train], ptr[n_train:n_train + n_val]
    ite = pte[:n_test]
    mu, sd = Xtr_all[itr].mean(0, keepdim=True), Xtr_all[itr].std(0, keepdim=True).clamp_min(1e-6)
    f = lambda X: ((X - mu) / sd).to(dev)
    return dict(Xtr=f(Xtr_all[itr]), Ytr=Ytr_all[itr].to(dev),
                Xva=f(Xtr_all[iva]), Yva=Ytr_all[iva].to(dev),
                Xte=f(Xte_all[ite]), Yte=Yte_all[ite].to(dev))


def moons(n_train, n_val, n_test, seed, dev, noise=0.15):
    """Two moons in 2D, with a constant coordinate appended.

    Why the constant. A bias-free positively homogeneous network satisfies f(lambda x) =
    lambda f(x), so its decision regions are CONES through the origin and two moons would be
    unlearnable. Appending a constant input restores affine capability on the original two
    coordinates while leaving the network bias-free, so f(x) = P(x) x and the whole operator
    framework remain exact. The visualisations below are drawn in the original 2D plane.
    """
    import numpy as _np
    def make(n, s):
        rng = _np.random.default_rng(s)
        t = rng.uniform(0, _np.pi, n // 2)
        a = _np.stack([_np.cos(t), _np.sin(t)], 1)
        b = _np.stack([1 - _np.cos(t), 1 - _np.sin(t) - 0.5], 1)
        X = _np.concatenate([a, b]) + rng.normal(0, noise, (n, 2))
        y = _np.array([0] * (n // 2) + [1] * (n - n // 2))
        q = rng.permutation(n)
        return X[q], y[q]
    out = {}
    for name, n, s in (("tr", n_train, seed), ("va", n_val, seed + 101), ("te", n_test, seed + 202)):
        X, y = make(n, s)
        X = _np.concatenate([X, _np.ones((len(X), 1))], 1)          # the constant coordinate
        out[f"X{name}"] = torch.tensor(X, dtype=torch.float32, device=dev)
        out[f"Y{name}" if name != "tr" else "Ytr"] = torch.tensor(y, dtype=torch.long, device=dev)
    return dict(Xtr=out["Xtr"], Ytr=out["Ytr"], Xva=out["Xva"], Yva=out["Yva"],
                Xte=out["Xte"], Yte=out["Yte"])


def get_task(cfg, dev):
    """Dispatch on cfg['task'], defaulting to mnist1d.

    The key is ABSENT from the configs of the runs completed before CIFAR was added, and must stay
    absent for them: the run id is the hash of the config, so introducing the key unconditionally
    would invalidate 125 finished runs and force a full recompute. New tasks carry the key; the
    original task is identified by its absence.
    """
    name = cfg.get("task", "mnist1d")
    args = (cfg["n_train"], cfg["n_val"], cfg["n_test"], cfg["seed"], dev)
    if name == "mnist1d":
        return mnist1d(*args)
    if name == "cifar10":
        return cifar10(*args)
    if name == "moons":
        return moons(*args)
    raise ValueError(f"unknown task {name!r}")


def mnist1d(n_train, n_val, n_test, seed, dev):
    from olo.tasks.mnist1d import MNIST1D
    t = MNIST1D(n_train=n_train, n_val=n_val, n_test=n_test, seed=seed)
    return dict(Xtr=t.Xtr.to(dev), Ytr=t.ytr.to(dev), Xva=t.Xva.to(dev), Yva=t.yva.to(dev),
                Xte=t.Xte.to(dev), Yte=t.yte.to(dev))


def ce(out, Y):
    n = out.shape[0]
    p = torch.softmax(out - out.max(1, keepdim=True).values, 1)
    idx = torch.arange(n, device=out.device)
    loss = float(-torch.log(p[idx, Y].clamp_min(1e-30)).mean())
    R = p.clone()
    R[idx, Y] -= 1.0
    return loss, R, float((out.argmax(1) == Y).float().mean())


# --------------------------------------------------------------------------- measurements
def operator_stats(Ws, X, arch):
    """Participation ratio of P(x) and the fraction of inputs annihilated outright."""
    W = [w.double() for w in Ws]
    Xd = X.double()
    _, gs = gpu.forward(W, Xd, arch)
    A, B = gpu.contexts(W, gs, Xd, arch)
    sv = torch.linalg.svdvals(A[0] @ W[0] @ B[0])
    live = sv[:, 0] > 1e-9 * sv[:, 0].max().clamp_min(1e-300)
    dead = float((~live).float().mean())
    if not bool(live.any()):
        return float("nan"), dead
    s = sv[live]
    return float(((s.sum(1) ** 2) / (s ** 2).sum(1)).mean()), dead


def gate_stats(Ws, X, arch, prev=None):
    """Occupancy, dead units, input-dependence (Hamming) and churn since the previous probe."""
    if arch in ("linear", "fgln", "crelu"):
        _, gs = gpu.forward(Ws, X, arch)
        on = [(g > 0) for g in gs] if arch == "crelu" else None
        if on is None:
            return dict(density=float("nan"), dead_units=float("nan"),
                        hamming=float("nan"), churn=float("nan")), None
    else:
        _, gs = gpu.forward(Ws, X, arch)
        on = [g > 0.5 for g in gs]
    flat = torch.cat([o.reshape(o.shape[0], -1) for o in on], dim=1)      # (n, total gates)
    density = float(flat.float().mean())
    dead_units = float((~flat.any(0)).float().mean())
    n = flat.shape[0]
    half = n // 2
    hamming = float((flat[:half] != flat[half:2 * half]).float().mean()) if half else float("nan")
    churn = float((flat != prev).float().mean()) if prev is not None and prev.shape == flat.shape \
        else float("nan")
    return dict(density=density, dead_units=dead_units, hamming=hamming, churn=churn), flat


def alignment(Ws, X, R, eta, arch, arm, hyper, state, step, k_ideal=3000,
              dtype="float64"):
    # `k_ideal` is a budget, not a constant. The step map has n*d_out*d_in equations, so it grows
    # 77x going from MNIST-1D (40k) to CIFAR-10 (3.07M) and each LSQR iteration grows with it. The
    # solve quality is recorded as `solve_ne` at every probe, so a reduced budget shows up in the
    # data as a worse residual rather than as a silently wrong number.
    """How close each arm's step is to the reference, in BOTH spaces. Expensive.

    Three objects, in plain terms:
        Dstar  what the loss asked the operator to do        = -eta * dL/dP, one per input
        PiD    the most of that the architecture can deliver = M M^+ Dstar. When the model is
               over-parameterised relative to the batch this equals Dstar, alpha = 1, and the
               distinction disappears entirely.
        ref    the weight update delivering PiD with the smallest norm = M^+ Dstar

    and for the step dW an arm would take from these same weights:

        cos_op = cos( M(dW), PiD )  operator space: did the operator move the way it was asked
        cos_w  = cos( dW,    ref )  weight space:   did the weights move the way that would have
                                    produced that operator motion
        ratio_op, ratio_w           the same comparisons in norm, which unlike the cosines do
                                    depend on the step size

    Both are recorded because they can disagree, and the disagreement is informative: dW and ref
    may differ by any element of ker M without changing the operator at all, so a low cos_w beside
    a high cos_op says the arm reached the right operator motion by a different route through
    weight space. That question is only askable because the reference is itself a weight update.
    """
    # Precision is a cost decision, not a detail. This GPU runs fp64 at 1/64 rate, and at CIFAR
    # scale one LSQR iteration is ~1e11 FLOP, so an fp64 probe costs ~0.17s per iteration and never
    # converges within a sane budget. fp32 was checked against fp64 at MNIST-1D scale on ReLU and
    # CReLU and agreed (alpha 0.1023 vs 0.0996); it is NOT safe for the input-independent
    # architectures, whose step maps are 83-93% null. `solve_ne` reports the outcome either way.
    dt = torch.float64 if dtype == "float64" else torch.float32
    W = [w.to(dt) for w in Ws]
    Xd, Rd = X.to(dt), R.to(dt)
    _, gs = gpu.forward(W, Xd, arch)
    A, B = gpu.contexts(W, gs, Xd, arch)
    shapes = [tuple(w.shape) for w in W]
    M = gpu.StepMap(A, B, shapes)
    Dstar = (-eta * (Rd.unsqueeze(2) * Xd.unsqueeze(1))).reshape(-1)
    ref = gpu.solve_min_norm(M, Dstar, iters=k_ideal, lam_rel=1e-7, stall=1e-10)
    PiD = M.mv(ref)
    ne = float(torch.linalg.vector_norm(M.rmv(PiD - Dstar))
               / torch.linalg.vector_norm(M.rmv(Dstar)).clamp_min(1e-300))
    alpha = float(torch.linalg.vector_norm(PiD)
                  / torch.linalg.vector_norm(Dstar).clamp_min(1e-300))
    out = dict(alpha=alpha, solve_ne=ne)
    nref = torch.linalg.vector_norm(ref)
    npi = torch.linalg.vector_norm(PiD)
    # score this arm's own step, and also gd's and adam's from the same weights, so the three are
    # comparable along whichever trajectory is being driven
    cands = {"": (arm, hyper), "gd_": ("gd", 1.0), "adam_": ("adam", 1.0)}
    for pre, (a, h) in cands.items():
        dW = step_for(a, W, Xd, Rd, h, arch, state, step, dtype_ok=True)
        flat = torch.cat([d.reshape(-1) for d in dW])
        r = M.mv(flat)
        nr, nw = torch.linalg.vector_norm(r), torch.linalg.vector_norm(flat)
        out[f"{pre}cos_op"] = float(r @ PiD / (nr * npi)) if nr > 0 and npi > 0 else float("nan")
        out[f"{pre}cos_w"] = float(flat @ ref / (nw * nref)) if nw > 0 and nref > 0 else float("nan")
        out[f"{pre}ratio_op"] = float(nr / npi) if npi > 0 else float("nan")
        out[f"{pre}ratio_w"] = float(nw / nref) if nref > 0 else float("nan")
    return out


# --------------------------------------------------------------------------- the arms
def _ckpt_state(state):
    """Every optimiser buffer, not just Adam's.

    `state` originally held only Adam's `m` and `v`, so the checkpoint hard-coded those two keys.
    The comparator arms in `baselines.py` keep their own buffers (Muon's momentum, Shampoo's and
    SOAP's accumulated factors, K-FAC's covariances); with the old code a resumed run would
    silently restart them from zero while the weights carried on, which no metric would flag.
    Runs here are killed and resumed routinely, so this had to be general.
    """
    out = {}
    for k, v in state.items():
        if isinstance(v, list) and v and all(hasattr(x, "detach") for x in v):
            out[k] = [x.detach().cpu() for x in v]
        elif isinstance(v, list) and all(x is None or hasattr(x, "detach") for x in v):
            out[k] = [None if x is None else x.detach().cpu() for x in v]
    return out


def _restore_state(z, dev):
    """Inverse of `_ckpt_state`, tolerant of checkpoints written before it existed."""
    st = {}
    for k, v in z.items():
        if k in ("Ws", "step"):
            continue
        if isinstance(v, list):
            st[k] = [None if x is None else x.to(dev) for x in v]
    st.setdefault("m", None)
    return st


# Which buffer each stateful arm needs before its step means anything. Scoring one of these from
# an empty state does not measure the rule: heavy ball with a zero momentum buffer takes the plain
# gradient step (which lies in range(M^T) BY CONSTRUCTION), Adam with m = v = 0 takes sign(g), and
# K-FAC with no accumulated covariances takes a single-batch natural gradient. That mistake has
# produced two wrong measurements in this project -- the ker M fractions of E4 and E18, where heavy
# ball read 0.0002 because it was scored as plain gradient descent -- so `step_for` now refuses
# rather than silently returning some other optimiser's step.
_ARM_STATE = {"opmom": "u", "heavyball": "hb", "muon": "muon", "shampoo": "sh_L",
              "soap": "so_m", "kfac": "kf_A"}


def requires_state(arm):
    """The state key `arm` must carry for a scored step to be that arm's step, or None."""
    return _ARM_STATE.get(arm)


def step_for(arm, Ws, X, R, hyper, arch, state, step, dtype_ok=False, allow_cold=False):
    """The update each arm would take. `state` holds the optimiser buffers, updated in place.

    SCORING A STATEFUL ARM: pass the state that arm's own trajectory produced. Scoring with an
    empty state silently evaluates a different rule (see `_ARM_STATE`), so that raises unless
    `allow_cold=True` says the cold-start step is genuinely what is wanted.

    `switch` hands over from one rule to another partway through training, to ask WHEN an
    optimiser's advantage is created: if Adam's benefit is established in the first few hundred
    steps and merely preserved thereafter, a run that starts with Adam and finishes with the
    reference should keep it. If the benefit requires Adam throughout, it should not.
    """
    # Only on the SCORING path, and never at step 0. Training legitimately starts without the
    # buffer and creates it on the first step; the alignment probe also runs at step 0, BEFORE
    # that first step, so a cold state there is correct rather than a mistake -- guarding it
    # raised on every cell of a stateful arm and killed a whole K-FAC sweep in 8.8 minutes.
    # What the guard is for is scoring a rule at step t > 0 with a state its own trajectory never
    # produced, which is how the ker M fractions of E4 and E18 came out wrong.
    need = _ARM_STATE.get(arm)
    if dtype_ok and step > 0 and need is not None and not allow_cold and need not in state:
        raise ValueError(
            f"step_for({arm!r}) called without its {need!r} buffer: a cold {arm} step is a "
            f"different optimiser, not {arm}. Pass the run's own state, or allow_cold=True if "
            f"the cold-start step is genuinely what you want.")
    n = X.shape[0]
    if arm == "opmom":
        # Momentum on the STEP, not on a gradient. The reference step already carries eta, so
        # there is no learning rate: u_t = beta u_{t-1} + (1-beta) d_t is applied as-is. The
        # (1-beta) keeps the steady-state magnitude equal to d_t, so beta=0 recovers the plain
        # reference exactly and beta only changes the direction, never the scale -- without it,
        # heavy-ball would inflate the step by 1/(1-beta) and confound momentum with step size.
        eta, k, beta = hyper
        d = gpu.op_step(Ws, X, R / R.norm().clamp_min(1e-12), eta, int(k), arch)
        out = []
        if dtype_ok:
            # Scoring must neither create nor mutate the training state. Creating it here was a
            # real bug: the alignment probe runs at step 0 *before* the first training step and
            # promotes weights to float64, so `setdefault` seeded the momentum buffer in float64;
            # the training path then added it to float32 weights, silently promoting them, and the
            # next forward died on a float/double matmul.
            u = state.get("u")
            for i, di in enumerate(d):
                prev = u[i].to(di.dtype) if u is not None else torch.zeros_like(di)
                out.append(beta * prev + (1 - beta) * di)
            return out
        u = state.setdefault("u", [torch.zeros_like(w) for w in Ws])
        for i, di in enumerate(d):
            u[i].mul_(beta).add_(di.to(u[i].dtype), alpha=1 - beta)
            out.append(u[i].clone())
        return out
    if arm == "opnoise":
        # The reference step plus noise, scaled relative to the step's own norm so that eps is
        # dimensionless and comparable across eta. `mode` decides WHERE the noise lives:
        #   iso    isotropic in weight space -- the standard noise-injection baseline
        #   ker    projected onto ker(M): moves the weights while leaving the operator EXACTLY
        #          unchanged, so it explores the function's level set and cannot help by
        #          changing what the network computes
        #   range  projected onto range(M^T): the complement, which changes the operator
        # The ker/range split is the part the step map makes possible and isotropic injection
        # cannot distinguish.
        eta, k, eps = hyper[0], int(hyper[1]), hyper[2]
        mode = hyper[3] if len(hyper) > 3 else "iso"
        d = gpu.op_step(Ws, X, R / R.norm().clamp_min(1e-12), eta, k, arch)
        flat = torch.cat([x.reshape(-1) for x in d])
        gen = torch.Generator(device=flat.device).manual_seed(hash((step, mode)) & 0x7fffffff)
        g = torch.randn(flat.shape, generator=gen, device=flat.device, dtype=flat.dtype)
        if mode in ("ker", "range"):
            _, gs = gpu.forward(Ws, X, arch)
            A, B = gpu.contexts(Ws, gs, X, arch)
            M = gpu.StepMap(A, B, [tuple(w.shape) for w in Ws])
            # M^+ M g is the projection of g onto range(M^T); the remainder lies in ker M
            # 300 iterations, not k: at k=50 the projection leaks 11% of the noise's operator
            # motion back in, which would make a "function-preserving" arm that visibly changes
            # the function. Measured leakage: 0.108 at 50 iters, 0.030 at 150, 0.012 at 300.
            g_rng = gpu.solve_min_norm(M, M.mv(g), iters=300, lam_rel=1e-7, stall=1e-10)
            g = (g - g_rng) if mode == "ker" else g_rng
        ng = torch.linalg.vector_norm(g).clamp_min(1e-30)
        noisy = flat + (eps * torch.linalg.vector_norm(flat) / ng) * g
        offs, out = 0, []
        for w in Ws:
            out.append(noisy[offs:offs + w.numel()].view(w.shape)); offs += w.numel()
        return out
    if arm == "switch":
        a, ha, b, hb, at = hyper
        sub, h = (a, ha) if step < at else (b, hb)
        return step_for(sub, Ws, X, R, tuple(h) if isinstance(h, list) else h,
                        arch, state, step, dtype_ok)
    if arm == "op":
        eta, k = hyper
        return gpu.op_step(Ws, X, R / R.norm().clamp_min(1e-12), eta, int(k), arch)
    g = gpu.coord_grad(Ws, X, R / n, arch)
    if arm == "gd":
        return [-hyper * x for x in g]
    if arm in ("muon", "shampoo", "soap", "kfac", "heavyball"):
        # The previous submission's comparators. Transcribed functionally in `baselines.py`;
        # `dtype_ok` is the scoring path and must leave every persistent buffer untouched.
        import baselines
        h = tuple(hyper) if isinstance(hyper, (list, tuple)) else (hyper,)
        mutate = not dtype_ok
        if arm == "heavyball":
            return baselines.heavyball(g, state, h, mutate)
        if arm == "muon":
            return baselines.muon(g, state, h, mutate)
        if arm == "shampoo":
            return baselines.shampoo(g, state, h, mutate)
        if arm == "soap":
            return baselines.soap(g, state, h, step, mutate)
        # R, not R/n. K-FAC's G factor is the covariance of the PER-EXAMPLE pre-activation
        # gradient, so it must not carry the batch mean: the original implementation writes
        # `z_grads[k] * B  # undo the batch mean` for exactly this reason. Passing R/n here made
        # G a factor n^2 too small while the damping stayed fixed, which multiplied the effective
        # damping by n^2 and left both factors effectively replaced by a multiple of the identity
        # -- measured: the step was then cos 0.89 to plain gradient descent instead of cos 0.16.
        # `g` is and must remain the MEAN gradient; only the Fisher factors change here.
        return baselines.kfac(Ws, X, R, g, state, h, arch, mutate)
    b1, b2, e = 0.9, 0.999, 1e-8
    m, v = state["m"], state["v"]
    out = []
    for i, gi in enumerate(g):
        if dtype_ok:                       # scoring only: do not mutate the driving state
            mh = (b1 * m[i] + (1 - b1) * gi.float()).to(gi.dtype) / (1 - b1 ** (step + 1))
            vh = (b2 * v[i] + (1 - b2) * gi.float() ** 2).to(gi.dtype) / (1 - b2 ** (step + 1))
        else:
            m[i].mul_(b1).add_(gi, alpha=1 - b1)
            v[i].mul_(b2).addcmul_(gi, gi, value=1 - b2)
            mh = m[i] / (1 - b1 ** (step + 1))
            vh = v[i] / (1 - b2 ** (step + 1))
        out.append(-hyper * mh / (vh.sqrt() + e))
    return out


# --------------------------------------------------------------------------- the run
def execute(experiment: str, cfg: dict, dev: str, force: bool = False) -> dict:
    d = run_dir(experiment, cfg)
    if not force and is_complete(d, cfg):
        return {"status": "skipped", "dir": str(d)}
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True))
    (d / "meta.json").write_text(json.dumps(
        {"status": "running", "git": git_sha(), "device": dev,
         "torch": torch.__version__, "started": time.time()}, indent=1))

    D = get_task(cfg, dev)
    d_out = int(D["Ytr"].max()) + 1
    Ws = gpu.init_net(D["Xtr"].shape[1], cfg["width"], d_out, cfg["depth"],
                      cfg["arch"], cfg["seed"], dev, init=cfg.get("init", "he"))
    state = {"m": [torch.zeros_like(w) for w in Ws], "v": [torch.zeros_like(w) for w in Ws]}
    start = 0
    ck = d / "ckpt.pt"
    if ck.exists() and not force:
        z = torch.load(ck, map_location=dev, weights_only=False)
        Ws = [w.to(dev) for w in z["Ws"]]
        state = _restore_state(z, dev)
        if state.get("m") is None:                  # pre-_ckpt_state checkpoints
            state = {"m": [w.to(dev) for w in z["m"]], "v": [w.to(dev) for w in z["v"]]}
        start = z["step"]
    mf = (d / "metrics.jsonl").open("a")

    hyper = tuple(cfg["hyper"]) if cfg["arm"] == "op" else cfg["hyper"]
    g = torch.Generator().manual_seed(cfg["seed"])
    if start:                                     # keep the batch stream reproducible on resume
        for _ in range(start):
            torch.randint(0, D["Xtr"].shape[0], (cfg["batch"],), generator=g)
    Xpr = D["Xtr"][:cfg["probe_n"]]
    prev_gates = None
    t0, err, last_step = time.time(), None, start
    try:
        for step in range(start, cfg["steps"] + 1):
            if step % cfg["every"] == 0:
                with torch.no_grad():
                    rec = {"step": step}
                    for split, (X, Y) in (("train", (D["Xtr"], D["Ytr"])),
                                          ("val", (D["Xva"], D["Yva"])),
                                          ("test", (D["Xte"], D["Yte"]))):
                        l, _, a = ce(gpu.forward(Ws, X, cfg["arch"])[0], Y)
                        rec[f"{split}_loss"], rec[f"{split}_acc"] = l, a
                    pr, dead = operator_stats(Ws, Xpr, cfg["arch"])
                    rec["pr"], rec["dead_inputs"] = pr, dead
                    gsx, prev_gates = gate_stats(Ws, Xpr, cfg["arch"], prev_gates)
                    rec.update(gsx)
                    if cfg.get("align_every") and step % cfg["align_every"] == 0:
                        nb = cfg.get("align_batch", cfg["batch"])
                        idx = torch.randint(0, D["Xtr"].shape[0], (nb,),
                                            generator=torch.Generator().manual_seed(1234)).to(dev)
                        o, _ = gpu.forward(Ws, D["Xtr"][idx], cfg["arch"])
                        _, R, _ = ce(o, D["Ytr"][idx])
                        rec.update(alignment(Ws, D["Xtr"][idx], R, cfg.get("eta_probe", 0.3),
                                             cfg["arch"], cfg["arm"], hyper, state, step,
                                             k_ideal=cfg.get("align_k", 3000),
                                             dtype=cfg.get("align_dtype", "float64")))
                    mf.write(json.dumps(rec) + "\n")
                    mf.flush()
                if step % cfg.get("snap_every", 10 ** 9) == 0:
                    torch.save({"Ws": [w.detach().cpu() for w in Ws], "step": step},
                               d / f"snap_{step}.pt")
            last_step = step
            if step == cfg["steps"]:
                break
            idx = torch.randint(0, D["Xtr"].shape[0], (cfg["batch"],), generator=g).to(dev)
            with torch.no_grad():
                out, _ = gpu.forward(Ws, D["Xtr"][idx], cfg["arch"])
                _, R, _ = ce(out, D["Ytr"][idx])
                dWs = step_for(cfg["arm"], Ws, D["Xtr"][idx], R, hyper, cfg["arch"], state, step)
                Ws = [w + dw for w, dw in zip(Ws, dWs)]
            if not all(torch.isfinite(w).all() for w in Ws):
                err = f"diverged at step {step}"
                break
            if step % cfg.get("ckpt_every", 2000) == 0 and step > start:
                torch.save({"Ws": [w.detach().cpu() for w in Ws],
                            **_ckpt_state(state), "step": step}, ck)
    except Exception as exc:                                   # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    mf.close()
    # Record the step actually reached, never cfg["steps"]. A run that diverged at step 370 once
    # wrote a checkpoint claiming step 12000; the next invocation resumed "from the end", emitted a
    # single probe of the diverged weights, and marked itself complete with a nan in the metrics.
    if err is None:
        torch.save({"Ws": [w.detach().cpu() for w in Ws],
                    **_ckpt_state(state), "step": last_step}, ck)
    elif ck.exists():
        ck.unlink()                      # a failed run must not be resumable into a false success
    if err is None:
        try:
            tail = [json.loads(l) for l in
                    (d / "metrics.jsonl").read_text().splitlines() if l.strip()][-1]
            if not all(np.isfinite(v) for v in tail.values() if isinstance(v, float)):
                err = "non-finite metrics at the final probe"
        except Exception:                                    # noqa: BLE001
            err = "unreadable metrics"
    (d / "meta.json").write_text(json.dumps(
        {"status": "complete" if err is None else "failed", "error": err, "git": git_sha(),
         "device": dev, "torch": torch.__version__, "seconds": time.time() - t0}, indent=1))
    return {"status": "complete" if err is None else "failed", "error": err,
            "seconds": time.time() - t0, "dir": str(d)}


def load(experiment: str):
    """Every completed run of an experiment, as (config, list-of-probe-records)."""
    out = []
    base = ROOT / experiment
    if not base.exists():
        return out
    for d in sorted(base.iterdir()):
        mp = d / "meta.json"
        if not mp.exists() or json.loads(mp.read_text()).get("status") != "complete":
            continue
        cfg = json.loads((d / "config.json").read_text())
        recs = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
        seen = {}
        for r in recs:                       # a resumed run may repeat a probe; keep the last
            seen[r["step"]] = r
        out.append((cfg, [seen[k] for k in sorted(seen)], d))
    return out
