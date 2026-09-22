"""Drives E5 (operator arm), E6 (endpoints) and E7 (gates and dynamics) as ONE sweep.

All three questions are answered from the same free-running trajectories, so the runs are shared
rather than repeated: E5 is the hyperparameter selection, E6 reads the endpoints at matched
training loss, and E7 reads the gate statistics and dynamics along the way. Storage is handled by
`exp.py`, keyed by configuration, so re-invoking this script costs nothing for work already done
and any new cell is simply added.

Reference configuration from E5: depth 8, width 128 on MNIST-1D, the deepest setting in which the
bias-free model both fits the training set and generalises.
"""
from __future__ import annotations
import argparse, itertools, time
import torch
import exp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["crelu", "relu"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--depths", type=int, nargs="*", default=None,
                    help="sweep over depths; absent means the single --depth")
    ap.add_argument("--widths", type=int, nargs="+", default=[128])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--adam-lrs", type=float, nargs="*", default=[1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--gd-lrs", type=float, nargs="*", default=[3e-3, 1e-2, 3e-2, 1e-1])
    ap.add_argument("--etas", type=float, nargs="*", default=[0.1, 0.3, 0.5, 1.0])
    ap.add_argument("--betas", type=float, nargs="*", default=[],
                    help="momentum on the reference step; beta=0 is the plain reference")
    ap.add_argument("--muon-lrs", type=float, nargs="*", default=[])
    ap.add_argument("--shampoo-lrs", type=float, nargs="*", default=[])
    ap.add_argument("--soap-lrs", type=float, nargs="*", default=[])
    ap.add_argument("--kfac-lrs", type=float, nargs="*", default=[])
    ap.add_argument("--heavyball-lrs", type=float, nargs="*", default=[])
    ap.add_argument("--noise-eps", type=float, nargs="*", default=[],
                    help="noise injected into the reference step, relative to its own norm")
    ap.add_argument("--noise-modes", nargs="*", default=["iso"],
                    choices=["iso", "ker", "range"])
    ap.add_argument("--ks", type=int, nargs="+", default=[50],
                    help="Krylov truncation levels: k=1 is exactly gradient descent, "
                         "large k the reference, so this is the bias dial")
    ap.add_argument("--task", default="mnist1d", choices=["mnist1d", "cifar10", "moons"])
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--align-every", type=int, default=2000)
    ap.add_argument("--align-k", type=int, default=3000)
    ap.add_argument("--align-batch", type=int, default=0, help="0 = same as --batch")
    ap.add_argument("--align-dtype", default="float64", choices=["float64","float32"])
    ap.add_argument("--experiment", default="ref_mnist1d")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    depths = a.depths if a.depths else [a.depth]
    jobs = []
    for depth, arch, seed, width in itertools.product(depths, a.archs, a.seeds, a.widths):
        for lr in a.adam_lrs:
            jobs.append((depth, arch, seed, width, "adam", lr))
        for lr in a.gd_lrs:
            jobs.append((depth, arch, seed, width, "gd", lr))
        for name in ("muon", "shampoo", "soap", "kfac", "heavyball"):
            for lr in getattr(a, f"{name}_lrs"):
                jobs.append((depth, arch, seed, width, name, lr))
        for e in a.etas:
            for k in a.ks:
                jobs.append((depth, arch, seed, width, "op", (e, k)))
        for e in a.etas:
            for k in a.ks:
                for b in a.betas:
                    jobs.append((depth, arch, seed, width, "opmom", (e, k, b)))
        for e in a.etas:
            for k in a.ks:
                for eps in a.noise_eps:
                    for mode in a.noise_modes:
                        jobs.append((depth, arch, seed, width, "opnoise", (e, k, eps, mode)))
    print(f"{len(jobs)} cells on {dev}", flush=True)

    t0 = time.time()
    for i, (depth, arch, seed, width, arm, hyper) in enumerate(jobs, 1):
        cfg = dict(arch=arch, init="he", arm=arm, hyper=hyper, depth=depth, width=width,
                   n_train=a.n_train, n_val=a.n_val, n_test=a.n_test,
                   batch=100, steps=a.steps, every=250,
                   probe_n=64, seed=seed, snap_every=4000, ckpt_every=4000,
                   align_every=a.align_every, eta_probe=0.3,
                   # bump when a MEASUREMENT changes: the run id is the hash of this config, so a
                   # new measurement must invalidate finished runs rather than silently be absent
                   metrics_version=2)
        if a.task != "mnist1d":          # see exp.get_task: the key must stay absent for mnist1d
            cfg["task"] = a.task
        if a.align_k != 3000:            # likewise: absent means the original 3000-iteration budget
            cfg["align_k"] = a.align_k
        if a.align_batch:
            cfg["align_batch"] = a.align_batch
        if a.align_dtype != "float64":
            cfg["align_dtype"] = a.align_dtype
        r = exp.execute(a.experiment, cfg, dev, force=a.force)
        h = hyper if not isinstance(hyper, tuple) else f"{hyper[0]:g}/k{hyper[1]}"
        print(f"[{i:>3}/{len(jobs)}] L{depth:<3} {arch:>5} s{seed} w{width:<3} {arm:>9} {str(h):<9} {r['status']:>8}"
              f" {r.get('seconds',0):6.0f}s  elapsed {(time.time()-t0)/60:5.1f}min"
              + (f"  ERROR {r['error']}" if r.get("error") else ""), flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
