"""When is an optimiser's generalisation advantage created?

Train with arm A for `switch_at` steps, then hand over to arm B for the remainder, with the total
budget held fixed. Sweeping the handover point localises WHERE the advantage is established:
early (the handover barely matters), late (only a full run of A keeps it), or throughout.

Both directions are run, since they answer different questions. adam->op asks whether the
reference can preserve what Adam built; op->adam asks whether Adam can repair what the reference
built.
"""
from __future__ import annotations
import argparse, itertools, time
import torch
import exp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["relu", "crelu"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--switch-at", type=int, nargs="+",
                    default=[0, 250, 1000, 3000, 6000, 12000])
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--experiment", default="switch_mnist1d")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    jobs = []
    for arch, seed, at in itertools.product(a.archs, a.seeds, a.switch_at):
        jobs.append((arch, seed, at, "adam", a.adam_lr, "op", [a.eta, a.k]))
        jobs.append((arch, seed, at, "op", [a.eta, a.k], "adam", a.adam_lr))
    print(f"{len(jobs)} cells on {dev}", flush=True)
    t0 = time.time()
    for i, (arch, seed, at, A, hA, B, hB) in enumerate(jobs, 1):
        cfg = dict(arch=arch, init="he", arm="switch", hyper=[A, hA, B, hB, at],
                   depth=8, width=128, n_train=4000, n_val=1000, n_test=1000, batch=100,
                   steps=a.steps, every=250, probe_n=64, seed=seed,
                   snap_every=4000, ckpt_every=4000, metrics_version=2)
        r = exp.execute(a.experiment, cfg, dev)
        print(f"[{i:>3}/{len(jobs)}] {arch:>5} s{seed} {A}->{B} @{at:<6} {r['status']:>8} "
              f"{r.get('seconds',0):5.0f}s  elapsed {(time.time()-t0)/60:5.1f}min", flush=True)
    print("SWITCH_DONE")


if __name__ == "__main__":
    main()
