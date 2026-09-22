"""Continue finished runs far past their original horizon, to look for delayed generalisation.

The original runs are left untouched: a new configuration (differing only in `steps`) hashes to a
new directory, and the finished checkpoint is COPIED into it so training resumes from step 12000
rather than restarting. Snapshots are dense enough to watch the decision slice evolve.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import torch
import exp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="relu")
    ap.add_argument("--arms", nargs="+", default=["adam", "op", "gd"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-steps", type=int, default=12000)
    ap.add_argument("--to-steps", type=int, default=300000)
    ap.add_argument("--snap-every", type=int, default=25000)
    ap.add_argument("--experiment", default="ref_mnist1d")
    ap.add_argument("--out-experiment", default="grok_mnist1d")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    src = exp.load(a.experiment)
    for arm in a.arms:
        cand = [(c, r, d) for c, r, d in src
                if c["arch"] == a.arch and c["arm"] == arm and c["seed"] == a.seed
                and c["steps"] == a.from_steps and c["width"] == 128]
        if not cand:
            print(f"  {arm}: no source run"); continue
        # the hyperparameter that generalises best under the stable estimator
        def stable(r):
            q = r[int(len(r) * 0.75):]
            return sum(x["test_acc"] for x in q) / len(q)
        c, r, d = max(cand, key=lambda z: stable(z[1]))
        cfg = dict(c); cfg["steps"] = a.to_steps; cfg["snap_every"] = a.snap_every
        cfg["ckpt_every"] = a.snap_every
        dst = exp.run_dir(a.out_experiment, cfg)
        dst.mkdir(parents=True, exist_ok=True)
        if (d / "ckpt.pt").exists() and not (dst / "ckpt.pt").exists():
            shutil.copy(d / "ckpt.pt", dst / "ckpt.pt")       # resume, do not restart
        h = cfg["hyper"] if not isinstance(cfg["hyper"], list) else cfg["hyper"][0]
        print(f"  {arm} (hyper {h:g}): {a.from_steps} -> {a.to_steps}", flush=True)
        res = exp.execute(a.out_experiment, cfg, dev)
        print(f"    {res['status']} {res.get('seconds',0):.0f}s {res.get('error') or ''}",
              flush=True)
    print("GROK_DONE")


if __name__ == "__main__":
    main()
