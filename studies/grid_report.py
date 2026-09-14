"""Read `runs/theory/grid.json` and print the tables the comparison is for.

Three questions, three tables:

1. **What does the spectrum do?** Effective rank `PR` at the start, at its extreme, and at
   the end. The extreme matters because the trajectory is non-monotone — reading only the
   endpoints hides a dip-and-recover.
2. **How big is the bias?** `sin` = the scale-free mismatch between the realised operator
   step and `-eta E_x[dL/dJ]`. 0 means the factorisation reproduces operator-space descent
   up to a step size; 1 means the realised step is orthogonal to it.
3. **Does the bias track the conditioning?** Correlation of `sin` with `log kappa` along each
   trajectory, which is the forward arrow of `theory/10` Cor. 5.2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def agg(runs, key, fn):
    v = [fn(r) for r in runs]
    v = [x for x in v if np.isfinite(x)]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"),) * 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default="runs/theory/grid.json")
    a = ap.parse_args()
    d = json.loads(Path(a.path).read_text())
    runs = d["runs"]
    archs = sorted({r["arch"] for r in runs})
    inits = [i for i in ("identity", "orthogonal", "xavier") if any(r["init"] == i for r in runs)]
    tasks = [t for t in ("teacher_isotropic", "teacher_lowrank", "mnist")
             if any(r["task"] == t for r in runs)]
    print(f"{len(runs)} runs  |  "
          f"converged: {sum(r['stop'] == 'converged' for r in runs)}  "
          f"max_steps: {sum(r['stop'] == 'max_steps' for r in runs)}  "
          f"diverged: {sum(r['stop'] == 'diverged' for r in runs)}\n")

    print("=" * 96)
    print("1. EFFECTIVE RANK  PR:  start -> minimum along the trajectory -> end   (mean over seeds)")
    print("=" * 96)
    print(f"{'task':<19}{'arch':<15}" + "".join(f"{i:>20}" for i in inits))
    for t in tasks:
        for A in archs:
            cells = []
            for i in inits:
                sel = [r for r in runs if r["task"] == t and r["arch"] == A and r["init"] == i]
                if not sel:
                    cells.append(f"{'-':>20}"); continue
                s0, _ = agg(sel, "", lambda r: r["trace"][0]["pr"])
                lo, _ = agg(sel, "", lambda r: min(x["pr"] for x in r["trace"]))
                s1, _ = agg(sel, "", lambda r: r["trace"][-1]["pr"])
                cells.append(f"{f'{s0:.1f}>{lo:.1f}>{s1:.1f}':>20}")
            print(f"{t:<19}{A:<15}" + "".join(cells))

    print("\n" + "=" * 96)
    print("2. SCALE-FREE GRADIENT BIAS  sin(dJ, -G):  start -> end   (mean +- sd over seeds)")
    print("=" * 96)
    print(f"{'task':<19}{'arch':<15}" + "".join(f"{i:>20}" for i in inits))
    for t in tasks:
        for A in archs:
            cells = []
            for i in inits:
                sel = [r for r in runs if r["task"] == t and r["arch"] == A and r["init"] == i]
                if not sel:
                    cells.append(f"{'-':>20}"); continue
                a0, _ = agg(sel, "", lambda r: r["trace"][0]["sin"])
                a1, sd = agg(sel, "", lambda r: r["trace"][-1]["sin"])
                cells.append(f"{f'{a0:.2f}->{a1:.2f}+-{sd:.2f}':>20}")
            print(f"{t:<19}{A:<15}" + "".join(cells))

    print("\n" + "=" * 96)
    print("3. DOES THE BIAS TRACK THE CONDITIONING?  rank-corr(sin, separation) along each run")
    print("=" * 96)
    print(f"{'task':<19}{'arch':<15}" + "".join(f"{i:>20}" for i in inits))
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    for t in tasks:
        for A in archs:
            cells = []
            for i in inits:
                sel = [r for r in runs if r["task"] == t and r["arch"] == A and r["init"] == i]
                if not sel:
                    cells.append(f"{'-':>20}"); continue
                cs = []
                for r in sel:
                    s = np.array([x["sin"] for x in r["trace"]])
                    k = np.array([x["separation"] for x in r["trace"]])
                    m = np.isfinite(s) & np.isfinite(k)
                    if m.sum() > 4 and np.ptp(k[m]) > 1e-9 and np.ptp(s[m]) > 1e-9:
                        cs.append(float(np.corrcoef(rk(s[m]), rk(k[m]))[0, 1]))
                cells.append(f"{f'{np.mean(cs):+.2f}' if cs else '-':>20}")
            print(f"{t:<19}{A:<15}" + "".join(cells))

    print("\n" + "=" * 96)
    print("4. FINAL LOSS and SCALE RATIO ||dJ|| / eta||G||   (mean over seeds)")
    print("=" * 96)
    print(f"{'task':<19}{'arch':<15}" + "".join(f"{i:>20}" for i in inits))
    for t in tasks:
        for A in archs:
            cells = []
            for i in inits:
                sel = [r for r in runs if r["task"] == t and r["arch"] == A and r["init"] == i]
                if not sel:
                    cells.append(f"{'-':>20}"); continue
                L, _ = agg(sel, "", lambda r: r["loss_final"])
                sc, _ = agg(sel, "", lambda r: r["trace"][-1]["scale"])
                cells.append(f"{f'{L:.2e} | {sc:.2f}':>20}")
            print(f"{t:<19}{A:<15}" + "".join(cells))


if __name__ == "__main__":
    main()
