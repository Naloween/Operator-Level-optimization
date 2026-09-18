"""Compare arms AT MATCHED TRAINING LOSS, not at a matched step budget.

The naive comparison in `studies/dial.py` is confounded: the operator arm and the baselines end at
very different points on the fitting curve (train 0.85 vs 0.06), so a test-loss gap between them
is indistinguishable from an early-stopping effect. The fix is to read every arm's test loss at
the SAME training loss, which is exactly the generalisation curve `test(train)`.

For each run we interpolate `test_loss` at a set of training-loss levels along its own trajectory,
then aggregate over seeds. An arm that is genuinely better generalising sits below the others on
this curve at every level it reaches.
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
import numpy as np


def at_level(trace, level, xkey="train_loss", ykey="test_loss"):
    """Interpolate y at the first crossing of x == level (x decreases along training)."""
    xs = [t[xkey] for t in trace if xkey in t and ykey in t]
    ys = [t[ykey] for t in trace if xkey in t and ykey in t]
    if not xs or min(xs) > level:
        return None                                   # never reached that training loss
    for i in range(1, len(xs)):
        if xs[i] <= level <= xs[i - 1]:
            span = xs[i - 1] - xs[i]
            w = 0.0 if span == 0 else (xs[i - 1] - level) / span
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[int(np.argmin(xs))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", default=["runs/theory/dial_fair.json"])
    ap.add_argument("--levels", type=float, nargs="+", default=[1.5, 1.0, 0.75, 0.5, 0.25])
    a = ap.parse_args()

    rows = []
    for f in a.files:
        rows += json.load(open(f))["rows"]

    g = defaultdict(list)
    for r in rows:
        key = (r["arm"], tuple(r["hyper"]) if isinstance(r["hyper"], (list, tuple)) else r["hyper"])
        g[key].append(r)

    hdr = "  ".join(f"{('tr='+str(l)):>9}" for l in a.levels)
    print(f"{'arm':<5} {'hyper':<16} {hdr}   {'best train':>11} {'best acc':>9}")
    print("-" * (24 + 11 * len(a.levels) + 24))
    table = []
    for key, rs in g.items():
        cells, reached = [], 0
        for lv in a.levels:
            vals = [at_level(r["trace"], lv) for r in rs]
            vals = [v for v in vals if v is not None]
            if vals:
                cells.append(f"{np.mean(vals):9.3f}")
                reached += 1
            else:
                cells.append(f"{'--':>9}")
        bt = np.mean([min(t["train_loss"] for t in r["trace"]) for r in rs])
        ba = np.mean([max(t["test_acc"] for t in r["trace"]) for r in rs])
        table.append((reached, key, cells, bt, ba))
    for reached, key, cells, bt, ba in sorted(table, key=lambda z: (-z[0], str(z[1]))):
        print(f"{key[0]:<5} {str(key[1]):<16} " + "  ".join(cells) + f"   {bt:11.4f} {ba:9.4f}")


if __name__ == "__main__":
    main()
