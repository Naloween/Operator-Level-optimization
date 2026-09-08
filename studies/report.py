"""Turn the theory studies' JSON into the tables in `theory/04-instability.md`.

Reads whichever of `runs/theory/{forcing_linear,seed_source,symmetrize}.json` exist and
prints the corresponding section. Every number in file 04 §5 comes from here, so the claims
are re-derivable rather than transcribed.

    python studies/report.py [--dir runs/theory]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(d: Path, name: str):
    p = d / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def forcing_tables(data) -> None:
    rows = [(r["depth"], r["p"], r["c"], r["summary"]) for r in data["runs"]]
    ps = sorted({p for _, p, _, _ in rows})
    Ls = sorted({L for L, _, _, _ in rows})
    get = lambda L, p, c: next(s for l, q, k, s in rows if l == L and q == p and k == c)

    print("\n### 5.1  psi measured / predicted (deep linear, c = +1)\n")
    print("   L  " + "".join(f"{'p=' + str(p):>18}" for p in ps))
    worst = 0.0
    for L in Ls:
        cells = []
        for p in ps:
            s = get(L, p, 1)
            m, q = s.get("psi_measured", float("nan")), s["psi_pred"]
            if np.isfinite(m):
                worst = max(worst, abs(m - q))
            cells.append(f"{m:>8.3f} /{q:>7.3f}")
        print(f"{L:>4}  " + "".join(f"{c:>18}" for c in cells))
    print(f"\nworst |measured - predicted| over {len(Ls) * len(ps)} cells: {worst:.4f}")

    errs = [get(L, p, 1)["law_max_log_err"] for L in Ls for p in ps]
    print(f"full-spectrum law error (Cor. 14.1): {min(errs):.1e} .. {max(errs):.1e}")

    print("\n### 5.2  sign: final r(t)/r(0)\n")
    print(f"{'L':>5} {'p':>6} {'psi':>8} {'c=+1':>11} {'c=-1':>11}   both signs correct?")
    ok = 0
    tested = 0
    for L in Ls:
        for p in ps:
            a, b = get(L, p, 1), get(L, p, -1)
            psi = a["psi_pred"]
            if abs(psi) < 1e-9:
                good = abs(a["sep_ratio"] - 1) < 0.01 and abs(b["sep_ratio"] - 1) < 0.01
            else:
                good = ((a["sep_ratio"] > 1) == (psi > 0)) and ((b["sep_ratio"] > 1) != (psi > 0))
            ok += bool(good)
            tested += 1
            if L in (4, 32, 256):
                print(f"{L:>5} {p:>6} {psi:>8.3f} {a['sep_ratio']:>11.3g} "
                      f"{b['sep_ratio']:>11.3g}   {'yes' if good else 'NO'}")
    print(f"\nsign predictions correct: {ok}/{tested}")


def seed_tables(data) -> None:
    runs = data["runs"]
    print("\n### 5.4  seed: fluctuation (slope -1/2) or systematic (slope 0)\n")
    print(f"{'source':<34}{'slope (mean +- sd)':>22}{'ratio at max B':>17}{'n':>4}")
    groups: dict[str, list] = {}
    for r in runs:
        key = (f"prescribed, mu={r['mu']:g}" if r["task"] == "prescribed" else r["task"])
        groups.setdefault(key, []).append(r)
    for key, rs in groups.items():
        sl = np.array([r["slope"] for r in rs])
        hi = np.array([r["rows"][-1]["ratio_mean"] for r in rs])
        print(f"{key:<34}{f'{sl.mean():+.3f} +- {sl.std():.3f}':>22}"
              f"{hi.mean():>17.3f}{len(rs):>4}")
    print("\n(-0.5 = a mean-zero finite-batch fluctuation, Cor. 16.1;"
          " 0 = systematic, Cor. 16.2)")


def symmetrize_tables(data) -> None:
    runs = data["runs"]
    print("\n### 4.1  symmetrized batches: Delta stays at machine zero\n")
    print(f"{'init':<14}{'arm':<14}{'mu':>5}{'max_l |Delta_l|':>18}")
    for arm in sorted({r["arm"] for r in runs}):
        for mu in sorted({r["mu"] for r in runs}):
            sel = [r for r in runs if r["arm"] == arm and r["mu"] == mu]
            if sel:
                init = sel[0].get("init", "looks_linear")
                print(f"{init:<14}{arm:<14}{mu:>5g}"
                      f"{max(r['delta_final'] for r in sel):>18.2e}")

    print("\n### and it does not remove the bias: separation at matched growth\n")
    print(f"{'init':<14}{'arm':<14}{'mu':>5}{'L':>5}{'steps':>8}{'sbar':>8}"
          f"{'sep':>9}{'eff_rank':>10}{'fb share':>10}{'reached?':>10}")
    for r in sorted(runs, key=lambda r: (r.get("init", ""), r["mu"], r["depth"], r["arm"])):
        reached = "yes" if r["sbar_final"] >= 1.79 else "NO"
        print(f"{r.get('init', 'looks_linear'):<14}{r['arm']:<14}{r['mu']:>5g}"
              f"{r['depth']:>5}{r['steps_to_growth']:>8}{r['sbar_final']:>8.3f}"
              f"{r['sep_final']:>9.4f}{r['eff_rank_final']:>10.3f}"
              f"{r['feedback_share']:>10.2f}{reached:>10}")
    print("\nRows marked NO did not reach the growth target within the step budget, so their"
          "\nseparations are not comparable across depth -- read only the reached ones.")

    nonmono = []
    for r in runs:
        sep = [row["sep"] for row in r["trace"]]
        if len(sep) > 3 and max(sep) > 1.05 * sep[-1]:
            nonmono.append((r.get("init", "looks_linear"), r["arm"], r["mu"], r["depth"],
                            max(sep), sep[-1]))
    if nonmono:
        print("\n### non-monotone separation (Cor. 13.2: growth separates, shrinkage undoes it)\n")
        print(f"{'init':<14}{'arm':<14}{'mu':>5}{'L':>5}{'peak sep':>11}{'final sep':>11}")
        for row in nonmono:
            print(f"{row[0]:<14}{row[1]:<14}{row[2]:>5g}{row[3]:>5}{row[4]:>11.4f}{row[5]:>11.4f}")
    else:
        print("\nNo non-monotone separation trajectories in this grid.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="runs/theory")
    a = ap.parse_args()
    d = Path(a.dir)

    for name, script, fn in (("forcing_linear", "forcing", forcing_tables),
                             ("seed_source", "seed_source", seed_tables),
                             ("symmetrize", "symmetrize", symmetrize_tables)):
        data = _load(d, name)
        if data is None:
            print(f"\n[{name}.json not found -- run studies/{script}.py]")
        else:
            fn(data)


if __name__ == "__main__":
    main()
