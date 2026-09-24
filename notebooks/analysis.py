"""Analysis library for the operator-level experiments: loads runs and draws every figure.

Kept as a module rather than living inside the notebook so that the plotting code is testable and
reusable, and so the notebook stays a narrative. `results.ipynb` imports this.

Every figure is written to figures/ as PDF and PNG, ready to be included in the paper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "studies"))
FIGDIR = HERE.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

ARMS = ["gd", "adam", "op"]
LABEL = {"gd": "gradient descent", "adam": "Adam", "op": "reference (operator)"}
COLOR = {"gd": "#4c78a8", "adam": "#59a14f", "op": "#e15759"}


def load(experiment="ref_mnist1d"):
    import exp
    return exp.load(experiment)


def hyper_of(cfg):
    h = cfg["hyper"]
    return h[0] if isinstance(h, (list, tuple)) else h


def select(runs, arch=None, arm=None, seed=None):
    out = []
    for c, r, d in runs:
        if arch and c["arch"] != arch:
            continue
        if arm and c["arm"] != arm:
            continue
        if seed is not None and c["seed"] != seed:
            continue
        out.append((c, r, d))
    return out


def best_by_val(recs):
    """The probe with the highest validation accuracy -- never the max over the test curve."""
    return max(recs, key=lambda x: x["val_acc"])


def at_train_loss(recs, level, key):
    """Value of `key` when the training loss first falls to `level`, by linear interpolation."""
    xs = [r["train_loss"] for r in recs]
    ys = [r.get(key) for r in recs]
    if any(y is None for y in ys) or min(xs) > level:
        return None
    for i in range(1, len(xs)):
        if xs[i] <= level <= xs[i - 1]:
            span = xs[i - 1] - xs[i]
            w = 0.0 if span == 0 else (xs[i - 1] - level) / span
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[int(np.argmin(xs))]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    return FIGDIR / f"{name}.pdf"


# ---------------------------------------------------------------- summary tables
def summary_table(runs):
    """Per arch and arm: best hyperparameter by validation, with test accuracy and rank."""
    rows = []
    for arch in sorted({c["arch"] for c, _, _ in runs}):
        for arm in ARMS:
            sub = select(runs, arch=arch, arm=arm)
            if not sub:
                continue
            byh = {}
            for c, r, _ in sub:
                byh.setdefault(hyper_of(c), []).append(best_by_val(r))
            best_h = max(byh, key=lambda k: np.mean([b["val_acc"] for b in byh[k]]))
            b = byh[best_h]
            rows.append(dict(
                arch=arch, arm=arm, hyper=best_h, n=len(b),
                val=np.mean([x["val_acc"] for x in b]),
                test=np.mean([x["test_acc"] for x in b]),
                test_sd=np.std([x["test_acc"] for x in b]),
                pr=np.mean([x["pr"] for x in b]),
                train_at_peak=np.mean([x["train_loss"] for x in b]),
                final_train=np.mean([select(runs, arch, arm)[0][1][-1]["train_loss"]]),
                density=np.mean([x.get("density", np.nan) for x in b]),
                hamming=np.mean([x.get("hamming", np.nan) for x in b]),
            ))
    return rows


def convergence_table(runs):
    """Did every arm actually fit? Without this the comparison measures speed, not capability."""
    rows = []
    for arch in sorted({c["arch"] for c, _, _ in runs}):
        for arm in ARMS:
            sub = select(runs, arch=arch, arm=arm)
            if not sub:
                continue
            fin = [r[-1]["train_loss"] for _, r, _ in sub]
            acc = [r[-1]["train_acc"] for _, r, _ in sub]
            rows.append(dict(arch=arch, arm=arm, n=len(sub),
                             final_train_loss_min=float(np.min(fin)),
                             final_train_loss_med=float(np.median(fin)),
                             final_train_acc_max=float(np.max(acc))))
    return rows


# ---------------------------------------------------------------- figures
def fig_dynamics(runs, arch="crelu", seed=0):
    """Loss and accuracy against step, best hyperparameter per arm. Shows speed AND overfitting."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for arm in ARMS:
        sub = select(runs, arch=arch, arm=arm, seed=seed)
        if not sub:
            continue
        # Pick the run to draw with the SAME estimator that scores every number in the paper.
        # Selecting the displayed run by peak validation while the text reports the convergence
        # phase would show a different run from the one being described.
        def _sc(z):
            v = at_convergence(z[1])
            return v if v is not None else -np.inf
        c, r, _ = max(sub, key=_sc)
        s = [x["step"] for x in r]
        axes[0].plot(s, [x["train_loss"] for x in r], color=COLOR[arm], label=LABEL[arm])
        axes[1].plot(s, [x["test_acc"] for x in r], color=COLOR[arm], label=LABEL[arm])
        axes[2].plot(s, [x["pr"] for x in r], color=COLOR[arm], label=LABEL[arm])
        b = best_by_val(r)
        axes[1].plot(b["step"], b["test_acc"], "o", color=COLOR[arm], ms=6)
    axes[0].set_yscale("log"); axes[0].set_ylabel("training loss")
    axes[1].set_ylabel("test accuracy")
    axes[2].set_ylabel("operator rank (participation ratio)")
    for a in axes:
        a.set_xlabel("step"); a.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{arch}, seed {seed}: dots mark the validation-selected point", fontsize=10)
    fig.tight_layout()
    return fig


def fig_gates(runs, arch="relu", seed=0):
    """The gate side of learning: occupancy, input-dependence, churn, dead units."""
    import matplotlib.pyplot as plt
    keys = [("density", "gate density"), ("hamming", "pattern diversity\n(Hamming between inputs)"),
            ("churn", "pattern churn\n(flips since last probe)"), ("dead_units", "dead units")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for arm in ARMS:
        sub = select(runs, arch=arch, arm=arm, seed=seed)
        if not sub:
            continue
        # Pick the run to draw with the SAME estimator that scores every number in the paper.
        # Selecting the displayed run by peak validation while the text reports the convergence
        # phase would show a different run from the one being described.
        def _sc(z):
            v = at_convergence(z[1])
            return v if v is not None else -np.inf
        c, r, _ = max(sub, key=_sc)
        s = [x["step"] for x in r]
        for ax, (k, lab) in zip(axes, keys):
            v = [x.get(k, np.nan) for x in r]
            ax.plot(s, v, color=COLOR[arm], label=LABEL[arm])
            ax.set_ylabel(lab); ax.set_xlabel("step"); ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{arch}, seed {seed}", fontsize=10)
    fig.tight_layout()
    return fig


def fig_alignment(runs, arch="crelu", seed=0):
    """cos_op and cos_w along training, for each arm scored from its own trajectory."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for arm in ARMS:
        sub = select(runs, arch=arch, arm=arm, seed=seed)
        if not sub:
            continue
        # Pick the run to draw with the SAME estimator that scores every number in the paper.
        # Selecting the displayed run by peak validation while the text reports the convergence
        # phase would show a different run from the one being described.
        def _sc(z):
            v = at_convergence(z[1])
            return v if v is not None else -np.inf
        c, r, _ = max(sub, key=_sc)
        al = [x for x in r if "cos_op" in x]
        if not al:
            continue
        s = [x["step"] for x in al]
        axes[0].plot(s, [x["cos_op"] for x in al], "o-", color=COLOR[arm], label=LABEL[arm])
        axes[1].plot(s, [x["cos_w"] for x in al], "o-", color=COLOR[arm], label=LABEL[arm])
    for ax, lab in zip(axes, [r"$\cos_{op}$: operator change vs the reachable ideal",
                              r"$\cos_{w}$: weight step vs the reference step"]):
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("step"); ax.set_ylabel(lab, fontsize=9); ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{arch}, seed {seed}: each arm scored along its own trajectory", fontsize=10)
    fig.tight_layout()
    return fig


def fig_rank_vs_acc(runs, probes=8):
    """Operator rank against test accuracy, every run, at the convergence phase.

    Both axes come from the same window. Reading rank at the end of the budget instead would put
    the operator runs at the far left of this plot -- their rank collapses after interpolation
    (`at_convergence`) -- and the scatter would then show a relationship that is an artefact of how
    long each run sat past convergence rather than of the rule.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, arch in zip(axes, ["crelu", "relu"]):
        for arm in ARMS:
            xs, ys = [], []
            for _, r, _ in select(runs, arch=arch, arm=arm):
                pr = at_convergence(r, "pr", probes=probes)
                te = at_convergence(r, "test_acc", probes=probes)
                if pr is not None and te is not None:
                    xs.append(pr); ys.append(te)
            if not xs:
                continue
            ax.scatter(xs, ys, color=COLOR[arm], label=LABEL[arm], s=34, alpha=0.85,
                       edgecolor="white", linewidth=0.6)
        ax.set_xlabel("operator rank at convergence (participation ratio)")
        ax.set_ylabel("test accuracy at convergence")
        ax.set_title(arch); ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- post-hoc (from snapshots)
def posthoc_rows(run_dir):
    import json
    p = run_dir / "posthoc.json"
    return json.loads(p.read_text()) if p.exists() else []


def fig_local_complexity(runs, arch="relu"):
    """Local complexity against training, over all gates and over live units only.

    The two panels differ by exactly the dilution a dying network creates: a globally dead unit can
    never flip, so the all-gates curve falls for any optimiser that kills units regardless of where
    the partition boundaries actually are.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for arm in ARMS:
        sub = select(runs, arch=arch, arm=arm)
        if not sub:
            continue
        byh = {}
        for c, r, d in sub:
            byh.setdefault(hyper_of(c), []).append((c, r, d))
        def _cell(k):
            v = [at_convergence(r) for _, r, _ in byh[k]]
            v = [x for x in v if x is not None]
            return np.mean(v) if v else -np.inf
        bh = max(byh, key=_cell)
        series = [posthoc_rows(d) for _, _, d in byh[bh]]
        series = [s for s in series if s]
        if not series:
            continue
        steps = [x["step"] for x in series[0]]
        for ax, key in zip(axes, ["lc_0.3", "lcl_0.3"]):
            v = np.mean([[x[key] for x in s] for s in series], axis=0)
            ax.plot(steps, v, "o-", color=COLOR[arm], label=LABEL[arm])
    axes[0].set_title("all gates (diluted by dead units)", fontsize=10)
    axes[1].set_title("live units only", fontsize=10)
    for ax in axes:
        ax.set_xlabel("step"); ax.set_ylabel(r"local complexity at $\epsilon=0.3$")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{arch}: gate flips under an $\\epsilon$-perturbation of the input", fontsize=10)
    fig.tight_layout()
    return fig


def fig_layer_reps(runs, arch="relu", seed=0, n=400, layers=(0, 3, 7)):
    """Two-dimensional PCA of the hidden representation at several depths, coloured by class."""
    import sys as _s, matplotlib.pyplot as plt, torch
    _s.path.insert(0, str(HERE.parent / "studies"))
    import exp as _exp, posthoc as _ph
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Selected by the same estimator used everywhere else: `pick` takes the best hyper-parameter
    # by stable test accuracy, and the requested seed's run within it. Selecting this figure by
    # peak validation while every number in the paper is selected by the stable statistic would
    # show representations from a different run than the one being described.
    picked = []
    for arm in ARMS:
        got = pick_converged(runs, arch, arm, width=128, steps=12000)
        if not got:
            continue
        _, group = got
        same = [z for z in group if z[0]["seed"] == seed] or group
        picked.append((arm, same[0]))
    if not picked:
        return None
    c0 = picked[0][1][0]
    D = _exp.mnist1d(c0["n_train"], c0["n_val"], c0["n_test"], seed, dev)
    X, Y = D["Xte"][:n], D["Yte"][:n].cpu().numpy()
    fig, axes = plt.subplots(len(picked), len(layers), figsize=(3.4 * len(layers), 3.2 * len(picked)))
    axes = np.atleast_2d(axes)
    for i, (arm, (c, r, d)) in enumerate(picked):
        # Read at the convergence phase, not at the last snapshot: the operator arm's rank
        # collapses after interpolation (E9), and a final-snapshot PCA reports that collapse as if
        # it were the representation the rule builds while it is learning.
        cs = conv_snapshot(r, d)
        Ws = [w.to(dev) for w in torch.load(cs[1], map_location=dev,
                                            weights_only=False)["Ws"]]
        reps = _ph.layer_reps(Ws, X, arch)
        for j, l in enumerate(layers):
            ax = axes[i, j]
            if l >= len(reps):
                ax.axis("off"); continue
            P, var = _ph.pca2(reps[l])
            ax.scatter(P[:, 0], P[:, 1], c=Y, cmap="tab10", s=7, alpha=0.8, linewidths=0)
            ax.set_title(f"{LABEL[arm]} | layer {l+1}  ({100*var:.0f}% var)"
                         + (f"  [step {cs[0]}]" if j == 0 else ""), fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{arch}: class structure by depth, PCA of hidden activations", fontsize=10)
    fig.tight_layout()
    return fig


def fig_operator_diversity(runs, seed=None):
    """Operator diversity against training, with the shared initialisation marked.

    Pointwise rank (how complex P(x) is at one input) and diversity (how much P varies between
    inputs) are different claims, and an optimiser can move them in opposite directions. All arms
    of a given seed start from the SAME weights, so the value at step 0 is a common baseline.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, arch in zip(axes, ["crelu", "relu"]):
        init_done = False
        for arm in ARMS:
            sub = select(runs, arch=arch, arm=arm, seed=seed)
            if not sub:
                continue
            byh = {}
            for c, r, d in sub:
                byh.setdefault(hyper_of(c), []).append((c, r, d))
            def _score(g):
                z = [at_convergence(r) for _, r, _ in g]
                z = [x for x in z if x is not None]
                return np.mean(z) if z else -np.inf
            bh = max(byh, key=lambda k: _score(byh[k]))
            series = [posthoc_rows(d) for _, _, d in byh[bh]]
            series = [s for s in series if s]
            if not series:
                continue
            steps = [x["step"] for x in series[0]]
            v = np.mean([[x["op_div_c"] for x in s] for s in series], axis=0)
            ax.plot(steps, v, "o-", color=COLOR[arm], label=LABEL[arm])
            # Mark where this arm fits the training set. Everything to the right of the marker is
            # post-interpolation, which is where the operator arm's diversity collapses and where
            # the endpoint reading of E9 is taken.
            ci = [converged_at(r) for _, r, _ in byh[bh]]
            ci = [r[i]["step"] for (_, r, _), i in zip(byh[bh], ci) if i is not None]
            if ci:
                cs = float(np.mean(ci))
                ax.axvline(cs, color=COLOR[arm], ls="--", lw=1.0, alpha=0.55)
                ax.plot([cs], [np.interp(cs, steps, v)], "*", color=COLOR[arm], ms=13,
                        markeredgecolor="white", markeredgewidth=0.7, zorder=5)
            if not init_done:
                ax.axhline(v[0], color="0.5", ls=":", lw=1)
                ax.text(steps[-1], v[0], "  shared init", va="bottom", ha="right",
                        fontsize=8, color="0.4")
                init_done = True
        ax.set_title(arch); ax.set_xlabel("step")
        ax.set_ylabel("operator diversity across inputs\n(effective rank, mean removed)")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def fig_pointwise_vs_diversity(runs):
    """The two notions of simplicity against each other, at the convergence phase."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, arch in zip(axes, ["crelu", "relu"]):
        for arm in ARMS:
            xs, ys, init = [], [], None
            for c, r, d in select(runs, arch=arch, arm=arm):
                ph = posthoc_rows(d)
                if not ph:
                    continue
                # Both axes in one window. This previously paired the pointwise rank at the
                # validation peak with the diversity at the LAST snapshot, which for the operator
                # arm are on opposite sides of its post-interpolation collapse.
                pr = at_convergence(r, "pr")
                cs = conv_snapshot(r, d)
                if pr is None:
                    continue
                row = next((x for x in ph if cs and x.get("step") == cs[0]), ph[-1])
                xs.append(pr); ys.append(row["op_div_c"])
                init = ph[0]["op_div_c"]
            if xs:
                ax.scatter(xs, ys, color=COLOR[arm], label=LABEL[arm], s=34, alpha=0.85,
                           edgecolor="white", linewidth=0.6)
        if init:
            ax.axhline(init, color="0.5", ls=":", lw=1)
            ax.text(ax.get_xlim()[1], init, "shared init  ", va="bottom", ha="right",
                    fontsize=8, color="0.4")
        ax.set_xlabel("pointwise rank of $P(x)$")
        ax.set_ylabel("diversity of $P$ across inputs")
        ax.set_title(arch); ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def fig_eta_k_dynamics(runs, arch="crelu", seed=0, width=128, steps=12000):
    """Training dynamics across the eta x k grid, against tuned gd and Adam.

    Four panels, all from runs already computed:
      (a) training loss, varying k at fixed eta -- what FAITHFULNESS does to the dynamics
      (b) training loss, varying eta at fixed k -- what STEP SIZE does, for contrast
      (c) test accuracy for the same runs, with the validation-selected point marked
      (d) gate density, where the operator arm's early transient is visible
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    # width/steps must be pinned: the capacity sweep produced op runs at widths 16/32/64 which
    # would otherwise appear as extra, identically-coloured k=50 curves
    sub = [(c, r) for c, r, d in select(runs, arch=arch, arm="op", seed=seed)
           if isinstance(c["hyper"], list) and c["width"] == width and c["steps"] == steps]
    if not sub:
        return None
    ks = sorted({c["hyper"][1] for c, _ in sub})
    es = sorted({c["hyper"][0] for c, _ in sub})
    fixed_e = 1.0 if 1.0 in es else es[len(es) // 2]
    fixed_k = 50 if 50 in ks else ks[-1]
    ck = {k: cm.viridis(i / max(len(ks) - 1, 1)) for i, k in enumerate(ks)}
    ce = {e: cm.plasma(i / max(len(es) - 1, 1)) for i, e in enumerate(es)}

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for c, r in sub:
        e, k = c["hyper"]
        s = [x["step"] for x in r]
        if abs(e - fixed_e) < 1e-9:
            ax[0, 0].plot(s, [x["train_loss"] for x in r], color=ck[k], label=f"k={k}")
            ax[1, 0].plot(s, [x["test_acc"] for x in r], color=ck[k], label=f"k={k}")
            ax[1, 1].plot(s, [x["density"] for x in r], color=ck[k], label=f"k={k}")
        if k == fixed_k:
            ax[0, 1].plot(s, [x["train_loss"] for x in r], color=ce[e], label=f"$\\eta$={e:g}")
    # tuned baselines for reference
    for arm, style in (("gd", "--"), ("adam", ":")):
        b = [z for z in select(runs, arch=arch, arm=arm, seed=seed)
             if z[0]["width"] == width and z[0]["steps"] == steps]
        if not b:
            continue
        c, r, _ = max(b, key=lambda z: stable_test(z[1]))
        s = [x["step"] for x in r]
        for a, key in ((ax[0, 0], "train_loss"), (ax[0, 1], "train_loss"),
                       (ax[1, 0], "test_acc"), (ax[1, 1], "density")):
            a.plot(s, [x[key] for x in r], style, color="0.25", lw=1.6,
                   label=f"{arm} (tuned)")
    ax[0, 0].set_title(f"training loss, varying $k$ at $\\eta$={fixed_e:g}", fontsize=10)
    ax[0, 1].set_title(f"training loss, varying $\\eta$ at $k$={fixed_k}", fontsize=10)
    ax[1, 0].set_title("test accuracy (varying $k$)", fontsize=10)
    ax[1, 1].set_title("gate density (varying $k$)", fontsize=10)
    for a in ax[0]:
        a.set_yscale("log"); a.set_ylabel("training loss")
    ax[1, 0].set_ylabel("test accuracy"); ax[1, 1].set_ylabel("gate density")
    for a in ax.ravel():
        a.set_xlabel("step"); a.grid(alpha=0.25)
        a.legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle(f"{arch}, seed {seed}: the operator arm across the $\\eta\\times k$ grid",
                 fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- selection
def stable_test(recs, frac=0.25):
    """Test accuracy averaged over the final `frac` of probes -- no selection, no cherry-picking.

    `best_by_val` takes the maximum over ~48 probes, which is an upward-biased estimator whenever
    the run oscillates. Measured on ReLU under Adam, whose test accuracy swings by 0.318 and whose
    training loss never falls below 0.68, the two estimators differ by 0.126 and they disagree
    about which optimiser wins. Prefer this one, and report the swing alongside it so an unconverged
    run is visible rather than silently flattering.
    """
    q = recs[int(len(recs) * (1 - frac)):]
    return float(np.mean([x["test_acc"] for x in q]))


def swing(recs, frac=0.5):
    q = recs[int(len(recs) * (1 - frac)):]
    return float(max(x["test_acc"] for x in q) - min(x["test_acc"] for x in q))


def pick(runs, arch, arm, width=None, steps=None, by="stable"):
    """Best hyperparameter for an (arch, arm) cell under the chosen estimator."""
    sub = [(c, r, d) for c, r, d in select(runs, arch=arch, arm=arm)
           if (width is None or c["width"] == width)
           and (steps is None or c["steps"] == steps)]
    if not sub:
        return None
    byh = {}
    for c, r, d in sub:
        byh.setdefault(hyper_of(c), []).append((c, r, d))
    score = (lambda g: np.mean([stable_test(r) for _, r, _ in g])) if by == "stable" \
        else (lambda g: np.mean([best_by_val(r)["val_acc"] for _, r, _ in g]))
    bh = max(byh, key=lambda k: score(byh[k]))
    return bh, byh[bh]


def fig_selection_bias(runs):
    """Peak-selected against the two windowed estimators, with the swing that explains the gap.

    Three bars per arm: the peak of the validation curve, the convergence-phase mean (what Part II
    reports), and the last-quarter mean. The point of the figure is that the three disagree by an
    amount comparable to the between-arm differences, so the estimator has to be fixed once and
    used for both selection and scoring.
    """
    import matplotlib.pyplot as plt
    archs = [a for a in ("crelu", "relu", "leaky") if select(runs, arch=a)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    w = 0.25
    xs = np.arange(len(archs))
    for j, arm in enumerate(ARMS):
        pk, cv, st, sw = [], [], [], []
        for arch in archs:
            got = pick_converged(runs, arch, arm, width=128, steps=12000)
            if not got:
                for a in (pk, cv, st, sw):
                    a.append(np.nan)
                continue
            _, g = got
            pk.append(np.mean([best_by_val(r)["test_acc"] for _, r, _ in g]))
            c = [at_convergence(r) for _, r, _ in g]
            cv.append(np.mean([x for x in c if x is not None]) if any(
                x is not None for x in c) else np.nan)
            st.append(np.mean([stable_test(r) for _, r, _ in g]))
            sw.append(np.mean([swing(r) for _, r, _ in g]))
        axes[0].bar(xs + (j - 1) * w - w * 0.30, pk, w * 0.28, color=COLOR[arm], alpha=0.35,
                    label=f"{LABEL[arm]} (peak)")
        axes[0].bar(xs + (j - 1) * w, cv, w * 0.28, color=COLOR[arm],
                    label=f"{LABEL[arm]} (convergence)")
        axes[0].bar(xs + (j - 1) * w + w * 0.30, st, w * 0.28, color=COLOR[arm], alpha=0.65,
                    hatch="///", edgecolor="white", linewidth=0.0,
                    label=f"{LABEL[arm]} (last 25%)")
        axes[1].bar(xs + (j - 1) * w, sw, w * 0.8, color=COLOR[arm], label=LABEL[arm])
    axes[0].set_ylabel("test accuracy"); axes[0].set_title(
        "peak (pale) | convergence phase (solid) | last quarter (hatched)", fontsize=10)
    axes[1].set_ylabel("test-accuracy swing over the final half")
    axes[1].set_title("how much the run is still oscillating", fontsize=10)
    for a in axes:
        a.set_xticks(xs); a.set_xticklabels(archs); a.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=7, ncol=3, frameon=False)
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def fig_eta_k_heatmap(runs):
    """Test accuracy over the eta x k grid, as a heatmap rather than a printed table."""
    import matplotlib.pyplot as plt
    archs = [a for a in ("crelu", "relu") if select(runs, arch=a, arm="op")]
    fig, axes = plt.subplots(1, len(archs), figsize=(5.6 * len(archs), 4.3), squeeze=False)
    for ax, arch in zip(axes[0], archs):
        cells = {}
        for c, r, d in select(runs, arch=arch, arm="op"):
            if not isinstance(c["hyper"], list) or c["width"] != 128 or c["steps"] != 12000:
                continue
            # Convergence phase, matching the E7 table. A cell that never fits contributes
            # nothing rather than an endpoint number, so it shows as a blank square -- which is
            # the honest rendering of the ReLU k=1 column.
            v = at_convergence(r)
            if v is not None:
                cells.setdefault(tuple(c["hyper"]), []).append(v)
        cells = {k: v for k, v in cells.items() if v}
        if not cells:
            ax.axis("off"); continue
        es = sorted({e for e, _ in cells}); ks = sorted({k for _, k in cells})
        Z = np.full((len(ks), len(es)), np.nan)
        for (e, k), v in cells.items():
            Z[ks.index(k), es.index(e)] = np.mean(v)
        im = ax.imshow(Z, cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks(range(len(es))); ax.set_xticklabels([f"{e:g}" for e in es])
        ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks)
        ax.set_xlabel(r"$\eta$ (step size)"); ax.set_ylabel("$k$ (faithfulness)")
        ax.set_title(f"{arch}: test accuracy at convergence", fontsize=10)
        for i in range(len(ks)):
            for j in range(len(es)):
                if np.isfinite(Z[i, j]):
                    ax.text(j, i, f"{Z[i,j]:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if Z[i, j] < np.nanmax(Z) * 0.93 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def fig_capacity(runs):
    """Test accuracy against width: does giving the operator arm LESS capacity close the gap?

    The question is about capacity, not about matched fit, so this is the unrestricted comparison:
    each arm's best cell over its whole grid, scored at the endpoint. At width 16 that includes
    cells which never fit (gradient descent tops out at 0.880 training accuracy there and Adam at
    0.906), and failing to fit at a given capacity is part of what the figure is reporting -- open
    markers say so. The convergence-phase value is overlaid where the arm has one.
    """
    import matplotlib.pyplot as plt
    archs = [a for a in ("crelu", "relu") if select(runs, arch=a)]
    fig, axes = plt.subplots(1, len(archs), figsize=(5.6 * len(archs), 4.2), squeeze=False)
    for ax, arch in zip(axes[0], archs):
        widths = sorted({c["width"] for c, _, _ in select(runs, arch=arch)})
        for arm in ARMS:
            xs, ys, es, fits, cx, cy = [], [], [], [], [], []
            for w in widths:
                got = pick(runs, arch, arm, width=w, steps=12000)
                if not got:
                    continue
                _, g = got
                v = [stable_test(r) for _, r, _ in g]
                xs.append(w); ys.append(np.mean(v)); es.append(np.std(v))
                fits.append(np.mean([max(x["train_acc"] for x in r) for _, r, _ in g]) >= 0.999)
                gc = pick_converged(runs, arch, arm, width=w, steps=12000)
                if gc:
                    cv = [at_convergence(r) for _, r, _ in gc[1]]
                    cv = [x for x in cv if x is not None]
                    if cv:
                        cx.append(w); cy.append(np.mean(cv))
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", color=COLOR[arm], label=LABEL[arm],
                            capsize=3)
                nf = ~np.array(fits)
                if nf.any():
                    ax.plot(np.array(xs)[nf], np.array(ys)[nf], "o", mfc="white",
                            mec=COLOR[arm], mew=1.8, ms=9, zorder=4)
                if cx:
                    ax.plot(cx, cy, "x", color=COLOR[arm], ms=7, alpha=0.75, zorder=3)
        ax.set_xscale("log", base=2); ax.set_xlabel("width"); ax.set_ylabel("test accuracy")
        ax.set_title(arch, fontsize=10); ax.grid(alpha=0.25)
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def fig_fit_vs_generalise(runs, width=128, steps=12000):
    """Final training loss against test accuracy, every run. Two regimes, not one trend.

    Below interpolation (gradient descent at small lr, Adam when it diverges) fitting better helps.
    Above it -- and every operator run and every well-tuned Adam run interpolates, at 100% training
    accuracy -- further loss reduction buys nothing, and among such runs the LESS converged ones
    generalise better. The operator arm drives the loss about three orders lower than Adam and gains
    nothing for it.
    """
    import matplotlib.pyplot as plt
    archs = [a for a in ("crelu", "relu", "leaky") if select(runs, arch=a)]
    fig, axes = plt.subplots(1, len(archs), figsize=(5.0 * len(archs), 4.2), squeeze=False)
    for ax, arch in zip(axes[0], archs):
        for arm in ARMS:
            xs, ys, interp = [], [], []
            for c, r, d in select(runs, arch=arch, arm=arm):
                if c["width"] != width or c["steps"] != steps:
                    continue
                i = converged_at(r)
                # Read the loss in the same window as the accuracy. Taking the FINAL loss beside a
                # convergence-phase accuracy would put the operator arm three orders further left
                # than where it was when that accuracy was measured.
                if i is not None:
                    tl = float(np.median([x["train_loss"] for x in r[i:i + 8]]))
                    ys.append(at_convergence(r))
                else:
                    tl = r[-1]["train_loss"]
                    ys.append(stable_test(r))
                if not np.isfinite(tl) or tl <= 0:
                    tl = 1e-12                       # log axis: floor exact zeros
                xs.append(tl)
                interp.append(i is not None)
            if not xs:
                continue
            xs, ys, interp = np.array(xs), np.array(ys), np.array(interp)
            ax.scatter(xs[interp], ys[interp], color=COLOR[arm], s=38, alpha=0.9,
                       edgecolor="white", linewidth=0.6, label=f"{LABEL[arm]} (interpolates)")
            if (~interp).any():
                ax.scatter(xs[~interp], ys[~interp], facecolor="none", edgecolor=COLOR[arm],
                           s=46, linewidth=1.4, label=f"{LABEL[arm]} (does not fit)")
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("training loss at convergence  (better fit $\\rightarrow$)")
        ax.set_ylabel("test accuracy at convergence")
        ax.set_title(arch, fontsize=10); ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    return fig


def fig_switch(runs_switch, arch="relu"):
    """Test accuracy against the handover step, both directions.

    switch_at = 0 is the pure second arm, switch_at = total is the pure first arm, so the two
    endpoints of each curve reproduce the unmixed optimisers and act as an internal check.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    style = {"adam->op": ("#e15759", "o-"), "op->adam": ("#59a14f", "s-")}
    for direction in ("adam->op", "op->adam"):
        pts = {}
        for c, r, d in runs_switch:
            if c["arch"] != arch or c["arm"] != "switch":
                continue
            A, hA, B, hB, at = c["hyper"]
            if f"{A}->{B}" != direction:
                continue
            pts.setdefault(at, []).append(stable_test(r))
        if not pts:
            continue
        xs = sorted(pts)
        m = [np.mean(pts[x]) for x in xs]
        s = [np.std(pts[x]) for x in xs]
        col, mk = style[direction]
        ax.errorbar(xs, m, yerr=s, fmt=mk, color=col, capsize=3,
                    label=f"{direction}  (n={len(pts[xs[0]])} seeds)")
    ax.set_xlabel("handover step"); ax.set_ylabel("test accuracy at convergence")
    ax.set_title(f"{arch}: when is the advantage created?", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- operator momentum
def momentum_cells(runs, width=128, steps=12000, k=50):
    """Index the momentum sweep by (arch, eta, beta), with beta=0 reproducing the plain reference.

    The `opmom` arm carries hyper = (eta, k, beta) and applies the smoothed update
    u_t = beta u_{t-1} + (1 - beta) d_t directly, with no learning rate, so beta = 0 is the
    unmodified reference step and the beta = 0 column is a bit-for-bit control.
    """
    cells = {}
    for c, r, _ in runs:
        if c["width"] != width or c["steps"] != steps:
            continue
        h = c["hyper"]
        if c["arm"] == "opmom" and h[1] == k:
            cells.setdefault((c["arch"], h[0], h[2]), []).append(r)
        elif c["arm"] == "op" and isinstance(h, (list, tuple)) and h[1] == k:
            cells.setdefault((c["arch"], h[0], "plain"), []).append(r)
    return cells


def _tail_mean(recs, key, frac=0.5):
    """Mean of `key` over the last `frac` of the probes -- the same estimator as stable_test."""
    v = [x[key] for x in recs if key in x]
    return float(np.mean(v[int(len(v) * frac):])) if v else np.nan


def fig_momentum(runs, etas=(0.5, 1.0), betas=(0.0, 0.5, 0.9, 0.99), archs=("relu", "crelu")):
    """Operator momentum: accuracy against beta, and the faithfulness it trades away.

    Left column: test accuracy, paired to the beta = 0 control on the same seed. Right column:
    operator-space deflection cos_op over the same cells. The point of putting them side by side
    is that beta moves the two in opposite directions from what the faithfulness story predicts.
    """
    import matplotlib.pyplot as plt
    cells = momentum_cells(runs)
    fig, axes = plt.subplots(len(archs), 2, figsize=(9.6, 3.5 * len(archs)), squeeze=False)
    mk = {0.5: ("o-", "#4c78a8"), 1.0: ("s-", "#e15759")}
    xs = np.arange(len(betas))
    for row, arch in enumerate(archs):
        for eta in etas:
            acc_m, acc_s, cos_m = [], [], []
            for b in betas:
                g = cells.get((arch, eta, b), [])
                # Convergence phase, matching the E15 table. Scored at the endpoint instead, the
                # beta = 0.5 cells read slightly positive and the experiment looks equivocal; at
                # matched phase every cell is negative.
                v = [at_convergence(r) for r in g]
                v = [x for x in v if x is not None]
                acc_m.append(np.mean(v) if v else np.nan)
                acc_s.append(np.std(v) if v else np.nan)
                c = [at_convergence_sparse(r, "cos_op") for r in g]
                c = [x for x in c if x is not None]
                cos_m.append(np.mean(c) if c else np.nan)
            fmt, col = mk[eta]
            axes[row][0].errorbar(xs, acc_m, yerr=acc_s, fmt=fmt, color=col, capsize=3,
                                  label=f"$\\eta$ = {eta:g}")
            axes[row][1].plot(xs, cos_m, fmt, color=col, label=f"$\\eta$ = {eta:g}")
        for col_i, (ylab, title) in enumerate(
                [("test accuracy at convergence", "generalisation"),
                 ("$\\cos_{op}$ at convergence", "operator-space faithfulness")]):
            ax = axes[row][col_i]
            ax.set_xticks(xs); ax.set_xticklabels([f"{b:g}" for b in betas])
            ax.set_xlabel("momentum $\\beta$"); ax.set_ylabel(ylab)
            ax.set_title(f"{arch}: {title}", fontsize=10)
            ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
        axes[row][1].set_ylim(0, 1)
    fig.tight_layout()
    return fig


def momentum_table(runs, etas=(0.5, 1.0), betas=(0.5, 0.9, 0.99), archs=("relu", "crelu")):
    """Per-seed paired differences against beta = 0, so seed variance cancels."""
    cells = momentum_cells(runs)
    rows = []
    for arch in archs:
        for eta in etas:
            base = {}
            for c, r, _ in runs:
                h = c["hyper"]
                if (c["arch"] == arch and c["arm"] == "opmom" and c["width"] == 128
                        and c["steps"] == 12000 and h[1] == 50 and h[0] == eta):
                    base.setdefault(h[2], {})[c["seed"]] = r
            if 0.0 not in base:
                continue
            for b in betas:
                if b not in base:
                    continue
                shared = sorted(set(base[b]) & set(base[0.0]))
                d = [stable_test(base[b][s]) - stable_test(base[0.0][s]) for s in shared]
                rows.append(dict(
                    arch=arch, eta=eta, beta=b, n=len(d),
                    delta_acc=float(np.mean(d)), delta_sd=float(np.std(d)),
                    cos_op=float(np.mean([_tail_mean(base[b][s], "cos_op") for s in shared])),
                    cos_op_beta0=float(np.mean([_tail_mean(base[0.0][s], "cos_op") for s in shared])),
                    final_train_loss=float(np.median([base[b][s][-1]["train_loss"] for s in shared])),
                ))
    return rows


# ---------------------------------------------------------------- adversarial robustness
def robustness_sweep(runs, archs=("relu", "crelu", "leaky"), width=128, steps=12000,
                     eps=(0.02, 0.05, 0.1, 0.2, 0.3), pgd_steps=50, restarts=3,
                     dev=None, max_seeds=3, at="convergence"):
    """Clean and CW-PGD accuracy of every (arch, arm, hyper) cell.

    Uses the CW margin attack, not cross-entropy: see `toy2d.pgd_cw` for why the cross-entropy
    version is unusable here. The logit margin and the operator (input-Jacobian) norm are carried
    alongside, because they are what the cross-entropy attack was actually measuring.

    `at` selects the window. "convergence" reads the snapshot at each run's own interpolation
    point, which is what the rest of Part II reports; "end" reads the final checkpoint. The two
    differ most in the operator-norm column, since the reference keeps stepping at constant size
    after the loss is fit and its ||P||_F grows in that regime.
    """
    import torch
    import exp, gpu
    import toy2d
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    _data = {}

    def dataset(seed):
        if seed not in _data:
            _data[seed] = exp.mnist1d(4000, 1000, 1000, seed, dev)
        return _data[seed]

    for arch in archs:
        cells = {}
        for c, r, d in runs:
            if c["arch"] != arch or c["width"] != width or c["steps"] != steps:
                continue
            h = c["hyper"]
            key = (c["arm"], tuple(h) if isinstance(h, (list, tuple)) else h)
            cells.setdefault(key, []).append((c, r, d))
        for (arm, h), group in sorted(cells.items(), key=lambda kv: str(kv[0])):
            acc = {k: [] for k in ("train", "clean", "pr", "margin", "fro")}
            adv = {e: [] for e in eps}
            for c, r, d in group[:max_seeds]:
                if at == "convergence":
                    cs = conv_snapshot(r, d)
                    if cs is None or converged_at(r) is None:
                        continue
                    path, wstep = cs[1], cs[0]
                    train_acc = at_convergence(r, "train_acc")
                else:
                    if not (d / "ckpt.pt").exists():
                        continue
                    path, wstep, train_acc = d / "ckpt.pt", r[-1]["step"], r[-1]["train_acc"]
                Ws = [w.to(dev) for w in
                      torch.load(path, map_location=dev, weights_only=False)["Ws"]]
                acc.setdefault("step", []).append(wstep)
                D = dataset(c["seed"])
                X, Y = D["Xte"], D["Yte"]
                with torch.no_grad():
                    o, _ = gpu.forward(Ws, X, arch)
                    acc["clean"].append(float((o.argmax(1) == Y).float().mean()))
                acc["train"].append(train_acc)
                acc["margin"].append(toy2d.logit_margin(Ws, X, Y, arch))
                acc["pr"].append(exp.operator_stats(Ws, D["Xtr"][:256], arch)[0])
                acc["fro"].append(_operator_norm(Ws, D["Xtr"][:256], arch))
                for e in eps:
                    adv[e].append(toy2d.pgd_cw(Ws, X, Y, arch, eps=e,
                                               steps=pgd_steps, restarts=restarts))
                del Ws
                if dev == "cuda":
                    torch.cuda.empty_cache()
            if not acc["clean"]:
                continue
            row = dict(arch=arch, arm=arm, hyper=h, n=len(acc["clean"]),
                       **{k: float(np.mean(v)) for k, v in acc.items() if k != "train"},
                       train_acc=float(np.mean(acc["train"])))
            for e in eps:
                row[f"pgd_{e}"] = float(np.mean(adv[e]))
                row[f"ratio_{e}"] = float(np.mean(adv[e]) / np.mean(acc["clean"]))
            rows.append(row)
    return rows


def _operator_norm(Ws, X, arch):
    """Mean Frobenius norm of P(x) -- the input Jacobian, since f(x) = P(x) x here."""
    import torch
    import gpu
    W = [w.double() for w in Ws]
    Xd = X.double()
    _, gs = gpu.forward(W, Xd, arch)
    A, B = gpu.contexts(W, gs, Xd, arch)
    sv = torch.linalg.svdvals(A[0] @ W[0] @ B[0])
    live = sv[:, 0] > 1e-9 * sv[:, 0].max().clamp_min(1e-300)
    return float((sv[live] ** 2).sum(1).sqrt().mean()) if bool(live.any()) else float("nan")


def fig_robustness(rows, eps=(0.02, 0.05, 0.1, 0.2, 0.3), archs=("relu", "crelu", "leaky")):
    """The corrected robustness picture: an epsilon curve under the CW attack, per arm.

    Left: adversarial accuracy as a fraction of clean accuracy, best hyperparameter per arm. The
    arms lie on top of each other -- the separation reported earlier was an artefact of attacking
    with cross-entropy. Right: what the cross-entropy attack was actually tracking, the logit
    margin, which differs between arms by two orders of magnitude.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    style = {"relu": "-", "crelu": "--", "leaky": ":"}
    for arch in archs:
        for arm in ARMS:
            sub = [r for r in rows if r["arch"] == arch and r["arm"] == arm
                   and r["train_acc"] > 0.999]
            if not sub:
                continue
            best = max(sub, key=lambda r: r["clean"])
            axes[0].plot(eps, [best[f"ratio_{e}"] for e in eps], style[arch],
                         color=COLOR[arm], marker="o", ms=4,
                         label=f"{arch}: {LABEL[arm]}" if arch == "relu" else None)
    axes[0].set_xlabel("$\\epsilon$  ($L_\\infty$)")
    axes[0].set_ylabel("CW-PGD accuracy / clean accuracy")
    axes[0].set_title("no separation survives a scale-covariant attack", fontsize=10)
    axes[0].grid(alpha=0.25); axes[0].legend(frameon=False, fontsize=8)

    # What actually predicts robustness across cells. Operator rank does not (rho = +0.04 at the
    # convergence phase); the logit margin does (rho = +0.87), which is expected since the CW
    # objective is scale-covariant.
    allsub = [r for r in rows if r["arch"] in archs and r["train_acc"] > 0.999
              and r["margin"] > 0]
    for arm in ARMS:
        sub = [r for r in allsub if r["arm"] == arm]
        if sub:
            axes[1].scatter([r["margin"] for r in sub], [r["ratio_0.1"] for r in sub], s=34,
                            color=COLOR[arm], label=LABEL[arm], alpha=0.85,
                            edgecolor="white", lw=0.6)
    if allsub:
        x = np.log([r["margin"] for r in allsub])
        y = np.array([r["ratio_0.1"] for r in allsub])
        b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        axes[1].plot(np.exp(xs), np.polyval(b, xs), "-", color="0.35", lw=1.2, zorder=1)
        axes[1].text(0.04, 0.93, rf"$\rho(\log\,\mathrm{{margin}}) = "
                                 rf"{np.corrcoef(x, y)[0, 1]:+.2f}$",
                     transform=axes[1].transAxes, fontsize=9)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("logit margin")
    axes[1].set_ylabel("CW-PGD$_{0.1}$ accuracy / clean accuracy")
    axes[1].set_title("margin, not operator rank, predicts robustness", fontsize=10)
    axes[1].grid(alpha=0.25); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- convergence-phase estimators
def converged_at(recs, thresh=0.999):
    """First probe index at which the run has fitted the training set.

    Defined on the TRAINING signal only, never on test accuracy, so that using it to select a
    hyper-parameter cannot bias the quantity being selected. Returns None if the run never fits.
    """
    for i, x in enumerate(recs):
        if x.get("train_acc", 0.0) >= thresh:
            return i
    return None


def at_convergence(recs, key="test_acc", probes=8, thresh=0.999):
    """Mean of `key` over the `probes` probes starting where the run first fits the training set.

    `stable_test` averages the FINAL quarter of a run, which asks where an optimiser ends up. That
    is the wrong question when a rule keeps moving after it has nothing left to fit: the operator
    arm's test accuracy peaks shortly after interpolation and then declines, and its operator rank
    collapses entirely in that regime, so a final-quarter statistic reports the decay rather than
    the solution. This estimator reads each arm shortly after its own convergence instead, which is
    a matched *phase* rather than a matched step count -- the arms reach it at very different times
    (gradient descent by step 1000, Adam not until 6000-9000).
    """
    i = converged_at(recs, thresh)
    if i is None:
        return None
    vals = [x[key] for x in recs[i:i + probes] if key in x and x[key] == x[key]]
    return float(np.mean(vals)) if vals else None


def at_convergence_sparse(recs, key, thresh=0.999):
    """Value of a SPARSE diagnostic at the convergence phase: the first probe carrying `key` at or
    after the run fits the training set.

    The alignment diagnostics (cos_op, cos_w, alpha, operator rank) run every 2000 steps, while the
    cheap probes run every 250, so the 8-probe window of `at_convergence` spans only one alignment
    probe -- averaging over it would silently report one measurement as if it were eight, or drop
    the seed entirely when the window falls between two. Reading the first alignment probe at or
    after the anchor is the same estimator stated honestly, and keeps the reading inside the
    pre-decay window (the operator arm's rank collapse lags interpolation by ~2500 steps).
    """
    i = converged_at(recs, thresh)
    if i is None:
        return None
    for x in recs[i:]:
        v = x.get(key)
        if v is not None and v == v:
            return float(v)
    return None


def conv_snapshot(recs, run_dir, thresh=0.999):
    """(step, path) of the earliest weight snapshot at or after this run's convergence anchor.

    Measures that need the weights rather than a logged scalar -- operator diversity, the gate
    split, the layer PCA -- can only be read where a snapshot exists, so the convergence phase is
    quantised to the snapshot grid. Falls back to the last snapshot if the run never converges,
    and returns None if there are no snapshots at all.
    """
    import sys as _s
    _s.path.insert(0, str(HERE.parent / "studies"))
    import posthoc as _ph
    snaps = _ph.snapshots(run_dir)
    if not snaps:
        return None
    i = converged_at(recs, thresh)
    if i is None:
        return snaps[-1]
    cstep = recs[i]["step"]
    later = [s for s in snaps if s[0] >= cstep]
    return later[0] if later else snaps[-1]


def pick_converged(runs, arch, arm, width=None, steps=None, probes=8):
    """Best hyper-parameter by the convergence-phase estimator, selected and scored alike."""
    sub = [(c, r, d) for c, r, d in select(runs, arch=arch, arm=arm)
           if (width is None or c["width"] == width) and (steps is None or c["steps"] == steps)]
    if not sub:
        return None
    byh = {}
    for c, r, d in sub:
        byh.setdefault(hyper_of(c), []).append((c, r, d))
    def score(g):
        v = [at_convergence(r, probes=probes) for _, r, _ in g]
        v = [x for x in v if x is not None]
        return np.mean(v) if v else -np.inf
    bh = max(byh, key=lambda k: score(byh[k]))
    return bh, byh[bh]


def depth_table(runs, arch, depths=(2, 4, 8, 16, 32), probes=8):
    """Per-depth convergence-phase summary for one architecture.

    Selection and scoring both use `at_convergence`, so a depth at which one arm interpolates late
    is not penalised for having a short post-interpolation tail. Returns {arm: {depth: dict}} with
    the seed count, so incomplete depths can be reported as incomplete rather than averaged over
    whichever seeds happen to have finished.
    """
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for L in depths:
            sub = [(c, r, d) for c, r, d in select(runs, arch=arch, arm=arm)
                   if c["depth"] == L and r and r[-1]["step"] >= c["steps"]]
            if not sub:
                continue
            byh = {}
            for c, r, d in sub:
                byh.setdefault(hyper_of(c), []).append((c, r, d))
            def score(g):
                v = [at_convergence(r, probes=probes) for _, r, _ in g]
                v = [x for x in v if x is not None]
                return np.mean(v) if v else -np.inf
            bh = max(byh, key=lambda k: score(byh[k]))
            g = byh[bh]
            rows = {k: [] for k in ("test", "train_acc", "train_loss", "cos_op", "step")}
            for c, r, _ in g:
                i = converged_at(r)
                if i is None:
                    continue
                w = r[i:i + probes]
                rows["test"].append(at_convergence(r, probes=probes))
                rows["train_acc"].append(at_convergence(r, "train_acc", probes=probes))
                rows["train_loss"].append(np.median([x["train_loss"] for x in w]))
                rows["step"].append(r[i]["step"])
                c_ = at_convergence_sparse(r, "cos_op")
                if c_ is not None:
                    rows["cos_op"].append(c_)
            if not rows["test"]:
                continue
            out[arm][L] = {"hyper": bh, "n": len(rows["test"]), "n_cells": len(g),
                           **{k: float(np.mean(v)) for k, v in rows.items() if v},
                           "test_sd": float(np.std(rows["test"]))}
    return out


def fig_depth(runs, archs=("relu", "crelu"), depths=(2, 4, 8, 16, 32)):
    """Test accuracy, training accuracy, training loss and cos_op against depth, per architecture.

    Every panel reads the convergence phase. A point is drawn only where all three seeds of that
    cell completed, so a partially finished depth leaves a gap rather than a point that would move
    when the remaining seeds land.
    """
    import matplotlib.pyplot as plt
    keys = [("test", "test accuracy", False), ("train_acc", "training accuracy", False),
            ("train_loss", "training loss", True), ("cos_op", r"$\cos_{op}$", False)]
    fig, axes = plt.subplots(len(archs), len(keys), figsize=(4.0 * len(keys), 3.3 * len(archs)))
    axes = np.atleast_2d(axes)
    for i, arch in enumerate(archs):
        T = depth_table(runs, arch, depths)
        for j, (key, name, logy) in enumerate(keys):
            ax = axes[i, j]
            for arm in ARMS:
                xs = [L for L in depths if L in T[arm] and T[arm][L]["n"] >= 3 and key in T[arm][L]]
                ys = [T[arm][L][key] for L in xs]
                if not xs:
                    continue
                ax.plot(xs, ys, "o-", color=COLOR[arm], label=LABEL[arm], ms=4, lw=1.4)
                if key == "test":
                    sd = [T[arm][L]["test_sd"] for L in xs]
                    ax.fill_between(xs, np.array(ys) - sd, np.array(ys) + sd,
                                    color=COLOR[arm], alpha=0.13, lw=0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(depths)); ax.set_xticklabels([str(d) for d in depths])
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("depth $L$")
            ax.set_title(f"{arch}: {name}", fontsize=10)
            ax.grid(alpha=0.25, lw=0.5)
            if i == 0 and j == 0:
                ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    return fig


def fig_training_dynamics(runs, arch="relu", width=128, steps=12000, probes=8):
    """Every tracked metric against training step, all three arms at their best configuration.

    One panel per metric, mean over seeds with a band at +/- one standard deviation, so the
    seed-to-seed spread is visible rather than hidden behind a mean. A vertical dashed line marks
    where each arm first reaches training accuracy 0.999: the panels read differently on either
    side of it, since everything to the right is post-interpolation behaviour and only the operator
    arm is still moving there.

    The alignment diagnostics run every 2000 steps rather than every 250, so they are drawn with
    markers on a sparse grid; the rest are dense.
    """
    import matplotlib.pyplot as plt
    dense = [("train_loss", "training loss", True),
             ("test_loss", "test loss", True),
             ("train_acc", "training accuracy", False),
             ("test_acc", "test accuracy", False),
             ("pr", "operator rank (participation ratio)", False),
             ("density", "gate density", False),
             ("hamming", "gate pattern diversity\n(Hamming between inputs)", False),
             ("churn", "gate churn (flips since last probe)", False)]
    sparse = [("cos_op", r"$\cos_{op}$: operator change vs the reachable ideal", False)]
    panels = dense + sparse
    sparse_keys = {k for k, _, _ in sparse}
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.5 * nrow), squeeze=False)
    flat = [a for row in axes for a in row]

    picked = {}
    for arm in ARMS:
        got = pick_converged(runs, arch, arm, width=width, steps=steps, probes=probes)
        if got:
            picked[arm] = got

    for ax, (key, label, logy) in zip(flat, panels):
        for arm, (hyper, group) in picked.items():
            series = []
            for _, r, _ in group:
                xs = [x["step"] for x in r if key in x and x[key] == x[key]]
                ys = [x[key] for x in r if key in x and x[key] == x[key]]
                if xs:
                    series.append((xs, ys))
            if not series:
                continue
            n = min(len(s[0]) for s in series)
            xs = series[0][0][:n]
            M = np.array([s[1][:n] for s in series])
            m, sd = M.mean(0), M.std(0)
            ax.plot(xs, m, "o-" if key in sparse_keys else "-", color=COLOR[arm], lw=1.5, ms=4,
                    label=f"{LABEL[arm]} ({hyper})")
            ax.fill_between(xs, m - sd, m + sd, color=COLOR[arm], alpha=0.18, lw=0)
        for arm, (_, group) in picked.items():
            ci = [converged_at(r) for _, r, _ in group]
            cs = [r[i]["step"] for (_, r, _), i in zip(group, ci) if i is not None]
            if cs:
                ax.axvline(float(np.mean(cs)), color=COLOR[arm], ls="--", lw=0.9, alpha=0.5)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel(label, fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in flat[len(panels):]:
        ax.axis("off")
    flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{arch}, width {width}, depth 8: dynamics at each arm's best configuration "
                 f"(band = 1 sd over seeds; dashed line = that arm reaches training accuracy "
                 f"$0.999$)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


COMPARATORS = ["heavyball", "muon", "kfac", "soap", "shampoo"]
COMPARATOR_LABEL = {"heavyball": "heavy ball", "muon": "Muon", "kfac": "K-FAC",
                    "soap": "SOAP", "shampoo": "Shampoo"}


def comparator_table(main="ref_mnist1d", comp="baselines_mnist1d", archs=("relu", "crelu"),
                     width=128, steps=12000, depth=8, probes=8):
    """E18: the three main arms and the five comparators in one table, same window, same estimator.

    The comparators live in their own experiment directory but were run at an identical
    configuration, so the two are merged here rather than re-run. Everything is read at the
    convergence phase and selected by it, exactly as in E5, so a comparator's row is comparable to
    gradient descent's without further qualification.

    `fits` is reported per cell because it decides how the row should be read: an optimiser whose
    best-scoring cell never reaches training accuracy 0.999 is being scored at matched budget, not
    matched fit, and E17 shows that is the difference between two quite different claims.
    """
    rows = []
    pool = {}
    for exp in (main, comp):
        try:
            for c, r, d in load(exp):
                pool.setdefault(c["arm"], []).append((c, r, d))
        except Exception:
            continue
    for arch in archs:
        for arm in ARMS + COMPARATORS:
            sub = [(c, r, d) for c, r, d in pool.get(arm, [])
                   if c["arch"] == arch and c.get("width") == width
                   and c.get("steps") == steps and c.get("depth") == depth
                   and r and r[-1]["step"] >= c["steps"]]
            if not sub:
                continue
            byh = {}
            for c, r, d in sub:
                byh.setdefault(hyper_of(c), []).append((c, r, d))

            def conv_score(g):
                v = [at_convergence(r, probes=probes) for _, r, _ in g]
                v = [x for x in v if x is not None]
                return np.mean(v) if v else -np.inf

            def end_score(g):
                return np.mean([stable_test(r) for _, r, _ in g])

            # ONE cell per row. Selecting `test` by the convergence estimator and `test_end` by the
            # endpoint one would put two different learning rates in the same row and invite the
            # reader to read the pair as a window effect, when it is a selection effect. The row is
            # the convergence-selected cell where one exists, and the endpoint-selected cell
            # otherwise (an arm no cell of which ever fits); `fits` says which.
            bh_c = max(byh, key=lambda k: conv_score(byh[k]))
            bh_e = max(byh, key=lambda k: end_score(byh[k]))
            fits = conv_score(byh[bh_c]) > -np.inf
            bh = bh_c if fits else bh_e
            g = byh[bh]
            alt = bh_e if (fits and bh_e != bh_c) else None
            # An arm that never fits has no convergence anchor, but its deflection is still the
            # quantity E18 is about -- a comparator can be diagnostically interesting without
            # being a good optimiser. Fall back to the second half of its alignment probes and let
            # the `fits` column say which reading a row carries.
            def _cos(r):
                v = at_convergence_sparse(r, "cos_op")
                if v is not None:
                    return v
                al = [x["cos_op"] for x in r if "cos_op" in x and x["cos_op"] == x["cos_op"]]
                return float(np.mean(al[len(al) // 2:])) if al else None

            def _pr(r):
                v = at_convergence(r, "pr", probes=probes)
                if v is not None:
                    return v
                al = [x["pr"] for x in r if "pr" in x and x["pr"] == x["pr"]]
                return float(np.mean(al[-max(1, len(al) // 4):])) if al else None

            cos_op = [x for x in (_cos(r) for _, r, _ in g) if x is not None]
            pr = [x for x in (_pr(r) for _, r, _ in g) if x is not None]
            rows.append(dict(
                arch=arch, arm=arm, hyper=bh, n=len(g), fits=fits, alt_hyper=alt,
                test=conv_score(g) if fits else float("nan"),
                test_end=end_score(g),
                test_end_best=end_score(byh[bh_e]),
                max_train=float(np.mean([max(x["train_acc"] for x in r) for _, r, _ in g])),
                cos_op=float(np.mean(cos_op)) if cos_op else float("nan"),
                pr=float(np.mean(pr)) if pr else float("nan"),
            ))
    return rows


METRICS = [
    ("cos_op", r"$\cos_{op}$", "operator-space faithfulness of the step"),
    ("cos_w", r"$\cos_w$", "weight-space agreement with the reference step"),
    ("alpha", r"$\alpha$", "fraction of the request that is reachable"),
    ("pr", "operator rank", "participation ratio of $P(x)$, averaged over inputs"),
    ("density", "gate density", "fraction of gates on"),
    ("hamming", "gate diversity", "Hamming distance between inputs' gate patterns"),
    ("churn", "gate churn", "gate flips since the previous probe"),
    ("dead_units", "dead units", "fraction of units never on"),
    ("train_loss", "training loss", "at the convergence phase, log scale"),
]


def metric_cells(main="ref_mnist1d", comp="baselines_mnist1d", archs=("relu", "crelu"),
                 width=128, steps=12000, depth=8, probes=8, fitting_only=True):
    """One row per (arch, arm, hyper) cell: test accuracy and every diagnostic, same window.

    The unit of analysis is a CELL, not an arm. Correlating a metric with accuracy over arms-at-
    their-best gives eight points and no way to tell a property of the optimiser from a property of
    the setting; over cells there are enough points to ask whether a metric tracks accuracy WITHIN
    an optimiser as well as across them, which is the distinction that matters for reading any of
    these numbers causally.
    """
    pool = {}
    for exp in (main, comp):
        try:
            for c, r, d in load(exp):
                pool.setdefault(c["arm"], []).append((c, r, d))
        except Exception:
            continue
    rows = []
    for arch in archs:
        for arm in ARMS + COMPARATORS:
            sub = [(c, r, d) for c, r, d in pool.get(arm, [])
                   if c["arch"] == arch and c.get("width") == width
                   and c.get("steps") == steps and c.get("depth") == depth
                   and r and r[-1]["step"] >= c["steps"]]
            byh = {}
            for c, r, d in sub:
                byh.setdefault(hyper_of(c), []).append(r)
            for h, g in byh.items():
                test = [at_convergence(r, probes=probes) for r in g]
                test = [x for x in test if x is not None]
                fits = bool(test)
                if fitting_only and not fits:
                    continue
                row = dict(arch=arch, arm=arm, hyper=h, n=len(g), fits=fits,
                           test=float(np.mean(test)) if test else float("nan"),
                           max_train=float(np.mean([max(x["train_acc"] for x in r) for r in g])))
                for key, _, _ in METRICS:
                    if key in ("cos_op", "cos_w", "alpha"):
                        v = [at_convergence_sparse(r, key) for r in g]
                    else:
                        v = [at_convergence(r, key, probes=probes) for r in g]
                    v = [x for x in v if x is not None and x == x]
                    row[key] = float(np.mean(v)) if v else float("nan")
                rows.append(row)
    return rows


def metric_correlations(rows, target="test"):
    """How each diagnostic relates to accuracy, ACROSS optimisers and WITHIN them.

    The two differ and the difference is the point. A metric can correlate strongly across all
    cells purely because it identifies which optimiser produced the cell -- Adam has a low operator
    rank and a high accuracy, so rank and accuracy correlate across arms without rank explaining
    anything. Residualising both on a full set of per-(arch, arm) dummies removes exactly that, and
    what survives is the relationship a practitioner could act on: within a given optimiser, does
    moving this metric move accuracy?
    """
    groups = sorted({(r["arch"], r["arm"]) for r in rows})
    idx = {g: i for i, g in enumerate(groups)}
    D = np.zeros((len(rows), len(groups)))
    for i, r in enumerate(rows):
        D[i, idx[(r["arch"], r["arm"])]] = 1.0
    y = np.array([r[target] for r in rows])
    out = []
    for key, label, _ in METRICS:
        x = np.array([r.get(key, np.nan) for r in rows])
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 6:
            out.append(dict(key=key, label=label, n=int(m.sum()),
                            across=float("nan"), within=float("nan")))
            continue
        xv, yv = x[m], y[m]
        if key == "train_loss":
            xv = np.log10(np.clip(xv, 1e-14, None))
        across = float(np.corrcoef(xv, yv)[0, 1])
        Dm = D[m]
        keep = Dm.sum(0) > 1                    # a group with one cell contributes no within info
        Dm = Dm[:, keep]
        if Dm.shape[1] == 0 or Dm.shape[0] - Dm.shape[1] < 3:
            within = float("nan")
        else:
            rx = xv - Dm @ np.linalg.lstsq(Dm, xv, rcond=None)[0]
            ry = yv - Dm @ np.linalg.lstsq(Dm, yv, rcond=None)[0]
            within = (float(np.corrcoef(rx, ry)[0, 1])
                      if rx.std() > 1e-12 and ry.std() > 1e-12 else float("nan"))
        out.append(dict(key=key, label=label, n=int(m.sum()), across=across, within=within))
    return out
