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
        c, r, _ = max(sub, key=lambda z: best_by_val(z[1])["val_acc"])
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
        c, r, _ = max(sub, key=lambda z: best_by_val(z[1])["val_acc"])
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
        c, r, _ = max(sub, key=lambda z: best_by_val(z[1])["val_acc"])
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


def fig_rank_vs_acc(runs):
    """The headline scatter: operator rank against test accuracy, every run."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, arch in zip(axes, ["crelu", "relu"]):
        for arm in ARMS:
            sub = select(runs, arch=arch, arm=arm)
            if not sub:
                continue
            b = [best_by_val(r) for _, r, _ in sub]
            ax.scatter([x["pr"] for x in b], [x["test_acc"] for x in b],
                       color=COLOR[arm], label=LABEL[arm], s=34, alpha=0.85,
                       edgecolor="white", linewidth=0.6)
        ax.set_xlabel("operator rank (participation ratio)")
        ax.set_ylabel("test accuracy"); ax.set_title(arch); ax.grid(alpha=0.25)
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
        bh = max(byh, key=lambda k: np.mean([best_by_val(r)["val_acc"] for _, r, _ in byh[k]]))
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
    picked = []
    for arm in ARMS:
        sub = select(runs, arch=arch, arm=arm, seed=seed)
        if sub:
            picked.append((arm, max(sub, key=lambda z: best_by_val(z[1])["val_acc"])))
    if not picked:
        return None
    c0 = picked[0][1][0]
    D = _exp.mnist1d(c0["n_train"], c0["n_val"], c0["n_test"], seed, dev)
    X, Y = D["Xte"][:n], D["Yte"][:n].cpu().numpy()
    fig, axes = plt.subplots(len(picked), len(layers), figsize=(3.4 * len(layers), 3.2 * len(picked)))
    axes = np.atleast_2d(axes)
    for i, (arm, (c, r, d)) in enumerate(picked):
        snaps = _ph.snapshots(d)
        Ws = [w.to(dev) for w in torch.load(snaps[-1][1], map_location=dev,
                                            weights_only=False)["Ws"]]
        reps = _ph.layer_reps(Ws, X, arch)
        for j, l in enumerate(layers):
            ax = axes[i, j]
            if l >= len(reps):
                ax.axis("off"); continue
            P, var = _ph.pca2(reps[l])
            ax.scatter(P[:, 0], P[:, 1], c=Y, cmap="tab10", s=7, alpha=0.8, linewidths=0)
            ax.set_title(f"{LABEL[arm]} | layer {l+1}  ({100*var:.0f}% var)", fontsize=8)
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
            bh = max(byh, key=lambda k: np.mean([best_by_val(r)["val_acc"] for _, r, _ in byh[k]]))
            series = [posthoc_rows(d) for _, _, d in byh[bh]]
            series = [s for s in series if s]
            if not series:
                continue
            steps = [x["step"] for x in series[0]]
            v = np.mean([[x["op_div_c"] for x in s] for s in series], axis=0)
            ax.plot(steps, v, "o-", color=COLOR[arm], label=LABEL[arm])
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
    """The two notions of simplicity against each other, at the endpoint, every run."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, arch in zip(axes, ["crelu", "relu"]):
        for arm in ARMS:
            xs, ys, init = [], [], None
            for c, r, d in select(runs, arch=arch, arm=arm):
                ph = posthoc_rows(d)
                if not ph:
                    continue
                xs.append(best_by_val(r)["pr"]); ys.append(ph[-1]["op_div_c"])
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
    """Peak-selected against last-quarter-mean, with the swing that explains the gap."""
    import matplotlib.pyplot as plt
    archs = [a for a in ("crelu", "relu", "leaky") if select(runs, arch=a)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    w = 0.25
    xs = np.arange(len(archs))
    for j, arm in enumerate(ARMS):
        pk, st, sw = [], [], []
        for arch in archs:
            got = pick(runs, arch, arm, width=128, steps=12000)
            if not got:
                pk.append(np.nan); st.append(np.nan); sw.append(np.nan); continue
            _, g = got
            pk.append(np.mean([best_by_val(r)["test_acc"] for _, r, _ in g]))
            st.append(np.mean([stable_test(r) for _, r, _ in g]))
            sw.append(np.mean([swing(r) for _, r, _ in g]))
        axes[0].bar(xs + (j - 1) * w, pk, w * 0.42, color=COLOR[arm], alpha=0.45,
                    label=f"{LABEL[arm]} (peak)")
        axes[0].bar(xs + (j - 1) * w + w * 0.44, st, w * 0.42, color=COLOR[arm],
                    label=f"{LABEL[arm]} (last 25%)")
        axes[1].bar(xs + (j - 1) * w, sw, w * 0.8, color=COLOR[arm], label=LABEL[arm])
    axes[0].set_ylabel("test accuracy"); axes[0].set_title(
        "peak-selected (pale) vs last-quarter mean (solid)", fontsize=10)
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
            cells.setdefault(tuple(c["hyper"]), []).append(stable_test(r))
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
        ax.set_title(f"{arch}: test accuracy", fontsize=10)
        for i in range(len(ks)):
            for j in range(len(es)):
                if np.isfinite(Z[i, j]):
                    ax.text(j, i, f"{Z[i,j]:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if Z[i, j] < np.nanmax(Z) * 0.93 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def fig_capacity(runs):
    """Test accuracy against width: does giving the operator arm LESS capacity close the gap?"""
    import matplotlib.pyplot as plt
    archs = [a for a in ("crelu", "relu") if select(runs, arch=a)]
    fig, axes = plt.subplots(1, len(archs), figsize=(5.6 * len(archs), 4.2), squeeze=False)
    for ax, arch in zip(axes[0], archs):
        widths = sorted({c["width"] for c, _, _ in select(runs, arch=arch)})
        for arm in ARMS:
            xs, ys, es = [], [], []
            for w in widths:
                got = pick(runs, arch, arm, width=w, steps=12000)
                if not got:
                    continue
                _, g = got
                v = [stable_test(r) for _, r, _ in g]
                xs.append(w); ys.append(np.mean(v)); es.append(np.std(v))
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", color=COLOR[arm], label=LABEL[arm],
                            capsize=3)
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
                tl = r[-1]["train_loss"]
                if not np.isfinite(tl) or tl <= 0:
                    tl = 1e-12                       # log axis: floor exact zeros
                xs.append(tl); ys.append(stable_test(r))
                interp.append(r[-1]["train_acc"] > 0.999)
            if not xs:
                continue
            xs, ys, interp = np.array(xs), np.array(ys), np.array(interp)
            ax.scatter(xs[interp], ys[interp], color=COLOR[arm], s=38, alpha=0.9,
                       edgecolor="white", linewidth=0.6, label=f"{LABEL[arm]} (interpolates)")
            if (~interp).any():
                ax.scatter(xs[~interp], ys[~interp], facecolor="none", edgecolor=COLOR[arm],
                           s=46, linewidth=1.4, label=f"{LABEL[arm]} (does not fit)")
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("final training loss  (better fit $\\rightarrow$)")
        ax.set_ylabel("test accuracy (last-quarter mean)")
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
    ax.set_xlabel("handover step"); ax.set_ylabel("test accuracy (last-quarter mean)")
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
                acc_m.append(np.mean([stable_test(r) for r in g]) if g else np.nan)
                acc_s.append(np.std([stable_test(r) for r in g]) if g else np.nan)
                cos_m.append(np.mean([_tail_mean(r, "cos_op") for r in g]) if g else np.nan)
            fmt, col = mk[eta]
            axes[row][0].errorbar(xs, acc_m, yerr=acc_s, fmt=fmt, color=col, capsize=3,
                                  label=f"$\\eta$ = {eta:g}")
            axes[row][1].plot(xs, cos_m, fmt, color=col, label=f"$\\eta$ = {eta:g}")
        for col_i, (ylab, title) in enumerate(
                [("test accuracy (last-quarter mean)", "generalisation"),
                 ("$\\cos_{op}$ (last-half mean)", "operator-space faithfulness")]):
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
                     dev=None, max_seeds=3):
    """Clean and CW-PGD accuracy at the final checkpoint of every (arch, arm, hyper) cell.

    Uses the CW margin attack, not cross-entropy: see `toy2d.pgd_cw` for why the cross-entropy
    version is unusable here. The logit margin and the operator (input-Jacobian) norm are carried
    alongside, because they are what the cross-entropy attack was actually measuring.
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
                if not (d / "ckpt.pt").exists():
                    continue
                Ws = [w.to(dev) for w in
                      torch.load(d / "ckpt.pt", map_location=dev, weights_only=False)["Ws"]]
                D = dataset(c["seed"])
                X, Y = D["Xte"], D["Yte"]
                with torch.no_grad():
                    o, _ = gpu.forward(Ws, X, arch)
                    acc["clean"].append(float((o.argmax(1) == Y).float().mean()))
                acc["train"].append(r[-1]["train_acc"])
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

    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm and r["arch"] in archs and r["train_acc"] > 0.999]
        if sub:
            axes[1].scatter([r["margin"] for r in sub], [r["pr"] for r in sub], s=34,
                            color=COLOR[arm], label=LABEL[arm], alpha=0.85,
                            edgecolor="white", lw=0.6)
    axes[1].set_xscale("symlog")
    axes[1].set_xlabel("logit margin  (what cross-entropy PGD was really measuring)")
    axes[1].set_ylabel("operator rank (participation ratio)")
    axes[1].set_title("margin and rank are confounded across arms", fontsize=10)
    axes[1].grid(alpha=0.25); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
