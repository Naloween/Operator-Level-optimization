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
