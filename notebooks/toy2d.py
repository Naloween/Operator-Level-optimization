"""Visualising what each optimiser does to the partition, on a 2D task.

The operator framework needs a bias-free positively homogeneous network, for which
f(lambda x) = lambda f(x) and the decision regions would be cones through the origin. A constant
input coordinate restores affine capability without breaking homogeneity, so the network stays
bias-free, f(x) = P(x) x stays exact, and the pictures below can be drawn in the original plane.

A ReLU network's linear regions are exactly the sets of inputs sharing a gate pattern, so the
partition is obtained by evaluating the pattern on a grid and finding where it changes. This is the
object Humayun et al. (2402.15555) track as "local complexity", drawn directly rather than
summarised by a scalar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "studies"))
import gpu                                                      # noqa: E402
import exp                                                      # noqa: E402

COLOR = {"gd": "#4c78a8", "adam": "#59a14f", "op": "#e15759"}
LABEL = {"gd": "gradient descent", "adam": "Adam", "op": "reference (operator)"}


def grid(lim=2.2, n=420, dev="cpu"):
    g = np.linspace(-lim, lim, n)
    A, B = np.meshgrid(g, g)
    X = np.stack([A.ravel(), B.ravel(), np.ones(A.size)], 1)     # constant coordinate appended
    return A, B, torch.tensor(X, dtype=torch.float32, device=dev)


def patterns(Ws, X, arch):
    _, gs = gpu.forward(Ws, X, arch)
    on = [g > 0 for g in gs] if arch == "crelu" else [g > 0.5 for g in gs]
    return torch.cat([o.reshape(o.shape[0], -1) for o in on], 1)


def partition_edges(Ws, X, arch, shape):
    """Where does the gate pattern change between neighbouring grid points?"""
    P = patterns(Ws, X, arch).reshape(*shape, -1)
    dx = (P[:, 1:] != P[:, :-1]).any(-1)
    dy = (P[1:, :] != P[:-1, :]).any(-1)
    E = torch.zeros(shape, dtype=torch.bool, device=P.device)
    E[:, :-1] |= dx
    E[:-1, :] |= dy
    return E.cpu().numpy()


def region_count(Ws, X, arch):
    """Number of distinct linear regions intersecting the grid."""
    P = patterns(Ws, X, arch)
    return int(torch.unique(P, dim=0).shape[0])


def decision(Ws, X, arch, shape):
    with torch.no_grad():
        out, _ = gpu.forward(Ws, X, arch)
    return out.argmax(1).reshape(shape).cpu().numpy(), \
        (out[:, 1] - out[:, 0]).reshape(shape).cpu().numpy()


def load_arm(runs, arch, arm, seed=0, by="stable"):
    import analysis
    got = analysis.pick(runs, arch, arm, by=by)
    if not got:
        return None
    _, g = got
    for c, r, d in g:
        if c["seed"] == seed:
            return c, r, d
    return g[0]


def snaps(d):
    import posthoc
    return posthoc.snapshots(d)


def fig_partitions(runs, arch="relu", seed=0, lim=2.2, n=420, dev=None):
    """Final partition and decision boundary for each arm, with the training data."""
    import matplotlib.pyplot as plt
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    A, B, X = grid(lim, n, dev)
    D = exp.moons(400, 200, 400, seed, dev)
    xt = D["Xtr"].cpu().numpy(); yt = D["Ytr"].cpu().numpy()
    arms = [a for a in ("gd", "adam", "op") if load_arm(runs, arch, a, seed)]
    fig, axes = plt.subplots(1, len(arms), figsize=(4.6 * len(arms), 4.5), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        c, r, d = load_arm(runs, arch, arm, seed)
        s = snaps(d)
        Ws = [w.to(dev) for w in torch.load(s[-1][1], map_location=dev,
                                            weights_only=False)["Ws"]]
        lab, marg = decision(Ws, X, arch, A.shape)
        E = partition_edges(Ws, X, arch, A.shape)
        ax.contourf(A, B, lab, levels=[-0.5, 0.5, 1.5], colors=["#cfe3f5", "#f7d6d6"], alpha=0.9)
        ax.contour(A, B, marg, levels=[0], colors="k", linewidths=2.0, zorder=4)
        # zorder matters: imshow defaults to 0, which puts it UNDERNEATH the contourf
        ax.imshow(np.where(E, 1.0, np.nan), extent=[-lim, lim, -lim, lim], origin="lower",
                  cmap="Greys", alpha=0.55, interpolation="nearest", zorder=2, vmin=0.0, vmax=1.0)
        ax.scatter(xt[yt == 0, 0], xt[yt == 0, 1], s=8, c="#1f4e79", linewidths=0, zorder=5)
        ax.scatter(xt[yt == 1, 0], xt[yt == 1, 1], s=8, c="#a01d1d", linewidths=0, zorder=5)
        import analysis
        acc = analysis.stable_test(r)
        ax.set_title(f"{LABEL[arm]}\ntest {acc:.3f} | {region_count(Ws, X, arch)} regions",
                     fontsize=10)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{arch}: decision boundary (black) over the linear-region partition (grey)",
                 fontsize=11)
    fig.tight_layout()
    return fig


def fig_partition_evolution(runs, arch="relu", arm="adam", seed=0, lim=2.2, n=340, dev=None):
    """The partition at successive snapshots: does it migrate away from the data?"""
    import matplotlib.pyplot as plt
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    A, B, X = grid(lim, n, dev)
    D = exp.moons(400, 200, 400, seed, dev)
    xt = D["Xtr"].cpu().numpy(); yt = D["Ytr"].cpu().numpy()
    got = load_arm(runs, arch, arm, seed)
    if not got:
        return None
    c, r, d = got
    s = snaps(d)
    pick = s if len(s) <= 5 else [s[0], s[len(s) // 4], s[len(s) // 2], s[3 * len(s) // 4], s[-1]]
    fig, axes = plt.subplots(1, len(pick), figsize=(3.3 * len(pick), 3.5))
    for ax, (step, path) in zip(np.atleast_1d(axes), pick):
        Ws = [w.to(dev) for w in torch.load(path, map_location=dev, weights_only=False)["Ws"]]
        lab, marg = decision(Ws, X, arch, A.shape)
        E = partition_edges(Ws, X, arch, A.shape)
        ax.contourf(A, B, lab, levels=[-0.5, 0.5, 1.5], colors=["#cfe3f5", "#f7d6d6"], alpha=0.9)
        ax.contour(A, B, marg, levels=[0], colors="k", linewidths=1.6, zorder=4)
        ax.imshow(np.where(E, 1.0, np.nan), extent=[-lim, lim, -lim, lim], origin="lower",
                  cmap="Greys", alpha=0.55, interpolation="nearest", zorder=2, vmin=0.0, vmax=1.0)
        ax.scatter(xt[:, 0], xt[:, 1], s=4, c=np.where(yt == 0, "#1f4e79", "#a01d1d"),
                   linewidths=0)
        ax.set_title(f"step {step}  ({region_count(Ws, X, arch)} regions)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{arch}, {LABEL[arm]}: partition over training", fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ slices through real data
def data_plane(D, idx, lim=(-0.6, 1.6), n=360, dev="cpu"):
    """The 2D affine plane through three real inputs, as a grid.

    Origin at x0, first axis towards x1, second axis the component of x2 orthogonal to the first.
    The three anchors therefore sit at (0,0), (1,0) and (p,q) in plane coordinates, so slices from
    different runs are directly comparable: it is the same plane in input space every time.
    """
    X = D["Xtr"][idx].double()
    x0, x1, x2 = X[0], X[1], X[2]
    u = x1 - x0
    nu = torch.linalg.vector_norm(u)
    u = u / nu
    w = x2 - x0
    v = w - (w @ u) * u
    nv = torch.linalg.vector_norm(v)
    v = v / nv
    a = torch.linspace(lim[0], lim[1], n, dtype=torch.float64, device=X.device) * float(nu)
    b = torch.linspace(lim[0], lim[1], n, dtype=torch.float64, device=X.device) * float(nu)
    A, B = torch.meshgrid(a, b, indexing="xy")
    G = x0.unsqueeze(0) + A.reshape(-1, 1) * u.unsqueeze(0) + B.reshape(-1, 1) * v.unsqueeze(0)
    anchors = np.array([[0.0, 0.0],
                        [float(nu), 0.0],
                        [float((w @ u)), float(nv)]]) / float(nu)
    return (A.cpu().numpy() / float(nu), B.cpu().numpy() / float(nu),
            G.to(torch.float32), anchors)


def fig_data_slice(runs, arch="relu", seed=0, anchors=(0, 1, 2), n=360, lim=(-0.6, 1.6),
                   dev=None, experiment="ref_mnist1d"):
    """Partition and decision landscape on a plane through three MNIST-1D training points.

    All three arms are drawn on the SAME plane, so the comparison is of the functions, not of the
    viewpoint. The anchors are chosen from different classes so the plane actually crosses a
    decision boundary.
    """
    import matplotlib.pyplot as plt
    import analysis
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    D = exp.mnist1d(4000, 1000, 1000, seed, dev)
    y = D["Ytr"].cpu().numpy()
    # three points from three distinct classes, for a plane that crosses boundaries
    want, idx = [], []
    for i, c in enumerate(y):
        if c not in want:
            want.append(c); idx.append(i)
        if len(idx) == 3:
            break
    A, B, G, anc = data_plane(D, idx, lim, n, dev)
    arms = []
    for arm in ("gd", "adam", "op"):
        got = analysis.pick(runs, arch, arm, width=128, steps=12000, by="stable")
        if not got:
            continue
        _, g = got
        cand = [z for z in g if z[0]["seed"] == seed] or g
        arms.append((arm, cand[0]))
    if not arms:
        return None
    fig, axes = plt.subplots(1, len(arms), figsize=(4.7 * len(arms), 4.7), squeeze=False)
    for ax, (arm, (c, r, d)) in zip(axes[0], arms):
        s = snaps(d)
        Ws = [w.to(dev) for w in torch.load(s[-1][1], map_location=dev,
                                            weights_only=False)["Ws"]]
        with torch.no_grad():
            out, _ = gpu.forward(Ws, G, arch)
        lab = out.argmax(1).reshape(A.shape).cpu().numpy()
        E = partition_edges(Ws, G, arch, A.shape)
        ax.pcolormesh(A, B, lab, cmap="tab10", vmin=0, vmax=9, alpha=0.75, shading="auto",
                      zorder=1)
        ax.imshow(np.where(E, 1.0, np.nan), extent=[A.min(), A.max(), B.min(), B.max()],
                  origin="lower", cmap="Greys", alpha=0.55, interpolation="nearest",
                  zorder=2, vmin=0.0, vmax=1.0)
        ax.scatter(anc[:, 0], anc[:, 1], s=90, facecolor="white", edgecolor="k",
                   linewidth=2, zorder=5)
        for j, (px, py) in enumerate(anc):
            ax.annotate(f"class {y[idx[j]]}", (px, py), textcoords="offset points",
                        xytext=(7, 7), fontsize=8, zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
        ax.set_title(f"{LABEL[arm]}\ntest {analysis.stable_test(r):.3f} | "
                     f"{region_count(Ws, G, arch)} regions", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{arch}: the same 2D plane through three MNIST-1D training points\n"
                 f"colour = predicted class, grey = linear-region boundaries", fontsize=11)
    fig.tight_layout()
    return fig


def fig_complexity_vs_distance(runs, arch="relu", seed=0, n=360, lim=(-0.6, 1.6), nbins=26,
                               dev=None):
    """Partition density as a function of distance from the data, on a plane through 3 real points.

    This is the quantitative form of the slice picture, and the claim it tests is Humayun et al.'s:
    linear regions MIGRATE AWAY from the training samples during training. Region *count* does not
    test that -- a network can have many regions and still concentrate them on the data. Here the
    partition-edge density is binned by distance to the nearest anchor, so a curve that rises with
    distance means the boundaries have moved off the data.
    """
    import matplotlib.pyplot as plt
    import analysis
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    D = exp.mnist1d(4000, 1000, 1000, seed, dev)
    y = D["Ytr"].cpu().numpy()
    want, idx = [], []
    for i, c in enumerate(y):
        if c not in want:
            want.append(c); idx.append(i)
        if len(idx) == 3:
            break
    A, B, G, anc = data_plane(D, idx, lim, n, dev)
    dist = np.min(np.stack([np.hypot(A - px, B - py) for px, py in anc]), 0).ravel()
    edges = np.linspace(0, np.percentile(dist, 97), nbins + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for arm in ("gd", "adam", "op"):
        got = analysis.pick(runs, arch, arm, width=128, steps=12000, by="stable")
        if not got:
            continue
        _, g = got
        cand = [z for z in g if z[0]["seed"] == seed] or g
        c, r, d = cand[0]
        Ws = [w.to(dev) for w in torch.load(snaps(d)[-1][1], map_location=dev,
                                            weights_only=False)["Ws"]]
        E = partition_edges(Ws, G, arch, A.shape).ravel().astype(float)
        dens = [E[(dist >= edges[i]) & (dist < edges[i + 1])].mean() for i in range(nbins)]
        ax.plot(ctr, dens, "o-", color=COLOR[arm], ms=3,
                label=f"{LABEL[arm]}  (test {analysis.stable_test(r):.3f})")
    ax.set_xlabel("distance from the nearest training point (plane units)")
    ax.set_ylabel("partition-edge density")
    ax.set_title(f"{arch}: are the linear regions on the data, or away from it?", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def knn_predict(Xtr, Ytr, Q, k=5, chunk=8192):
    """Plain k-NN, chunked so the distance matrix never has to fit at once."""
    out = []
    for i in range(0, Q.shape[0], chunk):
        d = torch.cdist(Q[i:i + chunk].double(), Xtr.double())
        nn = d.topk(k, largest=False).indices
        lab = Ytr[nn]
        out.append(torch.mode(lab, dim=1).values)
    return torch.cat(out)


def knn_accuracy(seed=0, ks=(1, 3, 5, 10, 20), dev=None):
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    D = exp.mnist1d(4000, 1000, 1000, seed, dev)
    out = {}
    for k in ks:
        p = knn_predict(D["Xtr"], D["Ytr"], D["Xte"], k)
        out[k] = float((p == D["Yte"]).float().mean())
    return out


def fig_slice_with_knn(runs, arch="relu", seed=0, n=360, lim=(-0.6, 1.6), knn_k=5, dev=None):
    """The same plane, for each optimiser and for k-NN.

    k-NN is a useful reference precisely because it has no linear regions and no training: its
    decision map on this plane shows what the data alone implies, against which the networks'
    partitions can be read as inductive bias rather than as signal.
    """
    import matplotlib.pyplot as plt
    import analysis
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    D = exp.mnist1d(4000, 1000, 1000, seed, dev)
    y = D["Ytr"].cpu().numpy()
    want, idx = [], []
    for i, c in enumerate(y):
        if c not in want:
            want.append(c); idx.append(i)
        if len(idx) == 3:
            break
    A, B, G, anc = data_plane(D, idx, lim, n, dev)
    panels = []
    for arm in ("gd", "adam", "op"):
        got = analysis.pick(runs, arch, arm, width=128, steps=12000, by="stable")
        if not got:
            continue
        _, g = got
        cand = [z for z in g if z[0]["seed"] == seed] or g
        panels.append((arm, cand[0]))
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.5 * (len(panels) + 1), 4.6),
                             squeeze=False)
    for ax, (arm, (c, r, d)) in zip(axes[0], panels):
        Ws = [w.to(dev) for w in torch.load(snaps(d)[-1][1], map_location=dev,
                                            weights_only=False)["Ws"]]
        with torch.no_grad():
            out, _ = gpu.forward(Ws, G, arch)
        lab = out.argmax(1).reshape(A.shape).cpu().numpy()
        E = partition_edges(Ws, G, arch, A.shape)
        ax.pcolormesh(A, B, lab, cmap="tab10", vmin=0, vmax=9, alpha=0.8, shading="auto", zorder=1)
        ax.imshow(np.where(E, 1.0, np.nan), extent=[A.min(), A.max(), B.min(), B.max()],
                  origin="lower", cmap="Greys", alpha=0.5, interpolation="nearest",
                  zorder=2, vmin=0.0, vmax=1.0)
        ax.set_title(f"{LABEL[arm]}  (test {analysis.stable_test(r):.3f})", fontsize=10)
    axk = axes[0][-1]
    lab = knn_predict(D["Xtr"], D["Ytr"], G, knn_k).reshape(A.shape).cpu().numpy()
    axk.pcolormesh(A, B, lab, cmap="tab10", vmin=0, vmax=9, alpha=0.8, shading="auto", zorder=1)
    acc = knn_accuracy(seed, (knn_k,))[knn_k]
    axk.set_title(f"{knn_k}-NN  (test {acc:.3f})\nno linear regions", fontsize=10)
    for ax in axes[0]:
        ax.scatter(anc[:, 0], anc[:, 1], s=90, facecolor="white", edgecolor="k", linewidth=2,
                   zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{arch}: the same plane through three MNIST-1D points, with k-NN for reference",
                 fontsize=11)
    fig.tight_layout()
    return fig


def _edges_binned(Ws, arch, A, B, G, anc, edges, chunk=40000):
    """Partition-edge density binned by distance to the nearest anchor, chunked over the grid."""
    pats = []
    for i in range(0, G.shape[0], chunk):
        pats.append(patterns(Ws, G[i:i + chunk], arch))
    P = torch.cat(pats).reshape(*A.shape, -1)
    dx = (P[:, 1:] != P[:, :-1]).any(-1)
    dy = (P[1:, :] != P[:-1, :]).any(-1)
    E = torch.zeros(A.shape, dtype=torch.bool, device=P.device)
    E[:, :-1] |= dx
    E[:-1, :] |= dy
    E = E.cpu().numpy().ravel().astype(float)
    dist = np.min(np.stack([np.hypot(A - px, B - py) for px, py in anc]), 0).ravel()
    return np.array([E[(dist >= edges[i]) & (dist < edges[i + 1])].mean()
                     for i in range(len(edges) - 1)])


def complexity_curves(runs, arch="relu", seeds=(0, 1, 2), n_planes=8, n=240,
                      lim=(-0.6, 1.6), nbins=20, dev=None, steps=12000, experiment=None):
    """Density-vs-distance averaged over many random planes and seeds.

    The single-plane version is suggestive but arbitrary: one plane through the first three
    distinct classes. This resamples the triple, so what survives is a property of the function
    rather than of the viewpoint.
    """
    import analysis
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    acc = {arm: [] for arm in ("gd", "adam", "op")}
    ctr = None
    for seed in seeds:
        D = exp.mnist1d(4000, 1000, 1000, seed, dev)
        y = D["Ytr"].cpu().numpy()
        rng = np.random.default_rng(1234 + seed)
        chosen = []
        for _ in range(n_planes):
            cls = rng.choice(np.unique(y), 3, replace=False)
            chosen.append([int(rng.choice(np.where(y == c)[0])) for c in cls])
        loaded = {}
        for arm in ("gd", "adam", "op"):
            got = analysis.pick(runs, arch, arm, width=128, steps=steps, by="stable")
            if not got:
                continue
            _, g = got
            cand = [z for z in g if z[0]["seed"] == seed] or g
            c, r, d = cand[0]
            s = snaps(d)
            if not s:
                continue
            loaded[arm] = [w.to(dev) for w in torch.load(s[-1][1], map_location=dev,
                                                         weights_only=False)["Ws"]]
        for idx in chosen:
            A, B, G, anc = data_plane(D, idx, lim, n, dev)
            edges = np.linspace(0, 1.25, nbins + 1)
            ctr = 0.5 * (edges[1:] + edges[:-1])
            for arm, Ws in loaded.items():
                acc[arm].append(_edges_binned(Ws, arch, A, B, G, anc, edges))
    return ctr, {a: np.array(v) for a, v in acc.items() if len(v)}


def fig_complexity_avg(runs, arch="relu", **kw):
    import matplotlib.pyplot as plt
    ctr, curves = complexity_curves(runs, arch, **kw)
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    for arm, M in curves.items():
        m, s = np.nanmean(M, 0), np.nanstd(M, 0)
        ax.plot(ctr, m, "-", color=COLOR[arm], label=f"{LABEL[arm]}  (n={M.shape[0]} planes)")
        ax.fill_between(ctr, m - s, m + s, color=COLOR[arm], alpha=0.18, linewidth=0)
    ax.set_xlabel("distance from the nearest training point (plane units)")
    ax.set_ylabel("partition-edge density")
    ax.set_title(f"{arch}: averaged over random planes and seeds (band = 1 sd)", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def fig_slice_evolution(arch="relu", arm="adam", seed=0, experiment="grok_mnist1d",
                        n=300, lim=(-0.6, 1.6), dev=None, max_panels=6):
    """The same data-plane slice at successive snapshots of an extended run."""
    import matplotlib.pyplot as plt
    import json as _json
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(exp.ROOT) / experiment
    run = None
    for d in sorted(base.iterdir()):
        if not (d / "config.json").exists():
            continue
        c = _json.loads((d / "config.json").read_text())
        if c["arch"] == arch and c["arm"] == arm and c["seed"] == seed:
            run = (c, d); break
    if run is None:
        return None, None
    c, d = run
    s = snaps(d)
    if not s:
        return None, None
    if len(s) > max_panels:
        take = np.linspace(0, len(s) - 1, max_panels).astype(int)
        s = [s[i] for i in take]
    D = exp.mnist1d(4000, 1000, 1000, seed, dev)
    y = D["Ytr"].cpu().numpy()
    want, idx = [], []
    for i, cl in enumerate(y):
        if cl not in want:
            want.append(cl); idx.append(i)
        if len(idx) == 3:
            break
    A, B, G, anc = data_plane(D, idx, lim, n, dev)
    recs = [_json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    by_step = {}
    for r in recs:
        by_step[r["step"]] = r
    fig, axes = plt.subplots(1, len(s), figsize=(3.5 * len(s), 3.9))
    stats = []
    for ax, (step, path) in zip(np.atleast_1d(axes), s):
        Ws = [w.to(dev) for w in torch.load(path, map_location=dev, weights_only=False)["Ws"]]
        with torch.no_grad():
            out, _ = gpu.forward(Ws, G, arch)
        lab = out.argmax(1).reshape(A.shape).cpu().numpy()
        E = partition_edges(Ws, G, arch, A.shape)
        ax.pcolormesh(A, B, lab, cmap="tab10", vmin=0, vmax=9, alpha=0.8, shading="auto", zorder=1)
        ax.imshow(np.where(E, 1.0, np.nan), extent=[A.min(), A.max(), B.min(), B.max()],
                  origin="lower", cmap="Greys", alpha=0.5, interpolation="nearest",
                  zorder=2, vmin=0.0, vmax=1.0)
        ax.scatter(anc[:, 0], anc[:, 1], s=70, facecolor="white", edgecolor="k",
                   linewidth=1.8, zorder=5)
        near = [k for k in by_step if abs(k - step) <= 250]
        acc = by_step[min(near, key=lambda k: abs(k - step))]["test_acc"] if near else float("nan")
        nreg = region_count(Ws, G, arch)
        stats.append((step, acc, nreg))
        ax.set_title(f"step {step//1000}k   test {acc:.3f}\n{nreg} regions", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    h = c["hyper"] if not isinstance(c["hyper"], list) else c["hyper"][0]
    fig.suptitle(f"{arch}, {LABEL[arm]} (hyper {h:g}): the same plane, over extended training",
                 fontsize=11)
    fig.tight_layout()
    return fig, stats


def fig_spikes(arch="relu", arm="adam", seed=0, experiment="grok_mnist1d", smooth=5,
               steps_cfg=None):
    """Loss spikes against gate reorganisation, over extended training.

    Humayun et al. report that grokking is accompanied by spikes in which the linear regions
    reorganise. `churn` -- the fraction of gates that flip between consecutive probes -- measures
    exactly that, and it is recorded every 250 steps, so the association can be tested directly
    rather than inferred from snapshots.
    """
    import matplotlib.pyplot as plt
    import json as _json
    base = Path(exp.ROOT) / experiment
    run = None
    for d in sorted(base.iterdir()):
        if not (d / "config.json").exists():
            continue
        c = _json.loads((d / "config.json").read_text())
        if c["arch"] == arch and c["arm"] == arm and c["seed"] == seed:
            if steps_cfg is not None and c["steps"] != steps_cfg:
                continue
            # prefer the LONGEST *completed* run; "largest steps" alone once picked a diverged
            # extension whose accuracy had collapsed to chance
            st = _json.loads((d / "meta.json").read_text()).get("status") \
                if (d / "meta.json").exists() else None
            key = (st == "complete", c["steps"])
            if run is None or key > run[2]:
                run = (c, d, key)
    if run is None:
        return None, None
    c, d, _ = run
    recs = [_json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    seen = {}
    for r in recs:
        seen[r["step"]] = r
    recs = [seen[k] for k in sorted(seen)]
    s = np.array([r["step"] for r in recs])
    tl = np.array([r["train_loss"] for r in recs])
    te = np.array([r["test_acc"] for r in recs])
    ch = np.array([r.get("churn", np.nan) for r in recs])
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    ax[0].semilogy(s, tl, color="0.3", lw=0.9)
    ax[0].set_ylabel("training loss")
    ax[1].plot(s, te, color=COLOR[arm], lw=0.9)
    ax[1].set_ylabel("test accuracy")
    ax[2].plot(s, ch, color="#b07aa1", lw=0.9)
    ax[2].set_ylabel("gate churn\n(flips since last probe)")
    ax[2].set_xlabel("step")
    # mark loss spikes: a probe whose loss exceeds the running median by 10x
    med = np.array([np.median(tl[max(0, i - 20):i + 1]) for i in range(len(tl))])
    spike = tl > 10 * np.maximum(med, 1e-12)
    for a in ax:
        for x in s[spike]:
            a.axvline(x, color="crimson", alpha=0.18, lw=0.8, zorder=0)
        a.grid(alpha=0.25)
    h = c["hyper"] if not isinstance(c["hyper"], list) else c["hyper"][0]
    fig.suptitle(f"{arch}, {LABEL[arm]} (hyper {h:g}): loss spikes (red) against gate churn",
                 fontsize=11)
    fig.tight_layout()
    stats = dict(n_spikes=int(spike.sum()),
                 churn_at_spike=float(np.nanmean(ch[spike])) if spike.any() else float("nan"),
                 churn_elsewhere=float(np.nanmean(ch[~spike])),
                 acc_at_spike=float(np.nanmean(te[spike])) if spike.any() else float("nan"),
                 acc_elsewhere=float(np.nanmean(te[~spike])))
    return fig, stats


def complexity_vs_boundary(Ws, arch, A, B, G, nbins=18, chunk=40000):
    """Partition density as a function of distance to the DECISION BOUNDARY.

    Humayun et al. make a two-part claim: regions migrate AWAY from the training samples and
    TOWARDS the decision boundary. Measuring distance-from-data tests only the first half. Here the
    boundary is located on the slice as the set of grid cells where the predicted class changes,
    and the partition density is binned by distance to it, which tests the second half.
    """
    outs, pats = [], []
    for i in range(0, G.shape[0], chunk):
        with torch.no_grad():
            o, _ = gpu.forward(Ws, G[i:i + chunk], arch)
        outs.append(o.argmax(1))
        pats.append(patterns(Ws, G[i:i + chunk], arch))
    lab = torch.cat(outs).reshape(A.shape)
    P = torch.cat(pats).reshape(*A.shape, -1)
    dx = (P[:, 1:] != P[:, :-1]).any(-1)
    dy = (P[1:, :] != P[:-1, :]).any(-1)
    E = torch.zeros(A.shape, dtype=torch.bool, device=P.device)
    E[:, :-1] |= dx
    E[:-1, :] |= dy
    bx = (lab[:, 1:] != lab[:, :-1])
    by = (lab[1:, :] != lab[:-1, :])
    Bd = torch.zeros(A.shape, dtype=torch.bool, device=P.device)
    Bd[:, :-1] |= bx
    Bd[:-1, :] |= by
    bnd = np.argwhere(Bd.cpu().numpy())
    if not len(bnd):
        return None, None
    pts = np.stack([A[tuple(bnd.T)], B[tuple(bnd.T)]], 1)
    gp = np.stack([A.ravel(), B.ravel()], 1)
    step = max(1, len(pts) // 4000)                     # subsample the boundary for the distance
    from scipy.spatial import cKDTree
    dist = cKDTree(pts[::step]).query(gp)[0]
    Ef = E.cpu().numpy().ravel().astype(float)
    edges = np.linspace(0, np.percentile(dist, 95), nbins + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    dens = [Ef[(dist >= edges[i]) & (dist < edges[i + 1])].mean() for i in range(nbins)]
    return ctr, np.array(dens)


def pgd_accuracy(Ws, X, Y, arch, eps=0.3, steps=10, alpha=None):
    """Adversarial accuracy under L-inf PGD -- the quantity 'delayed robustness' refers to."""
    alpha = alpha or (2.5 * eps / steps)
    X0 = X.clone()
    d = (torch.rand_like(X) * 2 - 1) * eps
    for _ in range(steps):
        d = d.detach().requires_grad_(True)
        out, _ = gpu.forward(Ws, X0 + d, arch)
        loss = torch.nn.functional.cross_entropy(out, Y)
        g, = torch.autograd.grad(loss, d)
        d = (d + alpha * g.sign()).clamp(-eps, eps)
    with torch.no_grad():
        out, _ = gpu.forward(Ws, X0 + d, arch)
    return float((out.argmax(1) == Y).float().mean())


def robustness_curve(arch="relu", arm="adam", seed=0, experiment="grok_mnist1d",
                     steps_cfg=None, eps=(0.1, 0.3), dev=None):
    """Clean and adversarial accuracy at every snapshot: does robustness arrive late?"""
    import json as _json
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(exp.ROOT) / experiment
    run = None
    for d in sorted(base.iterdir()):
        if not (d / "config.json").exists():
            continue
        c = _json.loads((d / "config.json").read_text())
        if c["arch"] != arch or c["arm"] != arm or c["seed"] != seed:
            continue
        if steps_cfg is not None and c["steps"] != steps_cfg:
            continue
        st = _json.loads((d / "meta.json").read_text()).get("status")
        key = (st == "complete", c["steps"])
        if run is None or key > run[2]:
            run = (c, d, key)
    if run is None:
        return None
    c, d, _ = run
    D = exp.mnist1d(4000, 1000, 1000, seed, dev)
    rows = []
    for step, path in snaps(d):
        Ws = [w.to(dev) for w in torch.load(path, map_location=dev, weights_only=False)["Ws"]]
        with torch.no_grad():
            clean = float((gpu.forward(Ws, D["Xte"], arch)[0].argmax(1) == D["Yte"])
                          .float().mean())
        rec = {"step": step, "clean": clean}
        for e in eps:
            rec[f"pgd_{e}"] = pgd_accuracy(Ws, D["Xte"], D["Yte"], arch, eps=e)
        rows.append(rec)
    return rows


def extended_stats(arch="relu", arm="op", seed=0, experiment="grok_mnist1d", steps_cfg=None,
                   n=300, lim=(-0.6, 1.6), dev=None, eps=0.3):
    """Every statistic we track, at each snapshot of an extended run.

    Combines what lives in metrics.jsonl (loss, accuracy, churn, density) with what must be
    recomputed from the saved weights (region count, switching, concentration contrast, adversarial
    accuracy). Returns one row per snapshot.
    """
    import json as _json
    import posthoc
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(exp.ROOT) / experiment
    run = None
    for d in sorted(base.iterdir()):
        if not (d / "config.json").exists():
            continue
        c = _json.loads((d / "config.json").read_text())
        if c["arch"] != arch or c["arm"] != arm or c["seed"] != seed:
            continue
        if steps_cfg is not None and c["steps"] != steps_cfg:
            continue
        st = _json.loads((d / "meta.json").read_text()).get("status")
        key = (st == "complete", c["steps"])
        if run is None or key > run[2]:
            run = (c, d, key)
    if run is None:
        return None, None
    c, d, _ = run
    recs = {r["step"]: r for r in
            (_json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip())}
    D = exp.mnist1d(4000, 1000, 1000, seed, dev)
    y = D["Ytr"].cpu().numpy()
    want, idx = [], []
    for i, cl in enumerate(y):
        if cl not in want:
            want.append(cl); idx.append(i)
        if len(idx) == 3:
            break
    A, B, G, anc = data_plane(D, idx, lim, n, dev)
    edges = np.linspace(0, 1.25, 21)
    rows = []
    for step, path in snaps(d):
        Ws = [w.to(dev) for w in torch.load(path, map_location=dev, weights_only=False)["Ws"]]
        near = [k for k in recs if abs(k - step) <= 250]
        m = recs[min(near, key=lambda k: abs(k - step))] if near else {}
        dens = _edges_binned(Ws, arch, A, B, G, anc, edges)
        sw = posthoc.switching_units(Ws, D["Xtr"][:256], arch)
        rows.append(dict(step=step,
                         train=m.get("train_loss", float("nan")),
                         test=m.get("test_acc", float("nan")),
                         churn=m.get("churn", float("nan")),
                         density=m.get("density", float("nan")),
                         pr=m.get("pr", float("nan")),
                         regions=region_count(Ws, G, arch),
                         switching=sw["switching"],
                         concentration=float(dens[0] - dens[-1]),
                         pgd=pgd_accuracy(Ws, D["Xte"], D["Yte"], arch, eps=eps)))
    return c, rows


def fig_extended(arch="relu", arms=("op", "gd", "adam"), seed=0, experiment="grok_mnist1d",
                 **kw):
    """Side-by-side trajectories of every statistic, for each extended run."""
    import matplotlib.pyplot as plt
    keys = [("test", "test accuracy"), ("train", "training loss"),
            ("churn", "gate churn"), ("switching", "switching units"),
            ("regions", "linear regions on the slice"),
            ("concentration", "region concentration on data"),
            ("pr", "operator rank"), ("pgd", "adversarial acc (PGD 0.3)")]
    data = {}
    for arm in arms:
        c, rows = extended_stats(arch, arm, seed, experiment, **kw)
        if rows:
            data[arm] = rows
    if not data:
        return None, None
    fig, axes = plt.subplots(2, 4, figsize=(19, 7.5))
    for ax, (k, lab) in zip(axes.ravel(), keys):
        for arm, rows in data.items():
            s = [r["step"] for r in rows]
            v = [r[k] for r in rows]
            ax.plot(s, v, "o-", ms=3, color=COLOR[arm], label=LABEL[arm])
        if k == "train":
            ax.set_yscale("log")
        ax.set_ylabel(lab); ax.set_xlabel("step"); ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{arch}: extended training, every tracked statistic", fontsize=11)
    fig.tight_layout()
    return fig, data
