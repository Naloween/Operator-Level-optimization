from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLE = {
    "heavyball": ("Heavy Ball", "#1f77b4"),
    "adam": ("Adam", "#ff7f0e"),
    "muon": ("Muon", "#2ca02c"),
    "kfac": ("K-FAC", "#e377c2"),
    "shampoo": ("Shampoo", "#d62728"),
    "soap": ("SOAP", "#8c564b"),
    "als_exact": ("ALS-Exact", "#9467bd"),
}


def choose_snapshot_steps(step_keys: list[int]) -> list[int]:
    uniq = sorted(set(step_keys))
    if not uniq:
        return []
    s0 = min(uniq)
    sf = max(uniq)
    if len(uniq) == 1:
        return [s0]
    target_mid = max(s0, (s0 + sf) // 2)
    mids = [s for s in uniq if s != s0 and s != sf]
    if not mids:
        return [s0, sf]
    sm = min(mids, key=lambda s: abs(s - target_mid))
    return [s0, sm, sf]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--output_name", type=str, default="spectrum_snapshots_3panel.png")
    ap.add_argument("--yscale", choices=["linear", "log"], default="linear")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    payload = json.loads((run_dir / "results.json").read_text())
    results = payload["results"]
    methods = list(payload["methods"].keys())

    all_steps = set()
    for m in methods:
        spec = results.get(m, {}).get("spectra", {})
        all_steps.update(int(k) for k in spec.keys())
    plot_steps = choose_snapshot_steps(sorted(all_steps))
    if not plot_steps:
        raise RuntimeError("No spectra found in results.json")

    fig, axs = plt.subplots(1, len(plot_steps), figsize=(4.2 * len(plot_steps), 3.4), sharey=True)
    if len(plot_steps) == 1:
        axs = [axs]

    for i, step in enumerate(plot_steps):
        ax = axs[i]
        for m in methods:
            spec_map = results.get(m, {}).get("spectra", {})
            if not spec_map:
                continue
            key = str(step)
            if key not in spec_map:
                avail = sorted(int(k) for k in spec_map.keys())
                earlier = [k for k in avail if k <= step]
                if not earlier:
                    continue
                key = str(earlier[-1])
            s = np.asarray(spec_map[key], dtype=float)
            name, color = METHOD_STYLE.get(m, (m, None))
            ax.plot(np.arange(1, len(s) + 1), s, linewidth=1.8, color=color, label=name if i == 0 else None)
        ax.axhline(1.0, color="k", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.set_title(f"step={step}")
        ax.set_yscale(args.yscale)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("index")

    axs[0].set_ylabel("singular value")
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.28, wspace=0.2)

    out = run_dir / args.output_name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
