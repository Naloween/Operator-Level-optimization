from __future__ import annotations

import argparse
import json
from pathlib import Path
import csv

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument(
        "--dc_run_dir",
        type=str,
        default=None,
        help="Optional run dir containing D&C results.json to merge into plots.",
    )
    ap.add_argument("--copy_to_notes_images", action="store_true", default=False)
    ap.add_argument(
        "--plot_spec_steps", type=int, nargs="+", default=None,
        help="Override which steps to show in the spectrum figure (subset of stored steps).",
    )
    ap.add_argument("--spec_yscale", choices=["log", "linear"], default="linear")
    ap.add_argument("--spec_ylim", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"),
                    help="Y-axis limits for spectrum panels, e.g. --spec_ylim 0.1 10")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    payload = json.loads((run_dir / "results.json").read_text())
    results = payload["results"]
    cfg = payload["config"]
    if args.dc_run_dir is not None:
        dc_dir = Path(args.dc_run_dir)
        dc_results = dc_dir / "results.json"
        if dc_results.exists():
            dc_payload = json.loads(dc_results.read_text())
            if "dc" in dc_payload.get("results", {}):
                results["dc"] = dc_payload["results"]["dc"]
                if "dc" not in cfg["methods"]:
                    cfg["methods"].append("dc")
        else:
            # FGLN fallback: load D&C history from convergence CSV and optional spectra json.
            dc_csvs = sorted(dc_dir.glob("*.csv"))
            if dc_csvs:
                with dc_csvs[0].open() as f:
                    r = csv.DictReader(f)
                    hist = []
                    for row in r:
                        step = int(row["step"])
                        rel = float(row["rel_err"])
                        hist.append([step, rel, np.nan])
                dc_spec = {}
                spec_file = dc_dir / f"spectrum_steps_L{cfg['depth']}.json"
                if spec_file.exists():
                    spec_payload = json.loads(spec_file.read_text())
                    dc_spec = {str(k): v for k, v in spec_payload.get("spectra", {}).items()}
                results["dc"] = {"history": hist, "spectra": dc_spec}
                if "dc" not in cfg["methods"]:
                    cfg["methods"].append("dc")

    # Convergence figure
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for m in cfg["methods"]:
        hist = np.array(results[m]["history"], dtype=float)
        steps = hist[:, 0]
        rel = hist[:, 1]
        name, color = METHOD_STYLE.get(m, (m, None))
        ax.plot(steps, rel, label=name, color=color, linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"Relative operator error $\|P_t-P^\star\|_F/\|P^\star\|_F$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    conv_path = run_dir / "convergence_l128_msegrad.png"
    fig.savefig(conv_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Spectrum panel at checkpoint steps
    requested_steps = cfg.get("spec_steps", [])
    if not requested_steps:
        requested_steps = [cfg["steps"]]
    steps = sorted(set(int(s) for s in requested_steps))
    if args.plot_spec_steps is not None:
        steps = sorted(set(args.plot_spec_steps))

    ncols = min(4, max(1, len(steps)))
    nrows = int(np.ceil(len(steps) / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(3.8 * ncols, 2.9 * nrows), sharey=True
    )
    axs = np.array(axs).reshape(-1)
    line_styles = {"heavyball": "-", "adam": "--", "muon": "-.", "kfac": (0,(3,1,1,1)), "shampoo": (0,(5,2)), "soap": (0,(1,1)), "als_exact": ":"}
    markers = {"heavyball": "o", "adam": "s", "muon": "^", "kfac": "P", "shampoo": "D", "soap": "X", "als_exact": "d"}

    for ax_idx, step in enumerate(steps):
        ax = axs[ax_idx]
        for m in cfg["methods"]:
            spec_map = results[m]["spectra"]
            key = str(step)
            if key not in spec_map:
                # fallback to nearest earlier checkpoint
                avail = sorted(int(k) for k in spec_map.keys())
                earlier = [k for k in avail if k <= step]
                if not earlier:
                    continue
                key = str(earlier[-1])
            s = np.array(spec_map[key], dtype=float)
            name, color = METHOD_STYLE.get(m, (m, None))
            ax.plot(
                np.arange(1, len(s) + 1),
                s,
                linestyle=line_styles.get(m, "-"),
                marker=markers.get(m, "o"),
                markersize=1.8,
                linewidth=1.4,
                color=color,
                alpha=0.95,
                label=name if ax_idx == 0 else None,
            )
        ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"step={step}", fontsize=10)
        ax.set_yscale(args.spec_yscale)
        if args.spec_ylim is not None:
            ax.set_ylim(bottom=args.spec_ylim[0], top=args.spec_ylim[1])
        ax.grid(True, alpha=0.2)
        if ax_idx == len(steps) - 1:
            ax.set_xlabel("index", fontsize=9)
        ax.tick_params(axis="both", labelsize=9)

    for j in range(len(steps), len(axs)):
        axs[j].axis("off")

    axs[0].set_ylabel("singular value", fontsize=10)
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(labels)),
            frameon=False,
            fontsize=10,
        )
    fig.subplots_adjust(top=0.93, bottom=0.22, wspace=0.15)
    spec_path = run_dir / "spectrum_l128_msegrad.png"
    fig.savefig(spec_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if args.copy_to_notes_images:
        notes_dir = Path("notes/operator_optimizer/images")
        notes_dir.mkdir(parents=True, exist_ok=True)
        (notes_dir / "deeplinear_l128_convergence.png").write_bytes(conv_path.read_bytes())
        (notes_dir / "deeplinear_l128_spectrum.png").write_bytes(spec_path.read_bytes())

    print(f"[done] wrote {conv_path}")
    print(f"[done] wrote {spec_path}")


if __name__ == "__main__":
    main()
