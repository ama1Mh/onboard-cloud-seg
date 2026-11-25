"""Figure: per-class IoU vs model size (4-band TinyUNets) against 13-band UNetMobV2.

Reads results/*_multiclass.csv and results/*_efficiency.csv (written by eval_pretrained.py
and train.py). Usage:  python scripts/plot_size_sweep.py  ->  figures/size_sweep.png
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import matplotlib.ticker           # noqa: E402
import pandas as pd               # noqa: E402

CLASSES = [("clear", "Clear", "#7A7A7A", "o"), ("thick", "Thick cloud", "#0072B2", "s"),
           ("thin", "Thin cloud", "#E69F00", "^"), ("shadow", "Cloud shadow", "#CC3311", "D")]
REFERENCE = "UNetMobV2_V1"


def load(results="results"):
    mc = pd.concat([pd.read_csv(f) for f in glob.glob(f"{results}/*multiclass.csv")])
    ef = pd.concat([pd.read_csv(f) for f in glob.glob(f"{results}/*efficiency.csv")])
    return mc.drop_duplicates("model").merge(ef[["model", "params", "GFLOPs_256"]], on="model")


def main(results="results", out="figures/size_sweep.png"):
    df = load(results)
    ref = df[df.model == REFERENCE].iloc[0]
    ours = df[df.model.str.startswith("tinyunet") & (df.n_bands == 4)].sort_values("params")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for key, label, color, marker in CLASSES:
        col = f"IoU_{key}"
        axes[0].plot(ours.params, ours[col], marker=marker, color=color, label=label, lw=1.8, ms=5)
        axes[0].scatter([ref.params], [ref[col]], marker=marker, facecolors="none",
                        edgecolors=color, s=45, lw=1.4)
        axes[1].plot(ours.params, 100 * ours[col] / ref[col], marker=marker, color=color,
                     label=label, lw=1.8, ms=5)
    axes[0].set_ylim(0.35, 0.93)

    axes[0].set_ylabel("Test IoU")
    axes[0].set_title("(a) Per-class IoU", fontsize=10, loc="left")
    axes[1].set_ylabel("% of 13-band UNetMobV2 IoU")
    axes[1].set_title("(b) Relative to UNetMobV2", fontsize=10, loc="left")
    axes[1].axhline(100, color="#999999", lw=0.8, ls="--")
    ticks = list(ours.params) + [ref.params]
    labels = [f"{v/1e6:.2f}M" if v < 1e6 else f"{v/1e6:.1f}M" for v in ticks]
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(ticks, labels)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("Parameters (log scale)")
        ax.grid(True, which="major", color="#E5E5E5", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].annotate("13-band\nUNetMobV2", (ref.params, ref["IoU_clear"]), xytext=(0, 10),
                     textcoords="offset points", fontsize=7, ha="center", color="#555555")
    axes[1].legend(fontsize=7.5, frameon=False, loc="lower right")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
