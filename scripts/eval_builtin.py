"""Score CloudSEN12's built-in cloud masks against the human labels (test split).

Usage (on Kaggle, after downloading the test split to /tmp/cloudsen12):
    python scripts/eval_builtin.py --check     # print the raw values each mask uses
    python scripts/eval_builtin.py             # score everything

Writes:
    results/baselines_builtin.csv          one row per method x experiment
    results/per_patch/<method>_<exp>.csv   per-patch metrics (for later statistics)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import load, crop, valid_mask, N_PATCHES          # noqa: E402
from src.harmonize import harmonize, NODATA, SOURCE_FILE        # noqa: E402
from src.metrics import EXPERIMENTS, evaluate                    # noqa: E402

# Methods and the experiments the paper reports for each (Table, p.14).
# Masks without a shadow class are scored only on experiment a.
METHODS = {
    "fmask":                  ["a_cloud", "b_shadow", "c_invalid"],
    "sen2cor":                ["a_cloud", "b_shadow", "c_invalid"],
    "sen2cor_dark_as_shadow": ["b_shadow", "c_invalid"],
    "kappamask_L1C":          ["a_cloud", "b_shadow", "c_invalid"],
    "kappamask_L2A":          ["a_cloud", "b_shadow", "c_invalid"],
    "s2cloudless":            ["a_cloud"],
    "qa60":                   ["a_cloud"],
    "cd_fcnn_rgbi":           ["a_cloud"],
    "cd_fcnn_rgbi_swir":      ["a_cloud"],
}

# Paper values (median BOA, Table p.14)
PAPER_BOA = {
    ("fmask", "a_cloud"): 0.84, ("fmask", "b_shadow"): 0.72, ("fmask", "c_invalid"): 0.81,
    ("sen2cor", "a_cloud"): 0.71, ("sen2cor", "b_shadow"): 0.51, ("sen2cor", "c_invalid"): 0.67,
    ("sen2cor_dark_as_shadow", "b_shadow"): 0.51, ("sen2cor_dark_as_shadow", "c_invalid"): 0.67,
    ("kappamask_L1C", "a_cloud"): 0.82, ("kappamask_L1C", "b_shadow"): 0.74,
    ("kappamask_L1C", "c_invalid"): 0.81,
    ("kappamask_L2A", "a_cloud"): 0.77, ("kappamask_L2A", "b_shadow"): 0.64,
    ("kappamask_L2A", "c_invalid"): 0.75,
    ("s2cloudless", "a_cloud"): 0.79, ("qa60", "a_cloud"): 0.58,
    ("cd_fcnn_rgbi", "a_cloud"): 0.72, ("cd_fcnn_rgbi_swir", "a_cloud"): 0.72,
}


def check_values(root, split, n=None):
    """Print which raw values appear in each mask (all patches by default)."""
    files = ["manual_hq"] + sorted({SOURCE_FILE.get(m, m) for m in METHODS})
    for name in files:
        arr = load(root, split, f"LABEL_{name}", "uint8")
        arr = arr if n is None else arr[:n]
        vals, counts = np.unique(np.asarray(crop(arr)), return_counts=True)
        share = (100 * counts / counts.sum()).round(1)
        shown = dict(zip(vals.tolist(), share.tolist()))
        if len(shown) > 15:   # probability maps: just show the range
            shown = f"{len(shown)} values, min {vals.min()}, max {vals.max()}"
        print(f"{name:20s}", shown)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/cloudsen12")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results")
    ap.add_argument("--check", action="store_true", help="only print raw mask values")
    args = ap.parse_args()

    if args.check:
        check_values(args.root, args.split)
        return

    n = N_PATCHES[args.split]
    gt = load(args.root, args.split, "LABEL_manual_hq", "uint8")
    band = load(args.root, args.split, "L1C_B2", "int16")
    gts = [np.asarray(crop(gt[i])) for i in range(n)]
    data_ok = [valid_mask(np.asarray(crop(band[i]))) for i in range(n)]

    os.makedirs(os.path.join(args.out, "per_patch"), exist_ok=True)
    rows = []
    for method, exps in METHODS.items():
        raw = load(args.root, args.split, f"LABEL_{SOURCE_FILE.get(method, method)}", "uint8")
        preds = [harmonize(method, crop(raw[i])) for i in range(n)]
        if method == "qa60" and not any(p.any() for p in preds):
            print("qa60               skipped: mask is empty (all zeros) in this split")
            continue
        valids = [ok & (p != NODATA) for ok, p in zip(data_ok, preds)]
        for exp in exps:
            pos = EXPERIMENTS[exp]
            per_patch, summary = evaluate(preds, gts, valids, pos, pos)
            per_patch.to_csv(os.path.join(args.out, "per_patch", f"{method}_{exp}.csv"),
                             index=False)
            summary.update(method=method, experiment=exp,
                           paper_BOA=PAPER_BOA.get((method, exp), np.nan))
            rows.append(summary)
            print(f"{method:23s} {exp:10s} BOA {summary['BOA_median']:.3f} "
                  f"(pos-only {summary['BOA_median_pos_only']:.3f})  "
                  f"paper {summary['paper_BOA']:.2f}  "
                  f"PA high {summary['PA_high']:5.1f}%  UA high {summary['UA_high']:5.1f}%")

    cols = ["method", "experiment", "BOA_median", "BOA_median_pos_only", "paper_BOA",
            "PA_low", "PA_middle", "PA_high", "UA_low", "UA_middle", "UA_high",
            "IoU", "n_patches", "n_patches_with_positive"]
    out = pd.DataFrame(rows)[cols].round(4)
    path = os.path.join(args.out, "baselines_builtin.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
