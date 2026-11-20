"""Score CloudSEN12's built-in cloud masks against the human labels (test split).

Usage (on Kaggle, after downloading the test split to /tmp/cloudsen12):
    python scripts/eval_builtin.py --root /tmp/cloudsen12 --out results

Writes:
    results/baselines_builtin.csv          one row per method x experiment
    results/per_patch/<method>_<exp>.csv   per-patch metrics (for later statistics)

Before trusting the numbers, run with --check to print the label values each mask uses.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import load, crop, valid_mask, N_PATCHES          # noqa: E402
from src.metrics import EXPERIMENTS, evaluate                    # noqa: E402

# Built-in masks and the experiments the paper reports for each (Table, p.14).
# Masks without a shadow class are scored only on experiment a.
METHODS = {
    "fmask":             ["a_cloud", "b_shadow", "c_invalid"],
    "sen2cor":           ["a_cloud", "b_shadow", "c_invalid"],
    "kappamask_L1C":     ["a_cloud", "b_shadow", "c_invalid"],
    "kappamask_L2A":     ["a_cloud", "b_shadow", "c_invalid"],
    "s2cloudless":       ["a_cloud"],
    "qa60":              ["a_cloud"],
    "cd_fcnn_rgbi":      ["a_cloud"],
    "cd_fcnn_rgbi_swir": ["a_cloud"],
}

# Paper values (median BOA) to compare against
PAPER_BOA = {
    ("fmask", "a_cloud"): 0.84, ("fmask", "b_shadow"): 0.72, ("fmask", "c_invalid"): 0.81,
    ("sen2cor", "a_cloud"): 0.71, ("sen2cor", "b_shadow"): 0.51, ("sen2cor", "c_invalid"): 0.67,
    ("kappamask_L1C", "a_cloud"): 0.82, ("kappamask_L1C", "b_shadow"): 0.74,
    ("kappamask_L1C", "c_invalid"): 0.81,
    ("kappamask_L2A", "a_cloud"): 0.77, ("kappamask_L2A", "b_shadow"): 0.64,
    ("kappamask_L2A", "c_invalid"): 0.75,
    ("s2cloudless", "a_cloud"): 0.79, ("qa60", "a_cloud"): 0.58,
    ("cd_fcnn_rgbi", "a_cloud"): 0.72, ("cd_fcnn_rgbi_swir", "a_cloud"): 0.72,
}


def check_values(root, split, n=200):
    """Print which label values appear in each mask (first n patches)."""
    names = ["manual_hq"] + list(METHODS)
    for name in names:
        arr = load(root, split, f"LABEL_{name}", "uint8")
        vals, counts = np.unique(np.asarray(crop(arr[:n])), return_counts=True)
        share = (100 * counts / counts.sum()).round(1)
        print(f"{name:20s}", dict(zip(vals.tolist(), share.tolist())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/cloudsen12")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results")
    ap.add_argument("--check", action="store_true", help="only print mask label values")
    args = ap.parse_args()

    if args.check:
        check_values(args.root, args.split)
        return

    n = N_PATCHES[args.split]
    gt = load(args.root, args.split, "LABEL_manual_hq", "uint8")
    band = load(args.root, args.split, "L1C_B2", "int16")

    # Load ground truth and valid masks once (cropped, in memory: ~975 x 509 x 509)
    gts = [np.asarray(crop(gt[i])) for i in range(n)]
    valids = [valid_mask(np.asarray(crop(band[i]))) for i in range(n)]

    os.makedirs(os.path.join(args.out, "per_patch"), exist_ok=True)
    rows = []
    for method, exps in METHODS.items():
        pred = load(args.root, args.split, f"LABEL_{method}", "uint8")
        preds = [np.asarray(crop(pred[i])) for i in range(n)]
        for exp in exps:
            pos = EXPERIMENTS[exp]
            per_patch, summary = evaluate(preds, gts, valids, pos, pos)
            per_patch.to_csv(os.path.join(args.out, "per_patch", f"{method}_{exp}.csv"),
                             index=False)
            summary.update(method=method, experiment=exp,
                           paper_BOA=PAPER_BOA.get((method, exp), np.nan))
            rows.append(summary)
            print(f"{method:18s} {exp:10s} BOA {summary['BOA_median']:.3f} "
                  f"(pos-only {summary['BOA_median_pos_only']:.3f}) "
                  f"paper {summary['paper_BOA']:.2f}  "
                  f"PA high {summary['PA_high']:.1f}%  UA high {summary['UA_high']:.1f}%")

    cols = ["method", "experiment", "BOA_median", "BOA_median_pos_only", "paper_BOA",
            "PA_low", "PA_middle", "PA_high", "UA_low", "UA_middle", "UA_high",
            "IoU", "n_patches", "n_patches_with_positive"]
    out = pd.DataFrame(rows)[cols].round(4)
    path = os.path.join(args.out, "baselines_builtin.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
