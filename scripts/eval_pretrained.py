"""Score the pretrained CloudSEN12 models (UNetMobV2 and the 4-band model) on the test split.

Usage (on Kaggle, GPU recommended, after downloading the 13-band test split):
    pip install -q segmentation-models-pytorch
    python scripts/eval_pretrained.py

Writes:
    results/pretrained.csv               PA/UA/BOA per model x experiment (paper format)
    results/pretrained_multiclass.csv    4-class IoU / F1 / mIoU per model
    results/pretrained_efficiency.csv    parameters and FLOPs (512x512 and 256x256)
    results/per_patch/<model>_<exp>.csv  per-patch metrics
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import load, crop, valid_mask, N_PATCHES                       # noqa: E402
from src.metrics import (EXPERIMENTS, evaluate, multiclass_confusion,         # noqa: E402
                         multiclass_scores)
from src.pretrained import MODELS, load_model, predict_batch, count_params, count_flops  # noqa

PAPER_BOA = {"a_cloud": 0.92, "b_shadow": 0.89, "c_invalid": 0.91}   # UNetMobV2, Table p.14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/cloudsen12")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N patches (quick test)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    n = args.limit or N_PATCHES[args.split]

    gt = load(args.root, args.split, "LABEL_manual_hq", "uint8")
    b2 = load(args.root, args.split, "L1C_B2", "int16")
    gts = [np.asarray(crop(gt[i])) for i in range(n)]
    valids = [valid_mask(np.asarray(crop(b2[i]))) for i in range(n)]

    os.makedirs(os.path.join(args.out, "per_patch"), exist_ok=True)
    rows, mc_rows, eff_rows = [], [], []

    for name in args.models:
        spec = MODELS[name]
        print(f"\n== {name} ({len(spec['bands'])} bands)")
        model = load_model(name, device)
        bands = {b: load(args.root, args.split, f"L1C_{b}", "int16") for b in spec["bands"]}

        # Predict all patches in batches
        t0 = time.time()
        preds = []
        for i in range(0, n, args.batch):
            j = min(i + args.batch, n)
            x = np.stack([np.asarray(crop(bands[b][i:j])) for b in spec["bands"]], axis=1)
            preds.extend(predict_batch(model, x.astype(np.float32) / 10_000, device))
        print(f"  predicted {n} patches in {time.time() - t0:.0f} s")

        # Paper-format scores for the three experiments
        for exp, pos in EXPERIMENTS.items():
            per_patch, s = evaluate(preds, gts, valids, pos, pos)
            per_patch.to_csv(os.path.join(args.out, "per_patch", f"{name}_{exp}.csv"), index=False)
            paper = PAPER_BOA[exp] if len(spec["bands"]) == 13 else np.nan
            s.update(model=name, n_bands=len(spec["bands"]), experiment=exp, paper_BOA=paper)
            rows.append(s)
            print(f"  {exp:10s} BOA {s['BOA_median']:.3f}  paper {paper:.2f}  "
                  f"PA high {s['PA_high']:5.1f}%  UA high {s['UA_high']:5.1f}%  IoU {s['IoU']:.3f}")

        # 4-class scores
        cm = sum(multiclass_confusion(p, g, v) for p, g, v in zip(preds, gts, valids))
        mc = multiclass_scores(cm)
        mc.update(model=name, n_bands=len(spec["bands"]))
        mc_rows.append(mc)
        print("  4-class: mIoU {mIoU:.3f} | IoU clear {IoU_clear:.3f}, thick {IoU_thick:.3f}, "
              "thin {IoU_thin:.3f}, shadow {IoU_shadow:.3f}".format(**mc))

        # Size
        eff = dict(model=name, n_bands=len(spec["bands"]), params=count_params(model),
                   GFLOPs_512=count_flops(model, len(spec["bands"]), 512, device) / 1e9,
                   GFLOPs_256=count_flops(model, len(spec["bands"]), 256, device) / 1e9)
        eff_rows.append(eff)
        print(f"  params {eff['params']:,}  GFLOPs 512: {eff['GFLOPs_512']:.2f}  "
              f"256: {eff['GFLOPs_256']:.2f}")

    first = ["model", "n_bands", "experiment", "BOA_median", "paper_BOA"]
    df = pd.DataFrame(rows)
    df[first + [c for c in df.columns if c not in first]].round(4).to_csv(
        os.path.join(args.out, "pretrained.csv"), index=False)
    pd.DataFrame(mc_rows).round(4).to_csv(
        os.path.join(args.out, "pretrained_multiclass.csv"), index=False)
    pd.DataFrame(eff_rows).round(4).to_csv(
        os.path.join(args.out, "pretrained_efficiency.csv"), index=False)
    print(f"\nSaved results to {args.out}/")


if __name__ == "__main__":
    main()
