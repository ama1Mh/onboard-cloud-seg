"""Train a cloud segmentation model on CloudSEN12, then score it on the test split.

Training recipe follows Aybar et al. (2022): Adam lr 1e-3, cross-entropy, batch 32,
learning rate x0.1 when validation loss has not improved for 4 epochs,
stop when it has not improved for 10 epochs; keep the best validation-loss model.

Quick test (a few minutes):
    python scripts/train.py --model tinyunet16 --limit 64 --epochs 1 --no-wandb
Full run:
    python scripts/train.py --model tinyunet16

Writes:
    checkpoints/<run>.pt
    results/trained_<run>.csv, results/trained_<run>_multiclass.csv,
    results/trained_<run>_efficiency.csv, results/per_patch/<run>_<exp>.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import load, crop, valid_mask, N_PATCHES                     # noqa: E402
from src.dataset import CloudSEN12, IGNORE                                  # noqa: E402
from src.metrics import (EXPERIMENTS, evaluate, multiclass_confusion,       # noqa: E402
                         multiclass_scores)
from src.models import build_model                                          # noqa: E402
from src.pretrained import predict_batch, count_params, count_flops         # noqa: E402


def run_epoch(model, loader, loss_fn, device, opt=None, scaler=None):
    """One pass over the data. Trains if opt is given. Returns mean loss and confusion matrix."""
    train = opt is not None
    model.train(train)
    total, count = 0.0, 0
    cm = np.zeros((4, 4), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=scaler is not None):
            logits = model(x)
            loss = loss_fn(logits, y)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward() if scaler else loss.backward()
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
        total += loss.item() * x.size(0)
        count += x.size(0)
        if not train:
            pred = logits.argmax(1)
            keep = y != IGNORE
            idx = (y[keep] * 4 + pred[keep]).cpu().numpy()
            cm += np.bincount(idx, minlength=16).reshape(4, 4)
    return total / count, cm


def evaluate_test(model, root, bands, device, run, out, batch=8, limit=None):
    """Score on the test split exactly like the baselines."""
    n = limit or N_PATCHES["test"]
    gt = load(root, "test", "LABEL_manual_hq", "uint8")
    arrays = [load(root, "test", f"L1C_{b}", "int16") for b in bands]
    gts = [np.asarray(crop(gt[i])) for i in range(n)]
    valids = [valid_mask(np.asarray(crop(arrays[0][i]))) for i in range(n)]
    model.eval()
    preds = []
    for i in range(0, n, batch):
        j = min(i + batch, n)
        x = np.stack([np.asarray(crop(a[i:j])) for a in arrays], axis=1).astype(np.float32) / 10_000
        preds.extend(predict_batch(model, x, device))

    os.makedirs(os.path.join(out, "per_patch"), exist_ok=True)
    rows = []
    for exp, pos in EXPERIMENTS.items():
        per_patch, s = evaluate(preds, gts, valids, pos, pos)
        per_patch.to_csv(os.path.join(out, "per_patch", f"{run}_{exp}.csv"), index=False)
        s.update(model=run, n_bands=len(bands), experiment=exp)
        rows.append(s)
        print(f"  TEST {exp:10s} BOA {s['BOA_median']:.3f}  PA high {s['PA_high']:5.1f}%  "
              f"UA high {s['UA_high']:5.1f}%  IoU {s['IoU']:.3f}")
    first = ["model", "n_bands", "experiment", "BOA_median"]
    df = pd.DataFrame(rows)
    df[first + [c for c in df.columns if c not in first]].round(4).to_csv(
        os.path.join(out, f"trained_{run}.csv"), index=False)

    cm = sum(multiclass_confusion(p, g, v) for p, g, v in zip(preds, gts, valids))
    mc = multiclass_scores(cm)
    mc.update(model=run, n_bands=len(bands))
    pd.DataFrame([mc]).round(4).to_csv(os.path.join(out, f"trained_{run}_multiclass.csv"), index=False)
    print("  TEST 4-class: mIoU {mIoU:.3f} | IoU clear {IoU_clear:.3f}, thick {IoU_thick:.3f}, "
          "thin {IoU_thin:.3f}, shadow {IoU_shadow:.3f}".format(**mc))
    return mc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/cloudsen12")
    ap.add_argument("--model", default="tinyunet16")
    ap.add_argument("--bands", nargs="+", default=["B2", "B3", "B4", "B8"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-patience", type=int, default=4)
    ap.add_argument("--stop-patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only N patches per split (quick test)")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = f"{args.model}_{len(args.bands)}b_s{args.seed}"
    print(f"run {run} on {device}")

    train_ds = CloudSEN12(args.root, "train", args.bands, augment=True, limit=args.limit)
    val_ds = CloudSEN12(args.root, "val", args.bands, augment=False, limit=args.limit)
    kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)

    model = build_model(args.model, len(args.bands)).to(device)
    print(f"parameters: {count_params(model):,}")
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=args.lr_patience)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    wandb = None
    if not args.no_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project="onboard-cloud-seg", name=run, config=vars(args))

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = f"checkpoints/{run}.pt"
    best, bad = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, train_dl, loss_fn, device, opt, scaler)
        with torch.no_grad():
            va_loss, cm = run_epoch(model, val_dl, loss_fn, device, scaler=scaler)
        sched.step(va_loss)
        va = multiclass_scores(cm)
        lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}  train {tr_loss:.4f}  val {va_loss:.4f}  val mIoU {va['mIoU']:.3f} "
              f"(thin {va['IoU_thin']:.3f}, shadow {va['IoU_shadow']:.3f})  lr {lr:.0e}  "
              f"{time.time() - t0:.0f}s")
        if wandb:
            wandb.log({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "lr": lr,
                       **{f"val_{k}": v for k, v in va.items()}})
        if va_loss < best:
            best, bad = va_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.stop_patience:
                print(f"early stop: no improvement for {args.stop_patience} epochs")
                break

    print(f"\nbest val loss {best:.4f}; scoring best model on test")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    mc = evaluate_test(model, args.root, args.bands, device, run, args.out, limit=args.limit)

    eff = dict(model=run, n_bands=len(args.bands), params=count_params(model),
               GFLOPs_512=count_flops(model, len(args.bands), 512, device) / 1e9,
               GFLOPs_256=count_flops(model, len(args.bands), 256, device) / 1e9)
    pd.DataFrame([eff]).round(4).to_csv(os.path.join(args.out, f"trained_{run}_efficiency.csv"),
                                        index=False)
    print(f"  params {eff['params']:,}  GFLOPs 512: {eff['GFLOPs_512']:.2f}  256: {eff['GFLOPs_256']:.2f}")
    if wandb:
        wandb.log({f"test_{k}": v for k, v in mc.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
