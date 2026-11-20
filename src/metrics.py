"""Evaluation metrics following CloudSEN12 (Aybar et al., 2022, Technical Validation).

Per patch: PA (producer's accuracy ~ recall), UA (user's accuracy ~ precision),
BOA (balanced overall accuracy = mean of TPR and TNR).
PA and UA are NaN for patches with no positive pixels in the ground truth.
Summary: median BOA, and % of patches with PA/UA < 0.1 (low), 0.1-0.9 (middle), > 0.9 (high).
We also report a dataset-level IoU for the positive class.
"""
import numpy as np
import pandas as pd

# The three CloudSEN12 experiments: which label values count as "positive"
EXPERIMENTS = {
    "a_cloud":   (1, 2),      # thick + thin cloud vs everything else
    "b_shadow":  (3,),        # cloud shadow vs everything else
    "c_invalid": (1, 2, 3),   # cloud + shadow vs clear
}


def to_binary(labels, positive_values):
    """Boolean array: True where labels is one of positive_values."""
    return np.isin(labels, positive_values)


def confusion(pred, gt, valid):
    """Counts of TP, FP, FN, TN over valid pixels. All inputs are boolean arrays."""
    p, g = pred[valid], gt[valid]
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    tn = int(np.sum(~p & ~g))
    return tp, fp, fn, tn


def patch_metrics(tp, fp, fn, tn):
    """PA, UA, BOA for one patch, from its confusion counts."""
    has_pos = (tp + fn) > 0
    has_neg = (tn + fp) > 0
    if has_pos:
        pa = tp / (tp + fn)
        ua = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # predicted no positives at all
    else:
        pa = ua = np.nan                                 # as in the paper
    tpr = pa if has_pos else np.nan
    tnr = tn / (tn + fp) if has_neg else np.nan
    boa = np.nanmean([tpr, tnr]) if (has_pos or has_neg) else np.nan
    return pa, ua, boa


def groups(values):
    """% of non-NaN values below 0.1, between 0.1 and 0.9, above 0.9."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return [np.nan] * 3
    return [100 * np.mean(v < 0.1),
            100 * np.mean((v >= 0.1) & (v <= 0.9)),
            100 * np.mean(v > 0.9)]


def summarise(per_patch):
    """per_patch: DataFrame with columns tp, fp, fn, tn, pa, ua, boa, has_pos."""
    pa_g, ua_g = groups(per_patch["pa"]), groups(per_patch["ua"])
    tp, fp, fn = per_patch["tp"].sum(), per_patch["fp"].sum(), per_patch["fn"].sum()
    return {
        "BOA_median": np.nanmedian(per_patch["boa"]),
        # variant: BOA only over patches that contain the positive class
        "BOA_median_pos_only": np.nanmedian(per_patch.loc[per_patch["has_pos"], "boa"]),
        "PA_low": pa_g[0], "PA_middle": pa_g[1], "PA_high": pa_g[2],
        "UA_low": ua_g[0], "UA_middle": ua_g[1], "UA_high": ua_g[2],
        "IoU": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan,
        "n_patches": len(per_patch),
        "n_patches_with_positive": int(per_patch["has_pos"].sum()),
    }


def evaluate(pred_labels, gt_labels, valids, positive_pred, positive_gt):
    """Evaluate a method over many patches.

    pred_labels, gt_labels: iterables of 2D label arrays (already cropped)
    valids: iterable of 2D boolean arrays
    positive_pred / positive_gt: label values that count as positive
    Returns (per_patch DataFrame, summary dict).
    """
    rows = []
    for i, (p, g, v) in enumerate(zip(pred_labels, gt_labels, valids)):
        tp, fp, fn, tn = confusion(to_binary(p, positive_pred), to_binary(g, positive_gt), v)
        pa, ua, boa = patch_metrics(tp, fp, fn, tn)
        rows.append(dict(patch=i, tp=tp, fp=fp, fn=fn, tn=tn,
                         pa=pa, ua=ua, boa=boa, has_pos=(tp + fn) > 0))
    df = pd.DataFrame(rows)
    return df, summarise(df)
