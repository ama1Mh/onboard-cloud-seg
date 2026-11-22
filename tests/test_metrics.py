"""Small checks that the metrics behave as defined. Run: python -m pytest tests"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import confusion, patch_metrics, groups, evaluate  # noqa: E402


def test_perfect_prediction():
    g = np.array([[1, 0], [1, 0]], bool)
    v = np.ones_like(g)
    pa, ua, boa = patch_metrics(*confusion(g, g, v))
    assert pa == 1 and ua == 1 and boa == 1


def test_all_wrong():
    g = np.array([[1, 0], [1, 0]], bool)
    pa, ua, boa = patch_metrics(*confusion(~g, g, np.ones_like(g)))
    assert pa == 0 and ua == 0 and boa == 0


def test_cloud_free_patch_gives_nan_pa_ua():
    g = np.zeros((2, 2), bool)
    p = np.array([[1, 0], [0, 0]], bool)
    pa, ua, boa = patch_metrics(*confusion(p, g, np.ones_like(g)))
    assert np.isnan(pa) and np.isnan(ua)
    assert boa == 0.75            # only TNR is defined: 3 of 4 clear pixels correct


def test_invalid_pixels_are_ignored():
    g = np.array([[1, 1], [0, 0]], bool)
    p = np.array([[1, 0], [0, 1]], bool)
    v = np.array([[1, 0], [1, 0]], bool)   # ignore the two wrong pixels
    assert confusion(p, g, v) == (1, 0, 0, 1)


def test_groups():
    assert groups([0.05, 0.5, 0.95, np.nan]) == pytest.approx([100 / 3, 100 / 3, 100 / 3])


def test_evaluate_uses_label_values():
    gt = np.array([[0, 1], [2, 3]], np.uint8)       # clear, thick, thin, shadow
    pred = np.array([[0, 2], [1, 0]], np.uint8)     # thick/thin swapped, shadow missed
    v = np.ones((2, 2), bool)
    _, s_cloud = evaluate([pred], [gt], [v], (1, 2), (1, 2))
    assert s_cloud["BOA_median"] == 1.0              # cloud vs not-cloud is perfect
    _, s_shadow = evaluate([pred], [gt], [v], (3,), (3,))
    assert s_shadow["PA_low"] == 100.0               # shadow completely missed


def test_harmonize_maps_native_codes():
    from src.harmonize import harmonize
    fm = np.array([0, 1, 2, 3, 4], np.uint8)            # land, water, shadow, snow, cloud
    assert harmonize("fmask", fm).tolist() == [0, 0, 3, 0, 1]
    km = np.array([0, 1, 2, 3, 4, 5], np.uint8)
    assert harmonize("kappamask_L1C", km).tolist() == [255, 0, 3, 2, 1, 255]
    sc = np.array([0, 3, 4, 8, 9, 10, 11], np.uint8)
    assert harmonize("sen2cor", sc).tolist() == [255, 3, 0, 1, 1, 2, 0]
    pr = np.array([0, 39, 40, 100], np.uint8)
    assert harmonize("s2cloudless", pr).tolist() == [0, 0, 1, 1]


def test_multiclass_scores():
    from src.metrics import multiclass_confusion, multiclass_scores
    gt = np.array([[0, 1], [2, 3]], np.uint8)
    pred = np.array([[0, 1], [1, 0]], np.uint8)       # thin -> thick, shadow -> clear
    cm = multiclass_confusion(pred, gt, np.ones((2, 2), bool))
    assert cm.sum() == 4 and cm[2, 1] == 1 and cm[3, 0] == 1
    s = multiclass_scores(cm)
    assert s["IoU_thin"] == 0 and s["IoU_shadow"] == 0
    assert s["IoU_clear"] == 0.5 and s["IoU_thick"] == 0.5
    assert s["mIoU"] == 0.25
