"""Convert each built-in cloud mask in CloudSEN12 to the human-label scheme.

Output values: 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow, 255 = no data (ignored).
Codes were checked on the test split (value counts, 21 Nov 2026).
"""
import numpy as np

NODATA = 255

# Category maps: original value -> harmonized value. Values not listed -> 0 (clear).
CLASS_MAPS = {
    # Fmask: 0 clear land, 1 clear water, 2 cloud shadow, 3 snow, 4 cloud
    "fmask": {2: 3, 4: 1},
    # Sen2Cor scene classification (SCL): 0 no data, 1 saturated/defective, 2 dark area,
    # 3 cloud shadow, 4 vegetation, 5 not vegetated, 6 water, 7 unclassified,
    # 8 cloud medium prob., 9 cloud high prob., 10 thin cirrus, 11 snow
    "sen2cor": {0: NODATA, 1: NODATA, 3: 3, 8: 1, 9: 1, 10: 2},
    # Variant: also count "dark area" (2) as shadow, to see which matches the paper
    "sen2cor_dark_as_shadow": {0: NODATA, 1: NODATA, 2: 3, 3: 3, 8: 1, 9: 1, 10: 2},
    # KappaMask: 0 missing, 1 clear, 2 cloud shadow, 3 semi-transparent cloud, 4 cloud, 5 undefined
    "kappamask_L1C": {0: NODATA, 2: 3, 3: 2, 4: 1, 5: NODATA},
    "kappamask_L2A": {0: NODATA, 2: 3, 3: 2, 4: 1, 5: NODATA},
}

# Probability maps stored as 0-100 (%): cloud if value >= threshold
PROB_THRESHOLDS = {
    "s2cloudless": 40,        # s2cloudless default threshold 0.4
    "cd_fcnn_rgbi": 50,
    "cd_fcnn_rgbi_swir": 50,
}

# The file each method is read from (variants reuse another method's file)
SOURCE_FILE = {"sen2cor_dark_as_shadow": "sen2cor"}


def harmonize(method, arr):
    """Return a uint8 array in the human-label scheme (0-3, 255 = no data)."""
    arr = np.asarray(arr)
    if method in CLASS_MAPS:
        lut = np.zeros(256, dtype=np.uint8)          # default: clear
        for src, dst in CLASS_MAPS[method].items():
            lut[src] = dst
        return lut[arr]
    if method in PROB_THRESHOLDS:
        return (arr >= PROB_THRESHOLDS[method]).astype(np.uint8)   # 1 = cloud
    if method == "qa60":
        return (arr > 0).astype(np.uint8)            # any flag = cloud
    raise ValueError(f"No mapping for {method}")
