"""Loading CloudSEN12 high-quality split (Hugging Face: csaybar/CloudSEN12-high).

Files are raw memory-mapped arrays of shape (N, 512, 512).
Real image data is 509x509: row 0, col 0 and rows/cols 510-511 are padding (always 0).
"""
import numpy as np

N_PATCHES = {"train": 8490, "val": 535, "test": 975}
SIZE = 512
CROP = (slice(1, 510), slice(1, 510))   # keep the real 509x509 area

# Label values in LABEL_manual_hq
CLEAR, THICK, THIN, SHADOW = 0, 1, 2, 3
CLASS_NAMES = ["clear", "thick", "thin", "shadow"]


def load(root, split, name, dtype):
    """Open one file as a read-only memmap of shape (N, 512, 512).

    name examples: "L1C_B2", "LABEL_manual_hq", "LABEL_fmask"
    dtype: "int16" for bands, "uint8" for labels/masks
    """
    path = f"{root}/{split}/{name}.dat"
    return np.memmap(path, dtype=dtype, mode="r", shape=(N_PATCHES[split], SIZE, SIZE))


def crop(arr):
    """Crop the last two dimensions to the real 509x509 area."""
    return arr[..., CROP[0], CROP[1]]


def valid_mask(band_patch):
    """True where the pixel has data. In Sentinel-2 L1C, 0 means no-data."""
    return band_patch > 0
