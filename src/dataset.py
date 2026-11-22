"""PyTorch dataset for CloudSEN12 (memory-mapped files).

Each item: x float32 (C, 512, 512) reflectance, y int64 (512, 512) with classes 0-3.
The real 509x509 image is reflect-padded to 512; padded and no-data pixels get label 255 (ignored).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import load, crop, N_PATCHES

IGNORE = 255


class CloudSEN12(Dataset):
    def __init__(self, root, split, bands, augment=False, limit=None):
        self.root, self.split, self.bands, self.augment = root, split, list(bands), augment
        self.n = min(limit, N_PATCHES[split]) if limit else N_PATCHES[split]
        self._arrays = None   # opened lazily inside each worker

    def _open(self):
        if self._arrays is None:
            self._arrays = ([load(self.root, self.split, f"L1C_{b}", "int16") for b in self.bands],
                            load(self.root, self.split, "LABEL_manual_hq", "uint8"))
        return self._arrays

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        bands, label = self._open()
        x = np.stack([np.asarray(crop(b[i])) for b in bands]).astype(np.float32)   # (C,509,509)
        y = np.asarray(crop(label[i])).astype(np.int64)
        y[x[0] == 0] = IGNORE                                   # no-data pixels
        x = np.pad(x / 10_000, ((0, 0), (1, 2), (1, 2)), mode="reflect")
        y = np.pad(y, ((1, 2), (1, 2)), constant_values=IGNORE)
        if self.augment:   # random flips and 90-degree rotations
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
            k = np.random.randint(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))
