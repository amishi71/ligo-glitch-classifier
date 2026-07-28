"""Gravity Spy HDF5 dataset loader.

Reads directly from the Gravity Spy training-set HDF5 file
(https://zenodo.org/records/1486046, file trainingsetv1d1.h5).
Structure inside the file: /<label>/<sample_type>/<gravityspy_id>/<duration>.png
"""
from collections import Counter

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# Well-separated 8-class starter subset (project brief, Phase 3).
# Pass classes=None anywhere below to use every class found in the file.
DEFAULT_CLASSES = [
    "Blip", "Whistle", "Koi_Fish", "Power_Line",
    "Violin_Mode", "Scattered_Light", "Chirp", "Low_Frequency_Burst",
]


class GravitySpyH5Dataset(Dataset):
    def __init__(self, h5_path, split, classes=None, duration="1.0", transform=None):
        self.h5_path = h5_path
        self.split = split
        self.duration = duration
        self.transform = transform
        self._h5 = None  # opened lazily, once per worker process

        with h5py.File(h5_path, "r") as f:
            labels = classes if classes is not None else sorted(f.keys())
            self.classes = sorted(labels)
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

            self.index = []
            for label in self.classes:
                if label not in f or split not in f[label]:
                    continue
                for gid in f[label][split].keys():
                    self.index.append((label, gid))

    def __len__(self):
        return len(self.index)

    def _h5file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx):
        label, gid = self.index[idx]
        f = self._h5file()
        img = f[label][self.split][gid][f"{self.duration}.png"][()]
        img = np.asarray(img, dtype=np.float32)
        if img.ndim == 3:  # (1, H, W) -> (H, W)
            img = img[0]
        if img.max() > 1.0:
            img = img / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        img = img.repeat(3, 1, 1)  # fake RGB channels for the pretrained CNN

        if self.transform:
            img = self.transform(img)

        return img, self.class_to_idx[label]


def class_distribution(dataset):
    return Counter(label for label, _ in dataset.index)


def get_dataloaders(h5_path, split_names, classes=None, duration="1.0",
                     batch_size=64, num_workers=2, image_size=224):
    """split_names: dict with keys 'train', 'val', 'test' mapping to the
    literal sample_type strings inside the h5 file (check these with the
    inspection snippet in the README before relying on the defaults)."""
    transform = T.Compose([T.Resize((image_size, image_size))])

    train_ds = GravitySpyH5Dataset(h5_path, split_names["train"], classes, duration, transform)
    val_ds = GravitySpyH5Dataset(h5_path, split_names["val"], classes, duration, transform)
    test_ds = GravitySpyH5Dataset(h5_path, split_names["test"], classes, duration, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
