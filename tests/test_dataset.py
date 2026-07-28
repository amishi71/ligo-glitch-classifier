"""Local sanity test for GravitySpyH5Dataset.

Builds a tiny fake HDF5 file with the same structure as the real Gravity Spy
training set, so this catches bugs in the Dataset class before you ever touch
Colab, a GPU, or the real 3GB download.

Run with: pytest tests/
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import GravitySpyH5Dataset  # noqa: E402


def _make_fake_h5(path):
    with h5py.File(path, "w") as f:
        for label in ["Blip", "Whistle"]:
            for split in ["train", "validation", "test"]:
                for i in range(3):
                    gid = f"{label[:2]}{i}"
                    grp = f.require_group(f"{label}/{split}/{gid}")
                    for duration in ["0.5", "1.0", "2.0", "4.0"]:
                        grp.create_dataset(
                            f"{duration}.png",
                            data=np.random.randint(0, 255, (1, 140, 170), dtype=np.uint8),
                        )


def test_dataset_loads(tmp_path):
    h5_path = tmp_path / "fake.h5"
    _make_fake_h5(str(h5_path))

    ds = GravitySpyH5Dataset(str(h5_path), split="train", classes=["Blip", "Whistle"], duration="1.0")
    assert len(ds) == 6  # 2 classes x 3 ids

    img, label_idx = ds[0]
    assert img.shape == (3, 140, 170)
    assert label_idx in (0, 1)


def test_missing_split_is_skipped(tmp_path):
    h5_path = tmp_path / "fake.h5"
    _make_fake_h5(str(h5_path))

    ds = GravitySpyH5Dataset(str(h5_path), split="nonexistent_split", classes=["Blip", "Whistle"])
    assert len(ds) == 0
