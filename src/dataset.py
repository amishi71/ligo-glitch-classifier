"""PyTorch Dataset for Gravity Spy spectrogram images, plus dataloader builders."""
import os
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GravitySpyDataset(Dataset):
    """Expects `root_dir/<class_name>/<image>.png` layout (as shipped by the
    Gravity Spy Training Set release on Zenodo)."""

    def __init__(self, root_dir: str, classes: list, transform=None):
        self.samples = []
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            folder = os.path.join(root_dir, c)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(folder, fname), self.class_to_idx[c]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under '{root_dir}'. Check that class subfolders "
                f"exist and contain .png/.jpg files."
            )

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(image_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def build_dataloaders(config: dict, classes: list):
    image_size = config["model"]["image_size"]
    train_tf, eval_tf = get_transforms(image_size)

    train_ds = GravitySpyDataset(config["data"]["train_dir"], classes, transform=train_tf)
    val_ds = GravitySpyDataset(config["data"]["val_dir"], classes, transform=eval_tf)
    test_ds = GravitySpyDataset(config["data"]["test_dir"], classes, transform=eval_tf)

    bs = config["train"]["batch_size"]
    nw = config["train"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def compute_class_weights(train_ds: GravitySpyDataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss, to counter class imbalance."""
    label_counts = Counter(label for _, label in train_ds.samples)
    counts = np.array([label_counts.get(i, 1) for i in range(num_classes)], dtype=np.float32)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)