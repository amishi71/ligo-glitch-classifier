"""Train the baseline single-view CNN glitch classifier.

Run this on Colab (GPU). Usage, from the repo root, after downloading data:
    python src/train.py --epochs 15
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from dataset import DEFAULT_CLASSES, class_distribution, get_dataloaders
from model import build_model


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser(description="Train the baseline Gravity Spy CNN.")
    parser.add_argument("--data-path", default="data/raw/trainingsetv1d1.h5")
    parser.add_argument("--classes", nargs="*", default=None, help="omit for the default 8-class subset")
    parser.add_argument("--all-classes", action="store_true", help="use all 22 classes in the file")
    parser.add_argument("--duration", default="1.0", choices=["0.5", "1.0", "2.0", "4.0"])
    parser.add_argument("--split-train", default="train")
    parser.add_argument("--split-val", default="validation")
    parser.add_argument("--split-test", default="test")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    if args.all_classes:
        classes = None
    else:
        classes = args.classes if args.classes else DEFAULT_CLASSES

    split_names = {"train": args.split_train, "val": args.split_val, "test": args.split_test}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cpu":
        print("WARNING: no GPU detected. On Colab: Runtime > Change runtime type > T4 GPU.")

    train_loader, val_loader, _, train_ds, _, _ = get_dataloaders(
        args.data_path, split_names, classes, args.duration, args.batch_size, image_size=args.image_size,
    )
    print(f"train: {len(train_ds)} samples across {len(train_ds.classes)} classes: {train_ds.classes}")

    counts = class_distribution(train_ds)
    weights = np.array([counts[c] for c in train_ds.classes], dtype=np.float32)
    class_weights = torch.tensor(weights.sum() / (len(weights) * weights), dtype=torch.float32).to(device)

    model = build_model(len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={tr_loss:.3f} train_acc={tr_acc:.3f}  "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save({"model_state": model.state_dict(), "classes": train_ds.classes}, ckpt_path)
            print("  -> saved new best checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()
