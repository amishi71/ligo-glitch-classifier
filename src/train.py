"""Train the single-view baseline classifier.

Usage:
    python -m src.train --config configs/config.yaml
"""
import argparse
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import build_dataloaders, compute_class_weights
from src.models import build_single_view_model
from src.utils import load_config, set_seed, get_device, list_classes


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["train"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    classes = list_classes(config["data"]["train_dir"])
    print(f"Found {len(classes)} classes: {classes}")

    train_loader, val_loader, _, train_ds, _, _ = build_dataloaders(config, classes)

    model = build_single_view_model(
        num_classes=len(classes),
        backbone_name=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
    ).to(device)

    if config["train"]["use_class_weights"]:
        weights = compute_class_weights(train_ds, len(classes)).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config["train"]["scheduler_patience"],
        factor=config["train"]["scheduler_factor"],
    )

    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(
        config["output"]["checkpoint_dir"], config["output"]["best_model_name"]
    )

    best_val_acc = 0.0
    for epoch in range(config["train"]["epochs"]):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{config['train']['epochs']}: "
            f"train_loss={train_loss:.3f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(), "classes": classes},
                checkpoint_path,
            )
            print(f"  -> new best val_acc={val_acc:.3f}, saved to {checkpoint_path}")

    print(f"Training done. Best val_acc={best_val_acc:.3f}. Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()