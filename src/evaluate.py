"""Evaluate the trained model on the test set: classification report + confusion matrix.

Usage:
    python -m src.evaluate --config configs/config.yaml
"""
import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import build_dataloaders
from src.models import build_single_view_model
from src.utils import load_config, get_device, list_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    checkpoint_path = os.path.join(
        config["output"]["checkpoint_dir"], config["output"]["best_model_name"]
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    print(f"Loaded checkpoint with {len(classes)} classes")

    _, _, test_loader, _, _, test_ds = build_dataloaders(config, classes)

    model = build_single_view_model(
        num_classes=len(classes),
        backbone_name=config["model"]["backbone"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs.to(device))
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
    print(report)

    os.makedirs(config["output"]["outputs_dir"], exist_ok=True)
    report_path = os.path.join(config["output"]["outputs_dir"], "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, xticklabels=classes, yticklabels=classes, cmap="Blues", annot=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(config["output"]["outputs_dir"], "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved confusion matrix to {cm_path}")

    # Also dump per-sample predictions for the benchmarking step
    import pandas as pd
    filenames = [os.path.basename(p) for p, _ in test_ds.samples]
    pred_df = pd.DataFrame({
        "filename": filenames,
        "true_label": [classes[i] for i in all_labels],
        "predicted_label": [classes[i] for i in all_preds],
    })
    preds_path = os.path.join(config["output"]["outputs_dir"], "test_predictions.csv")
    pred_df.to_csv(preds_path, index=False)
    print(f"Saved per-sample predictions to {preds_path}")


if __name__ == "__main__":
    main()