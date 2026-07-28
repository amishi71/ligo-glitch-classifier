"""Evaluate a trained checkpoint: accuracy, macro-F1, confusion matrix.

Run this on Colab, right after train.py. Usage:
    python src/evaluate.py --checkpoint checkpoints/best_model.pt
"""
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from dataset import get_dataloaders
from model import build_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Gravity Spy CNN checkpoint.")
    parser.add_argument("--data-path", default="data/raw/trainingsetv1d1.h5")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--duration", default="1.0", choices=["0.5", "1.0", "2.0", "4.0"])
    parser.add_argument("--split-train", default="train")
    parser.add_argument("--split-val", default="validation")
    parser.add_argument("--split-test", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]

    split_names = {"train": args.split_train, "val": args.split_val, "test": args.split_test}
    _, _, test_loader, _, _, test_ds = get_dataloaders(
        args.data_path, split_names, classes, args.duration, args.batch_size, image_size=args.image_size,
    )

    model = build_model(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y.numpy())

    acc = (np.array(all_preds) == np.array(all_true)).mean()
    macro_f1 = f1_score(all_true, all_preds, average="macro")
    print(f"Test accuracy: {acc:.4f}   Macro-F1: {macro_f1:.4f}")
    print(classification_report(all_true, all_preds, target_names=classes))

    os.makedirs(args.output_dir, exist_ok=True)
    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.colorbar(im)
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    print("Saved confusion matrix to", out_path)


if __name__ == "__main__":
    main()
