"""Benchmark a trained checkpoint against Gravity Spy ML and volunteer labels.

Joins the model's test-set predictions to the official Gravity Spy label
CSVs on `gravityspy_id` (confirmed join key across the training-set HDF5,
the ML-classification CSVs at https://zenodo.org/records/5649212, and the
volunteer-classification CSV at https://zenodo.org/records/13904422).

Usage (after train.py has produced a checkpoint, and download_data.py has
pulled the label CSVs):
    python src/benchmark.py \
        --checkpoint checkpoints/best_model.pt \
        --ml-labels data/labels/H1_O3a.csv data/labels/H1_O3b.csv \
        --volunteer-labels data/labels/classifications.csv

Either --ml-labels or --volunteer-labels can be omitted if you only have one.
"""
import argparse
import os

import pandas as pd
import torch

from dataset import get_dataloaders
from model import build_model


def load_ml_labels(paths):
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[["gravityspy_id", "ml_label"]].drop_duplicates("gravityspy_id")
    return df


def load_volunteer_labels(path):
    df = pd.read_csv(path)
    # Column name for the consensus label has varied slightly across Gravity
    # Spy Zenodo releases -- try the known candidates rather than hardcoding one.
    candidates = ["label", "ml_label", "final_label", "class"]
    label_col = next((c for c in candidates if c in df.columns), None)
    if label_col is None:
        raise ValueError(
            f"Couldn't find a label column in {path}. Columns present: "
            f"{list(df.columns)}. Update `candidates` in load_volunteer_labels()."
        )
    df = df.rename(columns={label_col: "volunteer_label"})
    return df[["gravityspy_id", "volunteer_label"]].drop_duplicates("gravityspy_id")


def main():
    parser = argparse.ArgumentParser(description="Benchmark predictions against Gravity Spy labels.")
    parser.add_argument("--data-path", default="data/raw/trainingsetv1d1.h5")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--duration", default="1.0", choices=["0.5", "1.0", "2.0", "4.0"])
    parser.add_argument("--split-train", default="train")
    parser.add_argument("--split-val", default="validation")
    parser.add_argument("--split-test", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--ml-labels", nargs="*", default=[], help="path(s) to Gravity Spy ML label CSVs")
    parser.add_argument("--volunteer-labels", default=None, help="path to the volunteer classifications CSV")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    if not args.ml_labels and not args.volunteer_labels:
        raise SystemExit("Pass at least one of --ml-labels or --volunteer-labels.")

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

    all_preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            all_preds.extend(model(x).argmax(1).cpu().numpy())

    # test_ds.index lists (label, gravityspy_id) in the exact order the
    # DataLoader iterated it (shuffle=False for val/test loaders), so we can
    # zip predictions back onto ids positionally.
    true_labels = [label for label, _ in test_ds.index]
    gids = [gid for _, gid in test_ds.index]
    pred_labels = [classes[p] for p in all_preds]

    results = pd.DataFrame({
        "gravityspy_id": gids,
        "true_label": true_labels,
        "model_pred": pred_labels,
    })

    if args.ml_labels:
        ml_df = load_ml_labels(args.ml_labels)
        results = results.merge(ml_df, on="gravityspy_id", how="left")
        matched = results["ml_label"].notna()
        print(f"ML labels matched: {matched.sum()}/{len(results)} test glitches found in the ML CSV(s) you passed")
        if matched.any():
            agree = (results.loc[matched, "model_pred"] == results.loc[matched, "ml_label"]).mean()
            ref_acc = (results.loc[matched, "ml_label"] == results.loc[matched, "true_label"]).mean()
            print(f"  Your model vs Gravity Spy ML label agreement: {agree:.4f}")
            print(f"  Gravity Spy ML label vs training-set ground truth: {ref_acc:.4f}  (sanity check / reference point)")
        else:
            print("  No matches -- likely means the CSV(s) you passed don't cover the detector/run your "
                  "test-set glitches came from. Check which H1_*/L1_* files you downloaded.")

    if args.volunteer_labels:
        vol_df = load_volunteer_labels(args.volunteer_labels)
        results = results.merge(vol_df, on="gravityspy_id", how="left")
        matched = results["volunteer_label"].notna()
        print(f"Volunteer labels matched: {matched.sum()}/{len(results)} test glitches found in the volunteer CSV")
        if matched.any():
            agree = (results.loc[matched, "model_pred"] == results.loc[matched, "volunteer_label"]).mean()
            print(f"  Your model vs volunteer-consensus label agreement: {agree:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "benchmark_results.csv")
    results.to_csv(out_path, index=False)
    print("\nSaved full per-glitch comparison table to", out_path)

    mismatches = results[results["model_pred"] != results["true_label"]]
    print(f"{len(mismatches)} model predictions disagree with the training-set label -- "
          f"inspect these rows first when doing error analysis.")


if __name__ == "__main__":
    main()