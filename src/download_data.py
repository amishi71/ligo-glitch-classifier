"""Helper for setting up the data/ directory. Downloading the Gravity Spy
Training Set itself is a manual step (Zenodo doesn't have a stable API
endpoint for large asset bundles that's worth scripting around) — this
script just creates the expected folder layout and checks what's missing.

Manual steps:
  1. Go to Zenodo and search "Gravity Spy Training Set" (Bahaadini et al.).
     Download the release archive.
  2. Extract it so you end up with:
       data/train/<class_name>/*.png
       data/validation/<class_name>/*.png
       data/test/<class_name>/*.png
  3. Also from Zenodo, download:
       - "Gravity Spy Machine Learning Classifications ... O1, O2, O3a, O3b"
         (DOI 10.5281/zenodo.5649211) -> save as data/labels/ml_classifications.csv
       - "Gravity Spy Volunteer Classifications ... O1, O2, O3a, O3b"
         (DOI 10.5281/zenodo.13904422) -> save as data/labels/volunteer_classifications.csv

Usage:
    python -m src.download_data --config configs/config.yaml
"""
import argparse
import os

from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    required_dirs = [
        config["data"]["train_dir"],
        config["data"]["val_dir"],
        config["data"]["test_dir"],
        os.path.dirname(config["data"]["ml_labels_csv"]),
    ]
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)

    print("Created expected data directories:")
    for d in required_dirs:
        print(f"  {d}")

    print("\nChecking for data...")
    train_dir = config["data"]["train_dir"]
    if os.path.isdir(train_dir) and len(os.listdir(train_dir)) > 0:
        n_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        print(f"  Found {n_classes} class folders in {train_dir}")
    else:
        print(f"  {train_dir} is empty. See the docstring at the top of this file "
              f"for manual download instructions from Zenodo.")

    for key in ["ml_labels_csv", "volunteer_labels_csv"]:
        path = config["data"][key]
        status = "found" if os.path.exists(path) else "MISSING"
        print(f"  {path}: {status}")


if __name__ == "__main__":
    main()