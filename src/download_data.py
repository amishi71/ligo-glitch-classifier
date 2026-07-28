"""Download Gravity Spy data from Zenodo — verified record IDs and filenames.

Run this on Colab, not locally (the training set alone is 3.1-5.5GB).

Usage:
    python src/download_data.py --training-set --ml-labels H1_O3a --volunteer-labels
"""
import argparse
import bz2
import os
import shutil
import subprocess

TRAINING_SET_H5_URL = "https://zenodo.org/records/1486046/files/trainingsetv1d1.h5?download=1"
TRAINING_SET_TARGZ_URL = "https://zenodo.org/records/1486046/files/trainingsetv1d1.tar.gz?download=1"
ML_CLASSIFICATIONS_BASE = "https://zenodo.org/records/5649212/files/{name}.csv?download=1"
VOLUNTEER_CLASSIFICATIONS_URL = "https://zenodo.org/records/13904422/files/classifications.csv.bz2?download=1"

VALID_ML_FILES = [
    "H1_O1", "H1_O2", "H1_O3a", "H1_O3b",
    "L1_O1", "L1_O2", "L1_O3a", "L1_O3b",
]


def _wget(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    subprocess.run(["wget", "-q", "--show-progress", "-O", out_path, url], check=True)


def download_training_set(data_dir="data/raw", fmt="h5"):
    if fmt == "h5":
        out = os.path.join(data_dir, "trainingsetv1d1.h5")
        _wget(TRAINING_SET_H5_URL, out)
    elif fmt == "tar":
        out = os.path.join(data_dir, "trainingsetv1d1.tar.gz")
        _wget(TRAINING_SET_TARGZ_URL, out)
        subprocess.run(["tar", "-xzf", out, "-C", data_dir], check=True)
    else:
        raise ValueError("fmt must be 'h5' or 'tar'")
    return out


def download_ml_labels(names, data_dir="data/labels"):
    paths = []
    for name in names:
        if name not in VALID_ML_FILES:
            raise ValueError(f"{name} is not one of {VALID_ML_FILES}")
        out = os.path.join(data_dir, f"{name}.csv")
        _wget(ML_CLASSIFICATIONS_BASE.format(name=name), out)
        paths.append(out)
    return paths


def download_volunteer_labels(data_dir="data/labels"):
    out_bz2 = os.path.join(data_dir, "classifications.csv.bz2")
    _wget(VOLUNTEER_CLASSIFICATIONS_URL, out_bz2)
    out_csv = out_bz2[:-4]
    with bz2.open(out_bz2, "rb") as src, open(out_csv, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Download Gravity Spy data from Zenodo.")
    parser.add_argument("--training-set", action="store_true", help="download the pre-rendered training set")
    parser.add_argument("--training-set-format", choices=["h5", "tar"], default="h5")
    parser.add_argument("--ml-labels", nargs="*", default=[], help=f"any of {VALID_ML_FILES}")
    parser.add_argument("--volunteer-labels", action="store_true")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    if args.training_set:
        path = download_training_set(os.path.join(args.data_dir, "raw"), args.training_set_format)
        print("Downloaded training set to", path)

    if args.ml_labels:
        paths = download_ml_labels(args.ml_labels, os.path.join(args.data_dir, "labels"))
        print("Downloaded ML labels:", paths)

    if args.volunteer_labels:
        path = download_volunteer_labels(os.path.join(args.data_dir, "labels"))
        print("Downloaded volunteer labels to", path)


if __name__ == "__main__":
    main()
