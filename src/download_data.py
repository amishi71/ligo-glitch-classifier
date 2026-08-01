"""Download Gravity Spy data from Zenodo — verified record IDs and filenames.

Run this on Colab, not locally (the training set alone is 3.1-5.5GB).

Usage:
    python src/download_data.py --training-set --ml-labels H1_O3a --volunteer-labels

Note on the volunteer-labels file: this downloads the *pre-aggregated*
consensus dataset (Zenodo record 5911227, `retired_fulldata_min2_max50_ret0p9.hdf5`),
which already has one row per glitch with a `gravityspy_id` and a `final_label`
(the combined ML + volunteer classification). There is a separate, much larger
Zenodo record (13904422) containing the *raw*, per-vote classification log --
that one has no `gravityspy_id` column at all (it keys on a Zooniverse
`Subject_id` instead) and would need aggregation before it's usable for
benchmarking, so it's intentionally not what this script downloads.
"""
import argparse
import os
import subprocess

TRAINING_SET_H5_URL = "https://zenodo.org/records/1486046/files/trainingsetv1d1.h5?download=1"
TRAINING_SET_TARGZ_URL = "https://zenodo.org/records/1486046/files/trainingsetv1d1.tar.gz?download=1"
ML_CLASSIFICATIONS_BASE = "https://zenodo.org/records/5649212/files/{name}.csv?download=1"
VOLUNTEER_CONSENSUS_URL = (
    "https://zenodo.org/records/5911227/files/retired_fulldata_min2_max50_ret0p9.hdf5?download=1"
)

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
    """Downloads one CSV per requested detector+run.

    The training-set HDF5 doesn't record which detector/run each glitch came
    from, so there's no way to know in advance which of the 8 files will
    actually cover a given test split. Pass names=["all"] to download every
    file and guarantee full coverage; benchmark.py's match-rate printout will
    tell you if a narrower selection missed glitches.
    """
    if names == ["all"]:
        names = VALID_ML_FILES
    paths = []
    for name in names:
        if name not in VALID_ML_FILES:
            raise ValueError(f"{name} is not one of {VALID_ML_FILES}")
        out = os.path.join(data_dir, f"{name}.csv")
        _wget(ML_CLASSIFICATIONS_BASE.format(name=name), out)
        paths.append(out)
    return paths


def download_volunteer_labels(data_dir="data/labels"):
    """Downloads the pre-aggregated volunteer+ML consensus dataset.

    This is a single 1.1GB HDF5 file (not split by detector/run, unlike the
    ML classifications), read with pandas:
        pd.read_hdf(path, key="image_db")
    giving one row per glitch, keyed on `gravityspy_id`, with `final_label`
    as the consensus classification. Requires the `tables` package.
    """
    out = os.path.join(data_dir, "retired_fulldata_min2_max50_ret0p9.hdf5")
    _wget(VOLUNTEER_CONSENSUS_URL, out)
    return out


def main():
    parser = argparse.ArgumentParser(description="Download Gravity Spy data from Zenodo.")
    parser.add_argument("--training-set", action="store_true", help="download the pre-rendered training set")
    parser.add_argument("--training-set-format", choices=["h5", "tar"], default="h5")
    parser.add_argument(
        "--ml-labels", nargs="*", default=[],
        help=f"any of {VALID_ML_FILES}, or 'all' to download every detector/run for full coverage",
    )
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