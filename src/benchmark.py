"""Benchmark a trained checkpoint against Gravity Spy ML and volunteer labels.

Joins the model's test-set predictions to the official Gravity Spy labels on
`gravityspy_id`:
  - ML-classification CSVs, one per detector+run: https://zenodo.org/records/5649212
  - Volunteer/ML consensus HDF5 (single file, `final_label` column):
    https://zenodo.org/records/5911227

Note: there is a *different* Zenodo record (13904422) that also has "volunteer
classifications" in its title, but it's a raw per-vote export keyed on a
Zooniverse `Subject_id`, not `gravityspy_id` -- it is not usable here without
a separate aggregation step, so this script expects the 5911227 file instead.
`download_data.py --volunteer-labels` already downloads the correct one.

Usage (after train.py has produced a checkpoint, and download_data.py has
pulled the label files):
    python src/benchmark.py \
        --checkpoint checkpoints/best_model.pt \
        --ml-labels data/labels/H1_O3a.csv data/labels/H1_O3b.csv \
        --volunteer-labels data/labels/retired_fulldata_min2_max50_ret0p9.hdf5

Either --ml-labels or --volunteer-labels can be omitted if you only have one.
Reading the volunteer HDF5 requires the `tables` package (pip install tables).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import tables
import torch

from dataset import get_dataloaders
from model import build_model


def load_ml_labels(paths):
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[["gravityspy_id", "ml_label"]].drop_duplicates("gravityspy_id")
    return df


def _decode_if_bytes(arr):
    """String columns round-tripped through a pickled numpy object block can
    come back as either `str` or `bytes` elements depending on how they were
    originally written; normalize to `str`."""
    arr = np.asarray(arr, dtype=object)
    if arr.size and isinstance(arr.flat[0], bytes):
        return np.array([x.decode() for x in arr])
    return arr


def _read_fixed_format_columns(path, group, columns):
    """Reads a set of named columns out of a pandas 'fixed'-format HDF5
    store, without loading the (much larger) numeric blocks.

    Background: pandas' 'fixed' HDF5 format groups a DataFrame's columns
    into blocks by dtype. Numeric blocks land as plain dense arrays and are
    cheap to read with either h5py or PyTables. Object/string blocks are
    different: pandas pickles the *entire* block (every string column,
    every row) as a single object and writes it as one row of a PyTables
    `VLArray` with `ObjectAtom`.

    Reading that block via PyTables (as this function does) hands back the
    already-deserialized array directly -- PyTables wrote it, so PyTables
    reads it back correctly. Reading it via raw h5py instead only gives you
    the opaque serialized bytes, and reconstructing those by hand (manual
    `pickle.loads`, guessing at `.tobytes()` and array orientation) means
    guessing at an internal, version-dependent pandas/PyTables encoding --
    which is what was hanging/misbehaving before this fix.

    Reads every requested column in a single pass over the file (no
    re-opening or re-unpickling the same block once per column).

    Returns a dict {column_name: 1-D array}.
    """
    remaining = set(columns)
    found = {}
    with tables.open_file(path, mode="r") as h5file:
        group_node = h5file.get_node("/" + group)
        i = 0
        while remaining and hasattr(group_node, f"block{i}_items"):
            items = [c.decode() if isinstance(c, bytes) else c
                     for c in getattr(group_node, f"block{i}_items")[:]]
            wanted_here = remaining & set(items)
            if wanted_here:
                values_node = getattr(group_node, f"block{i}_values")
                block = values_node[:]
                if block.dtype == object and block.shape[0] == 1:
                    # whole block was pickled together as one VLArray row;
                    # PyTables has already unpickled it for us here
                    block = np.asarray(block[0])
                for col in wanted_here:
                    col_idx = items.index(col)
                    if block.ndim == 1:
                        col_vals = block
                    elif block.shape[1] == len(items):
                        col_vals = block[:, col_idx]
                    elif block.shape[0] == len(items):
                        col_vals = block[col_idx]
                    else:
                        raise ValueError(
                            f"Unexpected block shape {block.shape} for items {items}"
                        )
                    found[col] = _decode_if_bytes(col_vals)
                remaining -= wanted_here
            i += 1
    if remaining:
        raise KeyError(f"Column(s) {remaining} not found in {group!r} of {path}")
    return found


def load_volunteer_labels(path):
    """Loads the pre-aggregated volunteer+ML consensus dataset.

    Expects the HDF5 file from https://zenodo.org/records/5911227
    (`retired_fulldata_min2_max50_ret0p9.hdf5`), which has one row per glitch
    with `gravityspy_id` and `final_label` columns already computed -- no
    per-vote aggregation needed on our end.

    The file is ~1GB, mostly from ~20 per-class ML confidence columns we
    don't need. A plain `pd.read_hdf(path)` materializes the whole table --
    every column, every row -- in memory before we get to subset it, which
    is exactly the kind of spike that gets a process silently OOM-killed on
    a constrained runtime like Colab's free tier (no Python traceback, the
    process just disappears -- indistinguishable from a hang from the
    outside). So this reads only the two needed columns directly.

    If the file is in pytables 'table' format, `HDFStore.select(...,
    chunksize=...)` streams it row-chunk by row-chunk with `columns=`
    dropping everything else before each chunk is held in memory. If it's
    in 'fixed' format (no chunked-read support at all), falls back to
    `_read_fixed_format_columns`, which reads only the two needed columns
    via PyTables directly rather than the whole table.
    """
    size_gb = os.path.getsize(path) / 1e9
    print(f"Loading volunteer consensus file ({size_gb:.2f} GB)...")
    t0 = time.time()

    with pd.HDFStore(path, mode="r") as store:
        storer = store.get_storer("image_db")
        is_table = storer.is_table
        group_name = storer.group._v_pathname.lstrip("/")

    if is_table:
        chunksize = 20000
        print(f"  Table format detected -- streaming in chunks of {chunksize}...")
        chunks = []
        with pd.HDFStore(path, mode="r") as store:
            for i, chunk in enumerate(
                store.select("image_db", columns=["gravityspy_id", "final_label"], chunksize=chunksize)
            ):
                chunks.append(chunk)
                print(f"  ...read chunk {i + 1} ({sum(len(c) for c in chunks)} rows so far, "
                      f"{time.time() - t0:.0f}s elapsed)")
        df = pd.concat(chunks, ignore_index=True)
    else:
        print("  Fixed format detected -- reading only the gravityspy_id and final_label "
              "columns directly (via PyTables), skipping the wide numeric blocks that make "
              "up most of the file.")
        cols = _read_fixed_format_columns(path, group_name, ["gravityspy_id", "final_label"])
        print(f"  ...read {len(cols['gravityspy_id'])} rows in {time.time() - t0:.1f}s, "
              "building DataFrame...")
        df = pd.DataFrame({"gravityspy_id": cols["gravityspy_id"], "final_label": cols["final_label"]})

    print(f"Loaded {len(df)} rows in {time.time() - t0:.1f}s")
    df = df.rename(columns={"final_label": "volunteer_label"})
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
    parser.add_argument("--volunteer-labels", default=None, help="path to the volunteer consensus HDF5")
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
        print(f"Volunteer labels matched: {matched.sum()}/{len(results)} test glitches found in the volunteer file")
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