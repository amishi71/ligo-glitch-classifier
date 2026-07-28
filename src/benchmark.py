"""Compare your model's test-set predictions against Gravity Spy's published
ML labels and volunteer consensus labels.

Assumes:
  - `outputs/test_predictions.csv` exists (produced by evaluate.py) with a
    `filename` column that contains or embeds the Gravity Spy `gravityspy_id`.
  - The Gravity Spy CSVs downloaded from Zenodo are at the paths given in
    configs/config.yaml (data.ml_labels_csv / data.volunteer_labels_csv).

NOTE: filename-to-gravityspy_id matching depends on how your image files are
named. Gravity Spy training-set filenames typically embed the ID directly
(e.g. "<gravityspy_id>_1.0.png"). Adjust `extract_id()` below to match your
actual filenames — check a few filenames first with:
    ls data/test/<any_class>/ | head

Usage:
    python -m src.benchmark --config configs/config.yaml
"""
import argparse
import os
import re

import pandas as pd

from src.utils import load_config


def extract_id(filename: str) -> str:
    """Best-effort extraction of the gravityspy_id from an image filename.
    Adjust this regex to match your actual dataset's naming convention."""
    match = re.match(r"([A-Za-z0-9]+)_", filename)
    return match.group(1) if match else filename.split(".")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    preds_path = os.path.join(config["output"]["outputs_dir"], "test_predictions.csv")
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"{preds_path} not found — run `python -m src.evaluate` first.")

    preds = pd.read_csv(preds_path)
    preds["gravityspy_id"] = preds["filename"].apply(extract_id)

    ml_path = config["data"]["ml_labels_csv"]
    volunteer_path = config["data"]["volunteer_labels_csv"]

    if not os.path.exists(ml_path) or not os.path.exists(volunteer_path):
        print(
            "Gravity Spy label CSVs not found at the configured paths.\n"
            f"  ml_labels_csv: {ml_path}\n"
            f"  volunteer_labels_csv: {volunteer_path}\n"
            "Download them from Zenodo (see README) and place them there."
        )
        return

    ml_labels = pd.read_csv(ml_path)
    volunteer_labels = pd.read_csv(volunteer_path)

    comparison = preds.merge(
        ml_labels[["gravityspy_id", "ml_label"]], on="gravityspy_id", how="left"
    ).merge(
        volunteer_labels[["gravityspy_id", "label"]].rename(columns={"label": "volunteer_label"}),
        on="gravityspy_id", how="left",
    )

    matched = comparison.dropna(subset=["ml_label", "volunteer_label"])
    print(f"Matched {len(matched)} / {len(comparison)} test predictions to Gravity Spy records.")

    if len(matched) == 0:
        print("No matches found — check `extract_id()` against your actual filenames.")
        return

    agreement_with_ml = (matched["predicted_label"] == matched["ml_label"]).mean()
    agreement_with_volunteers = (matched["predicted_label"] == matched["volunteer_label"]).mean()

    print(f"Agreement with Gravity Spy ML labels:      {agreement_with_ml:.3f}")
    print(f"Agreement with Gravity Spy volunteer labels: {agreement_with_volunteers:.3f}")

    disagreements = matched[matched["predicted_label"] != matched["volunteer_label"]]
    out_path = os.path.join(config["output"]["outputs_dir"], "disagreements_for_review.csv")
    disagreements.to_csv(out_path, index=False)
    print(f"Saved {len(disagreements)} disagreement cases to {out_path} for manual review.")


if __name__ == "__main__":
    main()