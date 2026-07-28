# LIGO Glitch Classifier

A spectrogram-based CNN classifier for LIGO gravitational-wave detector glitches, benchmarked against the [Gravity Spy](https://www.zooniverse.org/projects/zooniverse/gravity-spy) citizen-science labels.

LIGO's interferometers are sensitive enough to pick up large amounts of non-astrophysical noise ("glitches") from sources like scattered light, seismic activity, and electronics. This project trains a CNN on Omega-scan spectrogram images to classify glitches into the Gravity Spy taxonomy (22 classes), then compares model predictions against Gravity Spy's own published machine-learning labels and human volunteer consensus labels.

## Project structure

```
ligo-glitch-classifier/
├── configs/
│   └── config.yaml           # paths + hyperparameters
├── data/
│   ├── train/ validation/ test/   # class-labeled spectrogram images
│   └── labels/                    # Gravity Spy ML + volunteer label CSVs
├── src/
│   ├── dataset.py             # Dataset / DataLoader construction
│   ├── models.py              # single-view + multi-view CNN architectures
│   ├── train.py                # training loop
│   ├── evaluate.py            # test-set evaluation, confusion matrix
│   ├── benchmark.py           # compare predictions vs Gravity Spy labels
│   ├── gradcam_demo.py        # Grad-CAM interpretability check (stretch)
│   ├── download_data.py       # data folder setup + download instructions
│   └── utils.py                # config loading, seeding, device selection
├── checkpoints/                # saved model weights (gitignored)
├── outputs/                    # generated reports, plots, predictions (gitignored)
├── notebooks/                  # exploratory analysis
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the install:
```bash
python3 -c "import torch, gwpy; print('Torch:', torch.__version__); print('GWpy:', gwpy.__version__)"
```

## Getting the data

```bash
python -m src.download_data
```

This creates the expected `data/` folder layout and tells you what's missing. You then need to manually download (Zenodo, free/open access):

1. **Gravity Spy Training Set** — pre-rendered spectrogram images, sorted by class. Extract into `data/train/`, `data/validation/`, `data/test/`.
2. **Gravity Spy ML Classifications (O1–O3b)** — DOI `10.5281/zenodo.5649211`. Save as `data/labels/ml_classifications.csv`.
3. **Gravity Spy Volunteer Classifications (O1–O3b)** — DOI `10.5281/zenodo.13904422`. Save as `data/labels/volunteer_classifications.csv`.

## Usage

**Train the baseline model:**
```bash
python -m src.train --config configs/config.yaml
```
Saves the best checkpoint to `checkpoints/best_model.pt`.

**Evaluate on the test set:**
```bash
python -m src.evaluate --config configs/config.yaml
```
Produces `outputs/classification_report.txt`, `outputs/confusion_matrix.png`, and `outputs/test_predictions.csv`.

**Benchmark against Gravity Spy labels:**
```bash
python -m src.benchmark --config configs/config.yaml
```
Reports agreement rate with Gravity Spy's ML and volunteer labels, and saves disagreement cases to `outputs/disagreements_for_review.csv` for manual inspection.

**(Stretch) Grad-CAM interpretability check:**
```bash
python -m src.gradcam_demo --config configs/config.yaml --index 0
```

## Results

_Fill in after training:_

| Model                             | Accuracy | Macro-F1 | Agreement w/ GSpy ML | Agreement w/ Volunteers |
| --------------------------------- | -------- | -------- | -------------------- | ----------------------- |
| Gravity Spy published ML CNN      | ~97–98%  | —        | —                    | —                       |
| This model (single-view ResNet18) | TBD      | TBD      | TBD                  | TBD                     |

## References

- Bahaadini, S. et al. (2018), *Machine learning for Gravity Spy: Glitch classification and dataset*.
- Zevin, M. et al. (2017), *Gravity Spy: Integrating Advanced LIGO Detector Characterization, Machine Learning, and Citizen Science*.
- Glanzer, J. et al. (2023), *Data quality up to the third observing run of Advanced LIGO: Gravity Spy glitch classifications*.
- Data and labels: [Zenodo Gravity Spy releases](https://zenodo.org).

## License

Add a license of your choice (MIT is a common default for research/portfolio code).