# LIGO Glitch Classifier

CNN-based classifier for LIGO gravitational-wave detector glitches, benchmarked
against Gravity Spy citizen-science labels. See `project_brief.md` (not included
in this scaffold — keep your original brief alongside this) for the full
scientific background, timeline, and stretch goals.

## Repo structure

```
ligo-glitch-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── dataset.py       # GravitySpyH5Dataset + dataloader helper
│   ├── model.py          # ResNet18 transfer-learning baseline
│   ├── download_data.py  # Zenodo downloader (verified URLs)
│   ├── train.py           # training CLI
│   └── evaluate.py       # evaluation CLI (macro-F1, confusion matrix)
├── notebooks/
│   └── colab_run.ipynb   # thin notebook that drives the scripts above on Colab
├── tests/
│   └── test_dataset.py   # local, no-GPU, no-download sanity test
├── data/          # gitignored — created by download_data.py
├── checkpoints/   # gitignored — created by train.py
└── outputs/       # gitignored — created by evaluate.py
```

## What runs where

**Local (your laptop):** editing code, `git` commits/pushes, and the one test
file (`tests/test_dataset.py`) — it builds a tiny fake HDF5 file in memory, so
it needs no download and no GPU. This is where you catch bugs in the data
pipeline before burning Colab GPU time on them.

**Colab (`notebooks/colab_run.ipynb`):** everything that needs a GPU or the
real dataset — downloading the ~3.1GB training set, training, evaluating.
Nothing here needs authentication as long as the GitHub repo stays public.

Don't try to download the training set or train locally unless your laptop
has a real GPU and you don't mind a multi-GB download — that's what Colab's
free T4 is for.

## Local setup

```bash
git clone https://github.com/amishi71/ligo-glitch-classifier.git
cd ligo-glitch-classifier
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests/                     # should pass in a few seconds, no download needed
```

If `pytest tests/` doesn't pass, don't move on to Colab yet — fix it here first,
it's much faster to iterate on than a Colab session.

When you make changes:

```bash
git add -A
git commit -m "describe the change"
git push
```

## Running on Colab

Open `notebooks/colab_run.ipynb` in Colab (upload it, or open directly from
GitHub via *File > Open notebook > GitHub* and pasting the repo URL). Set
**Runtime > Change runtime type > T4 GPU**, then run the cells in order:

1. **Clone the repo** — pulls whatever you last pushed from local.
2. **Install requirements** — `pip install -r requirements.txt`.
3. **Inspect the `.h5` split names** — the Zenodo description confirms
   `validation` as a literal split name but not the other two; this cell
   prints the real ones so you can confirm `train`/`test` before relying on
   the defaults in `src/dataset.py` and `src/train.py`.
4. **Download data**:
   ```bash
   python src/download_data.py --training-set --ml-labels H1_O3a --volunteer-labels
   ```
5. **Train**:
   ```bash
   python src/train.py --epochs 15
   ```
   Useful flags: `--classes Blip Whistle Chirp` to pick your own subset,
   `--all-classes` for the full 22-class taxonomy, `--duration 2.0` to try a
   different spectrogram window.
6. **Evaluate**:
   ```bash
   python src/evaluate.py
   ```
   Prints accuracy, macro-F1, a full classification report, and saves a
   confusion matrix to `outputs/confusion_matrix.png`.
7. **(Optional) Back up to Drive** — Colab sessions are ephemeral;
   `checkpoints/` and `outputs/` disappear when the runtime recycles unless
   you copy them to Drive or download them.

Colab sessions reset their filesystem between sessions, so you'll re-run
steps 1–2 (and usually 4, unless you cache `data/` on Drive) every time you
come back.

## Data sources (verified against live Zenodo records)

| What | Record | File(s) |
|---|---|---|
| Pre-rendered training images (labeled, pre-split train/val/test) | [1486046](https://zenodo.org/records/1486046) | `trainingsetv1d1.h5` (3.1GB) or `trainingsetv1d1.tar.gz` (5.5GB) |
| ML classifications | [5649212](https://zenodo.org/records/5649212) | one CSV per detector+run, e.g. `H1_O3a.csv`, `L1_O3b.csv` |
| Volunteer classifications | [13904422](https://zenodo.org/records/13904422) | `classifications.csv.bz2` (covers all runs) |

`src/download_data.py` uses these exact URLs — if a download 404s in the
future, it means Zenodo has moved something, not that the script is wrong;
re-check the record page before changing the URL.

## Benchmarking against Gravity Spy

`src/download_data.py --ml-labels ...` and `--volunteer-labels` pull the
comparison data. Join your test-set predictions against those CSVs on
`gravityspy_id` to build the comparison table from the project brief's
evaluation plan — that step isn't automated here yet since it depends on
which run/detector you're comparing against.
