# Installation and Running of Project

## Pre-Installation
This project uses python 3.13. If you do not already have python installed, please feel free to follow the following instructions

1. (If not already installed) Install [homebrew](https://brew.sh/) using the instructions at the link

2. (Optional) Verify homebrew is installed
```bash
brew --version
```

3. Install python using homebrew
```bash
brew install python@3.13
```

4. (Optional) Verify python 3.13 is installed
```bash
python3.13 --version
```

## First Installation

The packages we used are stored in [requirements.txt](requirements.txt). To install these packages, after cloning this repository, follow the steps:  

1. Create a virtual environment using python 3.13
```bash
python3.13 -m venv .venv
```

2. Activate virtual environment (note: once environment is activated, we use python instead of python3)
```bash
source .venv/bin/activate
```

3.  Install necessary packages based on requirements.txt
```bash
python -m pip install -r requirements.txt
```

4. Configure nbstripout for this repository so that Jupyter Notebook outputs are automatically stripped before being stored in Git (you only need to do this once per local clone of the repository)
```bash
nbstripout --install
```

5. Download dataset from kaggle
```bash
python src/download_data.py
```

6. Clean dataset
```bash
python src/data_cleaning.py
```

7. Create the train/test split
```bash
python src/split_data.py
```

8. Run the majority-class baseline
```bash
python -m src.baseline_majority
```

9. (Optional) Check all the packages you have installed by running
```bash
python -m pip list
```

10. (Optional) Verify that your python is pointing to your virtual environment by running
```bash
which python
```
This should point to something ending with `.venv/bin/python`

## Preparing Initial Features

Join each existing review split to the cleaned movie details:

```python
import pandas as pd
from src.prepare_features import prepare_features, INITIAL_FEATURES

# Paths below assume the repository root as the working directory.
movies = pd.read_json("data/cleaned/IMDB_movie_details.json", lines=True, convert_dates=False)
train_reviews = pd.read_json("data/splits/IMDB_reviews_train.json", lines=True, convert_dates=False)
X_train = prepare_features(train_reviews, movies)
y_train = train_reviews["is_spoiler"]
movie_groups = train_reviews["movie_id"]
```

`X_train` contains only `review_text`, `review_summary`, `rating` (review rating),
`plot_summary`, `plot_synopsis`, `genre`, `duration` (minutes), and
`days_since_release` (review date minus movie release date). Its row order and
index match `train_reviews`; labels and grouping IDs stay separate from features.
Apply the same helper to validation/test reviews without changing their splits.

Incomplete/invalid dates produce missing `days_since_release` values. Negative
values are retained for reviews dated before release. Genre remains a list and
text remains text: encoding, imputation, and scaling still need to be fitted on
training folds only. This step does not fit preprocessing or run feature selection.
Pass the complete prepared DataFrame to the shared fold function:

```python
from src.text_features import generate_tfidf_folds

model_data = X_train.assign(
    is_spoiler=train_reviews["is_spoiler"],
    movie_id=train_reviews["movie_id"],
)
for fold in generate_tfidf_folds(
    model_data, feature_columns=INITIAL_FEATURES, oversample_positive=True,
):
    X_fold_train, X_fold_val = fold["X_train"], fold["X_val"]
    y_fold_train, y_fold_val = fold["y_train"], fold["y_val"]
    # Fit and evaluate your model here.
```

The function applies separate TF-IDF vectorizers to the four text fields,
binary-encodes genres, imputes/scales numeric columns, and combines everything
into sparse matrices. It fits all transformations on training rows only, then
optionally oversamples complete training rows. Labels and movie IDs are never
model features. `max_features` limits each text vocabulary independently.
Without `feature_columns`, the existing review-text-only behaviour is preserved.

For backward elimination, pass a smaller `feature_columns` list but keep the
original grouping columns in `model_data` so folds stay fixed. The returned
`feature_slices` identifies each selected feature's columns, and `transformer`
contains all fitted encoders. For final-model training, build fresh encoders with
`create_feature_transformer(training_data, feature_columns)` from
`src.text_features`, fit them on the full prepared training partition, and only
transform test data. See `notebooks/04_feature_preparation.ipynb` for the full flow.

## Logistic Regression

Run `notebooks/05_logistic_regression.ipynb` after preparing the shared splits.
Sections 1-8 use saved training reviews and cleaned movie metadata, excluding
the test set from model selection and fitting. Existing feature preparation, grouped folds, training-only
oversampling and metrics are reused by `src/logistic_regression.py`.

The initial search uses 10,000 randomly sampled training reviews, five folds,
L1/L2/elastic net with `C = [0.001, 0.01, 0.1, 1]`, one unpenalized model,
and TF-IDF n-gram ranges `(1, 1)`, `(1, 2)`.
Elastic net's mixing ratio is fixed at 0.5.
Each text field is capped at 5,000 features. All eight initial features are used.
Each tuning stage has 26 configurations and 130 fits; weakly regularized sparse models can
take noticeably longer. Change the notebook settings to adjust the grid or
fold count.

Mean validation macro-F1 selects the configuration, not averaged weights.
Precision, recall, positive-class F1 and average precision (AP, not trapezoidal PR-AUC)
are also recorded with fold standard deviations. Candidates that fail to
converge in any fold are excluded; their scores remain visible.
No ablation or threshold tuning is performed.

The notebook now runs three stages: tune all eight feature blocks, perform
greedy backward elimination with those hyperparameters fixed, then retune on
the selected features using the full saved training partition. Initial tuning
and elimination use `sample_size`; sections 7 and 8 always use all training rows.
Each elimination step compares every possible one-block
removal using identical folds. Independent encoders are fitted once per fold
per step and their sparse feature blocks are reused across removal candidates.
The original review text stays available for grouping even if excluded as an input.

The plotted elimination path continues down to one usable feature (or stops if
no removal converges). Select the smallest visited subset within
`max_macro_f1_drop = 0.005` of the best path macro-F1, including the baseline.
This is an absolute 0.5 percentage-point tolerance, not a per-step allowance.
It is a greedy search, not an exhaustive search or a statistical significance test.
The plot shows fold standard deviations, not confidence intervals.
With eight blocks, elimination adds up to 180 fits with five-fold CV.

Fresh encoders and model weights are then fitted on the full training partition.
The notebook saves the fitted pipeline, settings, full-training and selection-row indices
and both CV tables, elimination trials/path and plot under
`data/models/logistic_regression_<grouping>_full_backward/`.
Final tuning and fitting exclude the test split. Full-training CV requires
more time and memory than the sample-based stages. Its folds differ from those
on the sample, so the comparison table reports each stage's scope and row count;
score changes are not a paired estimate of feature-selection improvement.
Set `sample_size = None` only to also use full training data in the early stages.
Validation scores reused for feature selection and tuning are not unbiased final performance;
keep test evaluation for the end, after all modelling choices are frozen.

Section 9 reloads the saved full-training pipeline and predicts the matching
held-out test split in chunks. It does not fit, oversample, ablate or tune
anything. It reports macro-F1, spoiler-class precision/recall/F1, accuracy,
average precision (AP), and the confusion matrix over the complete test set.
It saves `test_predictions.csv`, `test_metrics.csv`, and
`test_confusion_matrix.csv` alongside the model. Do not use test scores for
further model selection.

Run the small synthetic leakage and refitting checks with:

```bash
python -m unittest discover -s tests -v
```

## Movie-Disjoint Evaluation

After cleaning the dataset, create the separate movie-disjoint split:

```bash
python src/split_data.py --group-by movie_id
```

This writes training and test files under `data/splits/movie_disjoint/`.
All reviews of a movie stay together. Movies sharing identical review text,
including text that becomes identical after spoiler-cue removal, stay together
as well. The 80/20 sizes and class proportions are approximate because groups
cannot be divided.

Load the training file from that directory and use the same grouping for CV:

```python
from src.text_features import generate_tfidf_folds

folds = generate_tfidf_folds(train_reviews, group_by="movie_id")
```

The default `python src/split_data.py` still writes the standard review-grouped
split to `data/splits/`. Regenerate existing splits and their majority baseline
after this update: grouping now also accounts for duplicates after cue removal.

## Spoiler-Warning Ablation

Use the same original training DataFrame, row order, grouping, seed, and number
of folds for both conditions:

```python
original_folds = generate_tfidf_folds(train_reviews, k=5)
ablated_folds = generate_tfidf_folds(train_reviews, k=5, remove_spoilers=True)
```

For RQ3, pass `group_by="movie_id"` to both calls. Each generator yields one fold
at a time; `train_idx` and `val_idx` expose the positional row assignments.
Both conditions use identical groups and folds. Only the ablated model inputs
have spoiler cues removed, and TF-IDF is fitted separately within each fold.
Do not resplit a separately cleaned dataset. Freeze `SPOILER_TERMS` before
creating splits and keep it unchanged throughout the comparison.

For multi-feature ablation, pass `feature_columns=INITIAL_FEATURES` and
`remove_spoilers=True` with the original prepared DataFrame. Review text and
review summary are cleaned internally; plot fields are unchanged. For the final
ablated model, pass the same options to `create_feature_transformer`, fit on
training data, and only transform test data. Keep empty reviews so both
conditions evaluate the same examples.

To export a cue-removed copy of the cleaned dataset, run:

```bash
python src/clean_spoilers.py
```

This removes cues from `review_text` and `review_summary` and writes
`data/processed/IMDB_reviews_remove.json`. Labels, metadata, and row order are
preserved. `--input` and `--output` accept other JSON-lines paths, including
individual split files. The script removes cues; it does not insert `[MASK]`.

Edit `SPOILER_TERMS` in `src/clean_spoilers.py` to adjust the phrase list before
splitting. Matching ignores case, uses whole words, and accepts spaces, hyphens,
or underscores between phrase words.

## Training-Only Oversampling

Enable random positive-class oversampling within each training fold:

```python
folds = generate_tfidf_folds(train_reviews, oversample_positive=True)
```

Positive training rows are sampled with replacement until they match the number
of negative rows. Splitting and TF-IDF fitting happen first; validation rows and
TF-IDF weights remain unchanged. Matrices stay sparse, and the existing
`random_state` makes sampling reproducible. `train_idx` includes the repeated
rows and stays aligned with `X_train` and `y_train`. Oversampling is off by default
and can be combined with `group_by` and `remove_spoilers`.

For the final model, fit TF-IDF on the full training partition first, then reuse
the same helper before fitting the classifier:

```python
from src.text_features import oversample_positive_indices

indices = oversample_positive_indices(y_train, random_state=67)
X_train, y_train = X_train[indices], y_train[indices]  # y_train: NumPy array
```

Never apply this helper to validation/test data or before creating CV folds.
Compare validation scores with oversampling on and off; balancing classes does
not guarantee improved performance.

## Adding New Package(s)

If you would like to add a new package e.g. pytest, follow these steps (on a new branch)

1. Activate virtual environment (if not already activated)
```bash
source .venv/bin/activate
```

2.  Install latest packages based on requirements.txt
```bash
python -m pip install -r requirements.txt
```

3. Install new package
```bash
python -m pip install pytest
```

4. Freeze current packages into requirements.txt
```bash
python -m pip freeze > requirements.txt
```

Then create a pull request for approval.

## Running Project Normally

For normal development (post-installation), feel free to just run this before beginning work:

1. Activate virtual environment
```bash
source .venv/bin/activate
```

2. (Optional) If you know new packages have been added by others, run this:
```bash
python -m pip install -r requirements.txt
```
