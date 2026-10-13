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

For the final ablated model, apply `clean_text` from `src.clean_spoilers` to the
full training text and held-out test text, fit TF-IDF and the classifier on the
cleaned training text, and only transform the cleaned test text. Keep empty
reviews so both conditions evaluate the same examples.

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

indices = oversample_positive_indices(y_train)
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
