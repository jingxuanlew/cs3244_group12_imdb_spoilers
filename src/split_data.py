"""Split cleaned reviews 80/20, stratified by is_spoiler, keeping identical
   review_text in one partition. Returns (reviews, train_reviews, test_reviews)."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 42
N_SPLITS = 5  # 1 of 5 folds as test -> 80/20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned" / "IMDB_reviews.json"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
TRAIN_PATH = SPLITS_DIR / "IMDB_reviews_train.json"
TEST_PATH = SPLITS_DIR / "IMDB_reviews_test.json"


def create_train_test_split():
    """Create and save the shared 80/20 train-test split.

    Rows with identical review_text are kept in the same partition
    (grouped), and the spoiler ratio is kept similar in both (stratified).
    """
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"{CLEANED_PATH} does not exist. Run data_cleaning.py first."
        )

    reviews = pd.read_json(CLEANED_PATH, lines=True)
    if reviews["review_text"].isna().any():
        raise ValueError("review_text has missing values; cannot group safely.")

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(
        splitter.split(
            X=reviews["review_text"],
            y=reviews["is_spoiler"],
            groups=reviews["review_text"],
        )
    )

    train = reviews.iloc[train_idx].reset_index(drop=True)
    test = reviews.iloc[test_idx].reset_index(drop=True)

    # Safety check: no identical review text in both partitions.
    overlap = set(train["review_text"]) & set(test["review_text"])
    if overlap:
        raise AssertionError(f"{len(overlap)} review texts appear in both splits.")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_json(TRAIN_PATH, orient="records", lines=True, force_ascii=False)
    test.to_json(TEST_PATH, orient="records", lines=True, force_ascii=False)

    n = len(reviews)
    print(f"Train: {len(train):,} ({len(train) / n:.2%}), "
          f"spoiler rate {train['is_spoiler'].mean():.2%}")
    print(f"Test:  {len(test):,} ({len(test) / n:.2%}), "
          f"spoiler rate {test['is_spoiler'].mean():.2%}")
    return reviews, train, test


if __name__ == "__main__":
    create_train_test_split()