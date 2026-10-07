from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "cleaned" / "IMDB_reviews.json"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "splits"
TRAIN_PATH = OUTPUT_DIRECTORY / "IMDB_reviews_train.json"
TEST_PATH = OUTPUT_DIRECTORY / "IMDB_reviews_test.json"


def create_train_test_split():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} does not exist. Run `python src/data_cleaning.py` first."
        )

    reviews = pd.read_json(INPUT_PATH, lines=True)
    required_columns = {"review_text", "is_spoiler", "movie_id"}
    missing_columns = required_columns.difference(reviews.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if reviews["review_text"].isna().any():
        raise ValueError("review_text contains missing values; cannot form leakage-safe groups.")

    # One row per exact review-text group. Each group must stay in one split.
    group_summary = (
        reviews.groupby("review_text", sort=False, dropna=False)
        .agg(
            group_size=("review_text", "size"),
            unique_labels=("is_spoiler", "nunique"),
        )
        .reset_index()
    )
    group_summary["group_label"] = (
        reviews.groupby("review_text", sort=False, dropna=False)["is_spoiler"]
        .first()
        .to_numpy()
    )

    stratifiable_groups = group_summary[group_summary["unique_labels"] == 1].copy()
    conflicting_groups = group_summary[group_summary["unique_labels"] > 1].copy()

    train_group_text, test_group_text = train_test_split(
        stratifiable_groups["review_text"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratifiable_groups["group_label"],
    )

    # Assign conflicting-label groups as indivisible groups to the smaller partition.
    train_group_text = list(train_group_text)
    test_group_text = list(test_group_text)
    group_sizes = stratifiable_groups.set_index("review_text")["group_size"]
    train_rows = int(group_sizes.loc[train_group_text].sum())
    test_rows = int(group_sizes.loc[test_group_text].sum())
    conflicting_groups = conflicting_groups.sample(frac=1, random_state=RANDOM_STATE)

    for _, group in conflicting_groups.iterrows():
        if train_rows <= test_rows:
            train_group_text.append(group["review_text"])
            train_rows += int(group["group_size"])
        else:
            test_group_text.append(group["review_text"])
            test_rows += int(group["group_size"])

    train_text = set(train_group_text)
    test_text = set(test_group_text)
    train_reviews = reviews[reviews["review_text"].isin(train_text)].copy()
    test_reviews = reviews[reviews["review_text"].isin(test_text)].copy()

    if set(train_reviews.index).intersection(test_reviews.index):
        raise AssertionError("A row appears in both train and test partitions.")
    if set(train_reviews["review_text"]).intersection(test_reviews["review_text"]):
        raise AssertionError("An exact review_text value appears in both partitions.")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    train_reviews.to_json(TRAIN_PATH, orient="records", lines=True, force_ascii=False)
    test_reviews.to_json(TEST_PATH, orient="records", lines=True, force_ascii=False)

    print(f"Total reviews: {len(reviews):,}")
    print(f"Training reviews: {len(train_reviews):,} ({len(train_reviews) / len(reviews):.2%})")
    print(f"Test reviews: {len(test_reviews):,} ({len(test_reviews) / len(reviews):.2%})")
    print(f"Conflicting-label text groups kept intact: {len(conflicting_groups):,}")
    print(f"Saved training data to: {TRAIN_PATH}")
    print(f"Saved test data to: {TEST_PATH}")
    return reviews, train_reviews, test_reviews


if __name__ == "__main__":
    create_train_test_split()