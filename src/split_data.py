"""Split cleaned reviews 80/20, stratified by is_spoiler, keeping identical
   review_text in one partition. Returns (reviews, train_reviews, test_reviews)."""

import argparse
from pathlib import Path

import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

if __package__:
    from .clean_spoilers import clean_text
else:
    from clean_spoilers import clean_text

RANDOM_STATE = 67
N_SPLITS = 5  # 1 of 5 folds as test -> 80/20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned" / "IMDB_reviews.json"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
TRAIN_PATH = SPLITS_DIR / "IMDB_reviews_train.json"
TEST_PATH = SPLITS_DIR / "IMDB_reviews_test.json"


def get_split_groups(reviews, group_by="review_text"):
    """Keep duplicate texts together, optionally also keeping movies together.

    Use cue-removed text for grouping in BOTH experimental conditions so that
    removing warnings cannot introduce duplicate-text leakage or change folds.
    Model input text is not changed here.
    """
    if group_by not in ("review_text", "movie_id"):
        raise ValueError("group_by must be 'review_text' or 'movie_id'.")
    if reviews["review_text"].isna().any():
        raise ValueError("review_text has missing values; cannot group safely.")
    texts = reviews["review_text"].map(clean_text)
    if group_by == "review_text":
        return texts
    if reviews["movie_id"].isna().any():
        raise ValueError("movie_id has missing values; cannot group safely.")

    # Connect movies sharing a review, including transitive connections.
    movie_codes, movies = pd.factorize(reviews["movie_id"])
    pairs = pd.DataFrame({"text": texts.to_numpy(), "movie": movie_codes})
    repeated = pairs[pairs.duplicated("text", keep=False)]
    anchors = repeated.groupby("text")["movie"].transform("first")
    graph = coo_matrix(
        ([1] * len(repeated), (repeated["movie"].to_numpy(), anchors.to_numpy())),
        shape=(len(movies), len(movies)),
    )
    _, components = connected_components(graph, directed=False)
    return components[movie_codes]


def create_train_test_split(group_by="review_text"):
    """Create and save the shared 80/20 train-test split.

    Reviews identical after cue removal are kept together. With movie_id
    grouping, movies linked by those reviews are also kept together.
    Stratification attempts to keep the spoiler ratios similar.
    """
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"{CLEANED_PATH} does not exist. Run data_cleaning.py first."
        )

    reviews = pd.read_json(CLEANED_PATH, lines=True)
    groups = get_split_groups(reviews, group_by)
    if len(pd.unique(groups)) < N_SPLITS:
        raise ValueError(f"Need at least {N_SPLITS} independent groups.")

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(
        splitter.split(
            X=reviews["review_text"],
            y=reviews["is_spoiler"],
            groups=groups,
        )
    )

    train = reviews.iloc[train_idx].reset_index(drop=True)
    test = reviews.iloc[test_idx].reset_index(drop=True)

    # Safety check: no identical review text in both partitions.
    overlap = set(train["review_text"]) & set(test["review_text"])
    if overlap:
        raise AssertionError(f"{len(overlap)} review texts appear in both splits.")
    if group_by == "movie_id":
        assert not set(train["movie_id"]) & set(test["movie_id"])

    output_dir = SPLITS_DIR / "movie_disjoint" if group_by == "movie_id" else SPLITS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_json(output_dir / TRAIN_PATH.name, orient="records", lines=True, force_ascii=False)
    test.to_json(output_dir / TEST_PATH.name, orient="records", lines=True, force_ascii=False)

    n = len(reviews)
    print(f"Train: {len(train):,} ({len(train) / n:.2%}), "
          f"spoiler rate {train['is_spoiler'].mean():.2%}")
    print(f"Test:  {len(test):,} ({len(test) / n:.2%}), "
          f"spoiler rate {test['is_spoiler'].mean():.2%}")
    return reviews, train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-by", choices=("review_text", "movie_id"), default="review_text")
    create_train_test_split(parser.parse_args().group_by)
