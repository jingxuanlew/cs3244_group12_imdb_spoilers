"""Baseline 0: majority-class predictor."""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from src.metrics import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned" / "IMDB_reviews.json"
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "IMDB_reviews_train.json"
TEST_PATH = PROJECT_ROOT / "data" / "splits" / "IMDB_reviews_test.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "baseline0_results.csv"
RANDOM_STATE = 42


def load_labels():
    """Return (y_train, y_test, split_name), using the team's shared split if present."""
    if TRAIN_PATH.exists() and TEST_PATH.exists():
        train = pd.read_json(TRAIN_PATH, lines=True)
        test = pd.read_json(TEST_PATH, lines=True)
        return train["is_spoiler"].astype(int), test["is_spoiler"].astype(int), "shared_text_grouped"

    print("WARNING: shared split not found in data/splits/; using a TEMPORARY stratified split.")
    reviews = pd.read_json(CLEANED_PATH, lines=True)
    y = reviews["is_spoiler"].astype(int)
    y_train, y_test = train_test_split(
        y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return y_train, y_test, "temporary_random"


def run_baseline0(y_train, y_test) -> dict:
    # "prior" always predicts the most common class in the training data and gives
    # the class share as its score, so every review gets the same score.
    model = DummyClassifier(strategy="prior")
    model.fit([[0]] * len(y_train), y_train)

    X_test = [[0]] * len(y_test)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return evaluate(y_test, y_pred, y_score)

def main():
    y_train, y_test, split_name = load_labels()

    results = {"model": "majority_baseline", "split": split_name, **run_baseline0(y_train, y_test)}
    results["spoiler_rate_test"] = y_test.mean()

    # Round decimals so the CSV is easy to read
    results = {k: round(v, 4) if isinstance(v, float) else v for k, v in results.items()}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(OUTPUT_PATH, index=False)
    print(pd.Series(results).to_string())

if __name__ == "__main__":
    main()