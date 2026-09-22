from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
CLEANED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "cleaned"

REVIEWS_FILE = "IMDB_reviews.json"
MOVIES_FILE = "IMDB_movie_details.json"


def clean_movie_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with movie IDs normalized by removing
    surrounding whitespace and trailing '/' characters.
    """
    cleaned = df.copy()

    cleaned["movie_id"] = (
        cleaned["movie_id"]
        .str.strip()
        .str.rstrip("/")
    )

    return cleaned


def clean_data():
    """Load raw data, apply cleaning, and save cleaned datasets."""

    reviews_path = RAW_DATA_DIRECTORY / REVIEWS_FILE
    movies_path = RAW_DATA_DIRECTORY / MOVIES_FILE

    # Check that raw data exists before attempting to clean it.
    if not reviews_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            "Raw IMDB data could not be found in data/raw/. "
            "Run the dataset download script first."
        )

    print("Loading raw data...")

    reviews = pd.read_json(
        reviews_path,
        lines=True,
    )

    movies = pd.read_json(
        movies_path,
        lines=True,
    )

    print("Cleaning movie IDs...")

    reviews_cleaned = clean_movie_ids(reviews)
    movies_cleaned = clean_movie_ids(movies)

    # Create data/cleaned if it does not already exist.
    CLEANED_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    reviews_cleaned.to_json(
        CLEANED_DATA_DIRECTORY / REVIEWS_FILE,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    movies_cleaned.to_json(
        CLEANED_DATA_DIRECTORY / MOVIES_FILE,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print("Cleaning complete.")
    print(f"Cleaned reviews saved to: {CLEANED_DATA_DIRECTORY / REVIEWS_FILE}")
    print(f"Cleaned movie details saved to: {CLEANED_DATA_DIRECTORY / MOVIES_FILE}")


if __name__ == "__main__":
    clean_data()