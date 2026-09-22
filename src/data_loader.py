from pathlib import Path

import pandas as pd

# Path(__file__).resolve() => /Users/bob/.../cs3244_group12_imdb_spoilers/src/data_loader.py
# .parents[1] gives cs3244_group12_imdb_spoilers
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw" # cs3244_group12_imdb_spoilers/data/raw


def load_reviews() -> pd.DataFrame:
    """Load the raw IMDB reviews dataset."""
    file_path = RAW_DATA_DIRECTORY / "IMDB_reviews.json"

    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} does not exist. "
            "Run `python scripts/download_data.py` first."
        )

    return pd.read_json(
        file_path,
        lines=True,
    )


def load_movie_details() -> pd.DataFrame:
    """Load the raw IMDB movie metadata dataset."""
    file_path = RAW_DATA_DIRECTORY / "IMDB_movie_details.json"

    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} does not exist. "
            "Run `python scripts/download_data.py` first."
        )

    return pd.read_json(
        file_path,
        lines=True,
    )