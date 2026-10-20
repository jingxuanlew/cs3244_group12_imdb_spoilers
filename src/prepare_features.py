"""Join review/movie data and select the initial model features."""

from pathlib import Path
import pandas as pd
from .data_cleaning import clean_movie_ids


INITIAL_FEATURES = [
    "review_text",
    "review_summary",
    "rating",  # Review rating, not the movie's aggregate rating.
    "plot_summary",
    "plot_synopsis",
    "genre",
    "duration",
    "days_since_release"
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def prepare_features(reviews: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Return exactly INITIAL_FEATURES, preserving review row order and index.

    Duration is in minutes. 
    days_since_release is review date minus release date. 
    Invalid/incomplete dates yield missing values, not guessed dates.
    Labels and grouping IDs remain in the original reviews DataFrame.
    """
    review_columns = ["movie_id", "review_date", "review_text", "review_summary", "rating"]
    movie_columns = ["movie_id", "plot_summary", "plot_synopsis", "genre", "duration", "release_date"]
    review_data = clean_movie_ids(reviews[review_columns])
    movie_data = clean_movie_ids(movies[movie_columns])

    if review_data.movie_id.isna().any() or movie_data.movie_id.isna().any():
        raise ValueError("movie_id must not be missing when joining reviews and movies.")

    movie_data["duration"] = (pd.to_timedelta(movie_data["duration"], errors="coerce").dt.total_seconds() / 60)
    movie_data["release_date"] = pd.to_datetime(movie_data["release_date"], format="%Y-%m-%d", errors="coerce")
    joined = review_data.merge(
        movie_data, on="movie_id", how="left", sort=False,
        validate="many_to_one", indicator=True,
    )
    if joined["_merge"].ne("both").any():
        raise ValueError("Some reviews have no matching movie details; check movie_id values.")

    review_dates = pd.to_datetime(joined["review_date"], format="%d %B %Y", errors="coerce")
    joined["days_since_release"] = (review_dates - joined["release_date"]).dt.days
    joined["rating"] = pd.to_numeric(joined["rating"], errors="coerce")
    joined.index = reviews.index
    return joined[INITIAL_FEATURES]
