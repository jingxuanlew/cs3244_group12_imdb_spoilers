from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from .clean_spoilers import clean_text
from .split_data import get_split_groups


def create_tfidf_vectorizer(
    ngram_range=(1, 1),
    min_df=1,
    max_df=1.0,
    max_features=None,
):
    """
    Create a configurable TF-IDF vectorizer.

    Parameters
    ----------
    ngram_range : tuple
        Range of n-grams to include.
        (1, 1) = unigrams only
        (1, 2) = unigrams + bigrams

    min_df : int or float
        Ignore terms that appear in fewer than this number/proportion
        of training documents.

    max_df : int or float
        Ignore terms that appear in more than this number/proportion
        of training documents.

    max_features : int or None
        Optional upper limit on the number of TF-IDF features.

    Returns
    -------
    TfidfVectorizer
        An unfitted sklearn TF-IDF vectorizer.
    """
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )

def generate_tfidf_folds(
    reviews,
    k=5,
    text_col="review_text",
    label_col="is_spoiler",
    random_state=67,
    ngram_range=(1, 1),
    min_df=1,
    max_df=1.0,
    max_features=None,
    group_by="review_text",
    remove_spoilers=False,
):
    """
    Generate stratified cross-validation folds with leakage-safe TF-IDF.

    A fresh TF-IDF vectorizer is fitted only on the training text
    within each fold. The validation text is transformed using the
    vectorizer fitted on that fold's training data.

    Parameters
    ----------
    reviews : pandas.DataFrame
        DataFrame containing review text and spoiler labels.

    k : int
        Number of cross-validation folds. Must be at least 2.

    text_col : str
        Name of the column containing review text.

    label_col : str
        Name of the target label column.

    random_state : int
        Random seed used for reproducible fold generation.

    ngram_range, min_df, max_df, max_features
        TF-IDF configuration passed to create_tfidf_vectorizer().

    group_by : str
        "review_text" for the standard split or "movie_id" for RQ3.
        Must match the outer split strategy.

    remove_spoilers : bool
        Remove warning phrases from model input after assigning folds.
        Pass the same original training DataFrame, k and random_state for
        both conditions; grouping always accounts for cue-removed duplicates.

    Yields
    ------
    dict
        Dictionary containing the fold number, transformed train/validation
        matrices, labels, fitted vectorizer, and positional train/val indices.
    """

    if k < 2:
        raise ValueError("k must be at least 2.")

    grouping_reviews = reviews.assign(review_text=reviews[text_col])
    groups = get_split_groups(grouping_reviews, group_by)
    if len(set(groups)) < k:
        raise ValueError(f"Need at least {k} independent groups.")
    sgkf = StratifiedGroupKFold(
        n_splits=k,
        shuffle=True,
        random_state=random_state
    )

    for fold_number, (train_idx, val_idx) in enumerate(
        sgkf.split(
            X=reviews[text_col],
            y=reviews[label_col],
            groups=groups,
        ),
        start=1,
    ):
        fold_train = reviews.iloc[train_idx]
        fold_val = reviews.iloc[val_idx]
        train_text = fold_train[text_col]
        val_text = fold_val[text_col]
        if remove_spoilers:
            train_text = train_text.map(clean_text)
            val_text = val_text.map(clean_text)

        vectorizer = create_tfidf_vectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )

        X_train = vectorizer.fit_transform(
            train_text
        )

        X_val = vectorizer.transform(
            val_text
        )

        y_train = fold_train[label_col].to_numpy()
        y_val = fold_val[label_col].to_numpy()

        yield {
            "fold": fold_number,
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
            "vectorizer": vectorizer,
            "train_idx": train_idx,
            "val_idx": val_idx,
        }
