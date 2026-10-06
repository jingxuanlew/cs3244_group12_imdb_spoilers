from sklearn.feature_extraction.text import TfidfVectorizer


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

from sklearn.model_selection import StratifiedGroupKFold


def generate_tfidf_folds(
    reviews,
    k=5,
    text_col="review_text",
    label_col="is_spoiler",
    random_state=42,
    ngram_range=(1, 1),
    min_df=1,
    max_df=1.0,
    max_features=None,
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

    Yields
    ------
    dict
        Dictionary containing the fold number, transformed train/validation
        matrices, labels, and fitted vectorizer.
    """

    if k < 2:
        raise ValueError("k must be at least 2.")

    # StratifiedGroupKFold serves two purposes:
    # 1. Stratification:
    #    tries to preserve the spoiler/non-spoiler class distribution
    #    across the folds
    # 2. Grouping:
    #    keeps rows with identical review_text together so the same exact
    #    review cannot appear in both training and validation within a fold
    sgkf = StratifiedGroupKFold(
        n_splits=k,
        shuffle=True,
        random_state=random_state
    )

    for fold_number, (train_idx, val_idx) in enumerate(
        sgkf.split(
            X=reviews[text_col],
            y=reviews[label_col],
            groups=reviews[text_col],
        ),
        start=1,
    ):
        fold_train = reviews.iloc[train_idx]
        fold_val = reviews.iloc[val_idx]

        vectorizer = create_tfidf_vectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )

        X_train = vectorizer.fit_transform(
            fold_train[text_col]
        )

        X_val = vectorizer.transform(
            fold_val[text_col]
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
        }