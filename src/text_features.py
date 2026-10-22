import numpy as np
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

from .clean_spoilers import clean_text
from .split_data import get_split_groups

TEXT_COLUMNS = ("review_text", "review_summary", "plot_summary", "plot_synopsis")
NUMERIC_COLUMNS = ("rating", "duration", "days_since_release")


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

def _prepare_text(values, remove_spoilers=False):
    values = values.fillna("")
    return values.map(clean_text) if remove_spoilers else values


def _genre_tokens(values):
    return values if isinstance(values, (list, tuple)) else []


def create_feature_transformer(
    training_data,
    feature_columns,
    text_col="review_text",
    remove_spoilers=False,
    ngram_range=(1, 1),
    min_df=1,
    max_df=1.0,
    max_features=None,
):
    """Build an unfitted encoder using only the current training partition.

    Text fields get independent TF-IDF vocabularies.
    Genre is binary encode.
    Numeric fields are median-imputed and scaled. 
    Duration are in minutes, as returned by prepare_features. 
    Empty training text/genre fields are omitted. 
    The returned transformer also handles missing text and ablation when transforming validation or test data.
    """
    transformers = []
    for column in feature_columns:
        if column in (*TEXT_COLUMNS, text_col):
            remove = remove_spoilers and column in (text_col, "review_text", "review_summary")
            vectorizer = create_tfidf_vectorizer(ngram_range, min_df, max_df, max_features)
            analyzer = vectorizer.build_analyzer()
            if any(analyzer(text) for text in _prepare_text(training_data[column], remove)):
                transformers.append((column, make_pipeline(
                    FunctionTransformer(_prepare_text, kw_args={"remove_spoilers": remove}),
                    vectorizer,
                ), column))
        elif column == "genre":
            if any(_genre_tokens(values) for values in training_data[column]):
                transformers.append((column, CountVectorizer(
                    analyzer=_genre_tokens, binary=True,
                ), column))
        else:
            transformers.append((column, make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                MaxAbsScaler(),
            ), [column]))
    return ColumnTransformer(transformers, sparse_threshold=1.0)


def oversample_positive_indices(y_train, random_state):
    """Return training-row indices, repeating class 1 to match class 0.

    Only does oversampling for positive examples, as EDA has shown that the proportion of spoilers is much smaller than non-spoilers.
    """
    y_train = np.asarray(y_train)
    indices = np.arange(len(y_train))
    positive = np.flatnonzero(y_train == 1)
    extra = np.count_nonzero(y_train == 0) - len(positive)
    if extra <= 0:
        return indices
    if not len(positive):
        raise ValueError("Cannot oversample training data with no positive examples.")
    rng = np.random.default_rng(random_state)
    indices = np.concatenate((indices, rng.choice(positive, size=extra, replace=True)))
    rng.shuffle(indices)
    return indices


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
    oversample_positive=False,
    feature_columns=None,
):
    """
    Generate grouped folds with training-only feature encoding and oversampling.

    Each selected text field gets its own TF-IDF vectorizer. Other selected
    fields are encoded and combined into the same sparse matrix. All encoders
    are fitted on the training portion before any oversampling.

    Parameters
    ----------
    reviews : pandas.DataFrame
        DataFrame containing feature columns, original review text and spoiler
        labels. Include movie_id for movie-disjoint grouping. Use prepare_features
        to join movie details and convert numeric fields before calling this.

    k : int
        Number of cross-validation folds. Must be at least 2.

    text_col : str
        Original review text used for grouping and the default review-only model.

    label_col : str
        Name of the target label column.

    random_state : int
        Random seed used for reproducible fold generation.

    ngram_range, min_df, max_df, max_features
        TF-IDF configuration applied independently to each selected text field.

    group_by : str
        "review_text" for the standard split or "movie_id" for RQ3.
        Must match the outer split strategy.

    remove_spoilers : bool
        Remove warning phrases from model input after assigning folds.
        Pass the same original training DataFrame, k and random_state for
        both conditions; grouping always accounts for cue-removed duplicates.

    oversample_positive : bool
        Repeat positive rows after ALL selected features are encoded. Validation
        data is unchanged. Defaults to False.

    feature_columns : list of str or None
        Model inputs to encode, e.g. INITIAL_FEATURES for all eight features.
        Defaults to [text_col] for compatibility with review-only notebooks.
        Keep the original grouping columns even when excluding them as features.

    Yields
    ------
    dict
        Dictionary containing the fold number, transformed train/validation
        matrices, labels, fitted transformer, feature_slices, and positional
        train/val indices. vectorizer is the primary text vectorizer (or None
        if not selected/usable), retained for review-only callers.
        train_idx includes repeated indices when oversampling is enabled,
        matching the rows of X_train and y_train.
    """

    if k < 2:
        raise ValueError("k must be at least 2.")
    feature_columns = [text_col] if feature_columns is None else list(feature_columns)
    if label_col in feature_columns or "movie_id" in feature_columns:
        raise ValueError("Labels and movie IDs must not be model features.")

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

        transformer = create_feature_transformer(
            fold_train,
            feature_columns,
            text_col=text_col,
            remove_spoilers=remove_spoilers,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )

        X_train = csr_matrix(transformer.fit_transform(fold_train))
        X_val = csr_matrix(transformer.transform(fold_val))
        feature_slices = {
            column: transformer.output_indices_.get(column, slice(0, 0))
            for column in feature_columns
        }
        primary = transformer.named_transformers_.get(text_col)
        vectorizer = primary.named_steps["tfidfvectorizer"] if primary is not None else None

        y_train = fold_train[label_col].to_numpy()
        y_val = fold_val[label_col].to_numpy()

        if oversample_positive:
            sampled = oversample_positive_indices(y_train, random_state)
            X_train = X_train[sampled]
            y_train = y_train[sampled]
            train_idx = train_idx[sampled]

        yield {
            "fold": fold_number,
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
            "vectorizer": vectorizer,
            "transformer": transformer,
            "feature_slices": feature_slices,
            "train_idx": train_idx,
            "val_idx": val_idx,
        }
