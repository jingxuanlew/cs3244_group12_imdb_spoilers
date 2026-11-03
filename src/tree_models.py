"""Decision-tree and random-forest experiments for IMDb spoiler reviews.

Uses the project's existing grouped folds, feature encoders, oversampling,
and metrics. The held-out test set is used only by the notebook after fitting.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .metrics import evaluate
from .prepare_features import INITIAL_FEATURES, prepare_features
from .text_features import (
    TEXT_COLUMNS,
    create_feature_transformer,
    generate_tfidf_folds,
    oversample_positive_indices,
)

METRICS = ["accuracy", "precision", "recall", "f1", "macro_f1", "average_precision"]


def load_training_data(train_path, movies_path, sample_size=10000, random_state=67):
    """
    Return dataframe of training data (reads all training reviews if sample_size=None)  

    For initial tuning, sample_size=10000  
    For full-training, sample_size=None
    """

    train_path = Path(train_path)

    # Count no. of training reviews
    with train_path.open(encoding="utf-8") as source:
        total_rows = sum(1 for _ in source)

    selected = None
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive or None")

        # Select random row positions
        selected = np.random.default_rng(random_state).choice(
            total_rows,
            size=min(sample_size, total_rows),
            replace=False, # sample without replacement
        )

    movies = pd.read_json(movies_path, lines=True, convert_dates=False)
    prepared, original_indices = [], []
    with pd.read_json(train_path, lines=True, convert_dates=False, chunksize=10000) as reader:
        for reviews in reader:
            if selected is not None:
                reviews = reviews.loc[reviews.index.isin(selected)]
            if reviews.empty:
                continue
            features = prepare_features(reviews, movies)

            # Retain is_spoiler label and movie_id
            features["is_spoiler"] = reviews["is_spoiler"].to_numpy()
            features["movie_id"] = reviews["movie_id"].to_numpy()
            prepared.append(features)
            original_indices.append(reviews.index.to_numpy())

    if not prepared:
        raise ValueError("No training rows were loaded")
    return (
        pd.concat(prepared, ignore_index=True),
        np.concatenate(original_indices),
        total_rows,
    )


def make_model(kind, params=None, random_state=67, n_jobs=2):
    """Create an unfitted scikit model (majority baseline, decision tree, or random forest)"""

    params = {} if params is None else params
    if kind == "majority":
        return DummyClassifier(strategy="prior")
    
    if kind == "decision_tree":
        return DecisionTreeClassifier(random_state=random_state, **params)
    
    if kind == "random_forest":
        return RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, **params)
    
    raise ValueError(f"Unknown model kind: {kind}")


def tune_tree_model(
    training_data, kind, model_grid, tfidf_params,
    feature_columns=INITIAL_FEATURES, feature_sets=None,
    k=5, group_by="review_text", random_state=67,
    oversample_positive=True, n_jobs=2, report_training=False, verbose=True,
):
    """
    Used to determine which hyperparameter configuration performs best under our cross-validation procedure  

    Used for (both) decision tree and random forest

    Five-fold CV, returns (ranked summary, fold results, winning specification).

    TF-IDF is fitted separately in each training fold. Feature-subset trials
    reuse that fold's encoded columns, so elimination doesn't refit TF-IDF.
    """

    feature_columns = list(feature_columns)
    feature_sets = [feature_columns] if feature_sets is None else [list(s) for s in feature_sets]
    if not feature_columns or {"is_spoiler", "movie_id"} & set(feature_columns):
        raise ValueError("Provide real input features, not labels or movie IDs")
    
    if any(not subset or not set(subset) <= set(feature_columns) for subset in feature_sets):
        raise ValueError("Feature subsets must be nonempty and drawn from feature_columns")

    # Generate candidate configs
    # e.g. tree_grid = { "max_depth": [10, 30], "min_samples_leaf": [5, 20],} => md 10, msl 5 | md 10, msl 20 | ...
    configs = [
        {"model": params, "features": subset}
        for params in ParameterGrid(model_grid)
        for subset in feature_sets
    ]
    rows = []

    # Generate fold, create the grouped train/validation partitions and fit TF-IDF separately within each training fold.
    folds = generate_tfidf_folds(
        training_data,
        k=k,
        group_by=group_by,
        random_state=random_state,
        feature_columns=feature_columns,
        remove_spoilers=False,
        oversample_positive=False,
        **tfidf_params,
    )

    # Prepare training data
    for fold in folds:
        X_original, y_original = fold["X_train"], fold["y_train"] # encoded feature matrix, correct spoiler labels respectively

        if len(np.unique(y_original)) != 2:
            raise ValueError("Each training fold must contain both labels")

        # (Supervised Models) Oversample training rows only, once per fold, and ever oversample the dummy.
        if kind != "majority" and oversample_positive:
            indices = oversample_positive_indices(y_original, random_state)
            X_fit, y_fit = X_original[indices], y_original[indices]
        else:
            X_fit, y_fit = X_original, y_original

        # Tree estimators accept sparse input. Avoid converting TF-IDF to dense.
        if kind != "majority":

            # Store sparse matrix such that column operations are still efficient
            X_fit = X_fit.astype(np.float32).tocsc() # convert to Compressed Sparse Column

        # Loop through each candidate and configuration
        for candidate, config in enumerate(configs):

            # Find numerical columns for this candidate
            columns = []

            for name in config["features"]:
                block = fold["feature_slices"][name]
                columns.extend(range(block.start, block.stop))

            # Check for unusable feature subset (i.e. matrix with 0 features)
            if len(columns) == 0:
                rows.append({
                    "candidate": candidate, "fold": fold["fold"],
                    **dict.fromkeys(METRICS + ["train_macro_f1", "fit_seconds"], np.nan),
                })
                continue

            # Create and train model
            model = make_model(kind, config["model"], random_state, n_jobs)

            start = perf_counter() # used to measure elapsed time
            model.fit(X_fit[:, columns], y_fit)
            fit_seconds = perf_counter() - start

            # Predict on validation set (selects the same numerical columns from the validation matrix, then produces class probabilities)
            # X_fit[:, columns] selects input features for training
            # predict_proba(...)[:, 1] selects the spoiler probability column from the prediction output
            probabilities = model.predict_proba(fold["X_val"][:, columns])[:, 1]

            # Calculate validation metrics
            scores = evaluate(
                fold["y_val"], # Actual labels
                (probabilities > 0.5).astype(int), # Predicted labels
                probabilities # Spoiler probability scores
                )
            scores["average_precision"] = scores.pop("pr_auc")

            train_macro_f1 = np.nan
            if report_training:
                # Calculate training macro-F1
                train_prob = model.predict_proba(X_original[:, columns])[:, 1]
                train_macro_f1 = f1_score(
                    y_original, (train_prob > 0.5).astype(int), average="macro", zero_division=0,
                )

            # Record candidate's results
            rows.append({
                "candidate": candidate, "fold": fold["fold"],
                **{metric: scores[metric] for metric in METRICS},
                "train_macro_f1": train_macro_f1, "fit_seconds": fit_seconds,
            })

            # For progress tracking
            if verbose:
                print(f"{kind}: candidate {candidate + 1}/{len(configs)}, "
                      f"fold {fold['fold']}/{k}, macro-F1={scores['macro_f1']:.4f}", flush=True)

    fold_scores = pd.DataFrame(rows)

    # Group by candidate, calculate mean, standard deviation
    summary = fold_scores.groupby("candidate")[METRICS + ["train_macro_f1", "fit_seconds"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]

    # Check that candidate is eligible (i.e. it worked on every fold)
    summary["valid"] = fold_scores.groupby("candidate")["macro_f1"].count().eq(k) # verify nonmissing macro-F1 values for each candidate == k (5)

    settings = pd.DataFrame([
        {"model_params": c["model"], "features": c["features"], "n_blocks": len(c["features"])}
        for c in configs
    ]).rename_axis("candidate")


    summary = settings.join(summary).sort_values(
        ["valid", "macro_f1_mean"], ascending=False, kind="stable",
    )

    # Select best configuration
    eligible = summary.loc[summary["valid"]]
    if eligible.empty:
        best = None
    else:
        best_candidate = int(eligible.index[0]) # Since we sorted from best to worst, first candidate is best

        best = configs[best_candidate].copy() # Retrieve config of best candidate
        best["tfidf"] = dict(tfidf_params) # Attach TF-IDF settings

    # summary - Average metrics across folds, with configurations ranked | 	Used to compare models
    # fold_scores - Individual results from every candidate/fold | Used for detailed analysis
    # best - Winning configuration and features| Used for next modelling stage
    return summary, fold_scores, best


def backward_elimination(
    training_data,
    kind,
    initial_best, # best from tune_tree_model
    max_macro_f1_drop=0.005, # Accept a feature subset whose macro-F1 >= 0.005 below the best score on the elimination path if it uses fewer features
    k=5,
    group_by="review_text",
    random_state=67,
    oversample_positive=True,
    n_jobs=2,
    verbose=True,
):
    """
    Performs feature selection by repeatedly removing one feature block and checking how the model performs
    """

    # Check that the tolerance is valid
    if not np.isfinite(max_macro_f1_drop) or not 0 <= max_macro_f1_drop <= 1:
        raise ValueError("max_macro_f1_drop must be between 0 and 1")
    
    current = list(initial_best["features"]) # store current features
    fixed_grid = {name: [value] for name, value in initial_best["model"].items()} # fix model's hyperparameters
    path, trials = [], []

    for step in range(len(current)):
        # Generate feature subsets
        if step == 0:
            subsets = [current.copy()]

        else:
            subsets = []

            for removed in current:
                subset = [
                    name
                    for name in current
                    if name != removed
                ]
                subsets.append(subset)

        # Evaluate every subset
        # summary - Results for all feature subsets
        # _	- (Discarded) per-fold results
        # winner - Feature subset with the best eligible mean macro-F1, along with the fixed model settings
        summary, _, winner = tune_tree_model(
            training_data,
            kind,
            fixed_grid, # use the same classifier hyperparameters for every feature-removal trial
            initial_best["tfidf"],
            feature_columns=current, # tells the feature transformer which blocks are available to encode
            feature_sets=subsets, # tells the function which combinations of those blocks to evaluate
            k=k,
            group_by=group_by,
            random_state=random_state,
            oversample_positive=oversample_positive,
            n_jobs=n_jobs,
            verbose=False,
        )

        # Record all attempted removals
        trial = summary.reset_index()
        trial["step"] = step

        removed_features = []
        for subset in trial["features"]:
            removed = None
            for name in current:
                if name not in subset:
                    removed = name
                    break
            removed_features.append(removed)

        trial["removed"] = removed_features
        trials.append(trial) # add complete df of this step's experiments to trials list

        if winner is None:
            if step == 0:
                raise ValueError("The starting feature set did not work in every fold")
            break

        
        removed = next((name for name in current if name not in winner["features"]), None) # Get feature removed by winning candidate
        current = winner["features"]
        score = summary.loc[summary["valid"]].iloc[0] # retrieve the winner's macro-F1

        path.append({
            "step": step, "n_blocks": len(current), "features": current.copy(),
            "removed": removed, "macro_f1_mean": score["macro_f1_mean"],
            "macro_f1_std": score["macro_f1_std"],
        })

        if verbose:
            print(f"{kind}: {len(current)} blocks, removed={removed}, "
                  f"macro-F1={score['macro_f1_mean']:.4f}", flush=True)

        # Stop after reaching 1 feature block
        if len(current) == 1:
            break

    path = pd.DataFrame(path)
    floor = path["macro_f1_mean"].max() - max_macro_f1_drop # Calculate the acceptable score threshold

    # Select the smallest acceptable subset
    selected_row = path.index[path["macro_f1_mean"] >= floor][-1] # last qualifying row => smallest acceptable feature subset along visited path
    path["selected"] = path.index == selected_row

    # Selected features - The final feature-block list
    # path - The chosen sequence of feature removals
    # All trials - Every candidate-removal result across every step
    return list(path.loc[selected_row, "features"]), path, pd.concat(trials, ignore_index=True)


def refit_tree_model(
    training_data, # complete training partition
    kind, # "decision_tree", "random_forest", or "majority"
    best, # Selected hyperparameters, features, and TF-IDF settings
    random_state=67,
    oversample_positive=True, # Whether to balance training examples
    n_jobs=2, # Parallel workers for random forest
):
    """Train the final model based on decided features and hyperparameters"""

    # fit - learns preprocessing information from the training dat e.g. TF-IDF vocabulary, IDF weights
    # transform - converts the data into numerical columns using learned information
    transformer = create_feature_transformer(
        training_data, best["features"], remove_spoilers=False, **best["tfidf"],
    )

    X_train = transformer.fit_transform(training_data)
    y_train = training_data["is_spoiler"].to_numpy()

    # oversample the training data
    if kind != "majority" and oversample_positive:
        indices = oversample_positive_indices(y_train, random_state)
        X_train, y_train = X_train[indices], y_train[indices]

    # Create final classifier
    model = make_model(kind, best["model"], random_state, n_jobs)
    model.fit(X_train, y_train) # Train classifier

    return Pipeline([("features", transformer), ("classifier", model)]) # Return preprocessing + classifier together


def get_feature_names(transformer):
    """
    Recover human-readable feature names from the fitted feature transformer so we can understand the tree's splits and feature importances
    
    We can use those names when plotting a decision tree
    """

    # Find the total number of encoded columns
    n_features = max(block.stop for block in transformer.output_indices_.values())

    names = np.empty(n_features, dtype=object) # Create space for the feature names

    # Loop through fitted encoders
    # name - Name of the original feature block
    # encoder - Fitted transformer for that block
    # columns - Original input column or columns used by the transformer
    for name, encoder, columns in transformer.transformers_:
        block = transformer.output_indices_[name]

        # Check if block produced 0 encoded columns
        if block.start == block.stop:
            continue

        # Recover text feature names
        if name in TEXT_COLUMNS:
            terms = encoder.named_steps["tfidfvectorizer"].get_feature_names_out()
            block_names = [f"{name}: {term}" for term in terms]

        # Recover genre feature names
        elif name == "genre":
            block_names = [f"genre: {term}" for term in encoder.get_feature_names_out()]

        # Recover numerical feature names
        else:
            block_names = [f"{column} (scaled)" for column in columns]

        # Assign names to the correct positions
        names[block] = block_names

    return names # np array with one name for each encoded numerical column
