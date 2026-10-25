"""Grouped, training-only logistic regression tuning and final refitting."""

import warnings
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from .metrics import evaluate
from .prepare_features import INITIAL_FEATURES
from .text_features import (
    create_feature_transformer,
    generate_tfidf_folds,
    oversample_positive_indices,
)

METRICS = ["precision", "recall", "f1", "macro_f1", "average_precision"]


def create_logistic_regression(penalty="l2", C=1.0, random_state=67, max_iter=1000, tol=1e-3):
    """Use the penalty names with the project's sklearn 1.9 parameter API."""
    ratios = {"l2": 0.0, "l1": 1.0, "elasticnet": 0.5, None: 0.0}
    if penalty not in ratios:
        raise ValueError("penalty must be 'l2', 'l1', 'elasticnet', or None.")
    return LogisticRegression(
        l1_ratio=ratios[penalty], C=np.inf if penalty is None else C,
        solver="saga" if penalty in ("l1", "elasticnet") else "lbfgs",
        random_state=random_state, max_iter=max_iter, tol=tol,
    )


def tune_logistic_regression(
    training_data, model_grid, tfidf_grid, feature_columns=INITIAL_FEATURES,
    k=5, group_by="review_text", random_state=67, oversample_positive=True,
    max_iter=1000, tol=1e-3, verbose=True,
):
    """Return (ranked CV summary, per-fold scores, best parameter dictionaries).

    Pass ONLY the saved outer training partition (or a development sample).
    Fold encoders are reused across classifiers, never across folds. 
    Ranking uses mean positive-class F1. 
    A candidate must converge in every fold.
    Validation predictions use the default 0.5 probability threshold. 
    No ablation or test-file access occurs here. Elastic net uses l1_ratio=0.5.
    """
    model_configs = list(ParameterGrid(model_grid))
    tfidf_configs = list(ParameterGrid(tfidf_grid))
    rows, configs = [], []
    for tfidf_params in tfidf_configs:
        offset = len(configs)
        configs.extend({"model": params, "tfidf": tfidf_params}
                       for params in model_configs)
        for fold in generate_tfidf_folds(
            training_data, feature_columns=feature_columns, k=k,
            group_by=group_by, random_state=random_state,
            oversample_positive=oversample_positive, remove_spoilers=False,
            **tfidf_params,
        ):
            for model_id, model_params in enumerate(model_configs):
                model = create_logistic_regression(
                    **model_params, random_state=random_state,
                    max_iter=max_iter, tol=tol,
                )
                start = perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(fold["X_train"], fold["y_train"])
                converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                for warning in caught:
                    if not issubclass(warning.category, ConvergenceWarning):
                        warnings.warn(str(warning.message), warning.category)
                scores = evaluate(
                    fold["y_val"],
                    model.predict(fold["X_val"]),
                    model.predict_proba(fold["X_val"])[:, 1],
                )
                scores["average_precision"] = scores.pop("pr_auc")
                rows.append({
                    "candidate": offset + model_id, 
                    "fold": fold["fold"],
                    **{metric: scores[metric] for metric in METRICS},
                    "converged": converged, 
                    "iterations": int(model.n_iter_.max()),
                    "seconds": perf_counter() - start
                })
                if verbose:
                    print(f"TF-IDF {tfidf_params.get('ngram_range', (1, 1))}, fold {fold['fold']}/{k}, {model_params}: F1={scores['f1']:.4f}, macro F1={scores['macro_f1']:.4f}, converged={converged}", flush=True)

    fold_scores = pd.DataFrame(rows)
    summary = fold_scores.groupby("candidate")[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["converged"] = fold_scores.groupby("candidate")["converged"].all()
    parameters = pd.DataFrame([
        {**config["model"], **config["tfidf"],
         "penalty": config["model"].get("penalty", "l2") or "none"}
        for config in configs
    ]).rename_axis("candidate")
    summary = parameters.join(summary).sort_values(["converged", "macro_f1_mean"], ascending=False, kind="stable",)
    eligible = summary[summary["converged"]]
    best = configs[int(eligible.index[0])] if not eligible.empty else None
    return summary, fold_scores, best


def refit_logistic_regression(
    training_data, best_params, feature_columns=INITIAL_FEATURES,
    random_state=67, oversample_positive=True, max_iter=1000, tol=1e-3,
):
    """Learn fresh encoders and weights on ALL supplied training rows, not test.

    Returns a fitted inference pipeline. 
    CV validation folds rotate through the training partition.
    After tuning, all those rows may be used for refitting.
    """
    if "is_spoiler" in feature_columns or "movie_id" in feature_columns:
        raise ValueError("Labels and movie IDs must not be model features.")
    transformer = create_feature_transformer(training_data, feature_columns, remove_spoilers=False, **best_params["tfidf"])
    X_train = csr_matrix(transformer.fit_transform(training_data))
    y_train = training_data["is_spoiler"].to_numpy()
    if oversample_positive:
        indices = oversample_positive_indices(y_train, random_state)
        X_train, y_train = X_train[indices], y_train[indices]
    model = create_logistic_regression(
        **best_params["model"], random_state=random_state,
        max_iter=max_iter, tol=tol,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X_train, y_train)
    return Pipeline([("features", transformer), ("classifier", model)])
