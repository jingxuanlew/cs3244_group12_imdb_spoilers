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
    max_iter=1000, tol=1e-3, verbose=True, feature_sets=None,
):
    """Return (ranked CV summary, per-fold scores, best parameter dictionaries).

    Pass ONLY the saved outer training partition (or a development sample).
    Fold encoders are reused across classifiers, never across folds. 
    Ranking uses mean macro-F1.
    A candidate must converge in every fold.
    Validation predictions use the default 0.5 probability threshold. 
    No ablation or test-file access occurs here. Elastic net uses l1_ratio=0.5.

    feature_sets optionally compares subsets of feature_columns while reusing each fold's encoded matrix. 
    This is valid because the existing encoders process each input field independently. 
    Returned best parameters include the chosen feature list under "features".
    """
    feature_columns = list(feature_columns)
    feature_sets = [feature_columns] if feature_sets is None else [list(s) for s in feature_sets]
    model_configs = list(ParameterGrid(model_grid))
    tfidf_configs = list(ParameterGrid(tfidf_grid))
    rows, configs = [], []
    for tfidf_params in tfidf_configs:
        offset = len(configs)
        configs.extend({"model": params, "tfidf": tfidf_params, "features": subset}
                       for params in model_configs for subset in feature_sets)
        for fold in generate_tfidf_folds(
            training_data, feature_columns=feature_columns, k=k,
            group_by=group_by, random_state=random_state,
            oversample_positive=oversample_positive, remove_spoilers=False,
            **tfidf_params,
        ):
            for model_id, config in enumerate(configs[offset:]):
                model_params = config["model"]
                X_train, X_val = fold["X_train"], fold["X_val"]
                if config["features"] != feature_columns:
                    columns = np.concatenate([
                        np.arange(fold["feature_slices"][name].start,
                                  fold["feature_slices"][name].stop)
                        for name in config["features"]
                    ])
                    X_train, X_val = X_train[:, columns], X_val[:, columns]
                if X_train.shape[1] == 0:
                    rows.append({
                        "candidate": offset + model_id, "fold": fold["fold"],
                        **dict.fromkeys(METRICS, np.nan), "converged": False,
                        "iterations": 0, "seconds": 0.0,
                    })
                    continue
                model = create_logistic_regression(
                    **model_params, random_state=random_state,
                    max_iter=max_iter, tol=tol,
                )
                start = perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X_train, fold["y_train"])
                converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                for warning in caught:
                    if not issubclass(warning.category, ConvergenceWarning):
                        warnings.warn(str(warning.message), warning.category)
                scores = evaluate(
                    fold["y_val"],
                    model.predict(X_val),
                    model.predict_proba(X_val)[:, 1],
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
                    print(f"TF-IDF {tfidf_params.get('ngram_range', (1, 1))}, fold {fold['fold']}/{k}, {model_params}, {len(config['features'])} features: F1={scores['f1']:.4f}, macro F1={scores['macro_f1']:.4f}, converged={converged}", flush=True)

    fold_scores = pd.DataFrame(rows)
    summary = fold_scores.groupby("candidate")[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["converged"] = fold_scores.groupby("candidate")["converged"].all()
    parameters = pd.DataFrame([
        {**config["model"], **config["tfidf"],
         "penalty": config["model"].get("penalty", "l2") or "none",
         "features": config["features"]}
        for config in configs
    ]).rename_axis("candidate")
    summary = parameters.join(summary).sort_values(["converged", "macro_f1_mean"], ascending=False, kind="stable",)
    eligible = summary[summary["converged"]]
    best = configs[int(eligible.index[0])] if not eligible.empty else None
    return summary, fold_scores, best


def backward_elimination(
    training_data, best_params, feature_columns=INITIAL_FEATURES,
    max_macro_f1_drop=0.005, k=5, group_by="review_text", random_state=67,
    oversample_positive=True, max_iter=1000, tol=1e-3, verbose=True,
):
    """Return (selected features, greedy path, all removal trials).

    Freeze model/TF-IDF settings from initial tuning. 
    At each step evaluate every one-block removal and follow the best mean validation macro-F1.
    Continue down to one block (or until no candidate converges), then select the smallest visited subset within max_macro_f1_drop of the best path score.
    The tolerance is absolute, not a per-step allowance or a significance test.
    This greedy path does not search every possible subset.
    """
    current = list(feature_columns)
    if not current or not np.isfinite(max_macro_f1_drop) or not 0 <= max_macro_f1_drop <= 1:
        raise ValueError("Provide features and a finite macro-F1 tolerance between 0 and 1.")
    model_grid = {name: [value] for name, value in best_params["model"].items()}
    tfidf_grid = {name: [value] for name, value in best_params["tfidf"].items()}
    path, trials = [], []
    for step in range(len(current)):
        subsets = [current] if step == 0 else [
            [name for name in current if name != removed] for removed in current
        ]
        summary, _, best = tune_logistic_regression(
            training_data, model_grid, tfidf_grid, feature_columns=current,
            feature_sets=subsets, k=k, group_by=group_by, random_state=random_state,
            oversample_positive=oversample_positive, max_iter=max_iter, tol=tol,
            verbose=False,
        )
        summary["step"] = step
        summary["removed_feature"] = [
            next((name for name in current if name not in subset), None)
            for subset in summary["features"]
        ]
        trials.append(summary.reset_index())
        if best is None:
            if step == 0:
                raise RuntimeError("The all-feature baseline must converge before elimination.")
            warnings.warn("No valid converged removal remains; stopping elimination.")
            break
        winner = summary.loc[summary["converged"]].iloc[0]
        current = best["features"]
        path.append({
            "step": step, "n_features": len(current), "features": current.copy(),
            "removed_feature": winner["removed_feature"],
            "macro_f1_mean": winner["macro_f1_mean"],
            "macro_f1_std": winner["macro_f1_std"],
        })
        if verbose:
            print(f"Step {step}: {len(current)} features, removed={winner['removed_feature']}, "
                  f"macro F1={winner['macro_f1_mean']:.4f}", flush=True)
        if len(current) == 1:
            break

    path = pd.DataFrame(path)
    floor = path["macro_f1_mean"].max() - max_macro_f1_drop
    selected_index = path.index[path["macro_f1_mean"] >= floor][-1]
    path["selected"] = path.index == selected_index
    return list(path.loc[selected_index, "features"]), path, pd.concat(trials, ignore_index=True)


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
