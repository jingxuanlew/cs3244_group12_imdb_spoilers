"""Linear SVM: grouped training-only tuning, feature elimination, refitting and
SVM-specific diagnostics (kernel check, margins, weights).

- only the saved TRAINING split is used for any choice; the test split is touched once, at the end;
- folds come from generate_tfidf_folds, so encoders are fitted on training rows only;
- a candidate must converge in every fold to be eligible;
- metrics (precision, recall, F1, macro-F1, average precision).
"""

import warnings
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from .metrics import evaluate
from .prepare_features import INITIAL_FEATURES
from .text_features import (
    create_feature_transformer,
    generate_tfidf_folds,
    oversample_positive_indices,
)

# Accuracy is reported for completeness only. With ~74% non-spoilers, predicting
# "non-spoiler" for everything already scores ~0.74, so it is never used for selection.
METRICS = ["precision", "recall", "f1", "macro_f1", "average_precision", "accuracy"]
SELECTION_METRICS = ["precision", "recall", "f1", "macro_f1", "average_precision"]
IMBALANCE_OPTIONS = ("class_weight", "oversample", "none")


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def create_linear_svm(
    C=0.1, penalty="l2", loss="squared_hinge", class_weight=None,
    random_state=67, max_iter=5000, tol=1e-4,
):
    """Return an unfitted LinearSVC.

    penalty="l2": usual soft-margin SVM (all words keep some weight).
    penalty="l1": pushes unhelpful weights to exactly zero (built-in feature selection).
    loss="hinge": the textbook soft-margin loss max(0, 1 - y f(x)) from the lecture.
    loss="squared_hinge": LinearSVC's default; punishes large violations more and
        is usually faster to optimise.
    scikit-learn supports every combination except L1 + hinge.
    """
    if penalty not in ("l1", "l2"):
        raise ValueError("penalty must be 'l1' or 'l2'.")
    if loss not in ("hinge", "squared_hinge"):
        raise ValueError("loss must be 'hinge' or 'squared_hinge'.")
    if penalty == "l1" and loss == "hinge":
        raise ValueError("L1 penalty with hinge loss is not supported by LinearSVC.")
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be a positive finite number.")
    return LinearSVC(
        C=C, penalty=penalty, loss=loss,
        # Hinge loss only has a dual solver; L1 only has a primal solver.
        dual=True if loss == "hinge" else (False if penalty == "l1" else "auto"),
        class_weight=class_weight, random_state=random_state,
        max_iter=max_iter, tol=tol,
    )


def _imbalance_settings(imbalance):
    """Translate the imbalance option into (class_weight, oversample_positive)."""
    if imbalance not in IMBALANCE_OPTIONS:
        raise ValueError(f"imbalance must be one of {IMBALANCE_OPTIONS}.")
    return ("balanced" if imbalance == "class_weight" else None), imbalance == "oversample"


def _fit_tracking_convergence(model, X, y):
    """Fit and return (converged, seconds). Other warnings are re-raised."""
    start = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    seconds = perf_counter() - start
    converged = True
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            converged = False
        else:
            warnings.warn(str(warning.message), warning.category)
    return converged, seconds


def _scores(model, X, y):
    """Evaluate at the SVM threshold (decision value 0) and rename PR-AUC to AP."""
    decision = model.decision_function(X)
    scores = evaluate(y, (decision > 0).astype(int), decision)
    scores["average_precision"] = scores.pop("pr_auc")
    return scores


# --------------------------------------------------------------------------- #
# SVM diagnostics
# --------------------------------------------------------------------------- #
def margin_diagnostics(model, X, y):
    """Describe the fitted margin on the given (training) rows.

    For a linear SVM f(x) = w.x + b, the margin is the band where |f(x)| <= 1.
    - margin_width = 2 / ||w||: larger means a "wider street" (stronger regularisation).
    - inside_margin: share of rows with y*f(x) < 1. These are the rows that contribute
      hinge loss, i.e. the support vectors of a hinge-loss SVM. Many rows inside the
      margin means the classes overlap heavily.
    - misclassified: share of rows on the wrong side (y*f(x) < 0).
    - nonzero_weights: how many features the model actually uses (smaller with L1).
    """
    w = model.coef_.ravel()
    signed_y = np.where(np.asarray(y) == 1, 1.0, -1.0)
    functional_margin = signed_y * model.decision_function(X)
    norm = float(np.linalg.norm(w))
    return {
        "weight_norm": norm,
        "margin_width": 2.0 / norm if norm > 0 else np.inf,
        "inside_margin": float(np.mean(functional_margin < 1)),
        "misclassified": float(np.mean(functional_margin < 0)),
        "nonzero_weights": int(np.count_nonzero(w)),
        "total_weights": int(w.size),
    }


def feature_names(transformer):
    """Return one readable name per encoded column, e.g. 'review_text: dies'."""
    names = np.full(sum(s.stop - s.start for s in transformer.output_indices_.values()),
                    "", dtype=object)
    for block, encoder, _ in transformer.transformers_:
        block_slice = transformer.output_indices_.get(block, slice(0, 0))
        if block_slice.stop == block_slice.start:
            continue
        if isinstance(encoder, Pipeline) and "tfidfvectorizer" in encoder.named_steps:
            vocabulary = encoder.named_steps["tfidfvectorizer"].get_feature_names_out()
        elif hasattr(encoder, "vocabulary_"):  # binary genre encoder
            vocabulary = encoder.get_feature_names_out()
        else:  # numeric block: one column
            vocabulary = [block]
        names[block_slice] = [f"{block}: {name}" for name in vocabulary]
    return names


def top_features(pipeline, n=20):
    """Return the n strongest spoiler and non-spoiler features of a fitted pipeline.

    Comparing weights across blocks is fair because every block is scaled to a
    similar range (TF-IDF rows are normalised, genres are 0/1, numbers use MaxAbsScaler).
    """
    names = feature_names(pipeline.named_steps["features"])
    weights = pipeline.named_steps["classifier"].coef_.ravel()
    table = pd.DataFrame({
        "feature": names,
        "block": [name.split(": ", 1)[0] for name in names],
        "weight": weights,
    })
    spoiler = table.nlargest(n, "weight").assign(direction="spoiler")
    safe = table.nsmallest(n, "weight").assign(direction="non-spoiler")
    return pd.concat([spoiler, safe], ignore_index=True)


def block_importance(pipeline):
    """Sum of absolute weights per feature block: a rough 'how much is each input used'."""
    names = feature_names(pipeline.named_steps["features"])
    weights = np.abs(pipeline.named_steps["classifier"].coef_.ravel())
    table = pd.DataFrame({"block": [n.split(": ", 1)[0] for n in names], "abs_weight": weights})
    summary = table.groupby("block")["abs_weight"].agg(["sum", "count"])
    summary["share_of_total"] = summary["sum"] / summary["sum"].sum()
    return summary.sort_values("sum", ascending=False)


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #
def _column_subset(fold, subset, all_features):
    """Slice the already-encoded fold matrices down to the blocks in `subset`.

    Valid because each input column is encoded independently, so dropping a
    block's columns equals re-encoding without that block.
    """
    X_train, X_val = fold["X_train"], fold["X_val"]
    if list(subset) == list(all_features):
        return X_train, X_val
    columns = np.concatenate([
        np.arange(fold["feature_slices"][name].start, fold["feature_slices"][name].stop)
        for name in subset
    ]) if subset else np.array([], dtype=int)
    return X_train[:, columns], X_val[:, columns]


def tune_linear_svm(
    training_data, model_grid, tfidf_grid, feature_columns=INITIAL_FEATURES,
    feature_sets=None, k=5, group_by="review_text", random_state=67,
    imbalance="class_weight", remove_spoilers=False, selection_metric="macro_f1",
    max_iter=5000, tol=1e-4, verbose=True,
):
    """Return (ranked CV summary, per-fold scores, best configuration dict).

    Pass ONLY the outer training partition (or a sample of it).
    Each fold's encoders are reused across SVM settings but never across folds.
    Ranking: converged candidates first, then mean validation `selection_metric`.
    Besides validation scores, each row records training macro-F1 (to spot
    over/underfitting) and margin diagnostics on the fold's unique training rows.
    """
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS} "
                         "(accuracy is reported but not allowed for selection).")
    class_weight, oversample = _imbalance_settings(imbalance)
    feature_columns = list(feature_columns)
    feature_sets = [feature_columns] if feature_sets is None else [list(s) for s in feature_sets]
    model_configs = list(ParameterGrid(model_grid))
    configs, rows = [], []

    for tfidf_params in ParameterGrid(tfidf_grid):
        first = len(configs)
        configs += [{"model": m, "tfidf": tfidf_params, "features": s}
                    for m in model_configs for s in feature_sets]
        folds = generate_tfidf_folds(
            training_data, feature_columns=feature_columns, k=k, group_by=group_by,
            random_state=random_state, oversample_positive=oversample,
            remove_spoilers=remove_spoilers, **tfidf_params,
        )
        for fold in folds:
            # Positions of each distinct training row (oversampling repeats rows).
            _, unique_rows = np.unique(fold["train_idx"], return_index=True)
            for candidate in range(first, len(configs)):
                config = configs[candidate]
                X_train, X_val = _column_subset(fold, config["features"], feature_columns)
                row = {"candidate": candidate, "fold": fold["fold"]}
                if X_train.shape[1] == 0:
                    rows.append({**row, **dict.fromkeys(METRICS, np.nan),
                                 "train_macro_f1": np.nan, "converged": False,
                                 "iterations": 0, "seconds": 0.0})
                    continue
                model = create_linear_svm(
                    **config["model"], class_weight=class_weight,
                    random_state=random_state, max_iter=max_iter, tol=tol,
                )
                converged, seconds = _fit_tracking_convergence(model, X_train, fold["y_train"])
                scores = _scores(model, X_val, fold["y_val"])
                X_unique, y_unique = X_train[unique_rows], fold["y_train"][unique_rows]
                train_scores = _scores(model, X_unique, y_unique)
                rows.append({
                    **row, **{m: scores[m] for m in METRICS},
                    "train_macro_f1": train_scores["macro_f1"],
                    **margin_diagnostics(model, X_unique, y_unique),
                    "converged": converged, "iterations": int(np.max(model.n_iter_)),
                    "seconds": seconds,
                })
                if verbose:
                    print(f"TF-IDF {tfidf_params.get('ngram_range', (1, 1))} | fold "
                          f"{fold['fold']}/{k} | {config['model']} | "
                          f"{len(config['features'])} blocks | F1={scores['f1']:.4f} "
                          f"macro-F1={scores['macro_f1']:.4f} | {seconds:.1f}s"
                          + ("" if converged else " | NOT CONVERGED"), flush=True)

    fold_scores = pd.DataFrame(rows)
    numeric = METRICS + ["train_macro_f1"]
    summary = fold_scores.groupby("candidate")[numeric].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    extra = [c for c in ("margin_width", "inside_margin", "nonzero_weights", "seconds")
             if c in fold_scores]
    summary = summary.join(fold_scores.groupby("candidate")[extra].mean())
    summary["converged"] = fold_scores.groupby("candidate")["converged"].all()
    parameters = pd.DataFrame([
        {**c["model"], **c["tfidf"], "features": c["features"], "n_blocks": len(c["features"])}
        for c in configs
    ]).rename_axis("candidate")
    summary = parameters.join(summary).sort_values(
        ["converged", f"{selection_metric}_mean"], ascending=False, kind="stable",
    )
    eligible = summary.index[summary["converged"]]
    best = configs[int(eligible[0])] if len(eligible) else None
    return summary, fold_scores, best


def compare_tfidf_settings(
    training_data, best_config, variants, k=5, group_by="review_text", random_state=67,
    imbalance="class_weight", remove_spoilers=False, selection_metric="macro_f1",
    max_iter=5000, tol=1e-4,
):
    """Change ONE TF-IDF setting at a time, keeping the SVM settings fixed.

    variants: dict of name -> settings to override, e.g.
        {"min_df=2": {"min_df": 2}, "sublinear_tf": {"sublinear_tf": True}}.
    The unchanged best configuration is always included as "current".
    Returns one row per variant with mean/std validation scores and the change
    in the selection metric relative to "current" (same folds for every row).
    """
    model_grid = {name: [value] for name, value in best_config["model"].items()}
    rows = []
    for name, override in {"current": {}, **variants}.items():
        tfidf = {**best_config["tfidf"], **override}
        summary, _, _ = tune_linear_svm(
            training_data, model_grid, {key: [value] for key, value in tfidf.items()},
            feature_columns=best_config["features"], k=k, group_by=group_by,
            random_state=random_state, imbalance=imbalance, remove_spoilers=remove_spoilers,
            selection_metric=selection_metric, max_iter=max_iter, tol=tol, verbose=False,
        )
        result = summary.iloc[0]
        rows.append({"variant": name, "tfidf": tfidf, "converged": bool(result["converged"]),
                     **{f"{m}_{s}": result[f"{m}_{s}"] for m in METRICS for s in ("mean", "std")},
                     "seconds": result["seconds"]})
    table = pd.DataFrame(rows)
    table["change_vs_current"] = (table[f"{selection_metric}_mean"]
                                  - table.loc[0, f"{selection_metric}_mean"])
    return table


# --------------------------------------------------------------------------- #
# Backward elimination
# --------------------------------------------------------------------------- #
def backward_elimination(
    training_data, best_config, feature_columns=INITIAL_FEATURES, max_drop=0.005,
    selection_metric="macro_f1", k=5, group_by="review_text", random_state=67,
    imbalance="class_weight", remove_spoilers=False, max_iter=5000, tol=1e-4,
    verbose=True,
):
    """Return (selected features, greedy path, all removal trials).

    SVM and TF-IDF settings are frozen from initial tuning. At each step, try
    removing every remaining block once and keep the removal with the best mean
    validation score. Continue down to one block. Finally choose the SMALLEST
    visited subset whose score is within `max_drop` of the best score on the path.
    The tolerance is measured from the best score (not per step), so small
    losses cannot pile up unnoticed. Greedy: it does not test every subset.
    """
    if not 0 <= max_drop <= 1:
        raise ValueError("max_drop must be between 0 and 1.")
    current = list(feature_columns)
    if not current:
        raise ValueError("Provide at least one feature block.")
    model_grid = {name: [value] for name, value in best_config["model"].items()}
    tfidf_grid = {name: [value] for name, value in best_config["tfidf"].items()}
    score_column = f"{selection_metric}_mean"
    path, trials = [], []

    for step in range(len(current)):
        subsets = [current] if step == 0 else [[f for f in current if f != r] for r in current]
        summary, _, best = tune_linear_svm(
            training_data, model_grid, tfidf_grid, feature_columns=current,
            feature_sets=subsets, k=k, group_by=group_by, random_state=random_state,
            imbalance=imbalance, remove_spoilers=remove_spoilers,
            selection_metric=selection_metric, max_iter=max_iter, tol=tol, verbose=False,
        )
        summary = summary.assign(step=step, removed_feature=[
            next((f for f in current if f not in subset), None) for subset in summary["features"]
        ])
        trials.append(summary.reset_index())
        if best is None:
            if step == 0:
                raise RuntimeError("The all-feature model must converge before elimination.")
            warnings.warn("No removal converged in every fold; stopping elimination.")
            break
        winner = summary[summary["converged"]].iloc[0]
        current = list(best["features"])
        path.append({
            "step": step, "n_features": len(current), "features": current.copy(),
            "removed_feature": winner["removed_feature"],
            "score_mean": winner[score_column],
            "score_std": winner[f"{selection_metric}_std"],
        })
        if verbose:
            print(f"Step {step}: {len(current)} blocks, removed={winner['removed_feature']}, "
                  f"{selection_metric}={winner[score_column]:.4f}", flush=True)
        if len(current) == 1:
            break

    path = pd.DataFrame(path)
    floor = path["score_mean"].max() - max_drop
    chosen = path.index[path["score_mean"] >= floor][-1]
    path["selected"] = path.index == chosen
    return list(path.loc[chosen, "features"]), path, pd.concat(trials, ignore_index=True)


# --------------------------------------------------------------------------- #
# Final model
# --------------------------------------------------------------------------- #
def refit_linear_svm(
    training_data, best_config, feature_columns=None, random_state=67,
    imbalance="class_weight", remove_spoilers=False, max_iter=5000, tol=1e-4,
):
    """Fit fresh encoders and SVM weights on ALL supplied training rows.

    Returns (fitted Pipeline, info dict). The pipeline takes a prepared-feature
    DataFrame and returns predictions / decision values; it never oversamples.
    info holds convergence and margin diagnostics on the (non-repeated) training rows.
    """
    feature_columns = list(feature_columns or best_config["features"])
    if "is_spoiler" in feature_columns or "movie_id" in feature_columns:
        raise ValueError("Labels and movie IDs must not be model features.")
    class_weight, oversample = _imbalance_settings(imbalance)
    transformer = create_feature_transformer(
        training_data, feature_columns, remove_spoilers=remove_spoilers, **best_config["tfidf"],
    )
    X_train = csr_matrix(transformer.fit_transform(training_data))
    y_train = training_data["is_spoiler"].to_numpy()
    X_fit, y_fit = X_train, y_train
    if oversample:
        rows = oversample_positive_indices(y_train, random_state)
        X_fit, y_fit = X_train[rows], y_train[rows]
    model = create_linear_svm(
        **best_config["model"], class_weight=class_weight,
        random_state=random_state, max_iter=max_iter, tol=tol,
    )
    converged, seconds = _fit_tracking_convergence(model, X_fit, y_fit)
    if not converged:
        warnings.warn("Final LinearSVC did not converge; consider a larger max_iter.",
                      ConvergenceWarning)
    info = {"converged": converged, "seconds": seconds, "training_rows": len(y_train),
            "fitted_rows": len(y_fit), **margin_diagnostics(model, X_train, y_train)}
    return Pipeline([("features", transformer), ("classifier", model)]), info


# --------------------------------------------------------------------------- #
# Kernel vs linear check
# --------------------------------------------------------------------------- #
def compare_kernels(
    training_data, feature_columns=("review_text",), sample_sizes=(2000, 4000, 8000),
    C_values=(0.1, 1.0, 10.0), tfidf_params=None, k=5, group_by="review_text",
    random_state=67, imbalance="class_weight", verbose=True,
):
    """Compare LinearSVC with kernel SVMs (SVC linear / RBF) on small samples.

    For each sample size: draw a random sample of the training partition, take
    the FIRST grouped fold, and fit every model/C on the same matrices.
    Records validation scores, fit/predict time and number of support vectors.
    Used only to justify the model family (all on training data, no test data).
    """
    class_weight, oversample = _imbalance_settings(imbalance)
    tfidf_params = tfidf_params or {}
    rng = np.random.default_rng(random_state)
    rows = []
    for size in sample_sizes:
        picked = np.sort(rng.choice(len(training_data), min(size, len(training_data)), replace=False))
        sample = training_data.iloc[picked].reset_index(drop=True)
        fold = next(generate_tfidf_folds(
            sample, feature_columns=list(feature_columns), k=k, group_by=group_by,
            random_state=random_state, oversample_positive=oversample, **tfidf_params,
        ))
        X_train, X_val = fold["X_train"], fold["X_val"]
        for C in C_values:
            models = {
                "LinearSVC": create_linear_svm(C=C, class_weight=class_weight,
                                               random_state=random_state),
                "SVC (linear kernel)": SVC(kernel="linear", C=C, class_weight=class_weight),
                "SVC (RBF kernel)": SVC(kernel="rbf", C=C, gamma="scale",
                                        class_weight=class_weight),
            }
            for name, model in models.items():
                converged, fit_seconds = _fit_tracking_convergence(model, X_train, fold["y_train"])
                start = perf_counter()
                scores = _scores(model, X_val, fold["y_val"])
                rows.append({
                    "sample_size": len(sample), "train_rows": X_train.shape[0],
                    "n_features": X_train.shape[1], "model": name, "C": C,
                    **{m: scores[m] for m in METRICS},
                    "fit_seconds": fit_seconds, "predict_seconds": perf_counter() - start,
                    "support_vectors": int(model.n_support_.sum()) if hasattr(model, "n_support_") else np.nan,
                    "converged": converged,
                })
                if verbose:
                    print(f"n={len(sample):>6} | {name:<20} C={C:<5} | macro-F1="
                          f"{scores['macro_f1']:.4f} | fit {fit_seconds:.2f}s", flush=True)
    return pd.DataFrame(rows)