"""Small synthetic checks; no dataset files are opened."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src import logistic_regression as lr
from src.prepare_features import INITIAL_FEATURES
from src.split_data import get_split_groups
from src.text_features import generate_tfidf_folds


class LogisticRegressionTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "review_text": [f"SPOILERS common uniqueword{i}" for i in range(60)],
            "review_summary": ["short summary"] * 60,
            "plot_summary": [f"movie story plotword{i}" for i in range(60)],
            "plot_synopsis": [""] * 60,
            "genre": [[f"genre{i}"] for i in range(60)],
            "rating": [np.nan if i % 7 == 0 else i % 10 for i in range(60)],
            "duration": np.arange(60) + 60,
            "days_since_release": np.arange(60),
            "is_spoiler": [int(i % 4 == 0) for i in range(60)],
            "movie_id": [f"movie{i}" for i in range(60)],
        })
        # A duplicate with a conflicting label must still stay in one fold.
        duplicate = self.data.iloc[[0]].assign(is_spoiler=0)
        self.data = pd.concat([self.data, duplicate], ignore_index=True)
        self.model_grid = {"penalty": ["l2"], "C": [0.1, 1.0]}
        self.tfidf_grid = {"ngram_range": [(1, 1), (1, 2)], "min_df": [1]}

    def test_penalty_mapping(self):
        for penalty, ratio in [("l2", 0), ("l1", 1), ("elasticnet", 0.5), (None, 0)]:
            model = lr.create_logistic_regression(penalty=penalty, C=0.1)
            self.assertEqual(model.l1_ratio, ratio)
            self.assertEqual(model.C, np.inf if penalty is None else 0.1)
        with self.assertRaises(ValueError):
            lr.create_logistic_regression(penalty="invalid")

    def test_tuning_keeps_folds_and_fitted_features_isolated(self):
        assignments = []
        groups = get_split_groups(self.data).to_numpy()

        def checked_folds(*args, **kwargs):
            for fold in generate_tfidf_folds(*args, **kwargs):
                train_idx, val_idx = fold["train_idx"], fold["val_idx"]
                self.assertFalse(set(groups[train_idx]) & set(groups[val_idx]))
                self.assertEqual(len(set(val_idx)), len(val_idx))
                np.testing.assert_array_equal(
                    fold["y_val"], self.data.iloc[val_idx]["is_spoiler"],
                )
                self.assertEqual(np.sum(fold["y_train"] == 0), np.sum(fold["y_train"] == 1))
                vocab = fold["vectorizer"].vocabulary_
                self.assertIn("spoilers", vocab)  # No ablation.
                for text in self.data.iloc[val_idx]["review_text"]:
                    self.assertNotIn(text.split()[-1], vocab)
                plot_vocab = fold["transformer"].named_transformers_[
                    "plot_summary"
                ].named_steps["tfidfvectorizer"].vocabulary_
                for text in self.data.iloc[val_idx]["plot_summary"]:
                    self.assertNotIn(text.split()[-1], plot_vocab)
                rating_encoder = fold["transformer"].named_transformers_["rating"]
                expected = self.data.iloc[np.unique(train_idx)]["rating"].median()
                self.assertEqual(rating_encoder.named_steps["simpleimputer"].statistics_[0], expected)
                assignments.append(val_idx.copy())
                yield fold

        with patch.object(lr, "generate_tfidf_folds", side_effect=checked_folds):
            summary, folds, best = lr.tune_logistic_regression(
                self.data, self.model_grid, self.tfidf_grid, k=3, verbose=False,
            )
        self.assertEqual(len(folds), 12)
        for first, second in zip(assignments[:3], assignments[3:], strict=True):
            np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.sort(np.concatenate(assignments[:3])), np.arange(61))
        for candidate, scores in folds.groupby("candidate"):
            self.assertAlmostEqual(summary.loc[candidate, "f1_mean"], scores["f1"].mean())
        self.assertTrue(summary.iloc[0]["converged"])
        self.assertEqual(best["model"]["C"], summary.iloc[0]["C"])

    def test_refit_encodes_before_oversampling(self):
        params = {"model": {"penalty": "l2", "C": 0.1}, "tfidf": {"min_df": 1}}
        with patch.object(LogisticRegression, "fit", autospec=True,
                          side_effect=LogisticRegression.fit) as fit:
            pipeline = lr.refit_logistic_regression(self.data, params)
        _, X, y = fit.call_args.args
        self.assertEqual(X.shape[0], 2 * (self.data["is_spoiler"] == 0).sum())
        self.assertEqual(np.sum(y == 0), np.sum(y == 1))
        scaler = pipeline["features"].named_transformers_["rating"].named_steps["maxabsscaler"]
        self.assertEqual(scaler.n_samples_seen_, len(self.data))
        self.assertEqual(pipeline.predict(self.data).shape, (len(self.data),))
        with self.assertRaises(ValueError):
            lr.refit_logistic_regression(self.data, params, INITIAL_FEATURES + ["is_spoiler"])
        with self.assertRaises(ConvergenceWarning):
            lr.refit_logistic_regression(self.data, params, max_iter=1)

    def test_unpenalized_result_is_labelled_explicitly(self):
        summary, _, best = lr.tune_logistic_regression(
            self.data, {"penalty": [None]}, {"min_df": [1]}, k=3, verbose=False,
        )
        self.assertEqual(summary.iloc[0]["penalty"], "none")
        self.assertIsNone(best["model"]["penalty"])

    def test_unconverged_candidates_cannot_win(self):
        summary, folds, best = lr.tune_logistic_regression(
            self.data, self.model_grid, {"min_df": [1]}, k=3,
            max_iter=1, tol=1e-12, verbose=False,
        )
        self.assertFalse(summary["converged"].any())
        self.assertFalse(folds["converged"].any())
        self.assertIsNone(best)


if __name__ == "__main__":
    unittest.main()
