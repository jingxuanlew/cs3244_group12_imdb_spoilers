"""Exercise notebook sections 7-8 against a tiny saved training split."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.logistic_regression import tune_logistic_regression, refit_logistic_regression
from src.prepare_features import INITIAL_FEATURES, prepare_features


class FullTrainingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = Path(__file__).resolve().parents[1] / "notebooks/05_logistic_regression.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        cls.code = {cell["id"]: "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"}

    def test_final_stages_use_all_training_rows_and_save_correct_scope(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/cleaned").mkdir(parents=True)
            (root / "data/splits").mkdir()
            movies = pd.DataFrame({
                "movie_id": ["movie0"], "plot_summary": ["A story"], "plot_synopsis": ["A longer story"],
                "genre": [["Drama"]], "duration": ["1h 30min"], "release_date": ["2000-01-01"],
            })
            reviews = pd.DataFrame({
                "movie_id": ["movie0"] * 36, "review_date": ["10 February 2006"] * 36,
                "review_text": [f"Unique review row{i}" for i in range(36)],
                "review_summary": ["Review summary"] * 36,
                "rating": [3 if i % 3 == 0 else 8 for i in range(36)],
                "is_spoiler": [int(i % 3 == 0) for i in range(36)],
            })
            train_path = root / "data/splits/IMDB_reviews_train.json"
            movies.to_json(root / "data/cleaned/IMDB_movie_details.json", orient="records", lines=True)
            reviews.to_json(train_path, orient="records", lines=True)
            indices = np.arange(0, 24, 2)
            sample = reviews.iloc[indices].reset_index(drop=True)
            model_data = prepare_features(sample, movies).assign(
                movie_id=sample["movie_id"], is_spoiler=sample["is_spoiler"],
            )
            selection_scores = pd.DataFrame({
                "converged": [True], "selected": [True],
                "macro_f1_mean": [0.5], "macro_f1_std": [0.01],
            })
            namespace = dict(
                Path=Path, pd=pd, np=np, joblib=joblib, sklearn=sklearn,
                PROJECT_ROOT=root, train_path=train_path, total_training_rows=len(reviews),
                model_data=model_data, source_row_indices=indices, selected_features=["rating"],
                initial_features=INITIAL_FEATURES, k=3, group_by="review_text", RANDOM_STATE=67,
                oversample_positive=True, max_iter=1000, tol=1e-3, max_macro_f1_drop=0.005,
                model_grid={"penalty": ["l2"], "C": [1.0]}, tfidf_grid={"min_df": [1]},
                baseline_params={}, baseline_summary=selection_scores,
                baseline_fold_scores=pd.DataFrame({"fold": [1], "macro_f1": [0.5]}),
                elimination_path=selection_scores, elimination_trials=selection_scores,
                fig=Mock(), display=lambda value: None,
                tune_logistic_regression=tune_logistic_regression,
                refit_logistic_regression=refit_logistic_regression,
                prepare_features=prepare_features,
            )
            read_json = pd.read_json
            reads = []

            def checked_read(path, **kwargs):
                self.assertIn(Path(path), [train_path, root / "data/cleaned/IMDB_movie_details.json"])
                reads.append(Path(path))
                if "chunksize" in kwargs:
                    kwargs["chunksize"] = 7  # Exercise multiple chunks.
                return read_json(path, **kwargs)

            with patch.object(pd, "read_json", side_effect=checked_read):
                for cell_id in ["lr-full-training", "lr-retune", "lr-10"]:
                    exec(compile(self.code[cell_id], cell_id, "exec"), namespace)

            full = namespace["full_model_data"]
            self.assertEqual(len(model_data), 12)
            pd.testing.assert_frame_equal(full[["movie_id", "is_spoiler"]], reviews[["movie_id", "is_spoiler"]])
            self.assertEqual(list(full), ["rating", "review_text", "movie_id", "is_spoiler"])
            self.assertEqual(reads, [root / "data/cleaned/IMDB_movie_details.json", train_path])
            self.assertEqual(namespace["comparison"]["n_rows"].tolist(), [12, 12, 36])
            scaler = namespace["model"]["features"].named_transformers_["rating"].named_steps["maxabsscaler"]
            self.assertEqual(scaler.n_samples_seen_, 36)
            folder = namespace["output_dir"]
            artifact = joblib.load(folder / "model.joblib")
            self.assertEqual(artifact["training_rows"], 36)
            self.assertEqual(artifact["feature_selection_rows"], 12)
            self.assertTrue(folder.name.endswith("_full_backward"))
            np.testing.assert_array_equal(np.load(folder / "training_row_indices.npy"), np.arange(36))
            np.testing.assert_array_equal(np.load(folder / "selection_row_indices.npy"), indices)

            # Held-out prediction must not refit encoders/classifier or balance labels.
            test_reviews = reviews.iloc[:11].copy().reset_index(drop=True)
            test_reviews["review_text"] = [f"Held out text row{i}" for i in range(11)]
            test_path = train_path.with_name("IMDB_reviews_test.json")
            test_reviews.to_json(test_path, orient="records", lines=True)
            frozen_hash = joblib.hash(artifact["pipeline"])

            def checked_test_read(path, **kwargs):
                self.assertIn(Path(path), [test_path, root / "data/cleaned/IMDB_movie_details.json"])
                if "chunksize" in kwargs:
                    kwargs["chunksize"] = 4
                return read_json(path, **kwargs)

            with patch.object(pd, "read_json", side_effect=checked_test_read), \
                 patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("Refitting test")), \
                 patch("sklearn.compose.ColumnTransformer.fit_transform", side_effect=AssertionError("Refitting test")), \
                 patch("src.logistic_regression.oversample_positive_indices", side_effect=AssertionError("Oversampling test")):
                exec(self.code["lr-test-evaluation"], namespace)
            predictions = namespace["test_predictions"]
            self.assertEqual(len(predictions), len(test_reviews))
            np.testing.assert_array_equal(predictions["test_row"], np.arange(11))
            np.testing.assert_array_equal(predictions["is_spoiler"], test_reviews["is_spoiler"])
            expected = artifact["pipeline"].predict(prepare_features(test_reviews, movies)[["rating"]])
            np.testing.assert_array_equal(predictions["predicted_spoiler"], expected)
            self.assertEqual(joblib.hash(namespace["test_model"]), frozen_hash)
            self.assertEqual(namespace["test_confusion"].to_numpy().sum(), 11)
            self.assertIn("average_precision", namespace["test_metrics"])
            for name in ["test_predictions.csv", "test_metrics.csv", "test_confusion_matrix.csv"]:
                self.assertTrue((folder / name).is_file())

            # An already-full initial dataframe must not be loaded a second time.
            namespace["model_data"] = full
            with patch.object(pd, "read_json", side_effect=AssertionError("Unexpected reload")):
                exec(self.code["lr-full-training"], namespace)
            self.assertIs(namespace["full_model_data"], full)


if __name__ == "__main__":
    unittest.main()
