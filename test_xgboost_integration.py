import unittest

import numpy as np
import pandas as pd

import model_utils as mu


class TestXGBoostIntegration(unittest.TestCase):

    def test_preprocessing_uses_training_statistics(self):
        train = pd.DataFrame({
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [10.0, np.nan, 30.0],
        })

        validation = pd.DataFrame({
            "feature_a": [1000.0],
            "feature_b": [np.nan],
        })

        prep = mu.Preprocessor().fit(train)
        original_medians = prep.medians.copy()
        original_means = prep.means.copy()

        transformed = prep.transform_X(validation)

        self.assertEqual(transformed.shape, (1, 2))
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())

        pd.testing.assert_series_equal(
            prep.medians, original_medians
        )
        pd.testing.assert_series_equal(
            prep.means, original_means
        )

    def test_regression_metrics(self):
        actual = np.array([10.0, 20.0, 30.0])
        predicted = np.array([11.0, 19.0, 29.0])

        metrics = mu.regression_metrics(actual, predicted)

        self.assertAlmostEqual(metrics["RMSE"], 1.0)
        self.assertAlmostEqual(metrics["MAE"], 1.0)
        self.assertEqual(metrics["n"], 3)


if __name__ == "__main__":
    unittest.main()
