import unittest
import pandas as pd
import numpy as np
from model_training import LogTransformer

class TestLogTransformer(unittest.TestCase):
    def setUp(self):
        # Prepare test data
        self.df_positive = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [10, 20, 30]
        })

        self.df_zero = pd.DataFrame({
            "feature1": [0, 1, 2],
            "feature2": [10, 0, 30]
        })

        self.df_negative = pd.DataFrame({
            "feature1": [-1, 1, 2],
            "feature2": [10, -5, 30]
        })

        self.exclude_features = ["feature2"]

    def test_positive_values(self):
        transformer = LogTransformer()
        transformed = transformer.fit_transform(self.df_positive)
        expected = np.log1p(self.df_positive)
        pd.testing.assert_frame_equal(transformed, expected, check_dtype=False)

    def test_zero_values(self):
        transformer = LogTransformer()
        transformed = transformer.fit_transform(self.df_zero)
        expected = self.df_zero.copy()
        expected["feature1"] = np.log1p(np.where(expected["feature1"] <= 0, 1e-5, expected["feature1"]))
        expected["feature2"] = np.log1p(np.where(expected["feature2"] <= 0, 1e-5, expected["feature2"]))
        pd.testing.assert_frame_equal(transformed, expected, check_dtype=False)

    def test_negative_values(self):
        transformer = LogTransformer()
        transformed = transformer.fit_transform(self.df_negative)
        expected = self.df_negative.copy()
        expected["feature1"] = np.log1p(np.where(expected["feature1"] <= 0, 1e-5, expected["feature1"]))
        expected["feature2"] = np.log1p(np.where(expected["feature2"] <= 0, 1e-5, expected["feature2"]))
        pd.testing.assert_frame_equal(transformed, expected, check_dtype=False)

    def test_exclude_features(self):
        transformer = LogTransformer(exclude_features=self.exclude_features)
        transformed = transformer.fit_transform(self.df_negative)
        expected = self.df_negative.copy()
        expected["feature1"] = np.log1p(np.where(expected["feature1"] <= 0, 1e-5, expected["feature1"]))
        # Ensure feature2 remains unchanged
        expected["feature2"] = self.df_negative["feature2"]
        pd.testing.assert_frame_equal(transformed, expected, check_dtype=False)

if __name__ == "__main__":
    unittest.main()