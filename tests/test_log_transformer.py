import unittest
import pandas as pd
import numpy as np
from model_training import LogTransformer 

# Covers different scenarios: positive values, 
# zero values, negative values, excluded features, 
# and NumPy array inputs.

class TestLogTransformer(unittest.TestCase):
    def test_positive_values(self):
        input_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [10, 20, 30]})
        expected_output = pd.DataFrame({"feature1": [np.log1p(1), np.log1p(2), np.log1p(3)],
                                        "feature2": [np.log1p(10), np.log1p(20), np.log1p(30)]})
        transformer = LogTransformer()
        transformed_data = transformer.transform(input_data)
        pd.testing.assert_frame_equal(transformed_data, expected_output)

    def test_zero_values(self):
        input_data = pd.DataFrame({"feature1": [0, 0, 0], "feature2": [0, 10, 20]})
        expected_output = pd.DataFrame({"feature1": [np.log1p(1e-5), np.log1p(1e-5), np.log1p(1e-5)],
                                        "feature2": [np.log1p(1e-5), np.log1p(10), np.log1p(20)]})
        transformer = LogTransformer()
        transformed_data = transformer.transform(input_data)
        pd.testing.assert_frame_equal(transformed_data, expected_output)

    def test_negative_values(self):
        input_data = pd.DataFrame({"feature1": [-1, -2, -3], "feature2": [10, -10, 20]})
        expected_output = pd.DataFrame({"feature1": [np.log1p(1e-5), np.log1p(1e-5), np.log1p(1e-5)],
                                        "feature2": [np.log1p(10), np.log1p(1e-5), np.log1p(20)]})
        transformer = LogTransformer()
        transformed_data = transformer.transform(input_data)
        pd.testing.assert_frame_equal(transformed_data, expected_output)

    def test_exclude_features(self):
        input_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [10, 20, 30]})
        expected_output = pd.DataFrame({"feature1": [np.log1p(1), np.log1p(2), np.log1p(3)],
                                        "feature2": [10, 20, 30]})
        transformer = LogTransformer(exclude_features=["feature2"])
        transformed_data = transformer.transform(input_data)
        pd.testing.assert_frame_equal(transformed_data, expected_output)

    def test_numpy_array(self):
        input_data = np.array([[1, 10], [2, 20], [3, 30]])
        expected_output = np.array([[np.log1p(1), np.log1p(10)],
                                    [np.log1p(2), np.log1p(20)],
                                    [np.log1p(3), np.log1p(30)]])
        transformer = LogTransformer()
        transformed_data = transformer.transform(input_data)
        np.testing.assert_array_almost_equal(transformed_data, expected_output)

    def test_fit(self):
        transformer = LogTransformer()
        transformer.fit(None)  # Fit should not raise errors
        self.assertTrue(True)


# Run the tests
if __name__ == "__main__":
    unittest.main()
