# Add a separate test script (test_custom_transformer.py) 
# to test the functionality of the HasDiabetes transformer.

import pytest
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from tuning import HasDiabetes

@pytest.mark.parametrize("threshold, glucose_levels, expected", [
    # Test case 1: Standard values with a threshold of 150
    (150, [100, 160, 120], [0, 1, 0]),

    # Test case 2: Lower threshold of 140
    (140, [130, 150, 110], [0, 1, 0]),

    # Test case 3: Higher threshold of 160
    (160, [155, 165, 170], [0, 1, 1]),

    # Test case 4: All glucose levels below the threshold
    (200, [100, 150, 160], [0, 0, 0]),

    # Test case 5: All glucose levels above the threshold
    (90, [100, 150, 160], [1, 1, 1])
])
def test_has_diabetes_threshold_behavior(threshold, glucose_levels, expected):
    """
    Test the HasDiabetes transformer with different thresholds and input values.
    """
    # Create a test dataframe
    data = pd.DataFrame({"avg_glucose_level": glucose_levels})

    # Initialize the transformer
    transformer = HasDiabetes(threshold=threshold)

    # Transform the data
    transformed = transformer.transform(data)

    # Assert the transformed results match the expected output
    assert (transformed["has_diabetes"].tolist() == expected)

def test_has_diabetes_edge_cases():
    """
    Test edge cases for the HasDiabetes transformer.
    """
    # Edge case: Empty dataframe
    data_empty = pd.DataFrame({"avg_glucose_level": []})
    transformer = HasDiabetes(threshold=150)
    transformed_empty = transformer.transform(data_empty)
    assert transformed_empty.empty

    # Edge case: All NaN values
    data_nan = pd.DataFrame({"avg_glucose_level": [float("nan"), float("nan")]})
    transformed_nan = transformer.transform(data_nan)
    assert transformed_nan["has_diabetes"].tolist() == [0, 0]  # NaNs default to 0 in logic


