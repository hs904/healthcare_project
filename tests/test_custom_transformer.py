# Test the custom transformer (add a new feature with certain threshold)

import pytest
import pandas as pd
from model_training import HasDiabetes

@pytest.mark.parametrize("threshold, glucose_levels, expected", [
    # Test case 1: Standard values with a threshold of 126
    (126, [100, 130, 120], [0, 1, 0]),

    # Test case 2: Lower threshold of 110
    (110, [105, 115, 95], [0, 1, 0]),

    # Test case 3: Higher threshold of 140
    (140, [130, 145, 150], [0, 1, 1]),

    # Test case 4: All glucose levels below the threshold
    (126, [100, 110, 120], [0, 0, 0]),

    # Test case 5: All glucose levels above the threshold
    (126, [127, 150, 160], [1, 1, 1])
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
    assert transformed["has_diabetes"].tolist() == expected


def test_has_diabetes_edge_cases():
    """
    Test edge cases for the HasDiabetes transformer.
    """
    # Edge case: Empty dataframe
    data_empty = pd.DataFrame({"avg_glucose_level": []})
    transformer = HasDiabetes(threshold=126)
    transformed_empty = transformer.transform(data_empty)
    assert transformed_empty.empty

    # Edge case: All NaN values
    data_nan = pd.DataFrame({"avg_glucose_level": [float("nan"), float("nan")]})
    transformer = HasDiabetes(threshold=126)
    transformed_nan = transformer.transform(data_nan)
    assert transformed_nan["has_diabetes"].tolist() == [0, 0]  # NaNs default to 0
