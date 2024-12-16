# This file summarized all the functions used in eda_cleaning.ipybn
import pandas as pd
import numpy as np

def handle_missing_values(df, column, replacement_value=None):
    """Handle missing values in the dataset."""
    # Drop rows with missing values in the specified column
    df = df.dropna(subset=[column])

    # Replace specific values with a replacement value (e.g., 'Unknown')
    if replacement_value is not None:
        most_frequent_value = df[column].mode()[0]
        df[column] = df[column].replace(replacement_value, most_frequent_value)

    return df

def handle_outliers(df, column, threshold):
    """Handle outliers in the dataset for a specified column."""
    # Remove rows where the column's value exceeds the threshold
    df = df[df[column] <= threshold]

    return df

def remove_unrepresentative_rows(df, column, valid_values):
    """Remove rows with unrepresentative or noisy data in a specified column."""
    # Keep only rows where the column's value is in the list of valid values
    df = df[df[column].isin(valid_values)]

    return df

def merge_work_types(df, column, values_to_merge, new_value):
    """Merge specific values in a categorical column into a single new value."""
    # Replace specified values with a new value in the column
    df[column] = df[column].replace(values_to_merge, new_value)

    return df

def save_to_parquet(df, file_path):
    """Save the prepared dataset to a .parquet file."""
    df.to_parquet(file_path, index=False)