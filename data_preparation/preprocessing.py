import pandas as pd
import numpy as np

def describe_data(df):
    """Describe the dataset."""
    return df.describe(include="all")

def detect_missing_values(df):
    """Detect missing values in the dataset."""
    return df.isnull().sum()

def most_frequent_values(df):
    """
    Calculate the most frequent values in each column of the dataset.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: A dataframe with the most frequent item, its frequency, 
                  and its percentage of the total for each column.
    """
    total = df.count()
    summary = pd.DataFrame(total)
    summary.columns = ['Total']

    most_frequent_items = []
    frequencies = []

    for col in df.columns:
        try:
            most_frequent_item = df[col].value_counts().idxmax()
            frequency = df[col].value_counts().max()
            most_frequent_items.append(most_frequent_item)
            frequencies.append(frequency)
        except Exception as ex:
            print(f"Error processing column {col}: {ex}")
            most_frequent_items.append(None)
            frequencies.append(0)

    summary['Most Frequent Item'] = most_frequent_items
    summary['Frequency'] = frequencies
    summary['Percent of Total'] = np.round(np.array(frequencies) / total * 100, 3)

    return summary.T

def unique_values(df):
    """
    Find the number of unique values in each column of the dataset.
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in df.columns:
        # Count unique values
        unique = df[col].nunique()
        uniques.append(unique)
    tt['Unique Values'] = uniques
    return np.transpose(tt)
