import pandas as pd
from pathlib import Path

def load_data(file_path: str = "data/raw_data/raw_data.csv") -> pd.DataFrame:
    """
    Load the raw dataset from the specified file path.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(
            f"The dataset was not found at {file_path}. Please place it in the specified directory."
        )
    return pd.read_csv(file)

def load_prepared_data(path='data/prepared_data.parquet'):
    """
    Load the prepared dataset from a .parquet file.
    
    Args:
        path (str): Path to the prepared dataset file.
        
    Returns:
        pd.DataFrame: The prepared dataset.
    """
    return pd.read_parquet(path)
