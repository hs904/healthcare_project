import numpy as np

 # This part is derived from the conclusions from EDA and data cleaning. 
 # One of the steps we found during data exploration to deal with outliers 
 # in Avg_glucose_level was to treat extremely high values as if they were 
 # people with diabetes. Therefore, we consider setting a new binary 
 # feature. But this should actually fall under feature engineering section.
def create_binary_features(df):
    """Create new binary features."""
    df['has_diabetes'] = np.where(df['avg_glucose_level'] >= 126, 1, 0)
    return df

  # Other feature engineering precess(e.g. scaling. encoding, feature interation
  # are shown in the model_training.py file)