import numpy as np

 # This part is derived from the conclusions from EDA and data cleaning. 
 # One of the steps we found during data exploration to deal with outliers 
 # in Avg_glucose_level was to treat extremely high values as if they were 
 # people with diabetes. Therefore, we consider setting a new binary 
 # feature. But this should actually fall under feature engineering section.
def create_binary_features(df):
    """Create new binary features."""
    df['has_diabetes'] = np.where(df['avg_glucose_level'] >= 150, 1, 0)
    return df

  # Log transform (normalize) numeric features and encode categorical features are shown in 
  # model_traning.py as processing pipeline