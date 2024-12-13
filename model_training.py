from data_preparation.feature_engineering import create_binary_features
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from glum import GeneralizedLinearRegressor
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

### processing pipline
# Load cleaned dataset
cleaned_data = pd.read_parquet('data/prepared_data.parquet')
print(cleaned_data.head())

# Feature Engineering 
# numeric features will exlude binary features,
# as we are going to normalize.
numeric_features = ["age", "avg_glucose_level", "bmi"]
binary_features = ["hypertension", "heart_disease", "has_diabetes"]
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

cleaned_data_data = create_binary_features(cleaned_data) 
print(cleaned_data.head())

# Train-test split
train, test = train_test_split(cleaned_data, test_size=0.2, random_state=42)
X_train = train.drop(columns=["stroke"])
y_train = train["stroke"]
X_test = test.drop(columns=["stroke"])
y_test = test["stroke"]

# Custom Transformers 
# Zero or negative values in numeric features will cause NaN issue.
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_features=None):
        self.exclude_features = exclude_features if exclude_features else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):  # If input is a pandas DataFrame
            X = X.copy()
            for col in X.columns:
                if col not in self.exclude_features:
                    # Replace zero or negative values with a small constant
                    X[col] = np.where(X[col] <= 0, 1e-5, X[col])
                    X[col] = np.log1p(X[col])
        else:  # If input is a NumPy array
            X = X.copy()
            for i in range(X.shape[1]):
                if i not in self.exclude_features:  # Handle by index
                    X[:, i] = np.where(X[:, i] <= 0, 1e-5, X[:, i])
                    X[:, i] = np.log1p(X[:, i])
        return X


# Define pipelines for each feature group:
# log transformation of numerical features (exclude binary features)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("log", LogTransformer(exclude_features=binary_features)),
    ("scaler", StandardScaler())
])

# No transformation needed for binary features
binary_transformer = "passthrough" 

# Encoding categorical features
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", sparse_output=False))
])

# Combine all transformers in a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("bin", binary_transformer, binary_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# Modeling pipelines
# GLM Pipeline
glm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GeneralizedLinearRegressor(family="binomial", l1_ratio=1, fit_intercept=True))
])

# LGBM Pipeline
lgbm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LGBMClassifier(objective="binary", n_estimators=100, learning_rate=0.1))
])

# Train GLM
glm_pipeline.fit(X_train, y_train)
print("GLM Train Score:", glm_pipeline.score(X_train, y_train))
print("GLM Test Score:", glm_pipeline.score(X_test, y_test))

# Train LGBM
lgbm_pipeline.fit(X_train, y_train)
print("LGBM Train Score:", lgbm_pipeline.score(X_train, y_train))
print("LGBM Test Score:", lgbm_pipeline.score(X_test, y_test))
