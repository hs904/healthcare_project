from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from glum import GeneralizedLinearRegressor
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from data_preparation.feature_engineering import create_binary_features
import pandas as pd
import numpy as np

### processing pipline
# Load cleaned dataset
cleaned_data = pd.read_parquet('data/prepared_data.parquet')
print(cleaned_data.head())

# Feature Engineering 
# numeric features will exlude binary features,
# as we are going to normalize.
numeric_features = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", "has_diabetes"]
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

cleaned_data_data = create_binary_features(cleaned_data) 
print(cleaned_data.head())

# Train-test split
train, test = train_test_split(cleaned_data, test_size=0.2, random_state=42)
X_train, y_train = train.drop(columns=["stroke"]), train["stroke"]
X_test, y_test = test.drop(columns=["stroke"]), test["stroke"]

# Custom LogTransformer 
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_features=None):
        self.exclude_features = exclude_features if exclude_features else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for col in X.columns:
                if col not in self.exclude_features:
                    # Replace zero or negative values with a small constant
                    X[col] = np.where(X[col] <= 0, 1e-5, X[col])
                    X[col] = np.log1p(X[col])
        else:  # If X is a numpy array
            for i in range(X.shape[1]):
                # Assume exclude_features is a list of indices if X is ndarray
                if i not in self.exclude_features:
                    X[:, i] = np.where(X[:, i] <= 0, 1e-5, X[:, i])
                    X[:, i] = np.log1p(X[:, i])
        return X



# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("scale", StandardScaler()),
            ("log", LogTransformer(exclude_features=["hypertension", "heart_disease", "has_diabetes"]))
        ]), numeric_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
    ]
)


# GLM Pipeline
glm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GeneralizedLinearRegressor(family="binomial", l1_ratio=1, fit_intercept=True)),
])

# LGBM Pipeline
lgbm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LGBMClassifier(objective="binary", n_estimators=100, learning_rate=0.1)),
])

# Before train and evaluate model...
# Check for Missing Values
# Zero or negative values in numeric features will cause NaN issue.
print("Check for zero or negative values:")
print(X_train[numeric_features].describe())
print((X_train[numeric_features] <= 0).sum())


# Train and evaluate GLM
glm_pipeline.fit(X_train, y_train)
print("GLM Train Score:", glm_pipeline.score(X_train, y_train))
print("GLM Test Score:", glm_pipeline.score(X_test, y_test))

# Train and evaluate LGBM
lgbm_pipeline.fit(X_train, y_train)
print("LGBM Train Score:", lgbm_pipeline.score(X_train, y_train))
print("LGBM Test Score:", lgbm_pipeline.score(X_test, y_test))

# Hyperparameter Tuning for LGBM
param_grid = {
    "classifier__n_estimators": [50, 100, 150],
    "classifier__learning_rate": [0.01, 0.1, 0.2],
}
grid_search = GridSearchCV(lgbm_pipeline, param_grid, cv=3, scoring="accuracy", verbose=1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("LGBM Best Train Score:", grid_search.score(X_train, y_train))
print("LGBM Best Test Score:", grid_search.score(X_test, y_test))