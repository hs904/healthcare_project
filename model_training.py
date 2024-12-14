from data_preparation.feature_engineering import create_binary_features
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import roc_auc_score, make_scorer
from glum import GeneralizedLinearRegressor
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

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
    ("regressor", GeneralizedLinearRegressor(family="binomial", fit_intercept=True))
])

# LGBM Pipeline
lgbm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LGBMClassifier(objective="binary", n_estimators=100, learning_rate=0.1, class_weight="balanced"))
])

### Hyperparameter tuning
# GLM parameter grid
glm_param_grid = {
    "regressor__alpha": [0.01, 0.1, 1.0, 10.0],
    "regressor__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]
}

# LGBM parameter grid
lgbm_param_grid = {
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__n_estimators": [1000],
    "classifier__num_leaves": [15, 31, 63],
    "classifier__min_child_weight": [0.001, 0.01, 0.1]
}

# Stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Custom AUC scorer for GLM
def glm_auc_scorer(y_true, y_pred, **kwargs):
    """Scorer function for AUC."""
    return roc_auc_score(y_true, y_pred)

glm_scorer = make_scorer(glm_auc_scorer)

# GLM GridSearchCV
glm_search = GridSearchCV(
    estimator=glm_pipeline,
    param_grid=glm_param_grid,
    scoring=glm_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=2
)

# Custom Wrapper for LGBM with Early Stopping
class LGBMClassifierWithEarlyStopping(lgb.LGBMClassifier):
    """Wrapper for LightGBM to include early stopping during GridSearchCV."""
    def fit(self, X, y, eval_set=None, early_stopping_rounds=10, **kwargs):
        super().fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="auc",
            **kwargs
        )

# Update the LGBM pipeline to use the wrapper
lgbm_pipeline.named_steps['classifier'] = LGBMClassifierWithEarlyStopping()

# LGBM GridSearchCV
lgbm_search = GridSearchCV(
    estimator=lgbm_pipeline,
    param_grid=lgbm_param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

# Fit GLM pipeline
print("Tuning GLM pipeline...")
glm_search.fit(X_train, y_train)
print(f"Best GLM Parameters: {glm_search.best_params_}")
print(f"Best GLM AUC: {glm_search.best_score_}")

# Fit LGBM pipeline with early stopping
print("Tuning LGBM pipeline...")
lgbm_search.fit(
    X_train, y_train,
    classifier__eval_set=[(X_test, y_test)],
    classifier__early_stopping_rounds=10
)

print(f"Best LGBM Parameters: {lgbm_search.best_params_}")
print(f"Best LGBM AUC: {lgbm_search.best_score_}")
