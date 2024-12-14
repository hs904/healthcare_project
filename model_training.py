
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold

### Load Cleaned Data and Split It
# Load cleaned dataset
cleaned_data = pd.read_parquet('data/prepared_data.parquet')
pd.set_option('display.max_columns', None)
print(cleaned_data.head())

# Split the data into train and test data
X = cleaned_data.drop(columns=['stroke']) 
y = cleaned_data['stroke'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data size: {X_train.shape}")
print(f"Test data size: {X_test.shape}")

### Feature Engineering and Setup Modelling pipelines
# Define a scikit-learn transformer to add new feature above certain threshold.
class HasDiabetes(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=150):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Create a new binary column where 1 means glucose > threshold, 0 means otherwise
        glucose_level_column = X['avg_glucose_level']
        X['has_diabetes'] = (glucose_level_column > self.threshold).astype(int)
        return X

# Apply the HasDiabetes transformation first to the original DataFrame (X_train)
has_diabetes_transformer = HasDiabetes(threshold=150)
X_train = has_diabetes_transformer.transform(X_train)
X_test = has_diabetes_transformer.transform(X_test)

# Define the feature columns
numeric_features = ["age", "avg_glucose_level", "bmi", "has_diabetes"]
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Numeric transformation: Impute and scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())  # Scale numerical features
])

# Categorical transformation: Impute missing values and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encode categorical features
])

# Full preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

### Model Pipelines for GLM and LGBM
# Define Model Pipelines for GLM (Logistic Regression) and LGBM (LightGBM)

# GLM pipeline (Logistic Regression)
glm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Apply the full preprocessor pipeline
    ('classifier', LogisticRegression(class_weight='balanced', solver='liblinear'))  # Handle imbalance with class_weight='balanced'
])

# LGBM pipeline (LightGBM)
lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Apply the full preprocessor pipeline
    ('classifier', LGBMClassifier(objective='binary', metric='auc', is_unbalance=True))  # Handle imbalance with is_unbalance=True
])

# Fit GLM model
glm_pipeline.fit(X_train, y_train)

# Fit LGBM model
lgbm_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_glm = glm_pipeline.predict(X_test)
y_pred_lgbm = lgbm_pipeline.predict(X_test)

# Evaluate models using classification report and ROC AUC score
print("GLM Model Evaluation:")
print(classification_report(y_test, y_pred_glm))
print("ROC AUC score (GLM):", roc_auc_score(y_test, y_pred_glm))

print("\nLGBM Model Evaluation:")
print(classification_report(y_test, y_pred_lgbm))
print("ROC AUC score (LGBM):", roc_auc_score(y_test, y_pred_lgbm))



# Hyperparameter tuning

# Define parameter grids for hyperparameter tuning
glm_param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'classifier__penalty': ['l1', 'l2'],  # Type of regularization
    'classifier__solver': ['liblinear']  # Solver for logistic regression
}

# Set up GridSearchCV for GLM (Logistic Regression)
glm_grid_search = GridSearchCV(glm_pipeline, glm_param_grid, cv=5, n_jobs=-1, scoring='roc_auc', verbose=2)
glm_grid_search.fit(X_train, y_train)

# LGBM parameter grid setup
lgbm_param_grid = {
    'classifier__learning_rate': [0.01, 0.05, 0.1],  # Learning rate
    'classifier__n_estimators': [50, 100, 200],  # Number of boosting iterations
    'classifier__num_leaves': [31, 50, 100],  # Number of leaves
    'classifier__min_child_weight': [1, 5, 10]  # Minimum child weight
}

# Initialize GridSearchCV for LGBM
lgbm_grid_search = GridSearchCV(
    lgbm_pipeline,  # Use the original pipeline
    lgbm_param_grid,
    cv=5,
    n_jobs=-1,
    scoring='roc_auc',
    verbose=2
)

# Perform GridSearchCV for LGBM (no early stopping)
lgbm_grid_search.fit(X_train, y_train)

# Print the best parameters and evaluation for both models
print("Best parameters for GLM (Logistic Regression):", glm_grid_search.best_params_)
print("Best ROC AUC score for GLM:", glm_grid_search.best_score_)

print("\nBest parameters for LGBM:", lgbm_grid_search.best_params_)
print("Best ROC AUC score for LGBM:", lgbm_grid_search.best_score_)

# Evaluate the models with the best parameters on the test set
y_pred_glm = glm_grid_search.best_estimator_.predict(X_test)
y_pred_lgbm = lgbm_grid_search.best_estimator_.predict(X_test)

# Print the evaluation reports
print("\nGLM Model Evaluation (Best Parameters):")
print(classification_report(y_test, y_pred_glm))
print("ROC AUC score (GLM):", roc_auc_score(y_test, y_pred_glm))

print("\nLGBM Model Evaluation (Best Parameters):")
print(classification_report(y_test, y_pred_lgbm))
print("ROC AUC score (LGBM):", roc_auc_score(y_test, y_pred_lgbm))

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer

### Load Cleaned Data and Split It
# Load cleaned dataset
cleaned_data = pd.read_parquet('data/prepared_data.parquet')
pd.set_option('display.max_columns', None)
print(cleaned_data.head())

# Split the data into train and test data
X = cleaned_data.drop(columns=['stroke']) 
y = cleaned_data['stroke'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data size: {X_train.shape}")
print(f"Test data size: {X_test.shape}")

### Feature Engineering and Setup Modelling pipelines
# Define a scikit-learn transformer to add a new feature based on a threshold
class HasDiabetes(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=150):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()  # Ensure the original DataFrame is not modified
        glucose_level_column = X['avg_glucose_level']
        X['has_diabetes'] = (glucose_level_column > self.threshold).astype(int)
        return X

# Define the feature columns
numeric_features = ["age", "avg_glucose_level", "bmi", "has_diabetes"]
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Numeric transformation: Impute and scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical transformation: Impute missing values and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Full preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

### Model Pipelines for GLM and LGBM
# GLM pipeline (Logistic Regression)
glm_pipeline = Pipeline(steps=[
    ('diabetes_transformer', HasDiabetes(threshold=150)),
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', solver='liblinear'))
])

# LGBM pipeline (LightGBM)
lgbm_pipeline = Pipeline(steps=[
    ('diabetes_transformer', HasDiabetes(threshold=150)),
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(objective='binary', metric='auc', is_unbalance=True))
])

### Cross-Validation for Robust Model Evaluation
# Perform cross-validation
glm_scores = cross_val_score(glm_pipeline, X, y, cv=StratifiedKFold(5), scoring='roc_auc')
lgbm_scores = cross_val_score(lgbm_pipeline, X, y, cv=StratifiedKFold(5), scoring='roc_auc')

print("Cross-Validated ROC AUC (GLM):", np.mean(glm_scores))
print("Cross-Validated ROC AUC (LGBM):", np.mean(lgbm_scores))

### Train and Evaluate Final Models
# Fit GLM model
glm_pipeline.fit(X_train, y_train)

# Fit LGBM model
lgbm_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_glm = glm_pipeline.predict(X_test)
y_pred_lgbm = lgbm_pipeline.predict(X_test)

# Evaluate models using classification report and ROC AUC score
print("\nGLM Model Evaluation:")
print(classification_report(y_test, y_pred_glm))
print("ROC AUC score (GLM):", roc_auc_score(y_test, glm_pipeline.predict_proba(X_test)[:, 1]))

print("\nLGBM Model Evaluation:")
print(classification_report(y_test, y_pred_lgbm))
print("ROC AUC score (LGBM):", roc_auc_score(y_test, lgbm_pipeline.predict_proba(X_test)[:, 1]))
