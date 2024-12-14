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


