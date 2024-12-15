import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import optuna
import warnings

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

### Load cleaned data and split it
data_path = 'data/prepared_data.parquet' 
cleaned_data = pd.read_parquet(data_path)

# Define features and target
X = cleaned_data.drop(columns=['stroke'])
y = cleaned_data['stroke']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Feature engineering
class HasDiabetes(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=150):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        glucose_level_column = X['avg_glucose_level']
        X['has_diabetes'] = (glucose_level_column > self.threshold).astype(int)
        return X

numeric_features = ["age", "avg_glucose_level", "bmi"]
categorical_features = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Apply preprocessing to training and testing data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Add feature engineering after preprocessing
has_diabetes_transformer = HasDiabetes(threshold=150)
X_train_preprocessed = np.hstack([X_train_preprocessed, has_diabetes_transformer.transform(X_train)[['has_diabetes']].values])
X_test_preprocessed = np.hstack([X_test_preprocessed, has_diabetes_transformer.transform(X_test)[['has_diabetes']].values])

# Handle data imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

### GLM Pipeline
glm_pipeline = Pipeline(steps=[
    ('classifier', LogisticRegression(class_weight='balanced', solver='saga'))
])

# Hyperparameter tuning for GLM
glm_param_grid = {
    'classifier__penalty': ['elasticnet'],
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__l1_ratio': [0.1, 0.5, 0.7, 0.9]
}

glm_grid_search = RandomizedSearchCV(
    glm_pipeline, glm_param_grid, n_iter=50, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42
)
glm_grid_search.fit(X_train_resampled, y_train_resampled)

# Evaluate GLM (only used to check if the model works here)
best_glm_model = glm_grid_search.best_estimator_
print("Best GLM Parameters:", glm_grid_search.best_params_)
y_pred_glm = best_glm_model.predict(X_test_preprocessed)
print("GLM Classification Report:\n", classification_report(y_test, y_pred_glm))
print("GLM ROC AUC Score:", roc_auc_score(y_test, y_pred_glm))

### LightGBM Pipeline using Native API
# Define Optuna objective function
def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'verbosity': -1
    }

    train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)
    valid_data = lgb.Dataset(X_test_preprocessed, label=y_test, reference=train_data)

    # Train LightGBM model
    model = lgb.train(
        param,
        train_set=train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
    )

    # Predict probabilities for AUC score
    y_pred = model.predict(X_test_preprocessed, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_test, y_pred)
    return auc

# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Retrieve best parameters
best_params = study.best_params
print("Best LightGBM Parameters:", best_params)

# Train final LightGBM model with best parameters
train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)
valid_data = lgb.Dataset(X_test_preprocessed, label=y_test, reference=train_data)

final_model = lgb.train(
    best_params,
    train_set=train_data,
    valid_sets=[train_data, valid_data],
    valid_names=["train", "valid"],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
)

# Evaluate final LightGBM model (only used to check if the model works here)
y_pred_lgbm = final_model.predict(X_test_preprocessed, num_iteration=final_model.best_iteration)
y_pred_lgbm_binary = (y_pred_lgbm > 0.5).astype(int)

# Print evaluation metrics
print("LightGBM Classification Report:\n", classification_report(y_test, y_pred_lgbm_binary))
print("LightGBM ROC AUC Score:", roc_auc_score(y_test, y_pred_lgbm))
