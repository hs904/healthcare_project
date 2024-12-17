import pandas as pd
import numpy as np
from data.load_data import load_prepared_data
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, roc_auc_score, classification_report, roc_curve, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import optuna
import os
import matplotlib.pyplot as plt
from dalex import Explainer

### This file contains Modelling and Evaluation parts.

#################################################################
##################  Modelling  ##################################
#################################################################

### Step 1: Load cleaned data and split it ###

# Load cleaned data
cleaned_data = load_prepared_data(path='data/prepared_data.parquet')

# Add interaction feature before splitting data
cleaned_data['bmi_glucose_interaction'] = cleaned_data['bmi'] * cleaned_data['avg_glucose_level']

# Define features and target
X = cleaned_data.drop(columns=['stroke'])
y = cleaned_data['stroke']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Step 2: Add more feature engineering precesses ###

# Custom transformer: add new feature with certain threshold.
class HasDiabetes(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=126):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        glucose_level_column = X['avg_glucose_level']
        X['has_diabetes'] = (glucose_level_column > self.threshold).astype(int)
        return X

numeric_features = ["age", "avg_glucose_level", "bmi", "bmi_glucose_interaction"]
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

### Step 3: Model pipelines ###

# GLM Pipeline
glm_pipeline = Pipeline(steps=[
    ('classifier', LogisticRegression(class_weight='balanced', solver='saga'))
])

# LightGBM Pipeline with optuna package
# Define Optuna objective function
def lightgbm_objective(trial):
    """
    Objective function for Optuna to tune LightGBM hyperparameters.
    """
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0, step=0.1),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'max_bin': trial.suggest_int('max_bin', 200, 500),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
        'verbosity': -1
    }

# Set up k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_index, valid_index in kf.split(X_train_resampled, y_train_resampled):
        X_train_fold, X_valid_fold = X_train_resampled[train_index], X_train_resampled[valid_index]
        y_train_fold, y_valid_fold = y_train_resampled[train_index], y_train_resampled[valid_index]
        
        # Train LightGBM model on each fold
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        valid_data = lgb.Dataset(X_valid_fold, label=y_valid_fold, reference=train_data)
   
# Train LightGBM with parameters
    model = lgb.train(
        params,
        train_set=train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
    )

# Predict probabilities for AUC score
    y_pred = model.predict(X_valid_fold, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_valid_fold, y_pred)
    auc_scores.append(auc)

# Return the average AUC score across all folds
    return np.mean(auc_scores)

### Step 4: Hyperparameter tuning ###

# Hyperparameter tuning for GLM
glm_param_grid = {
    'classifier__penalty': ['elasticnet'],
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__l1_ratio': [0.1, 0.5, 0.7, 0.9],
    'classifier__class_weight': ['balanced'],
}

glm_grid_search = RandomizedSearchCV(
    glm_pipeline, glm_param_grid, n_iter=50, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42
)
glm_grid_search.fit(X_train_resampled, y_train_resampled)

# Hyperparameter tuning for LightGBM continued with optuna package
# Run Optuna optimization
lightgbm_study = optuna.create_study(direction='maximize')
lightgbm_study.optimize(lightgbm_objective, n_trials=50)

# Retrieve best parameters
best_lightgbm_params = lightgbm_study.best_params
print("Best LightGBM Parameters:", best_lightgbm_params)

# Train Final LightGBM Model with Best Parameters, with early stopping
train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)
valid_data = lgb.Dataset(X_test_preprocessed, label=y_test, reference=train_data)

final_lightgbm_model = lgb.train(
    best_lightgbm_params,
    train_set=train_data,
    valid_sets=[train_data, valid_data],
    valid_names=["train", "valid"],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
)

# Get best predictions for both models
# Predictions for GLM model
best_glm_model = glm_grid_search.best_estimator_
y_pred_glm_proba = best_glm_model.predict_proba(X_test_preprocessed)[:, 1]
print("Best GLM Parameters:", glm_grid_search.best_params_)
y_pred_glm = (y_pred_glm_proba > 0.5).astype(int)

# Predictions for LightGBM model
y_pred_lgbm = final_lightgbm_model.predict(X_test_preprocessed, num_iteration=final_lightgbm_model.best_iteration)
y_pred_lgbm_binary = (y_pred_lgbm > 0.5).astype(int)



#################################################################
##################  Evaluation  ##################################
#################################################################

### Step 1: ROC AUC Score and Classification Reperot ###

def evaluate_model(y_true, y_pred, y_proba, model_name, save_path):
    """
    Generate classification report and calculate ROC AUC score.
    Save results to a file.
    """
    report = classification_report(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    print(f"==== {model_name} Evaluation ====")
    print(report)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    with open(save_path, "w") as f:
        f.write(f"{model_name} Classification Report:\n")
        f.write(report)
        f.write(f"\n{model_name} ROC AUC Score: {roc_auc:.4f}\n")

# Evaluate GLM model
evaluate_model(
    y_test, y_pred_glm, y_pred_glm_proba,
    "GLM", "evaluation_plots/GLM_Report.txt"
)

# Evaluate LightGBM model
evaluate_model(
    y_test, y_pred_lgbm_binary, y_pred_lgbm,
    "LightGBM", "evaluation_plots/LightGBM_Report.txt"
)


### Step 2: ROC Curve Visualisation ###

# Plot ROC Curve for GLM model
def plot_roc_curve(y_true, y_proba, model_name, save_path):
    """
    Plot ROC curve and save the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_true, y_proba):.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved: {save_path}")

# ROC Curve for GLM model
plot_roc_curve(
    y_test, y_pred_glm_proba, "GLM", "evaluation_plots/roc_curve_glm.png"
)

# ROC Curve for LightGBM model
plot_roc_curve(
    y_test, y_pred_lgbm, "LightGBM", "evaluation_plots/roc_curve_lgbm.png"
)


### Step 3: "Predicted vs. Actual" plot ###

def plot_predicted_vs_actual_scatter(y_true, y_proba, model_name, save_path):
    """
    Creates a scatter plot of predicted probabilities vs actual labels.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_proba, y_true + np.random.uniform(-0.02, 0.02, size=len(y_true)),
                alpha=0.6, color="blue", label="Predictions")
    plt.axhline(y=0, color="red", linestyle="--", linewidth=0.8, label="Class 0 (Negative)")
    plt.axhline(y=1, color="green", linestyle="--", linewidth=0.8, label="Class 1 (Positive)")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Binary Label (0 or 1)")
    plt.title(f"Predicted vs. Actual Scatter Plot - {model_name}")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Predicted vs. Actual scatter plot saved: {save_path}")

# Scatter Plot for GLM model
plot_predicted_vs_actual_scatter(
    y_test, y_pred_glm_proba, "GLM", "evaluation_plots/predicted_vs_actual_glm.png"
)

# Scatter Plot for LightGBM model
plot_predicted_vs_actual_scatter(
    y_test, y_pred_lgbm, "LightGBM", "evaluation_plots/predicted_vs_actual_lgbm.png"
)
 
# Addition:
# The Calibration Curve (Reliability Curve) is the most 
# appropriate and widely used method to visualize the 
# relationship between predicted probabilities and actual 
# outcomes for binary classification tasks. 

def plot_calibration_curve(y_true, y_proba, model_name, save_path):
    """
    Plot calibration curve and save the plot.
    """
    y_proba = np.clip(y_proba, 0, 1)
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker="o", label=model_name, linestyle="--")
    plt.plot([0, 1], [0, 1], "r--", label="Perfect Calibration")
    plt.title(f"Calibration Curve - {model_name}")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Fraction of Positives")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Calibration curve saved: {save_path}")

# Calibration Curve for GLM model
plot_calibration_curve(
    y_test, y_pred_glm_proba, "GLM", "evaluation_plots/calibration_curve_glm.png"
)

# Calibration Curve for LightGBM model
plot_calibration_curve(
    y_test, y_pred_lgbm, "LightGBM", "evaluation_plots/calibration_curve_lgbm.png"
)


### Step 4: Feature Importance ###

# Get Feature Importance for GLM model
# Get transformed feature names from the preprocessor
glm_transformed_features = list(preprocessor.get_feature_names_out())
glm_transformed_features.append("has_diabetes")

# Get coefficients from GLM model
glm_coefficients = best_glm_model.named_steps['classifier'].coef_[0]

# Ensure lengths match
assert len(glm_transformed_features) == len(glm_coefficients), "Mismatch in features and coefficients!"

# Create a DataFrame for GLM feature importance
glm_feature_importance = pd.DataFrame({
    "Feature": glm_transformed_features,
    "Importance": glm_coefficients
}).sort_values(by="Importance", ascending=False, key=abs)
print(glm_feature_importance)

# Print the top 5 features for GLM
glm_top_features = glm_feature_importance.head(5)
print("Top 5 Features (GLM):\n", glm_top_features)

# Get Feature Importance for LightGBM model
# Create a DataFrame for the preprocessed test set with proper column names
X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=glm_transformed_features)

# Clip predictions to the valid range [0, 1]
y_pred_lgbm = np.clip(y_pred_lgbm, 0, 1)

# Extract feature importance from LightGBM
lgbm_feature_importance = pd.DataFrame({
    "Feature": glm_transformed_features,  # Use GLM feature names for LGBM
    "Importance": final_lightgbm_model.feature_importance(importance_type="gain")
}).sort_values(by="Importance", ascending=False)
print(lgbm_feature_importance)

# Print the top 5 features for LightGBM model
lgbm_top_features = lgbm_feature_importance.head(5)
print("Top 5 Features (LGBM):\n", lgbm_top_features)

### Step 5: Partial Dependence Plot ###

# Generate Partial Dependence Plots for the top 5 LGBM features
# Select top 5 features for LGBM
top_features_lgbm = lgbm_top_features["Feature"].tolist()

# Create Explainer for LGBM using the DataFrame with proper feature names
explainer_lgbm = Explainer(final_lightgbm_model, X_test_preprocessed_df, y_test)

# Generate Partial Dependence Plots and save them
for feature in top_features_lgbm:
    if feature in X_test_preprocessed_df.columns:
        # Calculate model profile (Partial Dependence) using Dalex
        pd_profile_lgbm = explainer_lgbm.model_profile(variables=[feature])  # Removed 'variable_splits'

        # Extract partial dependence data from the result
        pd_data = pd_profile_lgbm.result
        feature_values = pd_data[pd_data["_vname_"] == feature]["_x_"]
        predictions = pd_data[pd_data["_vname_"] == feature]["_yhat_"]

        # Plot the partial dependence using Matplotlib
        plt.figure(figsize=(8, 6))
        plt.plot(feature_values, predictions, marker="o", linestyle="--", color="blue", label=f"{feature}")
        plt.xlabel(feature)
        plt.ylabel("Predicted Probability (Positive Class)")
        plt.title(f"Partial Dependence Plot - {feature} (LGBM)")
        plt.grid(alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()

        # Save the plot as a static image
        save_path = f"evaluation_plots/partial_dependence/partial_dependence_{feature}_lgbm.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Partial dependence plot saved: {save_path}")
    else:
        print(f"Feature '{feature}' not found in the DataFrame.")