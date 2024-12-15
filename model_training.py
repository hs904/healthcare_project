import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, classification_report, roc_curve, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import optuna
import warnings
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import PartialDependenceDisplay
from dalex import Explainer

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


#################################################################################################
### Evaluation
# Evaluate GLM
print("Best GLM Parameters:", glm_grid_search.best_params_)
y_pred_glm = best_glm_model.predict(X_test_preprocessed)
glm_classification_report = classification_report(y_test, y_pred_glm)
glm_roc_auc_score = roc_auc_score(y_test, y_pred_glm)
print("GLM Classification Report:\n", glm_classification_report)
print("GLM ROC AUC Score:", glm_roc_auc_score)

with open("evaluation_plots/GLM_Repoert.txt", "w") as f:
    f.write("GLM Classification Report:\n")
    f.write(glm_classification_report + "\n")
    f.write(f"GLM ROC AUC Score: {glm_roc_auc_score}\n")

# Evaluate final LightGBM model
y_pred_lgbm = final_model.predict(X_test_preprocessed, num_iteration=final_model.best_iteration)
y_pred_lgbm_binary = (y_pred_lgbm > 0.5).astype(int)
lgbm_classification_report = classification_report(y_test, y_pred_lgbm_binary)
lgbm_roc_auc_score = roc_auc_score(y_test, y_pred_lgbm)
print("LightGBM Classification Report:\n", lgbm_classification_report)
print("LightGBM ROC AUC Score:", lgbm_roc_auc_score)

with open("evaluation_plots/LightGBM_Report.txt", "w") as f:
    f.write("LightGBM Classification Report:\n")
    f.write(lgbm_classification_report + "\n")
    f.write(f"LightGBM ROC AUC Score: {lgbm_roc_auc_score}\n")

# Plot ROC Curve for GLM
roc_display_glm = RocCurveDisplay.from_estimator(
    best_glm_model, X_test_preprocessed, y_test, name="GLM (Logistic Regression)"
)
plt.title("ROC Curve - GLM")
plt.savefig("evaluation_plots/roc_curve_glm.png")
plt.close()

# Plot ROC Curve for LightGBM
fpr, tpr, _ = roc_curve(y_test, y_pred_lgbm)
plt.figure()
plt.plot(fpr, tpr, label="LightGBM (AUC = {:.2f})".format(lgbm_roc_auc_score))
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.title("ROC Curve - LightGBM")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="best")
plt.savefig("evaluation_plots/roc_curve_lgbm.png")
plt.close()

# "Predicted vs. Actual" plot
# The Calibration Curve (Reliability Curve) is the most 
# appropriate and widely used method to visualize the 
# relationship between predicted probabilities and actual 
# outcomes for binary classification tasks. 

# Calibration curve for GLM
y_pred_glm_proba = best_glm_model.predict_proba(X_test_preprocessed)[:, 1]
prob_true, prob_pred = calibration_curve(y_test, y_pred_glm_proba, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker="o", label="GLM", linestyle="--", color="blue")
plt.plot([0, 1], [0, 1], "r--", label="Perfect Calibration", alpha=0.7)
plt.title("Calibration Curve")
plt.xlabel("Predicted Probability")
plt.ylabel("Actual Fraction of Positives")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("evaluation_plots/calibration_curve_glm.png")
plt.show()

# Calibration curve for LGBM
y_pred_lgbm = final_model.predict(X_test_preprocessed, num_iteration=final_model.best_iteration)
print("Min value in y_pred_lgbm:", y_pred_lgbm.min())
print("Max value in y_pred_lgbm:", y_pred_lgbm.max())
y_pred_lgbm = np.clip(y_pred_lgbm, 0, 1)
prob_true_lgbm, prob_pred_lgbm = calibration_curve(y_test, y_pred_lgbm, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred_lgbm, prob_true_lgbm, marker="o", label="LGBM", linestyle="--", color="green")
plt.plot([0, 1], [0, 1], "r--", label="Perfect Calibration", alpha=0.7)
plt.title("Calibration Curve - LGBM")
plt.xlabel("Predicted Probability")
plt.ylabel("Actual Fraction of Positives")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("evaluation_plots/calibration_curve_lgbm.png")
plt.show()

# Feature Improtance
# Get Feature Importance for GLM
# Get transformed feature names from the preprocessor
glm_transformed_features = list(preprocessor.get_feature_names_out())
glm_transformed_features.append("has_diabetes")

# Get coefficients from the GLM model
glm_coefficients = best_glm_model.named_steps['classifier'].coef_[0]

# Ensure lengths match
assert len(glm_transformed_features) == len(glm_coefficients), "Mismatch in features and coefficients!"

# Create a DataFrame for GLM feature importance
glm_feature_importance = pd.DataFrame({
    "Feature": glm_transformed_features,
    "Importance": glm_coefficients
}).sort_values(by="Importance", ascending=False, key=abs)

# Print the top 5 features for GLM
glm_top_features = glm_feature_importance.head(5)
print("Top 5 Features (GLM):\n", glm_top_features)

# Get Feature Importance for LGBM
# Create a DataFrame for the preprocessed test set with proper column names
X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=glm_transformed_features)

# Clip predictions to the valid range [0, 1]
y_pred_lgbm = np.clip(y_pred_lgbm, 0, 1)

# Extract feature importance from LightGBM
lgbm_feature_importance = pd.DataFrame({
    "Feature": glm_transformed_features,  # Use GLM feature names for LGBM
    "Importance": final_model.feature_importance(importance_type="gain")
}).sort_values(by="Importance", ascending=False)

# Print the top 5 features for LGBM
lgbm_top_features = lgbm_feature_importance.head(5)
print("Top 5 Features (LGBM):\n", lgbm_top_features)

# Partial Dependence Plot
# Generate Partial Dependence Plots for the top 5 LGBM features
# Select top 5 features for LGBM
top_features_lgbm = lgbm_top_features["Feature"].tolist()

# Create Explainer for LGBM using the DataFrame with proper feature names
explainer_lgbm = Explainer(final_model, X_test_preprocessed_df, y_test)

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