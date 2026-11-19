"""
Marketing campaign response prediction with a class-balanced Random Forest.

Grid-searches 216 hyperparameter combinations (5-fold CV, F1 scoring), picks the
ROC-optimal decision threshold, and reports feature importance.

Run from the repository root:
    python src/random_forest_balanced.py [path/to/data.csv]
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                           roc_auc_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns


# Load and preprocess data
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/marketing_campaign_clean.csv"
df = pd.read_csv(DATA_PATH)
df.drop(columns=['ID'], inplace=True)

# Convert categorical columns to dummy variables
categorical_cols = df.select_dtypes(exclude='number').columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features and target
X = df.drop('Response', axis=1)
y = df['Response'].astype(int)

# Check class distribution
print("Class distribution:")
print(y.value_counts())
print(f"Class ratio (positive/negative): {y.sum()}/{len(y)-y.sum()}")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest with class balancing - ONLY THIS MODEL
print("\n" + "="*60)
print("Random Forest with Class Balancing")
print("="*60)

# Hyperparameter tuning for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Grid search with cross-validation
rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='f1',  # F1 score is good for imbalanced datasets
    n_jobs=-1,
    verbose=1
)

rf_grid_search.fit(X_train, y_train)

print(f"Best parameters: {rf_grid_search.best_params_}")
print(f"Best cross-validation F1 score: {rf_grid_search.best_score_:.4f}")

# Train best model
best_rf = rf_grid_search.best_estimator_

# Model evaluation
print(f"\n{'-'*40}")
print(f"Results for Random Forest (Balanced)")
print(f"{'-'*40}")

# Predictions
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

# Find optimal threshold using ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Apply optimal threshold
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

# Calculate metrics
auc_score = roc_auc_score(y_test, y_prob)
f1_optimal = f1_score(y_test, y_pred_optimal)

print(f"AUC Score: {auc_score:.4f}")
print(f"Accuracy (0.5 threshold): {(y_pred == y_test).mean():.4f}")
print(f"Accuracy (optimal threshold): {(y_pred_optimal == y_test).mean():.4f}")
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"F1 Score (optimal threshold): {f1_optimal:.4f}")

print("\nClassification Report (optimal threshold):")
print(classification_report(y_test, y_pred_optimal))

# Feature importance analysis
print(f"\n{'='*60}")
print(f"FEATURE IMPORTANCE ANALYSIS")
print(f"{'='*60}")

feature_importances = best_rf.feature_importances_

# Create feature importance dataframe
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("Top 15 Most Important Features:")
print(feature_importance_df.head(15))

# Visualization
plt.figure(figsize=(15, 10))

# 1. Feature Importance Plot
plt.subplot(2, 3, 1)
top_features = feature_importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importances')
plt.gca().invert_yaxis()

# 2. ROC Curve
plt.subplot(2, 3, 2)
plt.plot(fpr, tpr, label=f"Random Forest (Balanced) (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# 3. Precision-Recall Curve
plt.subplot(2, 3, 3)
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.plot(recall, precision, label='Random Forest (Balanced)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

# 4. Confusion Matrix - THIS IS THE ONE FROM YOUR RESULTS
plt.subplot(2, 3, 4)
cm = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest (Balanced)')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 5. Class distribution
plt.subplot(2, 3, 5)
y_test.value_counts().plot(kind='bar')
plt.title('Test Set Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)

# 6. Probability distribution
plt.subplot(2, 3, 6)
plt.hist(y_prob[y_test == 0], alpha=0.7, label='Negative Class', bins=30)
plt.hist(y_prob[y_test == 1], alpha=0.7, label='Positive Class', bins=30)
plt.axvline(x=optimal_threshold, color='red', 
           linestyle='--', label='Optimal Threshold')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Probability Distribution by Class')
plt.legend()

plt.tight_layout()
Path('figures/generated').mkdir(parents=True, exist_ok=True)
plt.savefig('figures/generated/random_forest_evaluation.png', dpi=130, bbox_inches='tight')
plt.show()

# Business insights
print(f"\n{'='*60}")
print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
print(f"{'='*60}")

print(f"Model: Random Forest (Balanced)")
print(f"- F1 Score: {f1_optimal:.4f}")
print(f"- AUC Score: {auc_score:.4f}")
print(f"- Optimal Threshold: {optimal_threshold:.4f}")

# Calculate business metrics
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix Breakdown:")
print(f"- True Positives (Correctly identified potential customers): {tp}")
print(f"- False Positives (Incorrectly identified as potential): {fp}")
print(f"- True Negatives (Correctly identified non-customers): {tn}")
print(f"- False Negatives (Missed potential customers): {fn}")

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)

print(f"\nKey Metrics:")
print(f"- Sensitivity (Recall): {sensitivity:.4f} - Ability to find potential customers")
print(f"- Specificity: {specificity:.4f} - Ability to avoid false positives")
print(f"- Precision: {precision:.4f} - Accuracy of positive predictions")

print(f"\nTop 5 Most Important Features for Customer Classification:")
for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows(), 1):
    print(f"{i}. {row['feature']}: {row['importance']:.4f}")

# Save the model
import joblib
Path('models').mkdir(exist_ok=True)
joblib.dump(best_rf, 'models/random_forest_balanced_model.pkl')
joblib.dump(optimal_threshold, 'models/optimal_threshold_rf_balanced.pkl')

print(f"\nModel saved as 'models/random_forest_balanced_model.pkl'")
print("Optimal threshold saved as 'models/optimal_threshold_rf_balanced.pkl'")

print(f"\n{'='*60}")
print("MODEL SUMMARY")
print(f"{'='*60}")
print("This Random Forest (Balanced) model uses:")
print("- Built-in class balancing to handle imbalanced data")
print("- Grid search optimization for best hyperparameters")
print("- ROC curve analysis to find optimal threshold")
print("- Comprehensive evaluation metrics for business insights")