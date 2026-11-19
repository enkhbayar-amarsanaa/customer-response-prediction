"""
Marketing campaign response prediction with logistic regression.

Trains and compares 7 logistic-regression variants on an imbalanced target (15% responders):
baseline, Lasso (L1), Ridge (L2), Elastic Net, SMOTE oversampling, a soft-voting ensemble,
and a probability-calibrated model. Each model's decision threshold is tuned for F1.

Run from the repository root:
    python src/logistic_regression_models.py [path/to/data.csv]

Every figure is saved to figures/generated/ as it is shown;
the comparison table is written to results/model_comparison_results.csv.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

_FIG_DIR = Path("figures/generated")
_FIG_DIR.mkdir(parents=True, exist_ok=True)
_fig_count = 0
_plt_show = plt.show

def _save_and_show(*args, **kwargs):
    """Save the current figure before displaying it."""
    global _fig_count
    _fig_count += 1
    plt.gcf().savefig(_FIG_DIR / f"fig_{_fig_count:02d}.png", dpi=130, bbox_inches="tight")
    _plt_show(*args, **kwargs)
    plt.close("all")

plt.show = _save_and_show
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                           roc_auc_score, precision_recall_curve, average_precision_score,
                           precision_score, recall_score, f1_score)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class VisualizationMixin:
    """Mixin class for all visualization methods"""
    
    def setup_plot_style(self):
        """Setup consistent plot styling"""
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def plot_class_distribution(self, y, title="Class Distribution"):
        """Plot class distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        y.value_counts().plot(kind='bar', ax=ax1, color=['lightcoral', 'lightblue'])
        ax1.set_title(f'{title} - Counts')
        ax1.set_xlabel('Response')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        
        # Percentage plot
        y.value_counts(normalize=True).plot(kind='bar', ax=ax2, color=['lightcoral', 'lightblue'])
        ax2.set_title(f'{title} - Percentages')
        ax2.set_xlabel('Response')
        ax2.set_ylabel('Percentage')
        ax2.tick_params(axis='x', rotation=0)
        
        # Add value labels on bars
        for ax in [ax1, ax2]:
            for i, v in enumerate(ax.patches):
                ax.text(v.get_x() + v.get_width()/2, v.get_height() + max(ax.get_ylim())*0.01,
                       f'{v.get_height():.3f}' if ax == ax2 else f'{int(v.get_height())}',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_correlation_heatmap(self, X, y, top_n=20):
        """Plot correlation heatmap of top features with target"""
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        top_features = correlations.head(top_n).index
        
        plt.figure(figsize=(14, 10))
        correlation_matrix = X[top_features].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Top {top_n} Features Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, model_name, top_n=20):
        """Plot feature importance from logistic regression coefficients"""
        if hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(top_n)
            
            sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
            plt.title(f'Top {top_n} Feature Importance - {model_name}')
            plt.xlabel('Absolute Coefficient Value')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.show()
            
            return feature_importance
        else:
            print(f"Model {model_name} doesn't have coefficient attributes")
            return None

class MarketingResponsePredictor(VisualizationMixin):
    """Enhanced predictor class with comprehensive visualizations"""
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.setup_plot_style()
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the dataset with visualizations"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(filepath)
        
        # Remove ID column if present
        if 'ID' in df.columns:
            df.drop(columns=['ID'], inplace=True)
        
        # Convert categorical variables to dummy variables
        categorical_cols = df.select_dtypes(exclude='number').columns
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Separate features and target
        X = df.drop('Response', axis=1)
        y = df['Response'].astype(int)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Feature columns: {X.shape[1]}")
        print(f"Class distribution:")
        print(y.value_counts())
        print(f"Class distribution (%):")
        print(y.value_counts(normalize=True) * 100)
        
        # Visualizations
        self.plot_class_distribution(y, "Original Dataset Class Distribution")
        self.plot_feature_correlation_heatmap(X, y)
        
        return X, y
    
    def split_and_scale_data(self, X, y, test_size=0.2):
        """Split data and apply scaling with visualizations"""
        print(f"\nSplitting data (train: {1-test_size:.0%}, test: {test_size:.0%})...")
        
        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
        
        print(f"Training set class distribution:")
        print(y_train.value_counts())
        print(f"Test set class distribution:")
        print(y_test.value_counts())
        
        # Plot train/test distribution comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        y_train.value_counts().plot(kind='bar', ax=ax1, color=['lightcoral', 'lightblue'])
        ax1.set_title('Training Set Class Distribution')
        ax1.set_xlabel('Response')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        
        y_test.value_counts().plot(kind='bar', ax=ax2, color=['lightcoral', 'lightblue'])
        ax2.set_title('Test Set Class Distribution')
        ax2.set_xlabel('Response')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def find_optimal_threshold(self, y_true, y_prob, metric='f1'):
        """Find optimal threshold based on specified metric"""
        if metric == 'f1':
            thresholds = np.arange(0.1, 1.0, 0.01)
            f1_scores = []
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                f1_scores.append(f1_score(y_true, y_pred))
            optimal_idx = np.argmax(f1_scores)
            return thresholds[optimal_idx]
        elif metric == 'youden':
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            optimal_idx = np.argmax(tpr - fpr)
            return thresholds[optimal_idx]
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot enhanced confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Response', 'Response'],
                   yticklabels=['No Response', 'Response'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                plt.text(j+0.5, i+0.7, f'({cm[i,j]/total:.1%})', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true, y_prob, model_name):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_prob, model_name):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'{model_name} (AUC = {pr_auc:.4f})')
        
        # Add baseline (random classifier)
        baseline = y_true.mean()
        plt.axhline(y=baseline, color='navy', linestyle='--', 
                   label=f'Random Baseline ({baseline:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_threshold_analysis(self, y_true, y_prob, model_name):
        """Plot threshold analysis showing precision, recall, F1 vs threshold"""
        thresholds = np.arange(0.1, 1.0, 0.01)
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            precisions.append(precision_score(y_true, y_pred))
            recalls.append(recall_score(y_true, y_pred))
            f1_scores.append(f1_score(y_true, y_pred))
        
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
        
        # Mark optimal threshold
        optimal_threshold = self.find_optimal_threshold(y_true, y_prob, 'f1')
        plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal Threshold ({optimal_threshold:.3f})')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Threshold Analysis - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_probability_distribution(self, y_true, y_prob, model_name):
        """Plot probability distribution by class"""
        plt.figure(figsize=(10, 6))
        
        # Separate probabilities by actual class
        pos_probs = y_prob[y_true == 1]
        neg_probs = y_prob[y_true == 0]
        
        plt.hist(neg_probs, bins=30, alpha=0.7, label='No Response', color='lightcoral')
        plt.hist(pos_probs, bins=30, alpha=0.7, label='Response', color='lightblue')
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title(f'Probability Distribution by Actual Class - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def train_baseline_model(self, X_train, X_test, y_train, y_test):
        """Train baseline logistic regression with class weighting"""
        print("\n" + "="*60)
        print("1. BASELINE MODEL (Class Weighted)")
        print("="*60)
        
        model = LogisticRegression(
            class_weight='balanced', 
            random_state=self.random_state, 
            max_iter=1000
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        optimal_threshold = self.find_optimal_threshold(y_test, y_prob, metric='f1')
        y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
        
        self.models['baseline'] = {
            'model': model,
            'threshold': optimal_threshold,
            'y_prob': y_prob,
            'y_pred': y_pred_optimal
        }
        
        self._evaluate_model('baseline', y_test, y_pred_optimal, y_prob)
        
        # Generate all visualizations for baseline model
        self.plot_confusion_matrix(y_test, y_pred_optimal, 'Baseline')
        self.plot_roc_curve(y_test, y_prob, 'Baseline')
        self.plot_precision_recall_curve(y_test, y_prob, 'Baseline')
        self.plot_threshold_analysis(y_test, y_prob, 'Baseline')
        self.plot_probability_distribution(y_test, y_prob, 'Baseline')
        self.plot_feature_importance(model, X_train.columns, 'Baseline')
        
        return model
    
    def train_regularized_models(self, X_train, X_test, y_train, y_test):
        """Train L1 and L2 regularized models with visualizations"""
        print("\n" + "="*60)
        print("2. REGULARIZED MODELS (L1 & L2)")
        print("="*60)
        
        C_range = np.logspace(-4, 2, 20)
        
        # L1 Regularization (LASSO)
        print("\nTraining L1 (LASSO) Regularized Model...")
        lasso_model = LogisticRegression(
            penalty='l1', solver='liblinear', class_weight='balanced',
            max_iter=1000, random_state=self.random_state
        )
        
        param_grid_l1 = {'C': C_range}
        grid_search_l1 = GridSearchCV(
            lasso_model, param_grid_l1, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='f1', n_jobs=-1
        )
        grid_search_l1.fit(X_train, y_train)
        
        best_lasso = grid_search_l1.best_estimator_
        y_prob_l1 = best_lasso.predict_proba(X_test)[:, 1]
        optimal_threshold_l1 = self.find_optimal_threshold(y_test, y_prob_l1, metric='f1')
        y_pred_l1 = (y_prob_l1 >= optimal_threshold_l1).astype(int)
        
        self.models['lasso'] = {
            'model': best_lasso,
            'threshold': optimal_threshold_l1,
            'y_prob': y_prob_l1,
            'y_pred': y_pred_l1,
            'best_C': grid_search_l1.best_params_['C']
        }
        
        print(f"Best C for L1: {grid_search_l1.best_params_['C']:.6f}")
        self._evaluate_model('lasso', y_test, y_pred_l1, y_prob_l1)
        
        # Visualizations for LASSO
        self.plot_confusion_matrix(y_test, y_pred_l1, 'LASSO (L1)')
        self.plot_roc_curve(y_test, y_prob_l1, 'LASSO (L1)')
        self.plot_feature_importance(best_lasso, X_train.columns, 'LASSO (L1)')
        
        # L2 Regularization (Ridge)
        print("\nTraining L2 (Ridge) Regularized Model...")
        ridge_model = LogisticRegression(
            penalty='l2', class_weight='balanced',
            max_iter=1000, random_state=self.random_state
        )
        
        param_grid_l2 = {'C': C_range}
        grid_search_l2 = GridSearchCV(
            ridge_model, param_grid_l2, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='f1', n_jobs=-1
        )
        grid_search_l2.fit(X_train, y_train)
        
        best_ridge = grid_search_l2.best_estimator_
        y_prob_l2 = best_ridge.predict_proba(X_test)[:, 1]
        optimal_threshold_l2 = self.find_optimal_threshold(y_test, y_prob_l2, metric='f1')
        y_pred_l2 = (y_prob_l2 >= optimal_threshold_l2).astype(int)
        
        self.models['ridge'] = {
            'model': best_ridge,
            'threshold': optimal_threshold_l2,
            'y_prob': y_prob_l2,
            'y_pred': y_pred_l2,
            'best_C': grid_search_l2.best_params_['C']
        }
        
        print(f"Best C for L2: {grid_search_l2.best_params_['C']:.6f}")
        self._evaluate_model('ridge', y_test, y_pred_l2, y_prob_l2)
        
        # Visualizations for Ridge
        self.plot_confusion_matrix(y_test, y_pred_l2, 'Ridge (L2)')
        self.plot_roc_curve(y_test, y_prob_l2, 'Ridge (L2)')
        self.plot_feature_importance(best_ridge, X_train.columns, 'Ridge (L2)')
        
        return best_lasso, best_ridge
    
    def train_sampling_models(self, X_train, X_test, y_train, y_test):
        """Train models with SMOTE sampling"""
        print("\n" + "="*60)
        print("3. SAMPLING-BASED MODELS")
        print("="*60)
        
        print("\nTraining SMOTE Model...")
        smote = SMOTE(random_state=self.random_state)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Plot SMOTE effect
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        y_train.value_counts().plot(kind='bar', ax=ax1, color=['lightcoral', 'lightblue'])
        ax1.set_title('Before SMOTE')
        ax1.set_xlabel('Response')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        
        pd.Series(y_train_smote).value_counts().plot(kind='bar', ax=ax2, color=['lightcoral', 'lightblue'])
        ax2.set_title('After SMOTE')
        ax2.set_xlabel('Response')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        smote_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        smote_model.fit(X_train_smote, y_train_smote)
        
        y_prob_smote = smote_model.predict_proba(X_test)[:, 1]
        optimal_threshold_smote = self.find_optimal_threshold(y_test, y_prob_smote, metric='f1')
        y_pred_smote = (y_prob_smote >= optimal_threshold_smote).astype(int)
        
        self.models['smote'] = {
            'model': smote_model,
            'threshold': optimal_threshold_smote,
            'y_prob': y_prob_smote,
            'y_pred': y_pred_smote
        }
        
        print(f"SMOTE resampled data shape: {X_train_smote.shape}")
        print(f"SMOTE class distribution: {np.bincount(y_train_smote)}")
        self._evaluate_model('smote', y_test, y_pred_smote, y_prob_smote)
        
        # Visualizations for SMOTE
        self.plot_confusion_matrix(y_test, y_pred_smote, 'SMOTE')
        self.plot_roc_curve(y_test, y_prob_smote, 'SMOTE')
        self.plot_feature_importance(smote_model, X_train.columns, 'SMOTE')
        
        return smote_model
    
    def _evaluate_model(self, model_name, y_true, y_pred, y_prob):
        """Comprehensive model evaluation"""
        accuracy = (y_pred == y_true).mean()
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        
        print(f"\n{model_name.upper()} MODEL RESULTS:")
        print(f"Optimal Threshold: {self.models[model_name]['threshold']:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"PR-AUC:    {pr_auc:.4f}")
        
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

class AdvancedMarketingPredictor(VisualizationMixin):
    """Advanced predictor with additional methods and visualizations"""
    def __init__(self, X_train, X_test, y_train, y_test, random_state=42):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random_state
        self.advanced_models = {}
        self.advanced_results = {}
        self.setup_plot_style()
        
    def find_optimal_threshold(self, y_true, y_prob, metric='f1'):
        """Find optimal threshold based on specified metric"""
        if metric == 'f1':
            thresholds = np.arange(0.1, 1.0, 0.01)
            f1_scores = []
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                f1_scores.append(f1_score(y_true, y_pred))
            optimal_idx = np.argmax(f1_scores)
            return thresholds[optimal_idx]
        elif metric == 'youden':
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            optimal_idx = np.argmax(tpr - fpr)
            return thresholds[optimal_idx]
    
    def plot_calibration_curve(self, y_true, y_prob, model_name):
        """Plot calibration curve"""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label=f"{model_name}", linewidth=2, markersize=8)
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(f"Calibration Plot - {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def method_1_elastic_net(self):
        """Elastic Net: Combines L1 and L2 regularization"""
        print("\n" + "="*60)
        print("METHOD 1: ELASTIC NET REGULARIZATION")
        print("="*60)
        
        param_grid = {
            'C': np.logspace(-3, 1, 20),
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        
        elastic_model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            class_weight='balanced',
            max_iter=2000,
            random_state=self.random_state
        )
        
        grid_search = GridSearchCV(
            elastic_model, param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        best_elastic = grid_search.best_estimator_
        
        y_prob = best_elastic.predict_proba(self.X_test)[:, 1]
        optimal_threshold = self.find_optimal_threshold(self.y_test, y_prob, 'f1')
        y_pred = (y_prob >= optimal_threshold).astype(int)
        
        self.advanced_models['elastic_net'] = {
            'model': best_elastic,
            'threshold': optimal_threshold,
            'y_prob': y_prob,
            'y_pred': y_pred,
            'best_params': grid_search.best_params_
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        self._evaluate_advanced_model('elastic_net', y_pred, y_prob)
        
        # Visualizations for Elastic Net
        self.plot_confusion_matrix(self.y_test, y_pred, 'Elastic Net')
        self.plot_roc_curve(self.y_test, y_prob, 'Elastic Net')
        self.plot_feature_importance(best_elastic, self.X_train.columns, 'Elastic Net')
        self.plot_calibration_curve(self.y_test, y_prob, 'Elastic Net')
        
        return best_elastic
    
    def method_2_ensemble_voting(self):
        """Ensemble of different logistic regression configurations"""
        print("\n" + "="*60)
        print("METHOD 2: ENSEMBLE VOTING CLASSIFIER")
        print("="*60)
        
        models = [
            ('lasso', LogisticRegression(penalty='l1', solver='liblinear', 
                                       C=0.1, class_weight='balanced', max_iter=1000,
                                       random_state=self.random_state)),
            ('ridge', LogisticRegression(penalty='l2', C=0.1, class_weight='balanced',
                                       max_iter=1000, random_state=self.random_state)),
            ('elastic', LogisticRegression(penalty='elasticnet', solver='saga',
                                         C=0.1, l1_ratio=0.5, class_weight='balanced',
                                         max_iter=2000, random_state=self.random_state))
        ]
        
        voting_clf = VotingClassifier(estimators=models, voting='soft')
        
        smote = SMOTE(random_state=self.random_state)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        
        voting_clf.fit(X_train_smote, y_train_smote)
        
        y_prob = voting_clf.predict_proba(self.X_test)[:, 1]
        optimal_threshold = self.find_optimal_threshold(self.y_test, y_prob, 'f1')
        y_pred = (y_prob >= optimal_threshold).astype(int)
        
        self.advanced_models['ensemble_voting'] = {
            'model': voting_clf,
            'threshold': optimal_threshold,
            'y_prob': y_prob,
            'y_pred': y_pred
        }
        
        self._evaluate_advanced_model('ensemble_voting', y_pred, y_prob)
        
        # Visualizations for Ensemble
        self.plot_confusion_matrix(self.y_test, y_pred, 'Ensemble Voting')
        self.plot_roc_curve(self.y_test, y_prob, 'Ensemble Voting')
        self.plot_calibration_curve(self.y_test, y_prob, 'Ensemble Voting')
        
        return voting_clf
    
    def method_3_calibrated_probabilities(self):
        """Calibrated probabilities for better threshold selection"""
        print("\n" + "="*60)
        print("METHOD 3: PROBABILITY CALIBRATION")
        print("="*60)
        
        base_model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=0.1,
            class_weight='balanced',
            max_iter=1000,
            random_state=self.random_state
        )
        
        smote = SMOTE(random_state=self.random_state)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        
        calibrated_clf = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
        calibrated_clf.fit(X_train_smote, y_train_smote)
        
        y_prob = calibrated_clf.predict_proba(self.X_test)[:, 1]
        optimal_threshold = self.find_optimal_threshold(self.y_test, y_prob, 'f1')
        y_pred = (y_prob >= optimal_threshold).astype(int)
        
        self.advanced_models['calibrated'] = {
            'model': calibrated_clf,
            'threshold': optimal_threshold,
            'y_prob': y_prob,
            'y_pred': y_pred
        }
        
        self._evaluate_advanced_model('calibrated', y_pred, y_prob)
        
        # Visualizations for Calibrated model
        self.plot_confusion_matrix(self.y_test, y_pred, 'Calibrated')
        self.plot_roc_curve(self.y_test, y_prob, 'Calibrated')
        self.plot_calibration_curve(self.y_test, y_prob, 'Calibrated')
        
        return calibrated_clf
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot enhanced confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Response', 'Response'],
                   yticklabels=['No Response', 'Response'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                plt.text(j+0.5, i+0.7, f'({cm[i,j]/total:.1%})', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true, y_prob, model_name):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _evaluate_advanced_model(self, model_name, y_pred, y_prob):
        """Evaluate advanced model performance"""
        accuracy = (y_pred == self.y_test).mean()
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_prob)
        pr_auc = average_precision_score(self.y_test, y_prob)
        
        self.advanced_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        
        print(f"\n{model_name.upper().replace('_', ' ')} RESULTS:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"PR-AUC:    {pr_auc:.4f}")
        
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"Confusion Matrix: TN:{cm[0,0]}, FP:{cm[0,1]}, FN:{cm[1,0]}, TP:{cm[1,1]}")
    
    def run_all_advanced_methods(self):
        """Run all advanced methods"""
        print("Starting Advanced Logistic Regression Methods...")
        
        try:
            self.method_1_elastic_net()
        except Exception as e:
            print(f"Elastic Net failed: {e}")
        
        try:
            self.method_2_ensemble_voting()
        except Exception as e:
            print(f"Ensemble Voting failed: {e}")
        
        try:
            self.method_3_calibrated_probabilities()
        except Exception as e:
            print(f"Probability Calibration failed: {e}")
    
    def plot_model_comparison_charts(self, original_results):
        """Create comprehensive comparison visualizations"""
        all_results = {**original_results, **self.advanced_results}
        
        # Metrics comparison bar chart
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        models = list(all_results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [all_results[model][metric] for model in models]
            bars = axes[i].bar(models, values, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # ROC curves comparison
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, len(all_results)))
        
        for i, (model_name, color) in enumerate(zip(all_results.keys(), colors)):
            if model_name in self.advanced_models:
                y_prob = self.advanced_models[model_name]['y_prob']
            else:
                continue  # Skip if we don't have probabilities stored
                
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            roc_auc = roc_auc_score(self.y_test, y_prob)
            plt.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison - All Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Performance radar chart
        from math import pi
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Select top 4 models for clarity
        sorted_models = sorted(all_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:4]
        
        angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]  # Complete the circle
        
        for model_name, results in sorted_models:
            values = [results[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart - Top 4 Models', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        plt.show()
    
    def plot_business_impact_analysis(self, original_results):
        """Create business impact visualizations"""
        all_results = {**original_results, **self.advanced_results}
        
        # Cost-Benefit Analysis (assuming some business metrics)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        models = list(all_results.keys())
        precisions = [all_results[model]['precision'] for model in models]
        recalls = [all_results[model]['recall'] for model in models]
        f1_scores = [all_results[model]['f1_score'] for model in models]
        
        # Precision vs Recall scatter plot
        scatter = ax1.scatter(recalls, precisions, s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
        ax1.set_xlabel('Recall (True Positive Rate)')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision vs Recall Trade-off')
        ax1.grid(True, alpha=0.3)
        
        for i, model in enumerate(models):
            ax1.annotate(model, (recalls[i], precisions[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        
        # F1-Score ranking
        sorted_models = sorted(zip(models, f1_scores), key=lambda x: x[1], reverse=True)
        models_sorted, f1_sorted = zip(*sorted_models)
        
        bars = ax2.barh(models_sorted, f1_sorted, color=plt.cm.RdYlGn(np.array(f1_sorted)))
        ax2.set_xlabel('F1-Score')
        ax2.set_title('F1-Score Ranking')
        ax2.grid(True, alpha=0.3)
        
        for i, (bar, score) in enumerate(zip(bars, f1_sorted)):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontsize=9)
        
        # Model complexity vs performance (proxy using number of features selected)
        complexities = []
        for model_name in models:
            if model_name in self.advanced_models:
                model = self.advanced_models[model_name]['model']
            else:
                continue
                
            if hasattr(model, 'coef_'):
                # Count non-zero coefficients as complexity measure
                non_zero_coefs = np.sum(np.abs(model.coef_[0]) > 1e-6)
                complexities.append(non_zero_coefs)
            else:
                complexities.append(len(self.X_train.columns))  # Default complexity
        
        if len(complexities) == len(f1_scores):
            ax3.scatter(complexities, f1_scores, s=100, alpha=0.7, c=range(len(models)), cmap='plasma')
            ax3.set_xlabel('Model Complexity (Non-zero Coefficients)')
            ax3.set_ylabel('F1-Score')
            ax3.set_title('Model Complexity vs Performance')
            ax3.grid(True, alpha=0.3)
            
            for i, model in enumerate(models):
                if i < len(complexities):
                    ax3.annotate(model, (complexities[i], f1_scores[i]), xytext=(5, 5),
                                textcoords='offset points', fontsize=9)
        
        # Business value estimation (hypothetical)
        # Assuming: TP = $100 value, FP = -$20 cost, FN = -$50 opportunity cost
        business_values = []
        for model_name in models:
            if model_name in self.advanced_models:
                y_pred = self.advanced_models[model_name]['y_pred']
            else:
                continue  # Skip if we don't have predictions
                
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Hypothetical business value calculation
            business_value = tp * 100 - fp * 20 - fn * 50
            business_values.append(business_value)
        
        if len(business_values) > 0:
            valid_models = models[:len(business_values)]
            bars = ax4.bar(valid_models, business_values, 
                          color=['green' if v > 0 else 'red' for v in business_values])
            ax4.set_ylabel('Business Value ($)')
            ax4.set_title('Estimated Business Value by Model')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, business_values):
                ax4.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (max(business_values) * 0.01 if value > 0 else min(business_values) * 0.01),
                        f'${value:,.0f}', ha='center', va='bottom' if value > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def compare_all_methods(self, original_results):
        """Compare all methods including original ones"""
        print("\n" + "="*80)
        print("COMPREHENSIVE COMPARISON: ORIGINAL + ADVANCED METHODS")
        print("="*80)
        
        all_results = {**original_results, **self.advanced_results}
        comparison_df = pd.DataFrame(all_results).T
        comparison_df = comparison_df.round(4)
        
        print("\nComplete Results Table:")
        print("-" * 80)
        print(comparison_df.to_string())
        
        print("\n" + "="*60)
        print("BEST MODELS BY OBJECTIVE")
        print("="*60)
        
        best_precision = comparison_df['precision'].idxmax()
        best_recall = comparison_df['recall'].idxmax()
        best_f1 = comparison_df['f1_score'].idxmax()
        best_roc_auc = comparison_df['roc_auc'].idxmax()
        
        print(f"\nBest for PRECISION: {best_precision} ({comparison_df.loc[best_precision, 'precision']:.4f})")
        print(f"Best for RECALL: {best_recall} ({comparison_df.loc[best_recall, 'recall']:.4f})")
        print(f"Best for F1-SCORE: {best_f1} ({comparison_df.loc[best_f1, 'f1_score']:.4f})")
        print(f"Best for ROC-AUC: {best_roc_auc} ({comparison_df.loc[best_roc_auc, 'roc_auc']:.4f})")
        
        # Generate comprehensive visualizations
        self.plot_model_comparison_charts(original_results)
        self.plot_business_impact_analysis(original_results)
        
        return comparison_df

def main():
    """Main function to run the complete analysis with all visualizations"""
    
    # Input data: first argument, or data/marketing_campaign_clean.csv by default
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/marketing_campaign_clean.csv"
    
    print("="*80)
    print("ADVANCED MARKETING RESPONSE PREDICTION ANALYSIS")
    print("WITH COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    
    try:
        # Step 1: Initialize base predictor and run original methods
        print("\nSTEP 1: RUNNING ORIGINAL METHODS")
        print("-" * 50)
        
        base_predictor = MarketingResponsePredictor(random_state=42)
        X, y = base_predictor.load_and_preprocess_data(filepath)
        X_train, X_test, y_train, y_test = base_predictor.split_and_scale_data(X, y)
        
        # Train original models with visualizations
        base_predictor.train_baseline_model(X_train, X_test, y_train, y_test)
        base_predictor.train_regularized_models(X_train, X_test, y_train, y_test)
        base_predictor.train_sampling_models(X_train, X_test, y_train, y_test)
        
        # Step 2: Run advanced methods with visualizations
        print("\n\nSTEP 2: RUNNING ADVANCED METHODS")
        print("-" * 50)
        
        advanced_predictor = AdvancedMarketingPredictor(
            X_train, X_test, y_train, y_test, random_state=42
        )
        
        advanced_predictor.run_all_advanced_methods()
        
        # Step 3: Compare all methods with comprehensive visualizations
        print("\n\nSTEP 3: COMPREHENSIVE COMPARISON")
        print("-" * 50)
        
        comparison_df = advanced_predictor.compare_all_methods(base_predictor.results)
        
        # Step 4: Generate business insights
        print("\n\nSTEP 4: BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("-" * 50)
        
        # Find overall best model
        all_results = {**base_predictor.results, **advanced_predictor.advanced_results}
        best_model_name = max(all_results.keys(), key=lambda x: all_results[x]['f1_score'])
        best_score = all_results[best_model_name]['f1_score']
        
        print(f"\nOVERALL BEST MODEL: {best_model_name.upper()}")
        print(f"F1-Score: {best_score:.4f}")
        
        # Business recommendations
        print("\nBUSINESS RECOMMENDATIONS:")
        print("1. Use the best model for customer targeting")
        print("2. Focus on high-probability customers first")
        print("3. Monitor model performance regularly")
        print("4. Consider cost-benefit analysis for campaign ROI")
        print("5. Implement A/B testing to validate model predictions")
        print("6. Regular model retraining with new data")
        
        # Save results
        print("\nSAVING RESULTS:")
        Path('results').mkdir(exist_ok=True)
        comparison_df.to_csv('results/model_comparison_results.csv')
        print("Results saved to 'results/model_comparison_results.csv'")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("All visualizations have been generated.")
        print("="*80)
        
        return base_predictor, advanced_predictor, comparison_df
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {filepath}")
        print("Please check the file path and try again.")
        return None, None, None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None, None

# Run the analysis
if __name__ == "__main__":
    base_pred, adv_pred, results_df = main()