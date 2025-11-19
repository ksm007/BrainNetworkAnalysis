"""
Comprehensive Validation and Testing
====================================

Complete validation suite with:
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Cross-validation analysis
- Statistical tests
- Model comparison
- Feature importance
- Error analysis

Author: Brain Network Analysis Team
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (cross_val_score, cross_val_predict, 
                                     StratifiedKFold, learning_curve)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, auc, precision_recall_curve,
                             accuracy_score, f1_score, matthews_corrcoef)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveValidator:
    """Complete validation and testing suite."""
    
    def __init__(self, task_name):
        self.task_name = task_name
        self.results = {}
        
    def create_best_model(self):
        """Create the best performing model (Stacking Ensemble)."""
        
        # Base models
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=3,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1
        )
        
        et = RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_split=2,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1
        )
        
        gb = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1
        )
        
        svm = SVC(
            C=10, kernel='rbf', probability=True,
            class_weight='balanced', random_state=42
        )
        
        # Stacking ensemble
        stacking = StackingClassifier(
            estimators=[('rf', rf), ('et', et), ('gb', gb), ('svm', svm)],
            final_estimator=LogisticRegression(C=1.0, class_weight='balanced', 
                                               random_state=42),
            cv=5,
            n_jobs=-1
        )
        
        return stacking
    
    def plot_confusion_matrix(self, y_true, y_pred, title):
        """Plot confusion matrix with detailed statistics."""
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   cbar_kws={'label': 'Count'})
        ax1.set_title(f'{title}\nConfusion Matrix (Counts)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=11)
        ax1.set_xlabel('Predicted Label', fontsize=11)
        ax1.set_xticklabels(['Negative', 'Positive'])
        ax1.set_yticklabels(['Negative', 'Positive'])
        
        # Plot 2: Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens', ax=ax2,
                   cbar_kws={'label': 'Percentage (%)'})
        ax2.set_title(f'{title}\nConfusion Matrix (Percentages)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=11)
        ax2.set_xlabel('Predicted Label', fontsize=11)
        ax2.set_xticklabels(['Negative', 'Positive'])
        ax2.set_yticklabels(['Negative', 'Positive'])
        
        plt.tight_layout()
        
        # Save
        filename = f'confusion_matrix_{self.task_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   Saved: {filename}")
        
        return cm

    
    def plot_roc_curve(self, y_true, y_proba, title):
        """Plot ROC curve with AUC."""
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=11)
        plt.ylabel('True Positive Rate', fontsize=11)
        plt.title(f'{title}\nROC Curve', fontsize=12, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        # Save
        filename = f'roc_curve_{self.task_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   Saved: {filename}")
        plt.close()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_proba, title):
        """Plot Precision-Recall curve."""
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall', fontsize=11)
        plt.ylabel('Precision', fontsize=11)
        plt.title(f'{title}\nPrecision-Recall Curve', fontsize=12, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        
        # Save
        filename = f'pr_curve_{self.task_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   Saved: {filename}")
        plt.close()
        
        return pr_auc
    
    def plot_learning_curve(self, estimator, X, y, title):
        """Plot learning curve to diagnose bias/variance."""
        
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy', random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='r')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color='g')
        
        plt.xlabel('Training Set Size', fontsize=11)
        plt.ylabel('Accuracy Score', fontsize=11)
        plt.title(f'{title}\nLearning Curve', fontsize=12, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        # Save
        filename = f'learning_curve_{self.task_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   Saved: {filename}")
        plt.close()
    
    def cross_validation_analysis(self, estimator, X, y):
        """Detailed cross-validation analysis."""
        
        print(f"\n{'='*70}")
        print(f"CROSS-VALIDATION ANALYSIS: {self.task_name}")
        print(f"{'='*70}")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Multiple metrics
        metrics = {
            'accuracy': 'accuracy',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall',
            'roc_auc': 'roc_auc'
        }
        
        results = {}
        
        for metric_name, metric in metrics.items():
            try:
                scores = cross_val_score(estimator, X, y, cv=cv, 
                                        scoring=metric, n_jobs=-1)
                results[metric_name] = scores
                print(f"\n{metric_name.upper()}:")
                print(f"  Fold scores: {scores}")
                print(f"  Mean: {scores.mean():.4f}")
                print(f"  Std:  {scores.std():.4f}")
                print(f"  Min:  {scores.min():.4f}")
                print(f"  Max:  {scores.max():.4f}")
            except Exception as e:
                print(f"\n{metric_name.upper()}: Could not compute ({e})")
        
        return results
    
    def statistical_tests(self, y_true, y_pred):
        """Perform statistical tests on predictions."""
        
        print(f"\n{'='*70}")
        print(f"STATISTICAL TESTS: {self.task_name}")
        print(f"{'='*70}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mcc = matthews_corrcoef(y_true, y_pred)
        
        print(f"\nCONFUSION MATRIX:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision:   {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:      {recall:.4f} ({recall*100:.2f}%)")
        print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"  F1 Score:    {f1:.4f}")
        print(f"  MCC:         {mcc:.4f}")
        
        # Chi-square test for independence
        chi2, p_value = stats.chi2_contingency(cm)[:2]
        print(f"\nCHI-SQUARE TEST:")
        print(f"  Chi-square statistic: {chi2:.4f}")
        print(f"  P-value: {p_value:.6f}")
        if p_value < 0.001:
            print(f"  Result: Highly significant (p < 0.001) ✓")
        elif p_value < 0.05:
            print(f"  Result: Significant (p < 0.05) ✓")
        else:
            print(f"  Result: Not significant (p >= 0.05) ✗")
        
        # Binomial test (is accuracy better than chance?)
        n_correct = tp + tn
        n_total = tp + tn + fp + fn
        binom_result = stats.binomtest(n_correct, n_total, 0.5, alternative='greater')
        p_binom = binom_result.pvalue
        
        print(f"\nBINOMIAL TEST (vs. random chance):")
        print(f"  Correct predictions: {n_correct}/{n_total}")
        print(f"  P-value: {p_binom:.6f}")
        if p_binom < 0.001:
            print("  Result: Significantly better than chance (p < 0.001) ✓")
        elif p_binom < 0.05:
            print("  Result: Better than chance (p < 0.05) ✓")
        else:
            print("  Result: Not significantly better than chance ✗")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'mcc': mcc,
            'chi2': chi2,
            'chi2_pvalue': p_value,
            'binom_pvalue': p_binom
        }
    
    def error_analysis(self, X, y_true, y_pred, feature_names):
        """Analyze misclassified samples."""
        
        print(f"\n{'='*70}")
        print(f"ERROR ANALYSIS: {self.task_name}")
        print(f"{'='*70}")
        
        # Find misclassified samples
        errors = y_true != y_pred
        n_errors = errors.sum()
        
        print(f"\nMISCLASSIFICATION SUMMARY:")
        print(f"  Total samples: {len(y_true)}")
        print(f"  Correct: {(~errors).sum()} ({(~errors).sum()/len(y_true)*100:.1f}%)")
        print(f"  Errors: {n_errors} ({n_errors/len(y_true)*100:.1f}%)")
        
        if n_errors > 0:
            # False positives and false negatives
            fp = (y_pred == 1) & (y_true == 0)
            fn = (y_pred == 0) & (y_true == 1)
            
            print(f"\nERROR BREAKDOWN:")
            print(f"  False Positives: {fp.sum()} ({fp.sum()/n_errors*100:.1f}% of errors)")
            print(f"  False Negatives: {fn.sum()} ({fn.sum()/n_errors*100:.1f}% of errors)")
            
            # Analyze feature distributions for errors
            if len(feature_names) > 0:
                print(f"\nFEATURE ANALYSIS FOR ERRORS:")
                
                X_errors = X[errors]
                X_correct = X[~errors]
                
                # Compare mean feature values
                print(f"\n  Top 5 features with largest differences:")
                mean_diff = np.abs(X_errors.mean(axis=0) - X_correct.mean(axis=0))
                top_features = np.argsort(mean_diff)[-5:][::-1]
                
                for i, feat_idx in enumerate(top_features, 1):
                    feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature {feat_idx}"
                    diff = mean_diff[feat_idx]
                    print(f"    {i}. {feat_name}: {diff:.4f}")
    
    def validate_complete(self, X, y, feature_names=None):
        """Run complete validation suite."""
        
        print(f"\n{'#'*70}")
        print(f"# COMPREHENSIVE VALIDATION: {self.task_name}")
        print(f"{'#'*70}")
        
        # Create model
        print(f"\nCreating Stacking Ensemble model...")
        model = self.create_best_model()
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('selector', SelectKBest(f_classif, k=min(25, X.shape[1]))),
            ('pca', PCA(n_components=min(15, X.shape[1]), random_state=42)),
            ('classifier', model)
        ])
        
        # Cross-validation predictions
        print(f"\nGenerating cross-validation predictions...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
        
        # Get probability predictions for ROC/PR curves
        print(f"Generating probability predictions...")
        y_proba = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
        
        # 1. Confusion Matrix
        print(f"\n1. CONFUSION MATRIX")
        cm = self.plot_confusion_matrix(y, y_pred, self.task_name)
        
        # 2. ROC Curve
        print(f"\n2. ROC CURVE")
        roc_auc = self.plot_roc_curve(y, y_proba, self.task_name)
        
        # 3. Precision-Recall Curve
        print(f"\n3. PRECISION-RECALL CURVE")
        pr_auc = self.plot_precision_recall_curve(y, y_proba, self.task_name)
        
        # 4. Learning Curve
        print(f"\n4. LEARNING CURVE")
        self.plot_learning_curve(pipeline, X, y, self.task_name)
        
        # 5. Cross-validation Analysis
        cv_results = self.cross_validation_analysis(pipeline, X, y)
        
        # 6. Statistical Tests
        stats_results = self.statistical_tests(y, y_pred)
        
        # 7. Error Analysis
        if feature_names is not None:
            self.error_analysis(X, y, y_pred, feature_names)
        
        # 8. Classification Report
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION REPORT: {self.task_name}")
        print(f"{'='*70}")
        print(classification_report(y, y_pred, target_names=['Negative', 'Positive']))
        
        # Summary
        print(f"\n{'='*70}")
        print(f"VALIDATION SUMMARY: {self.task_name}")
        print(f"{'='*70}")
        print(f"  Accuracy:     {stats_results['accuracy']:.4f} ({stats_results['accuracy']*100:.2f}%)")
        print(f"  F1 Score:     {stats_results['f1']:.4f}")
        print(f"  ROC AUC:      {roc_auc:.4f}")
        print(f"  PR AUC:       {pr_auc:.4f}")
        print(f"  MCC:          {stats_results['mcc']:.4f}")
        print(f"  Significance: p = {stats_results['binom_pvalue']:.6f}")
        
        return {
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'cv_results': cv_results,
            'stats_results': stats_results
        }


def main():
    """Run comprehensive validation on all tasks."""
    
    print("="*70)
    print("COMPREHENSIVE VALIDATION SUITE")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    try:
        features_df = pd.read_csv('advanced_brain_features.csv')
    except FileNotFoundError:
        print("❌ advanced_brain_features.csv not found!")
        print("   Run: python3 advanced_feature_extraction.py")
        return
    
    metainfo = pd.read_csv('metainfo.csv')
    
    # Merge
    merged_df = features_df.merge(metainfo[['URSI', 'Sex', 'Age', 'CCI', 'CAQ']], 
                                  on='URSI', how='inner')
    
    print(f"   Loaded: {merged_df.shape}")
    
    # Prepare features
    feature_cols = [c for c in merged_df.columns 
                   if c not in ['URSI', 'Sex', 'Age', 'CCI', 'CAQ']]
    X = merged_df[feature_cols].fillna(merged_df[feature_cols].median())
    
    # Define tasks
    tasks = {
        'Creativity_75th': (merged_df['CAQ'] >= merged_df['CAQ'].quantile(0.75)).astype(int).values,
        'Math_75th': (merged_df['CCI'] >= merged_df['CCI'].quantile(0.75)).astype(int).values,
        'Sex': merged_df['Sex'].values,
    }
    
    # Run validation for each task
    all_results = {}
    
    for task_name, y in tasks.items():
        validator = ComprehensiveValidator(task_name)
        results = validator.validate_complete(X, y, feature_cols)
        all_results[task_name] = results
    
    # Final summary
    print(f"\n{'#'*70}")
    print(f"# FINAL SUMMARY - ALL TASKS")
    print(f"{'#'*70}")
    
    for task_name, results in all_results.items():
        print(f"\n{task_name}:")
        print(f"  ROC AUC: {results['roc_auc']:.4f}")
        print(f"  PR AUC:  {results['pr_auc']:.4f}")
        print(f"  Accuracy: {results['stats_results']['accuracy']:.4f}")
        print(f"  F1 Score: {results['stats_results']['f1']:.4f}")
    
    print(f"\n✅ Validation complete! Check generated PNG files for visualizations.")


if __name__ == "__main__":
    main()
