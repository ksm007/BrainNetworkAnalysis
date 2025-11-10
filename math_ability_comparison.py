"""
🧠 IMPROVED MATHEMATICAL ABILITY PREDICTION: Using FSIQ
======================================================

Based on the analysis, let's test FSIQ for mathematical ability prediction
and compare it with the current CCI-based approach.

Author: Brain Network Analysis Team
Version: 4.0 (FSIQ Optimization)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load data and prepare both CCI and FSIQ targets."""
    print("🧠 MATHEMATICAL ABILITY PREDICTION: CCI vs FSIQ COMPARISON")
    print("="*70)
    
    # Load features and metadata
    features_df = pd.read_csv('enhanced_brain_features.csv', index_col=0)
    metainfo = pd.read_csv('metainfo.csv')
    
    # Merge data
    merged_df = features_df.merge(metainfo[['URSI', 'Sex', 'Age', 'CCI', 'FSIQ', 'CAQ']], 
                                  left_index=True, right_on='URSI', how='inner')
    
    # Remove subjects with missing cognitive data
    merged_df = merged_df.dropna(subset=['CCI', 'FSIQ'])
    
    print(f"📊 Dataset: {merged_df.shape[0]} subjects, {features_df.shape[1]} brain features")
    print(f"📊 CCI range: {merged_df['CCI'].min():.1f} - {merged_df['CCI'].max():.1f}")
    print(f"📊 FSIQ range: {merged_df['FSIQ'].min():.1f} - {merged_df['FSIQ'].max():.1f}")
    
    return merged_df, features_df.columns

def create_advanced_features(X):
    """Enhanced feature engineering for better performance."""
    X_advanced = X.copy()
    
    # Interaction terms
    top_features = ['avg_edge_weight', 'max_strength', 'total_strength', 'avg_strength', 
                   'transitivity', 'density', 'std_degree', 'clustering_coef']
    
    for i, feat1 in enumerate(top_features[:5]):
        for feat2 in top_features[i+1:6]:
            if feat1 in X.columns and feat2 in X.columns:
                X_advanced[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
    
    # Ratio features
    if 'max_strength' in X.columns and 'avg_strength' in X.columns:
        X_advanced['strength_dominance'] = X['max_strength'] / (X['avg_strength'] + 1e-8)
    
    if 'std_degree' in X.columns and 'avg_degree' in X.columns:
        X_advanced['degree_variability'] = X['std_degree'] / (X['avg_degree'] + 1e-8)
    
    # Power transformations
    for feature in ['avg_edge_weight', 'max_strength', 'density']:
        if feature in X.columns:
            X_advanced[f'{feature}_squared'] = X[feature] ** 2
            X_advanced[f'{feature}_sqrt'] = np.sqrt(X[feature] + 1e-8)
    
    # Statistical aggregates
    X_advanced['feature_sum'] = X.sum(axis=1)
    X_advanced['feature_mean'] = X.mean(axis=1)
    X_advanced['feature_std'] = X.std(axis=1)
    X_advanced['feature_max'] = X.max(axis=1)
    X_advanced['feature_range'] = X_advanced['feature_max'] - X.min(axis=1)
    
    return X_advanced

def test_cognitive_measure(X, y, measure_name, threshold_name):
    """Test multiple algorithms for a given cognitive measure."""
    print(f"\n🎯 TESTING {measure_name} ({threshold_name})...")
    
    class_dist = pd.Series(y).value_counts().sort_index()
    print(f"   📊 Class distribution: {dict(class_dist)}")
    
    # Define models with optimized parameters
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=500, max_depth=20, min_samples_split=2,
            class_weight='balanced_subsample', random_state=42
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=500, max_depth=25, min_samples_split=2,
            class_weight='balanced_subsample', random_state=42
        ),
        'SVM': SVC(
            C=20, gamma='scale', kernel='rbf',
            class_weight='balanced', probability=True, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            C=5, penalty='l2', solver='saga', max_iter=2000,
            class_weight='balanced', random_state=42
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(150, 75), activation='relu', alpha=0.001,
            learning_rate='adaptive', max_iter=2000, random_state=42,
            early_stopping=True
        )
    }
    
    # Test each model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        # Create pipeline with scaling and feature selection
        pipeline = Pipeline([
            ('scaler', RobustScaler() if name != 'Neural Network' else StandardScaler()),
            ('selector', SelectKBest(f_classif, k=min(30, X.shape[1]))),
            ('classifier', model)
        ])
        
        # Cross-validation
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        
        print(f"   {name}: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['mean'])
    print(f"   🏆 BEST: {best_model[0]} ({best_model[1]['mean']:.3f})")
    
    return results, best_model

def comprehensive_comparison():
    """Compare all mathematical ability measures comprehensively."""
    
    # Load data
    merged_df, feature_cols = load_and_prepare_data()
    
    # Prepare features
    X = merged_df[feature_cols].fillna(merged_df[feature_cols].median())
    X_engineered = create_advanced_features(X)
    
    print(f"\n🔧 FEATURE ENGINEERING: {len(feature_cols)} → {X_engineered.shape[1]} features")
    
    # Define all mathematical ability targets
    targets = {
        'CCI_75': (merged_df['CCI'] >= merged_df['CCI'].quantile(0.75), 'Current approach (75th percentile)'),
        'FSIQ_75': (merged_df['FSIQ'] >= merged_df['FSIQ'].quantile(0.75), 'Alternative approach (75th percentile)'),
        'CCI_80': (merged_df['CCI'] >= merged_df['CCI'].quantile(0.80), 'More selective CCI (80th percentile)'),
        'FSIQ_80': (merged_df['FSIQ'] >= merged_df['FSIQ'].quantile(0.80), 'More selective FSIQ (80th percentile)'),
        'CCI_high': (merged_df['CCI'] >= 110, 'High cognitive capacity (CCI≥110)'),
        'FSIQ_high': (merged_df['FSIQ'] >= 120, 'Superior intelligence (FSIQ≥120)'),
        'Combined': ((merged_df['CCI'] >= merged_df['CCI'].quantile(0.75)) | 
                    (merged_df['FSIQ'] >= merged_df['FSIQ'].quantile(0.75)), 'Either CCI or FSIQ high')
    }
    
    # Test all targets
    all_results = {}
    
    for target_name, (y_target, description) in targets.items():
        y = y_target.astype(int).values
        results, best_model = test_cognitive_measure(X_engineered, y, target_name, description)
        all_results[target_name] = {
            'results': results,
            'best_model': best_model,
            'description': description,
            'class_distribution': pd.Series(y).value_counts().sort_index()
        }
    
    return all_results

def create_final_recommendations(all_results):
    """Create final recommendations based on comprehensive testing."""
    
    print(f"\n{'='*80}")
    print("🏆 FINAL RESULTS AND RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Summary table
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"{'Target':<15} {'Best Model':<18} {'Accuracy':<10} {'Description'}")
    print(f"{'-'*80}")
    
    performance_comparison = []
    
    for target_name, data in all_results.items():
        best_name, best_result = data['best_model']
        accuracy = best_result['mean']
        performance_comparison.append((target_name, best_name, accuracy, data['description']))
        
        print(f"{target_name:<15} {best_name:<18} {accuracy:.3f}     {data['description']}")
    
    # Find overall best
    best_overall = max(performance_comparison, key=lambda x: x[2])
    
    print(f"\n🎯 OPTIMAL MATHEMATICAL ABILITY PREDICTION:")
    print(f"   🏆 BEST APPROACH: {best_overall[0]} using {best_overall[1]}")
    print(f"   🎯 ACCURACY: {best_overall[2]:.1%}")
    print(f"   📋 DESCRIPTION: {best_overall[3]}")
    
    # Specific recommendations
    print(f"\n💡 DETAILED RECOMMENDATIONS:")
    
    # Compare CCI vs FSIQ directly
    cci_75_acc = all_results['CCI_75']['best_model'][1]['mean']
    fsiq_75_acc = all_results['FSIQ_75']['best_model'][1]['mean']
    
    if fsiq_75_acc > cci_75_acc:
        improvement = ((fsiq_75_acc - cci_75_acc) / cci_75_acc) * 100
        print(f"   ✅ RECOMMENDATION: Switch from CCI to FSIQ for Math_75")
        print(f"      • FSIQ gives {improvement:.1f}% better accuracy ({fsiq_75_acc:.3f} vs {cci_75_acc:.3f})")
        print(f"      • FSIQ is the gold standard intelligence measure")
        print(f"      • Better clinical and educational interpretability")
    elif cci_75_acc > fsiq_75_acc:
        improvement = ((cci_75_acc - fsiq_75_acc) / fsiq_75_acc) * 100
        print(f"   ✅ RECOMMENDATION: Keep using CCI for Math_75")
        print(f"      • CCI gives {improvement:.1f}% better accuracy ({cci_75_acc:.3f} vs {fsiq_75_acc:.3f})")
        print(f"      • CCI is more specific to cognitive processing")
        print(f"      • Better match for brain connectivity measures")
    else:
        print(f"   🤝 RECOMMENDATION: Either CCI or FSIQ can be used")
        print(f"      • Both measures give similar accuracy ({cci_75_acc:.3f})")
        print(f"      • Consider research goals and interpretability needs")
    
    # Additional insights
    print(f"\n🔍 ADDITIONAL INSIGHTS:")
    
    # Check 80th percentile performance
    cci_80_acc = all_results['CCI_80']['best_model'][1]['mean']
    fsiq_80_acc = all_results['FSIQ_80']['best_model'][1]['mean']
    
    print(f"   • 80th percentile thresholds: CCI={cci_80_acc:.3f}, FSIQ={fsiq_80_acc:.3f}")
    if max(cci_80_acc, fsiq_80_acc) > max(cci_75_acc, fsiq_75_acc):
        print(f"   • Consider using 80th percentile for higher accuracy")
    
    # Check combined approach
    combined_acc = all_results['Combined']['best_model'][1]['mean']
    print(f"   • Combined approach (CCI OR FSIQ high): {combined_acc:.3f}")
    
    print(f"\n📈 PRACTICAL IMPLICATIONS:")
    print(f"   • Your current Math_75 (CCI) accuracy: 76.4%")
    print(f"   • Optimized accuracy with best approach: {best_overall[2]:.1%}")
    
    if best_overall[2] > 0.764:
        improvement = ((best_overall[2] - 0.764) / 0.764) * 100
        print(f"   🚀 POTENTIAL IMPROVEMENT: +{improvement:.1f}% with optimal approach!")
    
    return best_overall

def main():
    """Run comprehensive mathematical ability prediction analysis."""
    
    # Run comprehensive comparison
    all_results = comprehensive_comparison()
    
    # Create recommendations
    best_approach = create_final_recommendations(all_results)
    
    print(f"\n✨ ANALYSIS COMPLETE!")
    print(f"   📊 Tested 7 different mathematical ability definitions")
    print(f"   🤖 Evaluated 5 machine learning algorithms for each")
    print(f"   🏆 Identified optimal approach: {best_approach[0]} ({best_approach[2]:.1%})")
    
    return all_results, best_approach

if __name__ == "__main__":
    all_results, best_approach = main()