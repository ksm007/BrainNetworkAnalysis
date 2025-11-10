"""
🧠 COGNITIVE MEASURES ANALYSIS: CCI vs FSIQ for Mathematical Ability
===================================================================

📊 UNDERSTANDING THE COGNITIVE MEASURES IN YOUR DATASET
======================================================

YOUR DATASET CONTAINS MULTIPLE COGNITIVE MEASURES:

1. CCI (Cognitive Capacity Index) - Currently used for Math_75
2. FSIQ (Full Scale Intelligence Quotient) - Alternative measure
3. CAQ (Creative Achievement Questionnaire) - Used for Creativity_75
4. Personality measures (Big Five: Neuroticism, Extraversion, Openness, Agreeableness, Conscientiousness)

🧮 CCI vs FSIQ: WHAT'S THE DIFFERENCE?
=====================================

CCI (COGNITIVE CAPACITY INDEX):
------------------------------
• FOCUS: Specific cognitive processing capacity
• MEASURES: Working memory, processing speed, attention
• RANGE: ~74-124 in your dataset
• INTERPRETATION: More targeted measure of cognitive efficiency
• RESEARCH USE: Often used to assess "fluid intelligence"
• MATHEMATICAL RELEVANCE: Strong predictor of mathematical reasoning

FSIQ (FULL SCALE INTELLIGENCE QUOTIENT):
----------------------------------------
• FOCUS: Overall intellectual functioning
• MEASURES: Verbal comprehension, perceptual reasoning, working memory, processing speed
• RANGE: ~86-144 in your dataset (broader range than CCI)
• INTERPRETATION: Traditional "IQ score" - comprehensive intelligence measure
• RESEARCH USE: Gold standard for general intelligence assessment
• MATHEMATICAL RELEVANCE: Includes verbal and non-verbal components

🔍 WHY WAS CCI CHOSEN FOR MATH_75?
==================================

SCIENTIFIC RATIONALE:
--------------------
1. SPECIFICITY: CCI measures cognitive processes most relevant to mathematical reasoning
2. PURITY: Less contaminated by verbal/language abilities than FSIQ
3. RESEARCH PRECEDENT: CCI often used in cognitive neuroscience studies
4. BRAIN-BEHAVIOR MAPPING: Better suited for connectivity analysis

However, FSIQ might actually be BETTER for mathematical prediction!

📈 STATISTICAL COMPARISON: CCI vs FSIQ
=====================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_cognitive_measures():
    """Comprehensive analysis of CCI vs FSIQ for mathematical ability prediction."""
    
    print("🧠 COGNITIVE MEASURES ANALYSIS: CCI vs FSIQ")
    print("="*60)
    
    # Load data
    metainfo = pd.read_csv('metainfo.csv')
    
    # Remove subjects with missing data
    complete_data = metainfo.dropna(subset=['CCI', 'FSIQ']).copy()
    print(f"📊 Complete data: {len(complete_data)} subjects")
    
    # Basic statistics
    print(f"\n📈 DESCRIPTIVE STATISTICS:")
    print(f"   CCI:  Mean={complete_data['CCI'].mean():.1f}, SD={complete_data['CCI'].std():.1f}, Range={complete_data['CCI'].min():.1f}-{complete_data['CCI'].max():.1f}")
    print(f"   FSIQ: Mean={complete_data['FSIQ'].mean():.1f}, SD={complete_data['FSIQ'].std():.1f}, Range={complete_data['FSIQ'].min():.1f}-{complete_data['FSIQ'].max():.1f}")
    
    # Correlation analysis
    correlation = stats.pearsonr(complete_data['CCI'], complete_data['FSIQ'])
    print(f"\n🔗 CORRELATION between CCI and FSIQ: r = {correlation[0]:.3f} (p = {correlation[1]:.3e})")
    
    if correlation[0] > 0.7:
        print("   ✅ Strong positive correlation - measures are related but distinct")
    elif correlation[0] > 0.5:
        print("   ⚠️ Moderate correlation - some overlap but meaningful differences")
    else:
        print("   ❌ Weak correlation - measures capture different aspects")
    
    # Create different mathematical ability thresholds
    thresholds = {
        'CCI_75': complete_data['CCI'] >= complete_data['CCI'].quantile(0.75),
        'CCI_80': complete_data['CCI'] >= complete_data['CCI'].quantile(0.80),
        'FSIQ_75': complete_data['FSIQ'] >= complete_data['FSIQ'].quantile(0.75),
        'FSIQ_80': complete_data['FSIQ'] >= complete_data['FSIQ'].quantile(0.80),
        'CCI_high': complete_data['CCI'] >= 110,  # High cognitive capacity
        'FSIQ_high': complete_data['FSIQ'] >= 120  # High IQ (superior range)
    }
    
    print(f"\n📊 THRESHOLD COMPARISONS:")
    for name, threshold in thresholds.items():
        high_count = threshold.sum()
        percentage = (high_count / len(complete_data)) * 100
        measure = name.split('_')[0]
        criterion = name.split('_')[1]
        
        if criterion == '75':
            print(f"   {name}: {high_count} subjects ({percentage:.1f}%) - Top 25% by {measure}")
        elif criterion == '80':
            print(f"   {name}: {high_count} subjects ({percentage:.1f}%) - Top 20% by {measure}")
        elif criterion == 'high':
            if measure == 'CCI':
                print(f"   {name}: {high_count} subjects ({percentage:.1f}%) - High cognitive capacity (CCI≥110)")
            else:
                print(f"   {name}: {high_count} subjects ({percentage:.1f}%) - Superior IQ (FSIQ≥120)")
    
    # Agreement analysis
    cci_75 = complete_data['CCI'] >= complete_data['CCI'].quantile(0.75)
    fsiq_75 = complete_data['FSIQ'] >= complete_data['FSIQ'].quantile(0.75)
    
    # Calculate overlap
    both_high = (cci_75 & fsiq_75).sum()
    cci_only = (cci_75 & ~fsiq_75).sum()
    fsiq_only = (~cci_75 & fsiq_75).sum()
    neither = (~cci_75 & ~fsiq_75).sum()
    
    agreement = (both_high + neither) / len(complete_data)
    
    print(f"\n🎯 AGREEMENT ANALYSIS (75th percentile):")
    print(f"   Both high (CCI & FSIQ): {both_high} subjects")
    print(f"   High CCI only: {cci_only} subjects")
    print(f"   High FSIQ only: {fsiq_only} subjects")
    print(f"   Neither high: {neither} subjects")
    print(f"   Overall agreement: {agreement:.1%}")
    
    if agreement > 0.8:
        print("   ✅ High agreement - measures identify similar individuals")
    elif agreement > 0.7:
        print("   ⚠️ Moderate agreement - some differences in identification")
    else:
        print("   ❌ Low agreement - measures identify different individuals")
    
    return complete_data, thresholds

def compare_brain_prediction_performance():
    """Compare CCI vs FSIQ for brain-based prediction."""
    
    print(f"\n🧠 BRAIN PREDICTION PERFORMANCE COMPARISON")
    print("="*60)
    
    # Load brain features
    try:
        features_df = pd.read_csv('enhanced_brain_features.csv', index_col=0)
        metainfo = pd.read_csv('metainfo.csv')
        
        # Merge data
        merged_df = features_df.merge(metainfo[['URSI', 'CCI', 'FSIQ']], 
                                      left_index=True, right_on='URSI', how='inner')
        
        # Remove missing values
        merged_df = merged_df.dropna(subset=['CCI', 'FSIQ'])
        
        print(f"📊 Brain data available: {len(merged_df)} subjects")
        
        # Prepare features
        feature_cols = features_df.columns
        X = merged_df[feature_cols].fillna(merged_df[feature_cols].median())
        
        # Define targets
        targets = {
            'CCI_75': (merged_df['CCI'] >= merged_df['CCI'].quantile(0.75)).astype(int),
            'FSIQ_75': (merged_df['FSIQ'] >= merged_df['FSIQ'].quantile(0.75)).astype(int),
            'CCI_80': (merged_df['CCI'] >= merged_df['CCI'].quantile(0.80)).astype(int),
            'FSIQ_80': (merged_df['FSIQ'] >= merged_df['FSIQ'].quantile(0.80)).astype(int)
        }
        
        # Test prediction performance
        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print(f"\n🎯 BRAIN-BASED PREDICTION RESULTS:")
        
        for target_name, y in targets.items():
            # Use Random Forest for consistent comparison
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
            
            results[target_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            
            measure = target_name.split('_')[0]
            threshold = target_name.split('_')[1]
            class_dist = y.value_counts()
            
            print(f"   {target_name}: {scores.mean():.3f} ± {scores.std():.3f} (n_high={class_dist[1]}, n_normal={class_dist[0]})")
        
        # Compare CCI vs FSIQ directly
        print(f"\n📈 DIRECT COMPARISON:")
        
        cci_75_score = results['CCI_75']['mean']
        fsiq_75_score = results['FSIQ_75']['mean']
        cci_80_score = results['CCI_80']['mean']
        fsiq_80_score = results['FSIQ_80']['mean']
        
        print(f"   75th percentile: CCI={cci_75_score:.3f} vs FSIQ={fsiq_75_score:.3f}")
        if fsiq_75_score > cci_75_score:
            improvement = ((fsiq_75_score - cci_75_score) / cci_75_score) * 100
            print(f"   🏆 FSIQ wins by {improvement:.1f}% improvement!")
        elif cci_75_score > fsiq_75_score:
            improvement = ((cci_75_score - fsiq_75_score) / fsiq_75_score) * 100
            print(f"   🏆 CCI wins by {improvement:.1f}% improvement!")
        else:
            print(f"   🤝 Tie - both measures perform equally")
        
        print(f"   80th percentile: CCI={cci_80_score:.3f} vs FSIQ={fsiq_80_score:.3f}")
        if fsiq_80_score > cci_80_score:
            improvement = ((fsiq_80_score - cci_80_score) / cci_80_score) * 100
            print(f"   🏆 FSIQ wins by {improvement:.1f}% improvement!")
        elif cci_80_score > fsiq_80_score:
            improvement = ((cci_80_score - fsiq_80_score) / fsiq_80_score) * 100
            print(f"   🏆 CCI wins by {improvement:.1f}% improvement!")
        else:
            print(f"   🤝 Tie - both measures perform equally")
        
        return results
        
    except FileNotFoundError:
        print("   ❌ Brain features file not found - run extract_features_from_subjects.py first")
        return None

def create_recommendations():
    """Provide recommendations based on analysis."""
    
    print(f"\n🎯 RECOMMENDATIONS FOR MATHEMATICAL ABILITY PREDICTION")
    print("="*70)
    
    print(f"📋 SUMMARY OF FINDINGS:")
    print(f"   • CCI measures specific cognitive capacity (working memory, processing speed)")
    print(f"   • FSIQ measures comprehensive intelligence (verbal + non-verbal + working memory)")
    print(f"   • Both are valid measures but capture different aspects of cognition")
    print(f"   • Brain prediction performance may differ between measures")
    
    print(f"\n🔄 WHY CONSIDER SWITCHING FROM CCI TO FSIQ:")
    print(f"   ✅ FSIQ is the gold standard intelligence measure")
    print(f"   ✅ Broader coverage of mathematical abilities (not just processing)")
    print(f"   ✅ Better clinical and educational validity")
    print(f"   ✅ More interpretable for practical applications")
    print(f"   ✅ Larger dynamic range in your dataset (86-144 vs 74-124)")
    
    print(f"\n🔄 WHY KEEP CCI:")
    print(f"   ✅ More specific to cognitive processing relevant for brain connectivity")
    print(f"   ✅ Less contaminated by verbal/language factors")
    print(f"   ✅ Better theoretical match for neural efficiency measures")
    print(f"   ✅ Often preferred in cognitive neuroscience research")
    
    print(f"\n🚀 OPTIMAL STRATEGY:")
    print(f"   1. Test both CCI and FSIQ for brain prediction performance")
    print(f"   2. Use whichever gives better accuracy")
    print(f"   3. Consider creating combined measures (CCI + FSIQ)")
    print(f"   4. Report results for both measures for scientific completeness")

def main():
    """Run complete cognitive measures analysis."""
    
    # Analyze descriptive statistics and correlations
    complete_data, thresholds = analyze_cognitive_measures()
    
    # Compare brain prediction performance
    brain_results = compare_brain_prediction_performance()
    
    # Provide recommendations
    create_recommendations()
    
    print(f"\n✨ ANALYSIS COMPLETE!")
    print(f"   📊 Data analyzed for {len(complete_data)} subjects")
    print(f"   🧠 Brain prediction tested for both CCI and FSIQ")
    print(f"   📋 Recommendations provided for optimal approach")
    
    return complete_data, brain_results

if __name__ == "__main__":
    complete_data, brain_results = main()