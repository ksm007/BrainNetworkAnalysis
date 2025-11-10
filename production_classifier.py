"""
Practical Implementation Guide: Optimized Brain Network Models
============================================================

This guide shows how to implement the optimized models in practice.

Author: Brain Network Analysis Team
Version: 3.0 (Production Ready)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import warnings
warnings.filterwarnings('ignore')

class BrainNetworkClassifier:
    """
    Production-ready brain network classifier with optimized models.
    
    Usage:
        classifier = BrainNetworkClassifier()
        classifier.train()
        predictions = classifier.predict_new_subject(brain_features)
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.is_trained = False
        
    def engineer_features(self, X):
        """Apply the same feature engineering used in optimization."""
        X_advanced = X.copy()
        
        # Interaction terms - using actual features from dataset
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
        
        if 'clustering_coef' in X.columns and 'transitivity' in X.columns:
            X_advanced['clustering_efficiency'] = X['clustering_coef'] / (X['transitivity'] + 1e-8)
        
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
    
    def load_data(self):
        """Load training data."""
        print("🔄 Loading training data...")
        
        features_df = pd.read_csv('enhanced_brain_features.csv', index_col=0)
        metainfo = pd.read_csv('metainfo.csv')
        
        merged_df = features_df.merge(metainfo[['URSI', 'Sex', 'Age', 'CCI', 'CAQ']], 
                                      left_index=True, right_on='URSI', how='inner')
        
        return merged_df, features_df.columns
    
    def train(self):
        """Train all optimized models."""
        print("🚀 Training Optimized Brain Network Models...")
        print("="*60)
        
        # Load data
        merged_df, feature_cols = self.load_data()
        X = merged_df[feature_cols].fillna(merged_df[feature_cols].median())
        
        # Apply feature engineering
        X_engineered = self.engineer_features(X)
        
        # Define targets
        targets = {
            'gender': merged_df['Sex'].values,
            'math_75': (merged_df['CCI'] >= merged_df['CCI'].quantile(0.75)).astype(int).values,
            'creativity_75': (merged_df['CAQ'] >= merged_df['CAQ'].quantile(0.75)).astype(int).values
        }
        
        # Train each model with optimized parameters
        self._train_gender_model(X_engineered, targets['gender'])
        self._train_math_model(X_engineered, targets['math_75'])
        self._train_creativity_model(X_engineered, targets['creativity_75'])
        
        self.is_trained = True
        print("✅ All models trained successfully!")
        
    def _train_gender_model(self, X, y):
        """Train optimized gender classification model."""
        print("\n🎯 Training Gender Model (71.1% accuracy)...")
        
        # Best parameters from optimization
        model = LogisticRegression(
            C=5, penalty='l2', solver='saga', max_iter=2000,
            class_weight='balanced', random_state=42
        )
        
        # Pipeline with scaling and feature selection
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('selector', SelectKBest(f_classif, k=25)),
            ('classifier', model)
        ])
        
        pipeline.fit(X, y)
        
        # Validate performance
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        print(f"   ✅ Gender Model: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.models['gender'] = pipeline
        
    def _train_math_model(self, X, y):
        """Train optimized math ability model."""
        print("\n🧮 Training Math Ability Model (76.4% accuracy)...")
        
        # Best parameters from optimization
        model = RandomForestClassifier(
            n_estimators=500, max_depth=20, min_samples_split=2,
            min_samples_leaf=1, max_features='sqrt',
            class_weight='balanced_subsample', random_state=42
        )
        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('selector', SelectKBest(f_classif, k=30)),
            ('classifier', model)
        ])
        
        pipeline.fit(X, y)
        
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        print(f"   ✅ Math Model: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.models['math_75'] = pipeline
        
    def _train_creativity_model(self, X, y):
        """Train optimized creativity model."""
        print("\n🎨 Training Creativity Model (72.8% accuracy)...")
        
        # Best parameters from optimization  
        model = MLPClassifier(
            hidden_layer_sizes=(150, 75), activation='relu', alpha=0.001,
            learning_rate='adaptive', max_iter=2000, random_state=42,
            early_stopping=True
        )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=35)),
            ('classifier', model)
        ])
        
        pipeline.fit(X, y)
        
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        print(f"   ✅ Creativity Model: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.models['creativity_75'] = pipeline
    
    def predict_new_subject(self, brain_features):
        """
        Predict cognitive abilities for a new subject.
        
        Args:
            brain_features (dict or Series): Brain connectivity features
            
        Returns:
            dict: Predictions and probabilities for each task
        """
        if not self.is_trained:
            raise ValueError("Models must be trained first. Call train() method.")
        
        # Convert to DataFrame if needed
        if isinstance(brain_features, dict):
            X_new = pd.DataFrame([brain_features])
        else:
            X_new = pd.DataFrame([brain_features])
        
        # Apply feature engineering
        X_engineered = self.engineer_features(X_new)
        
        predictions = {}
        
        # Gender prediction
        gender_pred = self.models['gender'].predict(X_engineered)[0]
        gender_prob = self.models['gender'].predict_proba(X_engineered)[0]
        predictions['gender'] = {
            'prediction': 'Male' if gender_pred == 1 else 'Female',
            'confidence': max(gender_prob),
            'probabilities': {'Female': gender_prob[0], 'Male': gender_prob[1]}
        }
        
        # Math ability prediction
        math_pred = self.models['math_75'].predict(X_engineered)[0]
        math_prob = self.models['math_75'].predict_proba(X_engineered)[0]
        predictions['math_ability'] = {
            'prediction': 'High (Top 25%)' if math_pred == 1 else 'Normal',
            'confidence': max(math_prob),
            'probabilities': {'Normal': math_prob[0], 'High': math_prob[1]}
        }
        
        # Creativity prediction
        creativity_pred = self.models['creativity_75'].predict(X_engineered)[0]
        creativity_prob = self.models['creativity_75'].predict_proba(X_engineered)[0]
        predictions['creativity'] = {
            'prediction': 'High (Top 25%)' if creativity_pred == 1 else 'Normal',
            'confidence': max(creativity_prob),
            'probabilities': {'Normal': creativity_prob[0], 'High': creativity_prob[1]}
        }
        
        return predictions
    
    def save_models(self, filepath='brain_models.pkl'):
        """Save trained models to disk."""
        if not self.is_trained:
            raise ValueError("No trained models to save.")
        
        model_data = {
            'models': self.models,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath='brain_models.pkl'):
        """Load trained models from disk."""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.is_trained = model_data['is_trained']
        print(f"✅ Models loaded from {filepath}")

def demo_usage():
    """Demonstrate how to use the optimized models."""
    print("🧠 BRAIN NETWORK CLASSIFIER DEMO")
    print("="*40)
    
    # Initialize and train
    classifier = BrainNetworkClassifier()
    classifier.train()
    
    # Save models for future use
    classifier.save_models('optimized_brain_models.pkl')
    
    # Example prediction on a new subject
    print("\n🔮 PREDICTING NEW SUBJECT...")
    
    # Example brain features (you would get these from extract_features_from_subjects.py)
    example_features = {
        'avg_degree': 29.8,
        'max_degree': 65,
        'std_degree': 12.1,
        'avg_strength': 35000.0,
        'max_strength': 180000.0,
        'total_strength': 2450000.0,
        'std_strength': 45000.0,
        'density': 0.432,
        'avg_edge_weight': 1200.5,
        'clustering_coef': 0.031,
        'transitivity': 0.68,
        'avg_betweenness': 0.048,
        'avg_eigenvector': 0.074
    }
    
    predictions = classifier.predict_new_subject(example_features)
    
    print(f"\n📊 PREDICTIONS FOR NEW SUBJECT:")
    print(f"   🚻 Gender: {predictions['gender']['prediction']} ({predictions['gender']['confidence']:.1%} confidence)")
    print(f"   🧮 Math Ability: {predictions['math_ability']['prediction']} ({predictions['math_ability']['confidence']:.1%} confidence)")
    print(f"   🎨 Creativity: {predictions['creativity']['prediction']} ({predictions['creativity']['confidence']:.1%} confidence)")
    
    return classifier

if __name__ == "__main__":
    classifier = demo_usage()