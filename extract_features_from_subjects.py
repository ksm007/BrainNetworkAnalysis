"""
Brain Network Feature Extraction
================================

Extracts network features from all .mat files in mat_subjects/ directory.
Generates enhanced_brain_features.csv for analysis.

Author: Brain Network Analysis Team
Version: 1.0
"""

import glob
import os
import pandas as pd
from convert_to_dense_matrix import extract_features_from_subject


subject_files = glob.glob("mat_subjects/*.mat")
all_features = []

for file in subject_files:
    subject_id = os.path.basename(file).split('_')[0]
    features = extract_features_from_subject(file)
    features['URSI'] = subject_id
    all_features.append(features)
    
features_dataframe = pd.DataFrame(all_features)
#print(features_dataframe.head(10))