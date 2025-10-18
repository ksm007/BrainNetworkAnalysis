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