from extract_features_from_subjects import features_dataframe
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

meta_df = pd.read_csv('metainfo.csv')

data = pd.merge(features_dataframe, meta_df, on='URSI')
#86, no small
print(data.head())

X = data[['avg_degree', 'max_degree', 'std_degree', 'avg_strength', 
          'max_strength', 'std_strength', 'total_strength', 'density', 
          'avg_edge_weight', 'clustering_coef', 'transitivity',
          'avg_betweenness', 'avg_eigenvector']]
y_sex = data['Sex']

X_train, X_test, y_train, y_test = train_test_split(X, y_sex, test_size=0.3, random_state=42)
classification = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
classification.fit(X_train, y_train)

y_pred = classification.predict(X_test)
print(f"Gender classification Accuracy:{accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))