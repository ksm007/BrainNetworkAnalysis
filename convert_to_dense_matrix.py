"""
Brain Connectivity Matrix Processing
====================================

Utilities for processing .mat brain connectivity files and extracting
network topology features.

Author: Brain Network Analysis Team
Version: 1.0
"""

import scipy.io
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import networkx as nx


def extract_features_from_subject(mat_file):
    mat_data = scipy.io.loadmat(mat_file)
    connectivity = mat_data['fibergraph']
    connectivity_dense = connectivity.toarray()
    #print(f"Shape : {connectivity_dense.shape}")
    #print(f"Non-zero connections: {np.count_nonzero(connectivity_dense)}")
    #print(connectivity_dense)
    connectivity_symmetric = (connectivity_dense + connectivity_dense.T)/2
    #print(connectivity_symmetric)
    features = {}
    degree = np.sum(connectivity_symmetric > 0, axis = 1)
    features['avg_degree'] = np.mean(degree)
    features['max_degree'] = np.max(degree)
    features['std_degree'] = np.std(degree)
    
    strength = np.sum(connectivity_symmetric, axis=1)
    features['avg_strength'] = np.mean(strength)
    features['max_strength'] = np.max(strength)
    features['total_strength'] = np.sum(strength)
    features['std_strength'] = np.std(strength)
    
    n = connectivity_symmetric.shape[0]
    features['density'] = np.count_nonzero(connectivity_symmetric)/(n * (n-1))
    features['avg_edge_weight'] = np.mean(connectivity_symmetric[connectivity_symmetric>0])
    G = nx.from_numpy_array(connectivity_symmetric)
    features['clustering_coef'] = nx.average_clustering(G, weight='weight')
    features['transitivity'] = nx.transitivity(G)
    
    betweenness = nx.betweenness_centrality(G, weight='weight')
    features['avg_betweenness'] = np.mean(list(betweenness.values()))
    
    eigenvector = nx.eigenvector_centrality_numpy(G, weight='weight')
    features['avg_eigenvector'] = np.mean(list(eigenvector.values()))
    
    return features