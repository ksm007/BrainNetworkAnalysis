"""
Advanced Brain Network Feature Extraction
=========================================

Enhanced feature extraction with:
- Advanced graph metrics (efficiency, modularity, rich club)
- Topological features (k-core, assortativity)
- Spectral features (eigenvalues, Laplacian)
- Community detection features
- Motif analysis

Author: Brain Network Analysis Team
Version: 2.0 (Advanced)
"""

import scipy.io
import numpy as np
import networkx as nx
from scipy import stats
from scipy.sparse import csgraph
import warnings
warnings.filterwarnings('ignore')


def extract_advanced_features(mat_file):
    """Extract comprehensive brain network features."""
    
    # Load connectivity matrix
    mat_data = scipy.io.loadmat(mat_file)
    connectivity = mat_data['fibergraph']
    connectivity_dense = connectivity.toarray()
    connectivity_symmetric = (connectivity_dense + connectivity_dense.T) / 2
    
    features = {}
    
    # ===== BASIC NETWORK METRICS =====
    degree = np.sum(connectivity_symmetric > 0, axis=1)
    features['avg_degree'] = np.mean(degree)
    features['max_degree'] = np.max(degree)
    features['std_degree'] = np.std(degree)
    features['median_degree'] = np.median(degree)
    
    strength = np.sum(connectivity_symmetric, axis=1)
    features['avg_strength'] = np.mean(strength)
    features['max_strength'] = np.max(strength)
    features['total_strength'] = np.sum(strength)
    features['std_strength'] = np.std(strength)
    features['median_strength'] = np.median(strength)
    
    n = connectivity_symmetric.shape[0]
    features['density'] = np.count_nonzero(connectivity_symmetric) / (n * (n-1))
    
    edge_weights = connectivity_symmetric[connectivity_symmetric > 0]
    features['avg_edge_weight'] = np.mean(edge_weights)
    features['std_edge_weight'] = np.std(edge_weights)
    features['median_edge_weight'] = np.median(edge_weights)
    features['max_edge_weight'] = np.max(edge_weights)
    
    # ===== GRAPH OBJECT =====
    G = nx.from_numpy_array(connectivity_symmetric)
    
    # ===== CLUSTERING & TRANSITIVITY =====
    features['clustering_coef'] = nx.average_clustering(G, weight='weight')
    features['transitivity'] = nx.transitivity(G)
    
    # Local clustering distribution
    local_clustering = list(nx.clustering(G, weight='weight').values())
    features['std_clustering'] = np.std(local_clustering)
    features['max_clustering'] = np.max(local_clustering)
    
    # ===== CENTRALITY MEASURES =====
    betweenness = nx.betweenness_centrality(G, weight='weight')
    features['avg_betweenness'] = np.mean(list(betweenness.values()))
    features['std_betweenness'] = np.std(list(betweenness.values()))
    features['max_betweenness'] = np.max(list(betweenness.values()))
    
    eigenvector = nx.eigenvector_centrality_numpy(G, weight='weight')
    features['avg_eigenvector'] = np.mean(list(eigenvector.values()))
    features['std_eigenvector'] = np.std(list(eigenvector.values()))
    
    # Closeness centrality
    try:
        closeness = nx.closeness_centrality(G, distance='weight')
        features['avg_closeness'] = np.mean(list(closeness.values()))
        features['std_closeness'] = np.std(list(closeness.values()))
    except:
        features['avg_closeness'] = 0
        features['std_closeness'] = 0
    
    # ===== EFFICIENCY MEASURES =====
    try:
        # Global efficiency
        features['global_efficiency'] = nx.global_efficiency(G)
        
        # Local efficiency
        local_eff = nx.local_efficiency(G)
        features['local_efficiency'] = local_eff
    except:
        features['global_efficiency'] = 0
        features['local_efficiency'] = 0
    
    # ===== ASSORTATIVITY =====
    try:
        features['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
    except:
        features['degree_assortativity'] = 0
    
    # ===== MODULARITY & COMMUNITY DETECTION =====
    try:
        # Greedy modularity communities
        communities = nx.community.greedy_modularity_communities(G, weight='weight')
        features['num_communities'] = len(communities)
        features['modularity'] = nx.community.modularity(G, communities, weight='weight')
        
        # Community size statistics
        community_sizes = [len(c) for c in communities]
        features['avg_community_size'] = np.mean(community_sizes)
        features['std_community_size'] = np.std(community_sizes)
    except:
        features['num_communities'] = 0
        features['modularity'] = 0
        features['avg_community_size'] = 0
        features['std_community_size'] = 0
    
    # ===== SPECTRAL FEATURES =====
    try:
        # Laplacian eigenvalues
        laplacian = nx.laplacian_matrix(G).toarray()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero
        
        features['spectral_gap'] = eigenvalues[1] if len(eigenvalues) > 1 else 0
        features['algebraic_connectivity'] = eigenvalues[1] if len(eigenvalues) > 1 else 0
        features['avg_eigenvalue'] = np.mean(eigenvalues)
        features['max_eigenvalue'] = np.max(eigenvalues)
        
        # Adjacency eigenvalues
        adj_eigenvalues = np.linalg.eigvalsh(connectivity_symmetric)
        features['spectral_radius'] = np.max(np.abs(adj_eigenvalues))
        features['trace'] = np.trace(connectivity_symmetric)
    except:
        features['spectral_gap'] = 0
        features['algebraic_connectivity'] = 0
        features['avg_eigenvalue'] = 0
        features['max_eigenvalue'] = 0
        features['spectral_radius'] = 0
        features['trace'] = 0
    
    # ===== RICH CLUB COEFFICIENT =====
    try:
        rich_club = nx.rich_club_coefficient(G, normalized=False)
        if rich_club:
            features['rich_club_coef'] = np.mean(list(rich_club.values()))
            features['max_rich_club'] = np.max(list(rich_club.values()))
        else:
            features['rich_club_coef'] = 0
            features['max_rich_club'] = 0
    except:
        features['rich_club_coef'] = 0
        features['max_rich_club'] = 0
    
    # ===== K-CORE DECOMPOSITION =====
    try:
        core_numbers = nx.core_number(G)
        features['max_k_core'] = np.max(list(core_numbers.values()))
        features['avg_k_core'] = np.mean(list(core_numbers.values()))
    except:
        features['max_k_core'] = 0
        features['avg_k_core'] = 0
    
    # ===== SMALL-WORLD PROPERTIES =====
    try:
        # Average shortest path (on largest component)
        if nx.is_connected(G):
            features['avg_shortest_path'] = nx.average_shortest_path_length(G, weight='weight')
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_cc)
            features['avg_shortest_path'] = nx.average_shortest_path_length(G_largest, weight='weight')
        
        # Small-world coefficient
        features['small_world_sigma'] = features['clustering_coef'] / features['avg_shortest_path'] if features['avg_shortest_path'] > 0 else 0
    except:
        features['avg_shortest_path'] = 0
        features['small_world_sigma'] = 0
    
    # ===== DEGREE DISTRIBUTION STATISTICS =====
    features['degree_skewness'] = stats.skew(degree)
    features['degree_kurtosis'] = stats.kurtosis(degree)
    features['strength_skewness'] = stats.skew(strength)
    features['strength_kurtosis'] = stats.kurtosis(strength)
    
    # ===== NETWORK RESILIENCE =====
    # Measure based on degree distribution
    features['network_heterogeneity'] = np.std(degree) / np.mean(degree) if np.mean(degree) > 0 else 0
    
    # ===== WEIGHTED NETWORK FEATURES =====
    features['strength_degree_correlation'] = np.corrcoef(strength, degree)[0, 1] if len(strength) > 1 else 0
    features['avg_neighbor_degree'] = np.mean([np.mean([G.degree(neighbor) for neighbor in G.neighbors(node)]) 
                                                for node in G.nodes() if G.degree(node) > 0])
    
    return features


def extract_features_batch(mat_files, verbose=True):
    """Extract features from multiple subjects."""
    import os
    
    all_features = []
    
    for i, file in enumerate(mat_files):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(mat_files)} subjects...")
        
        try:
            subject_id = os.path.basename(file).split('_')[0]
            features = extract_advanced_features(file)
            features['URSI'] = subject_id
            all_features.append(features)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    return all_features


if __name__ == "__main__":
    import glob
    import pandas as pd
    
    print("Extracting advanced features from brain networks...")
    subject_files = glob.glob("mat_subjects/*.mat")
    
    all_features = extract_features_batch(subject_files)
    features_df = pd.DataFrame(all_features)
    
    print(f"\nExtracted {len(features_df.columns) - 1} features from {len(features_df)} subjects")
    print(f"\nFeature names:")
    print(features_df.columns.tolist())
    
    # Save features
    features_df.to_csv('advanced_brain_features.csv', index=False)
    print("\n✅ Saved to advanced_brain_features.csv")
