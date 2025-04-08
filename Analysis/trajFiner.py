# import module from the parent directory
import sys
import os
from tqdm import tqdm
from datetime import date
from math import floor
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import folium
import scipy
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import networkx as nx
from networkx.algorithms.community import louvain_communities
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score
# from hdbscan import HDBSCAN
from sklearn.cluster import HDBSCAN

working_directory = os.getcwd()
working_directory = os.path.abspath('.')
sys.path.append(working_directory)

from Analysis.topoMap import clusterLocations, cogTopoGraph, topoResPath
import SCBIRL_Global_PE.utils as SIRLU

def trajWeekSubset(who, week_order: int):
    visit_dates = SIRLU.visited_date(who)
    traj_total = SIRLU.load_all_traj(who)
    
    _, week_code = SIRLU.extract_week_ends(visit_dates)
    week_order_indices = [i for i, w in enumerate(week_code) if w == week_order]
    if week_order_indices == []:
        return []
    start_date = visit_dates[week_order_indices[0]]
    end_date = visit_dates[week_order_indices[-1]]
    tcs = list(filter(lambda tc: start_date <= tc.date <= end_date, traj_total))
    return tcs

def traj4Visualize(who, week_order: int,):
    tcs = trajWeekSubset(who, week_order)
    od_pairs = []
    for tc in tcs:
        coord_seq = tc.travel_chain
        if len(coord_seq) < 2:
            continue
        for i in range(len(coord_seq) - 1):
            flow = (tuple(coord_seq[i]), tuple(coord_seq[i+1]))
            od_pairs.append(flow)
    return od_pairs
    
def cogGraph4Visualize(who, week_order: int):
    tcs = trajWeekSubset(who, week_order)
    tc_week_end_date = max(tcs, key=lambda tc: tc.date)
    week_end_date = tc_week_end_date.date
    return cogTopoGraph(who, week_end_date)


def eigenDecomposition(A, plot=True):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigenvalues for visual inspection
    :return A tuple containing:
        - the optimal number of clusters by eigengap heuristic
        - all eigenvalues
        - all eigen vectors

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D^-1/2 A D^-1/2.
    2. Find the eigenvalues and their associated eigen vectors.
    3. Identify the maximum gap which corresponds to the number of clusters
       by eigengap heuristic.

    References:
    - https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    - http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4485%5B0%5D.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]

    # k parameter: Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in
    # the euclidean norm of complex numbers.
    eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)

    if plot:
        plt.title("Largest eigenvalues of input matrix")
        plt.scatter(range(len(eigenvalues)), eigenvalues)
        plt.grid()

    # Identify the optimal number of clusters as the index corresponding
    # to the largest gap between eigenvalues
    index_largest_gap = np.argmax(np.diff(eigenvalues))
    nb_clusters = index_largest_gap + 1

    return nb_clusters, eigenvalues, eigenvectors

def silhouette_optimal_k(similarity, k_range, plot=True):
    """
    Automatically select the optimal number of clusters using Silhouette score.
    
    Parameters:
    -----------
    similarity : array-like
        Similarity matrix
    k_range : list or range
        Range of possible cluster numbers to evaluate
    plot : bool, default=True
        Whether to plot the Silhouette scores
        
    Returns:
    --------
    best_k : int
        Optimal number of clusters
    """
    best_k = None
    best_score = -1
    print(f'Silhouette score computation:')
    
    scores_record = []
    for k in k_range:
        # Perform spectral clustering with current k
        clustering = SpectralClustering(n_clusters=k, affinity='precomputed', 
                                      assign_labels='cluster_qr', random_state=42)
        labels = clustering.fit_predict(similarity)
        
        # Convert similarity to dissimilarity for Silhouette score calculation
        disimilarity = 1 - similarity
        np.fill_diagonal(disimilarity, 0)
        score = silhouette_score(disimilarity, labels, metric='precomputed')
        print(f'k = {k}, score = {score}')    
        scores_record.append(score)
        if score > best_score:
            best_score = score
            best_k = k
    print(f'The optimal number of clusters is {best_k}.\n------------------')
    
    if plot:
        plt.title("Records of Silhouette scores")
        plt.scatter(k_range, scores_record)
        plt.grid()
        plt.show()
        
    return best_k

def topoNodeCluster(who, method='spectral', optimal='silhouette'):
    """
    Perform topological clustering on nodes based on their co-occurrence patterns.
    
    Parameters:
    -----------
    who : int
        User ID
    method : str, default='spectral'
        Clustering method ('spectral', 'louvain', 'hdbscan', or 'dbscan')
    optimal : str or int, default='silhouette'
        Method for determining optimal cluster number ('gap', 'silhouette', or specific number)
        
    Returns:
    --------
    labels : array-like
        Cluster labels for each node
    """
    # Load user's visit data and location mappings
    visit_dates = SIRLU.visited_date(who)
    id_coords_mapping = SIRLU.load_id_coords_mapping(who)
    total_loc_number = len(id_coords_mapping)
    
    # Extract week-end dates
    week_end_dates, _ = SIRLU.extract_week_ends(visit_dates)
    
    # Load or compute weekly clustering results
    res_dir = topoResPath(who)
    if not os.path.exists(res_dir):
        cluster_by_week = [clusterLocations(who, week_end_date) for week_end_date in week_end_dates]
    else:
        compute_res = [file for file in os.listdir(res_dir) if file.endswith('.pickle')]
        cluster_by_week = [SIRLU.load_pickle_binary(os.path.join(res_dir, file)) for file in compute_res]
    _, id_coorders_mapping_edit_list, _, cluster_labels_list = list(zip(*cluster_by_week))
    
    # Initialize similarity matrix and appearance counter
    similarity = np.zeros((total_loc_number, total_loc_number))
    appearance = np.zeros(total_loc_number)
    
    # Calculate co-occurrence based similarity
    for id_coorders_mapping_edit, cluster_label in zip(id_coorders_mapping_edit_list, cluster_labels_list):
        location_ids = np.array(list(id_coorders_mapping_edit.keys()))
        appearance[location_ids] += 1
        
        # Create co-occurrence matrix for current week
        cluster_label_row = np.array(cluster_label)
        cluster_label_column = np.array(cluster_label).reshape(-1, 1)
        cluster_label_judge = cluster_label_row == cluster_label_column
        
        # Create upper triangular mask
        seg_loc_len = len(cluster_label)
        masker = np.full((seg_loc_len, seg_loc_len), False)
        masker[np.triu_indices(seg_loc_len, 1)] = True
        
        # Update similarity matrix based on co-occurrences
        co_idx = np.where(cluster_label_judge & masker)
        co_ids = [(location_ids[i], location_ids[j]) for i, j in zip(*co_idx)]
        for co_id in co_ids:
            similarity[co_id] += 1
            
    # Normalize similarity by appearance frequency
    appearance_2d = appearance[:, np.newaxis]
    appearance_base = np.minimum(appearance_2d, appearance_2d.T)
    similarity_corrected = similarity / appearance_base
    
    # Create symmetric affinity matrix
    affinity = similarity_corrected + similarity_corrected.T
    affinity[np.where(affinity == 0)] += 1e-6
    np.fill_diagonal(affinity, 0)
    
    # Perform clustering based on specified method
    if method == 'spectral':
        if optimal == 'gap':
            n_clusters, *_ = eigenDecomposition(affinity, plot=True)
        elif optimal == 'silhouette':
            start_number, end_number = floor(0.1 * total_loc_number), floor(0.9 * total_loc_number)
            n_clusters = silhouette_optimal_k(affinity, range(start_number, end_number + 1))
        elif isinstance(optimal, int):
            n_clusters = optimal
        clusterer = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                     eigen_solver='arpack', assign_labels='cluster_qr',
                                     random_state=42)
        labels = clusterer.fit_predict(affinity)
    elif method == 'louvain':
        G = nx.from_numpy_matrix(affinity)
        communes = louvain_communities(G)
        communes = list(communes)
        
        labels = np.full(len(affinity), -1)
        for i, within_same_communes in enumerate(communes):
            cluster_idx = np.array(list(within_same_communes))
            labels[cluster_idx] = i
    elif method == 'hdbscan':
        clusterer = HDBSCAN(min_cluster_size=2, min_samples=2, metric='precomputed')
        disimilarity = 1 - affinity
        np.fill_diagonal(disimilarity, 0)
        labels = clusterer.fit_predict(disimilarity)
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=1/3, min_samples=2, metric='precomputed')
        disimilarity = 1 - affinity
        np.fill_diagonal(disimilarity, 0)
        labels = clusterer.fit_predict(disimilarity)
    return labels
    
def nodeVerTraj(who, labels):
    """
    Convert location IDs in trajectories to their corresponding cluster labels.
    
    Parameters:
    -----------
    who : int
        User ID
    labels : array-like
        Cluster labels for each location
        
    Returns:
    --------
    node_chain_list : list
        List of trajectories where locations are replaced by their cluster labels
    """
    visit_dates = SIRLU.visited_date(who)
    traj_total = SIRLU.load_all_traj(who)
    
    # Map location IDs to cluster labels
    id_chain_list = [traj.id_chain for traj in traj_total]
    id_node_map = {i: lab for i, lab in enumerate(labels)}
    node_chain_list = [[id_node_map[iden] for iden in id_chain] for id_chain in id_chain_list]
    assert len(visit_dates) == len(node_chain_list), "The date sequence does not match chain sequence."
    
    return node_chain_list

def nodeVisitScan(who, labels):
    """
    Create a binary matrix indicating presence/absence of each cluster in each trajectory.
    
    Parameters:
    -----------
    who : int
        User ID
    labels : array-like
        Cluster labels for each location
        
    Returns:
    --------
    node_scan : list of bool arrays
        Binary matrix where each row represents a cluster and each column represents a trajectory
    """
    node_num = np.max(labels)
    node_chain_list = nodeVerTraj(who, labels)
    node_scan = [[node in node_chain for node_chain in node_chain_list] for node in range(node_num)]
    return node_scan

def nodeTrajCount(who, labels):
    """
    Count the frequency of each unique trajectory pattern in terms of cluster sequences.
    
    Parameters:
    -----------
    who : int
        User ID
    labels : array-like
        Cluster labels for each location
        
    Returns:
    --------
    node_count : Counter
        Counter object containing frequencies of each trajectory pattern
    """
    node_chain_list = nodeVerTraj(who, labels)
    node_count = Counter(node_chain_list)
    return node_count



if __name__ == "__main__":
    who = 58124481
    cluster_res = topoNodeCluster(who)
    seq_res = nodeVisitScan(who, cluster_res)
    print(seq_res)
    