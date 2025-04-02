# import module from the parent directory
import sys
import os
from tqdm import tqdm
import numpy as np
# ref working direcotry、file path和environment path的关系：
# ref 工作目录是当前项目运行，即vscode打开的目录
# ref 文件路径是当前文件的路径
# ref 环境路径是python调用模块检索的路径

# ref 工作目录不会自动添加到sys.path检索路径中
# ref 当前文件路径会自动添加到sys.path检索路径中
# 方法一
# script_directory = os.path.dirname(os.path.realpath(__file__))
# parent_directory = os.path.dirname(script_directory)
# sys.path.append(parent_directory)

# 方法二
working_directory = os.getcwd()
working_directory = os.path.abspath('.')
sys.path.append(working_directory)

# 方法三
# working_directory = os.path.dirname(sys.path[0])
# sys.path.append(working_directory)

import pickle
import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import SCBIRL_Global_PE.migrationProcess as SIRLP
import SCBIRL_Global_PE.utils as SIRLU
from TRAJ_PROCESS.prepareChain import Traveler
from SCBIRL_Global_PE.utils import UserDataPart

from scipy.spatial import distance_matrix
# note: scipy wasserstein function is too slow.
from scipy.stats import wasserstein_distance_nd, lognorm
from scipy.stats import lognorm
import ot
import jax
jax.config.update('jax_platform_name', 'cpu')

from itertools import combinations
from multiprocessing import Pool, cpu_count
import time
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering
from shapely.geometry import Point
from geopy.distance import geodesic
from numba import njit, prange


def coords2compression(model, coords, depth: int):
    '''
    Transform the coordinates to the compressed representation by the model.
    '''
    # given the state return the location codes
    gc_vectors = [SIRLU.globalPE(coord, depth) for coord in coords]
    gc_vectors = np.squeeze(np.array(gc_vectors), axis=-1)

    # apply the model to the gc_vectors
    key = model.key
    c_params = model.c_params
    pe_real_compressed, pe_imag_compressed = model.compress_pe_code_complex.apply(c_params, key, gc_vectors, target_dim=depth)
    # transform the gc patterns to compressed representation, vector with dimension of depth
    pe_compressed = pe_real_compressed + pe_imag_compressed
    return pe_compressed

def compute_geodistace(coords):
    dist_matrix = np.zeros((len(coords), len(coords)))
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            coord_i = tuple(reversed(coords[i]))
            coord_j = tuple(reversed(coords[j]))
            dist_matrix[i, j] = geodesic(coord_i, coord_j).kilometers
    return dist_matrix + dist_matrix.T

def removeNonVisited(transitionProbs, id_coorders_mapping):
    '''
    Remove the non-visited locations from the transition matrix and coordinates mapping.
    Then, normalize the transition by scale each rows to sum to one.
    ------
    transitionProbs: the policy-transition computed by the model.
    id_coorders_mapping: the mapping between location id and its coordinates. It must have been sorted on id.
    '''
    # remove the last column: ending state
    transitionProbs = transitionProbs[:, :-1] 
    # find the non-visited locations
    non_visited = np.where(np.isnan(transitionProbs).all(axis=1))[0]
    # filter the corresponding column and row
    transitionProbs = np.delete(transitionProbs, non_visited, axis=0)
    transitionProbs = np.delete(transitionProbs, non_visited, axis=1)
    transitionProbsEdit = transitionProbs / np.sum(transitionProbs, axis=1, keepdims=True)

    # filter the id_coorders_mapping
    id_coorders_mapping_edit = {k: v for k, v in id_coorders_mapping.items() if k not in non_visited}
    # return the result
    return transitionProbsEdit, id_coorders_mapping_edit

def computeTransLimit(transition: np.array):
    # conduct eigen-decomposition to the mar
    eig_val, eig_vec = np.linalg.eig(transition.T)
    # check the largest eigenvalue is one
    eig_vec1_pos, = np.where(np.isclose(eig_val, 1.0))
    # find the eigen value 1
    final_eigen = eig_vec[:, eig_vec1_pos]
    # find the stationary distribution
    stationary = final_eigen / np.sum(final_eigen)
    return stationary

def compute_wasserstein(pe_compressed_reduced, transitionProbs, i, j, method='pot', M=None):
    distribution_i = transitionProbs[i, :]
    distribution_j = transitionProbs[j, :]
    method_set = ('pot', 'scipy', 'geomloss')
    assert method in method_set, "The method is not supported."

    if np.isnan(distribution_i).all() or np.isnan(distribution_j).all():
        return np.inf
    else:
        if method == 'scipy':
            entry = wasserstein_distance_nd(pe_compressed_reduced, pe_compressed_reduced, distribution_i, distribution_j)
        elif method == 'pot':
            # below is necessary: because the pot library requires the input to be contiguous
            if not distribution_i.flags.c_contiguous:
                distribution_i = np.ascontiguousarray(distribution_i)
            if not distribution_j.flags.c_contiguous:
                distribution_j = np.ascontiguousarray(distribution_j)
            # compute the optimal transport plan
            if M is None:
                raise ValueError("The cost matrix M is not provided.")
            entry = ot.emd2(distribution_i, distribution_j, M)
        else:
            pass
        return entry

def clusterThres(spatial_distrib_params, social_distrib_params, alpha = 1.5):
    s_1, _, scale_1 = spatial_distrib_params
    s_2, _, scale_2 = social_distrib_params
    sigma_agg = np.sqrt(s_1 ** 2 + s_2 ** 2)
    mu_agg = np.log(scale_1) + np.log(scale_2)
    thres_log = mu_agg - alpha * sigma_agg
    return np.exp(thres_log)

def computeTransitionProb(model, who, date):
    """
    Compute transition probabilities for each state using a given model and save to a CSV file.
    """
    who_string = SIRLU.toWhoString(who)
        
    state_attribute = SIRLU.load_state_attrs(who)
    
    all_traj_src = UserDataPart + '%s/all_traj.json' % (who_string,)
    all_traj_dict = SIRLU.loadJsonFile(all_traj_src)
    chains_dict = filter(lambda x: x['date'] <= date, all_traj_dict)

    visited_id = list()
    for d in chains_dict:
        visited_id.extend(d['id_chain'])
    visited_id = set(visited_id)
    
    computeFunc = SIRLP.getComputeFunction(model, 'transition_prob')
    size = model.a_dim
        
    coords_id = SIRLU.load_id_coords_mapping(who)
    coords_fnid = SIRLU.load_fnid_coords_mapping(who)

    # ref sort the dictionary by value
    coords_id = dict(sorted(coords_id.items()))
    transitionProbs = []
    coordsIdx = []
    for id, coords in tqdm(coords_id.items(), total=len(coords_id),desc="compute transition probability"):
        fnid = coords_fnid[coords]
        if id not in visited_id:
            transProbVec = np.full(size, np.nan)
        else:
        # get state attribute of this fnid
            state = SIRLU.getStateRow(state_attribute, fnid)
            pe_code = SIRLU.globalPE(coords,len(state)).flatten()
            # add three dimension
            pe_code = np.expand_dims(np.expand_dims(np.expand_dims(pe_code, axis=0), axis=0), axis = 0)
            state = np.expand_dims(np.expand_dims(np.expand_dims(state, axis=0), axis=0), axis = 0)
            # get transition probability
            transProbVec = computeFunc(state,pe_code)
        transitionProbs.append(transProbVec)
        coordsIdx.append(coords)
    
    res = (np.array(transitionProbs), coordsIdx)
    return res

def topoResPath(who):
    who_string = SIRLU.toWhoString(who)
    save_dir = f'./product/topoMap/{who_string}/' 
    return save_dir

def topoResSave(res, who, date):
    """
    Save the topological results to a pickle file.
    """
    save_dir = topoResPath(who)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + f'topo_res_{date:d}.pickle', 'wb') as f:
        pickle.dump(res, f)

@njit
def compute_emd_numba(dist_i, dist_j, cost_matrix):
    """
    Compute the Earth Mover's Distance (EMD) between two distributions using a greedy algorithm.
    Args:
        dist_i (ndarray): First distribution.
        dist_j (ndarray): Second distribution.
        cost_matrix (ndarray): Precomputed cost matrix.
    Returns:
        float: The EMD (Wasserstein distance) between the two distributions.
    """
    remaining_mass_i = dist_i.copy()
    remaining_mass_j = dist_j.copy()
    total_cost = 0.0

    for k in range(len(cost_matrix)):
        min_mass = min(remaining_mass_i[k], remaining_mass_j[k])
        total_cost += min_mass * cost_matrix[k, k]
        remaining_mass_i[k] -= min_mass
        remaining_mass_j[k] -= min_mass

    return total_cost


@njit(parallel=True)
def compute_circ_wasserstein_numba(angular_distribution, angular_cost):
    """
    使用numba加速的圆周Wasserstein距离计算
    
    参数:
    angular_distribution: shape (n, 96) 的数组，表示n个位置的时间分布
    angular_cost: shape (96, 96) 的数组，表示时间槽之间的成本矩阵
    
    返回:
    wasserstein_matrix: shape (n, n) 的数组，表示位置对之间的Wasserstein距离
    """
    n = angular_distribution.shape[0]
    wasserstein_matrix = np.zeros((n, n))
    
    # 对每对位置计算Wasserstein距离
    for i in prange(n):
        for j in range(i+1, n):
            # 获取两个位置的时间分布
            dist_i = angular_distribution[i]
            dist_j = angular_distribution[j]
            
            # Skip if either distribution is invalid
            if np.isnan(dist_i).all() or np.isnan(dist_j).all():
                wasserstein_matrix[i, j] = np.inf
                continue
            
            wasserstein_matrix[i, j] = compute_emd_numba(dist_i, dist_j, angular_cost)
    
    wasserstein_matrix += wasserstein_matrix.T
    return wasserstein_matrix

@njit(parallel=True)
def compute_hd_wasserstein_Numba(transition_probs, cost_matrix):
    """
    Compute the Wasserstein distance matrix using a greedy algorithm.
    Args:
        transition_probs (ndarray): Transition probability matrix.
        cost_matrix (ndarray): Precomputed cost matrix (spatial distance matrix).
    Returns:
        ndarray: Wasserstein distance matrix.
    """
    num_locs = len(transition_probs)
    wasserstein_matrix = np.zeros((num_locs, num_locs))

    for i in prange(num_locs):
        for j in range(i + 1, num_locs):
            dist_i = transition_probs[i]
            dist_j = transition_probs[j]

            # Skip if either distribution is invalid
            if np.isnan(dist_i).all() or np.isnan(dist_j).all():
                wasserstein_matrix[i, j] = np.inf
                continue

            # Use the compute_emd_numba function to calculate EMD
            wasserstein_matrix[i, j] = compute_emd_numba(dist_i, dist_j, cost_matrix)

    # Symmetrize the matrix
    wasserstein_matrix += wasserstein_matrix.T
    return wasserstein_matrix


def clusterLocations(who, date, res_save=True):
    who_string = SIRLU.toWhoString(who) + '/'
    data_dir = UserDataPart + who_string
    model_dir = './model/' + who_string
    save_dir = './product/' + who_string
    iter_start_date = SIRLU.load_traveler(who).iter_start_date

    # Paths for data files
    full_trajectory_path = data_dir + 'all_traj.json'

    if date >= iter_start_date:
        params_path = model_dir + f'evolution_model/iterated_model_{date:d}.pickle'
    else:
        params_path = model_dir + f'initial_model.pickle'
    
    inputs, targets_action, pe_code, action_dim, state_dim = SIRLU.loadTrajChain(data_dir, type='before', start_date=iter_start_date)
    logging.debug(inputs.shape,targets_action.shape,pe_code.shape)
    model = SIRLT.avril(inputs, targets_action, pe_code, state_dim, action_dim, state_only=True)

    # model.loadParams(model_dir + 'params_transformer_pe.pickle')
    model.loadParams(params_path)
    
    # read the list of location codes 
    id_coords_mapping = SIRLU.load_id_coords_mapping(who)
    id_tempo_mapping = SIRLU.load_id_tempo_mapping(who)
    id_coorders_mapping = dict(sorted(id_coords_mapping.items()))


    res = computeTransitionProb(model, who, date)
    transitionProbs, coordsIdx = np.array(res[0]), res[1]

    # Edit the transiton matrix and coordinates to remove the non-visited locations.
    transitionProbsEdit, id_coorders_mapping_edit = removeNonVisited(transitionProbs, id_coorders_mapping)
    logging.debug(f"The shape of the transition matrix is: {transitionProbsEdit.shape}")
    # compute the stationary distribution
    stationary = computeTransLimit(transitionProbsEdit)

    # distance computation
    num_locs = len(id_coorders_mapping_edit)
    logging.info(f"There are {num_locs} locations in total.")

    # Compute spatial distance matrix using calculate_geo_distance
    coords_list = list(id_coorders_mapping_edit.values())
    spatial_dist = compute_geodistace(coords_list)
    logging.info("Geographic distance computation finished.")

    # Compute spatial distance matrix
    # fulfill the representation by function
    pe_compressed = coords2compression(model, id_coorders_mapping.values(), depth=state_dim)
    pe_compressed_filtered = pe_compressed[list(id_coorders_mapping_edit.keys()), :]
    spatial_cost = distance_matrix(pe_compressed_filtered, pe_compressed_filtered, p=2)
    # print("Spatial distance computation finished.")

    # Compute Wasserstein distance matrix using Numba
    social_dist = compute_hd_wasserstein_Numba(transitionProbsEdit, spatial_cost)
    logging.info("Social distance computation finished.")

    # Compute the temporal distribution
    angular_distribution = np.array([id_tempo_mapping[id] for id in id_coorders_mapping_edit])
    slot_num = 24 * 4
    angular_cost = np.empty((slot_num, slot_num))
    init_cost = np.roll(np.abs(np.arange(slot_num) - slot_num // 2), slot_num // 2)
    for i in range(slot_num):
        angular_cost[i] = np.roll(init_cost, i)
    angular_cost /= 4
    temporal_dist = compute_circ_wasserstein_numba(angular_distribution, angular_cost)
    logging.info("Temporal distance computation finished.")
    
    # compute the total matrix
    # compute the similarity
    eps = 1e-12
    positive_smooth = lambda m: np.where(m <= 0, np.min(m[m > 0]), m)
    bandwidth_in_kilometers = 1.0
    
    # standardize the social distance matrix to fit standard log-normal distribution
    social_dist = positive_smooth(social_dist)
    social_dist_upper = social_dist[np.triu_indices_from(social_dist, k=1)]
    log_social_dist = np.log(social_dist_upper)
    log_mean = np.mean(log_social_dist)
    log_std = np.std(log_social_dist)
    log_social_dist_std = (log_social_dist - log_mean) / log_std
    target_mu = 0
    target_sigma = 1
    social_dist_upper_scaled = np.exp(target_mu + target_sigma * log_social_dist_std)
    social_dist_standard = np.zeros_like(social_dist)
    social_dist_standard[np.triu_indices_from(social_dist, k=1)] = social_dist_upper_scaled
    social_dist_standard += social_dist_standard.T  # 保持对称性
    
    # spatial similarity is computed by a gaussian kernel
    spatial_similarity = np.exp(-spatial_dist ** 2 / (2 * bandwidth_in_kilometers ** 2))
    # According to the 3 sigma rule, the left rare tail point for standard log-normal distribution is exp(-3)
    # then its corresponding quotient is eps(3) near to 20
    social_similarity = np.clip(1 / (social_dist_standard), eps, 20) 
    temporal_similarity = np.maximum(-np.log(temporal_dist / 12), eps)
    total_similarity = spatial_similarity * social_similarity * temporal_similarity

    total_sim = total_similarity[np.triu_indices_from(total_similarity, k=1)]
    total_sim_max = np.max(total_sim)
    total_sim = total_sim / total_sim_max
    # 对相似度值进行排序
    total_sim = np.sort(total_sim)
    
    # 计算CDF（使用均匀分布）
    cdf = np.arange(1, len(total_sim) + 1) / len(total_sim)
    
    # 使用kneedle算法找到knee point
    knee_locator = KneeLocator(total_sim, cdf, curve="concave", direction="increasing")
    sim_threshold = knee_locator.knee
    dist_threshold = 1 / (sim_threshold)
    # clustering the locations by agglomerative clustering
    aggClusterer = AgglomerativeClustering(None, metric='precomputed', 
                                          distance_threshold=dist_threshold, linkage='average')
    total_dist = 1 / (total_similarity / total_sim_max)
    np.fill_diagonal(total_dist, 0)
    aggClusterer.fit(total_dist)
    
    # get the cluster labels
    cluster_labels = aggClusterer.labels_
    # get the number of clusters
    num_clusters = len(np.unique(cluster_labels))
    print("There are {} clusters in total.".format(num_clusters))
    
    # return the result
    res = (transitionProbsEdit, id_coorders_mapping_edit, stationary, cluster_labels)
    
    if res_save:
        topoResSave(res, who, date)
    
    return res

def clusterLocationsDeprec(who, date, res_save = True):
    who_string = SIRLU.toWhoString(who) + '/'
    data_dir = UserDataPart + who_string
    model_dir = './model/' + who_string
    save_dir = './product/' + who_string
    iter_start_date = SIRLU.load_traveler(who).iter_start_date

    # Paths for data files
    full_trajectory_path = data_dir + 'all_traj.json'

    if date >= iter_start_date:
        params_path = model_dir + f'evolution_model/iterated_model_{date:d}.pickle'
    else:
        params_path = model_dir + f'initial_model.pickle'
    
    inputs, targets_action, pe_code, action_dim, state_dim = SIRLU.loadTrajChain(data_dir, type='before', start_date=iter_start_date)
    print(inputs.shape,targets_action.shape,pe_code.shape)
    model = SIRLT.avril(inputs, targets_action, pe_code, state_dim, action_dim, state_only=True)

    # model.loadParams(model_dir + 'params_transformer_pe.pickle')
    model.loadParams(params_path)
    
    MAXCORES = cpu_count() - 1

    # read the list of location codes 
    id_coords_mapping = SIRLU.load_id_coords_mapping(who)
    id_coorders_mapping = dict(sorted(id_coords_mapping.items()))

    # fulfill the representation by function
    pe_compressed = coords2compression(model, id_coorders_mapping.values(), depth=state_dim)
    print(pe_compressed.shape)

    res = computeTransitionProb(model, who, date)
    transitionProbs, coordsIdx = np.array(res[0]), res[1]

    # Edit the transiton matrix and coordinates to remove the non-visited locations.
    transitionProbsEdit, id_coorders_mapping_edit = removeNonVisited(transitionProbs, id_coorders_mapping)
    print("The shape of the transition matrix is: ", transitionProbsEdit.shape)
    # compute the stationary distribution
    stationary = computeTransLimit(transitionProbsEdit)

    # distance computation
    num_locs = len(id_coorders_mapping_edit)
    print("There are {} locations in total.".format(num_locs))
    # compute spatial distance matrix
    pe_compressed_filtered = pe_compressed[list(id_coorders_mapping_edit.keys()), :]
    spatial_dist = distance_matrix(pe_compressed_filtered, pe_compressed_filtered, p=2)
    print("Spatial distance computation finished.")

    # compute the transition relation distinction between points
    # compute the combination
    combine_pairs = [i for i in combinations(range(num_locs), 2)]

    # compute the wasserstein distance, based on spatial distribution computed before
    with Pool(processes=MAXCORES) as pool:
        results = pool.starmap(compute_wasserstein, [(pe_compressed_filtered, transitionProbsEdit, i, j, 'pot', spatial_dist) for i, j in combine_pairs])
    
    # fill the upper triangular matrix
    triu_idx = np.triu_indices(num_locs, 1)
    tril_idx = np.tril_indices(num_locs, 0)
    # create a numpy 2D array to record the wasserstein distance between each location
    social_dist = np.empty((num_locs, num_locs))
    social_dist[triu_idx] = results
    social_dist[tril_idx] = 0.0
    social_dist = social_dist + social_dist.T   
    print("Social distance computation finished.")

    # compute the total matrix
    total_dist = spatial_dist * social_dist
    # total_dist = np.exp(spatial_dist) * social_dist

    # calculate the distribution of the distance, start from 0.
    spatial_distrib_params = lognorm.fit(spatial_dist[triu_idx], floc=0.0)
    social_distrib_params = lognorm.fit(social_dist[triu_idx], floc=0.0)
    # determine the clustering threshold by accounting for the distribution
    threshold = clusterThres(spatial_distrib_params, social_distrib_params)

    # clustering the locations by agglomerative clustering
    aggClusterer = AgglomerativeClustering(None, metric='precomputed', distance_threshold=threshold,
        linkage='complete' )
    aggClusterer.fit(total_dist)

    # get the cluster labels
    cluster_labels = aggClusterer.labels_
    # get the number of clusters
    num_clusters = len(np.unique(cluster_labels))
    print("There are {} clusters in total.".format(num_clusters))
    
    # return the result
    res = (transitionProbsEdit, id_coorders_mapping_edit, stationary, cluster_labels)
    
    if res_save:
        topoResSave(res, who, date)
    
    return res

def cogTopoGraph(who, date):
    transitionProbsEdit, id_coorders_mapping_edit, stationary, cluster_labels = clusterLocations(who, date)
    keylist = list(id_coorders_mapping_edit.keys())
    
    # average the coordinates of the points in each cluster
    labpos = [np.where(cluster_labels == lab)[0] for lab in np.unique(cluster_labels)]
    transitionClust = np.empty((len(labpos), len(labpos)))
    for i in range(len(labpos)):
        source = transitionProbsEdit[labpos[i], :].mean(axis=0)
        for j in range(len(labpos)):
            transitionClust[i, j] = source.take(labpos[j]).sum()
            
    transitionClust /= transitionClust.sum(axis=1)[:, None]
    weight = computeTransLimit(transitionClust).real
    transitionCorrected = transitionClust * weight

    # ref 矩阵中的累积最小值
    # search the position where sits the smallest values whose cumulated sum is larger than 0.05
    threshold = 0.5
    transitionCorrectedFlatten = transitionCorrected.flatten()
    sorted = np.sort(transitionCorrectedFlatten)
    cut_pos = len(np.where(np.cumsum(sorted) <= threshold)[0])
    smalllest_idx = np.argsort(transitionCorrectedFlatten)[:cut_pos]
    # transitionCorrectedFlatten[smalllest_idx] = np.nan
    smalllest_idx_2d = np.unravel_index(smalllest_idx, transitionCorrected.shape)

    weighgFlatten = weight.flatten()
    sorted = np.sort(weighgFlatten)
    cut_pos = len(np.where(np.cumsum(sorted) <= threshold)[0])
    smalllest_idx_weight = np.argsort(weighgFlatten)[:cut_pos]

    transAdj = transitionCorrected.copy()
    transAdj[smalllest_idx_2d] = 0

    # get the average position of the points in each cluster
    cluster_coords_array = []
    for onelab2pos in labpos:
        cluster_coords_list = [id_coorders_mapping_edit[keylist[onepos]] for onepos in onelab2pos]
        cluster_coords = np.array(cluster_coords_list).mean(axis=0)
        cluster_coords = tuple(cluster_coords.tolist())
        cluster_coords_array.append(cluster_coords)

    # Create a graph from the CSV data
    G = nx.DiGraph()
    for i, coord in enumerate(cluster_coords_array):
        if not np.isin(i, smalllest_idx_weight):
            G.add_node(str(coord), pos=coord, weight=weight[i, 0])

    # count = 10
    size = len(cluster_coords_array)
    for i in range(size):
        if np.isin(i, smalllest_idx_weight):
            continue
        for j in range(size):
            if i != j and not np.isin(j, smalllest_idx_weight):
                weight_value = transAdj[i, j]
                if weight_value > 0:
                    G.add_edge(str(cluster_coords_array[i]), str(cluster_coords_array[j]), weight=weight_value)
    
    return G


if __name__ == '__main__':
    model_dir = "./model/"
    save_dir = "./product/topoMap/"
    user_list = [name for name in os.listdir(model_dir) if name.isdigit()]
    user_list = user_list[:1]
    res_dict = dict()
    for user in user_list:
        user_int = int(user)
        migrt_date = SIRLU.load_traveler(user_int).iter_start_date
        visit_dates = SIRLU.visited_date(user_int)
        # 获取iter_start_date之后每周的最后一天
        week_end_dates, _ = SIRLU.extract_week_ends(visit_dates)
        recording_dates = [date for date in week_end_dates if date >= migrt_date]
        for date in recording_dates:
            res = clusterLocations(user_int, date)
            # res_dict[(user, date)] = res
            # with open(save_dir + f'topo_cluster.pkl', 'wb') as f:
            #     pickle.dump(res_dict, f)
