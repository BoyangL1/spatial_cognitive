# import module from the parent directory
import sys
import os
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
import SCBIRL_Global_PE.utils as SIRLU

from scipy.spatial import distance_matrix
# note: scipy wasserstein function is too slow.
from scipy.stats import wasserstein_distance_nd, lognorm
import ot

from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
from multiprocessing import Pool, cpu_count
import time

def coords2compression(model, coords, depth: int):
    '''
    Transform the coordinates to the compressed representation by the model.
    '''
    # given the state return the location codes
    gc_vectors = [SIRLU.globalPE(coord, state_dim) for coord in coords]
    gc_vectors = np.squeeze(np.array(gc_vectors), axis=-1)

    # apply the model to the gc_vectors
    key = model.key
    c_params = model.c_params
    pe_real_compressed, pe_imag_compressed = model.compress_pe_code_complex.apply(c_params, key, gc_vectors, target_dim=depth)
    # transform the gc patterns to compressed representation, 14D vector
    pe_compressed = pe_real_compressed + pe_imag_compressed
    return pe_compressed

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


if __name__ == '__main__':
    data_dir = './data/'
    model_dir = './model/'
    save_dir = './data_pe/'

    # Paths for data files
    before_migration_path = data_dir + 'before_migrt.json'
    after_migration_path = data_dir + 'after_migrt.json'
    full_trajectory_path = data_dir + 'all_traj.json'

    inputs, targets_action, pe_code, action_dim, state_dim = SIRLU.loadTrajChain(before_migration_path, full_trajectory_path)
    print(inputs.shape,targets_action.shape,pe_code.shape)
    model = SIRLT.avril(inputs, targets_action, pe_code, state_dim, action_dim, state_only=True)

    # model.loadParams(model_dir + 'params_transformer_pe.pickle')
    model.loadParams(save_dir + 'after_migrt/model/20190801.pickle')
    
    MAXCORES = cpu_count() - 1

    # read the list of location codes 
    with open(data_dir + 'id_coords_mapping.pkl', 'rb') as f:
        id_coords_mapping = pickle.load(f)
    id_coorders_mapping = dict(sorted(id_coords_mapping.items()))

    # fulfill the representation by function
    pe_compressed = coords2compression(model, id_coorders_mapping.values(), depth=state_dim)
    print(pe_compressed.shape)

    # read the policy-transition matrix
    with open(save_dir + 'before_migrt_transition_prob.pkl', 'rb') as f:
        res = pickle.load(f)
    transitionProbs, coordsIdx = np.array(res[0]), res[1]

    # Edit the transiton matrix and coordinates to remove the non-visited locations.
    transitionProbsEdit, id_coorders_mapping_edit = removeNonVisited(transitionProbs, id_coorders_mapping)
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
    # deprecated: reduce the dimension of the compressed representation to speed up
    # pca = PCA(n_components=.95)
    # pe_compressed_reduced = pca.fit_transform(pe_compressed)

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
    threshold = clusterThres(spatial_distrib_params, social_distrib_params, alpha = 2)

    # clustering the locations by agglomerative clustering
    aggClusterer = AgglomerativeClustering(None, metric='precomputed', distance_threshold=threshold,
        linkage='complete' )
    aggClusterer.fit(total_dist)

    # get the cluster labels
    cluster_labels = aggClusterer.labels_
    # get the number of clusters
    num_clusters = len(np.unique(cluster_labels))
    print("There are {} clusters in total.".format(num_clusters))

    # save the cluster labels
    with open(save_dir + 'cluster_results_august.pkl', 'wb') as f:
        res = (transitionProbsEdit, id_coorders_mapping_edit, stationary, cluster_labels)
        pickle.dump(res, f)