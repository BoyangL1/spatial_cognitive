# import module from the parent directory
import sys
import os
import numpy as np

working_directory = os.getcwd()
sys.path.append(working_directory)

import pickle
import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import SCBIRL_Global_PE.utils as SIRLU
import SCBIRL_Global_PE.migrationProcess as SIRLM
from SCBIRL_Global_PE.utils import TravelData


def loadModel(path = None, tabular = False):
    '''
        Load the model from the model directory.
        Must correctly set the directory at first.
    '''
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
    if tabular: 
        return model

    if path is None:
        path = model_dir + 'params_transformer_pe.pickle'
    model.loadParams(path)
    return model

def modelPredict(X: np.array, model = None, attribute_type = 'reward'):
    '''
        load the matrix of features, return the reward predicted by the model
    '''
    if model is None:
        model = loadModel()
    # model = loadModel()
    feature_num = model.s_dim
    assert X.shape[1] == 7 * feature_num, "The input matrix does not have the correct number of features."
    state = X[:, :feature_num]
    pecode_real = X[:, feature_num:4 * feature_num]
    pecode_imag = X[:, 4 * feature_num:]
    # combine pecode_real and pecode_imag into a complex matrix
    pecode = np.empty_like(pecode_real, dtype=complex)
    pecode.real = pecode_real
    pecode.imag = pecode_imag
    # predict the reward
    predict_function = SIRLM.getComputeFunction(model, attribute_type)
    state = state[np.newaxis, np.newaxis, np.newaxis, :, :]
    pecode = pecode[np.newaxis, np.newaxis, np.newaxis, :, :]

    y_pred = list()
    for row in range(len(X)):
        # ref numpy take函数使用
        state_current = np.take(state, indices=row, axis=-2)
        pecode_current = np.take(pecode, indices=row, axis=-2)
        res_val = predict_function(state_current, pecode_current)
        y_pred.append(res_val)
    
    y_pred = np.array(y_pred)
    return y_pred

def backgroundData(date = None):
    migration = migrationDate()

    full_traj_path = './data/all_traj.json'
    chains_dict = SIRLU.loadJsonFile(full_traj_path)
    if date is not None:
        chains_dict =filter(lambda x: x['date'] <= date, chains_dict)
    all_chains = SIRLU.loadTravelDataFromDicts(chains_dict)

    # feature dataframe for query
    features_before, _ = SIRLU.preprocessStateAttributes('./data/before_migrt.json')
    features_after, _ = SIRLU.preprocessStateAttributes('./data/after_migrt.json')

    total_array_list = []
    # unstack the visits
    for chain in all_chains:
        features_query = features_before if chain.date < migration else features_after
        
        feature_array = []
        for fnid in chain.fnid_chain:
            # search the feature vector.
            feature_target = features_query.loc[features_query['fnid'] == fnid, :]
            # ref tuple 单元素解包
            fidx, = np.where(feature_target.columns != 'fnid')
            feature_vector = feature_target.iloc[0, fidx].to_numpy()
            feature_array.append(feature_vector)
        feature_array = np.array(feature_array)

        # calculate pe code vector 
        state_dim = feature_array.shape[1]        
        # note 复用于topoMap.coords2compression
        gc_vectors = [SIRLU.globalPE(coord, state_dim) for coord in chain.travel_chain]
        gc_vectors = np.squeeze(np.array(gc_vectors), axis=-1)
        gc_array = np.concatenate((gc_vectors.real, gc_vectors.imag), axis=1)

        one_chain_array = np.concatenate((feature_array, gc_array), axis=1)
        total_array_list.append(one_chain_array)

    total_array = np.vstack(total_array_list)
    return total_array
        

def migrationDate():
    # using the os path to get the after traj path
    after_traj_path = './data/after_migrt.json'

    after_traj = SIRLU.loadJsonFile(after_traj_path)
    migration_date = after_traj[0]['date']
    return migration_date


if __name__ == '__main__':
    import shap
    model = loadModel()
    migrationDate()
    