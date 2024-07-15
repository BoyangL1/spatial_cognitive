# import module from the parent directory
import sys
import os
working_directory = os.getcwd()
sys.path.append(working_directory)

import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import SCBIRL_Global_PE.utils as SIRLU
import SCBIRL_Global_PE.migrationProcess as SIRLM
from SCBIRL_Global_PE.utils import TravelData, Traveler

import numpy as np
import pandas as pd
import shap
from datetime import date, timedelta
from itertools import repeat, chain
from functools import partial
import pickle
import multiprocessing as mp

# if gpu is not useful, force to use cpu
import jax
jax.config.update('jax_platform_name', 'cpu')

# count the cpu number
# MAX_CPU_COUNT = mp.cpu_count() - 1
MAX_CPU_COUNT = 48

 
def loadModel(who, date = None, prior = True, accumulate = False, tabular = False):
    '''
        Load the model from the model directory.
        Must correctly set the directory at first.
    '''
    data_dir = './data/user_data/' + SIRLU.toWhoString(who) + '/'
    model_dir = './model/' + SIRLU.toWhoString(who) + '/'
    
    iter_start_date = SIRLU.load_traveler(who).iter_start_date
    inputs, targets_action, pe_code, action_dim, state_dim = SIRLU.loadTrajChain(data_dir, type='before', start_date=iter_start_date)
    print(inputs.shape, targets_action.shape, pe_code.shape)
    model = SIRLT.avril(inputs, targets_action, pe_code, state_dim, action_dim, state_only=True)
    if tabular: 
        return model
    
    if date is None or date < iter_start_date:
        path = model_dir + 'initial_model.pickle'
    else:
        if prior:
            modeltype = 'evolution_model/iterated_model_'        
        elif accumulate:
            modeltype = 'empirical_model/increased_model_'
        else:
            modeltype = 'no_prior_model/ignorant_model_'
        path = model_dir + modeltype + '{date}.pickle'.format(date=date)
    model.loadParams(path)
    return model

def modelPredict(X: np.array, model = None, attribute_type = 'reward'):
    '''
        load the matrix of features, return the reward predicted by the model
    '''
    # ! change here
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

def backgroundData(who: int, date = None):
    # feature dataframe for query
    features_query = SIRLU.load_state_attrs(who)
    
    data_dir = './data/user_data/' + SIRLU.toWhoString(who) + '/'
    full_traj_path = data_dir + 'all_traj.json'
    chains_dict = SIRLU.loadJsonFile(full_traj_path)
    if date is not None:
        chains_dict = filter(lambda x: x['date'] <= date, chains_dict)
    all_chains = SIRLU.loadTravelDataFromDicts(chains_dict)

    total_array_list = []
    # unstack the visits
    for chain in all_chains:
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
        
def grouped_shap(shap_vals, features, groups):
    revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))    
    
    groupmap = revert_dict(groups)
    shap_Tdf = pd.DataFrame(shap_vals, columns=pd.Index(features, name='features')).T
    shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
    shap_grouped = shap_Tdf.groupby('group').sum().T
    return shap_grouped


def sparseBackground(dataset: np.array):
    column_name = ['C{:02d}'.format(i) for i in range(dataset.shape[1])]
    background_df = pd.DataFrame(dataset, columns=column_name)
    # count the unique values with its frequency
    background_val_freq = background_df.groupby(column_name).size().reset_index(name='freq')
    background_uni = background_val_freq.drop(columns=['freq']).to_numpy()
    background_weight = background_val_freq['freq'].to_numpy()
    return background_uni, background_weight
    

def modelRewardExplain(date: int, who: int):
    '''
        Give the SHAP value by grouping the type.
    '''
    
    print('Explaining person: {who:9d}, date: {date}'.format(who=who, date=date))
    model = loadModel(who=who, date=date)
    modelPredWrapper = partial(modelPredict, model=model, attribute_type='reward')

    dataset = backgroundData(who=who, date=date)
    dataset_uni, dataset_freq = sparseBackground(dataset)
    print('Data with {k} rows'.format(k=dataset_uni.shape[0]))

    built_bench = np.zeros(model.s_dim).reshape(1, -1)
    locat_bench = np.mean(dataset[:, model.s_dim:], axis=0).reshape(1, -1)
    # zero_bench = np.zeros(dataset.shape[1]).reshape(1, -1)
    home_bench = np.hstack((built_bench, locat_bench))

    explainer = shap.PermutationExplainer(modelPredWrapper, home_bench)
    shap_values = explainer(dataset_uni)
    
    # below: group the shape var names
    varchr = 'LU_Business,LU_City_Road,LU_Consumption,LU_Culture,LU_Industry,LU_Medical,LU_Park_&_Scenery,LU_Public,LU_Residence,LU_Science_&_Education,LU_Special,LU_Transportation,LU_Wild'
    varname_BE = varchr.split(',')
    varname_PE = ['PE%02d' % i for i in range(6 * len(varname_BE))]
    varname = varname_BE + varname_PE
    groupmap = {
        'BuiltAttr': varname[:len(varname_BE)],
        'Location': varname[len(varname_BE):]
    }

    shap_grouped_by_classes = grouped_shap(shap_vals=shap_values.values, features=varname, groups=groupmap)
    return shap_grouped_by_classes, dataset_freq

def modelUserDateCombination():
    model_dir = './model/'
    # list all users with folder name consisting of all digits.
    # note: change here
    user_list = [name for name in os.listdir(model_dir) if name.isdigit()]
    user_list.remove('001102234')
    
    combination = []
    for user in user_list:
        evolution_model_path = model_dir + user + '/' + 'evolution_model/'
        date_list = [int(params.rstrip('.pickle')[-8:]) for params in os.listdir(evolution_model_path)]
        date_list = date_list[::7]
        for date in date_list:
            combination.append((int(user), date))
    return combination


def modelDateOfUser(user):
    model_dir = './model/'
    user = SIRLU.toWhoString(user)
    evolution_model_path = model_dir + user + '/' + 'evolution_model/'
    date_list = [int(params.rstrip('.pickle')[-8:]) for params in os.listdir(evolution_model_path)]
    date_list = date_list[::7]
    return date_list

def explainOneUser(user, parallel=False):
    date_list = modelDateOfUser(user)
    
    if not parallel:
        shap_dict = dict()
        # add reverse to mitigate the load balancing problem.
        for date in reversed(date_list):
            shap_dict[date] = modelRewardExplain(date, who=user)
    else:
        # parallel version
        CPU_COUNT = len(date_list)
        combination = [(date, user) for date in reversed(date_list)]
        with mp.Pool(CPU_COUNT) as pool:
            shap_dict_values = pool.starmap(modelRewardExplain, combination)
        shap_dict = dict(zip(date_list, shap_dict_values))
    return shap_dict


def explainAllRewards(parallel = False):
    combination = modelUserDateCombination()
    shap_dict = dict()
    
    if not parallel:
        for user, date in combination:
            shap_dict[(user, date)] = modelRewardExplain(date, who=user)
    else:
        # parallel version
        combination_switch = [(date, user) for user, date in combination]
        with mp.Pool(MAX_CPU_COUNT) as pool:
            shap_dict_values = pool.starmap(modelRewardExplain, combination_switch)
        for idx, (user, date) in enumerate(combination):
            shap_dict[(user, date)] = shap_dict_values[idx]
    return shap_dict

def modelRewardCalculation(date: int, who: int = 36384703):
    '''
        Give the SHAP value by grouping the type.
    '''
    print('Explaining person: {who:8d}, date: {date}'.format(who=who, date=date))
    model = loadModel(who=who, date=date)
    modelPredWrapper = partial(modelPredict, model=model, attribute_type='reward')

    dataset = backgroundData(who=who, date = date)
    dataset_uni, dataset_freq = sparseBackground(dataset)
    reward_comp = modelPredWrapper(dataset_uni)
    res = np.average(np.abs(reward_comp), weights=dataset_freq)
    return res


if __name__ == '__main__':
    '''
    Full Parallel Version
    '''
    # res = explainAllRewards(parallel=True)
    # with open('./product/shap_res.pkl', 'wb') as f:
    #     pickle.dump(res, f)
    
    '''
    Half Parallel Version
    '''
    model_dir = './model/'
    user_list = [int(name) for name in os.listdir(model_dir) if name.isdigit()]
    user_list.sort()
    for user in user_list:
        res = explainOneUser(user, parallel=True)
        with open('./product/shap_res_{:09d}.pkl'.format(user), 'wb') as f:
            pickle.dump(res, f)
    '''
    By Hand
    '''
    # model_dir = './model/'
    # user_list = [int(name) for name in os.listdir(model_dir) if name.isdigit()]
    # user_list.sort()

    # user = user_list[0]
    # res = explainOneUser(user, parallel=True)
    # with open('./product/shap_res_{:98d}.pkl'.format(user), 'wb') as f:
    #     pickle.dump(res, f)
    '''
    Inspect the baseline.
    '''
    # model_dir = './model/'
    # user_list = [int(name) for name in os.listdir(model_dir) if name.isdigit()]
    # user_list.sort()
    # reward_dict = dict()
    # for user in user_list:
    #     date_list = modelDateOfUser(user)
    #     for date in date_list:
    #         reward_dict[(user, date)] = modelRewardCalculation(date, who=user)
    #         with open('./product/reward_res.pkl', 'wb') as f:
    #                 pickle.dump(reward_dict, f)
    '''
    Test area
    '''
    # shap_dict = dict()
    # date = 20230507
    # shap_dict[date] = modelRewardExplain(date, who=1102234)