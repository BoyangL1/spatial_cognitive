# import module from the parent directory
import sys
import os
working_directory = os.getcwd()
sys.path.append(working_directory)

import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import SCBIRL_Global_PE.utils as SIRLU
import SCBIRL_Global_PE.migrationProcess as SIRLM
from SCBIRL_Global_PE.utils import TravelData, Traveler, UserDataPart

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
MAX_CPU_COUNT = mp.cpu_count() - 1
# MAX_CPU_COUNT = 48


def modelPredict(X: np.ndarray[float, float], model, standardize = False,
                 attribute_type = 'reward', mu = None, sigma = None):
    '''
        Calculate the reward of the grid cells.
        Load the matrix of processed features, return the reward predicted by the model.
        The reward can be either standardize or not.
    '''
    feature_num = model.s_dim
    assert X.shape[1] == feature_num + 2, "The input matrix does not have the correct number of features."
    state = X[:, :feature_num]
    positions = X[:, feature_num:]
    # predict the reward
    predict_function = SIRLM.getComputeFunction(model, attribute_type)
    
    state = state[np.newaxis, np.newaxis, np.newaxis, :, :]
    positions = positions[np.newaxis, np.newaxis, np.newaxis, :, :]

    y_pred = list()
    for row in range(len(X)):
        # ref numpy take函数使用
        state_current = np.take(state, indices=row, axis=-2)
        position_current = np.take(positions, indices=row, axis=-2)
        res_val = predict_function(state_current, position_current)
        # note browser
        y_pred.append(res_val)
    
    y_pred = np.array(y_pred)
    if standardize:
        y_pred = (y_pred - mu) / sigma
    return y_pred

def backgroundData(who: int, date = None):
    '''
    Construct the training feature array for the model prediction.
    The array records all visited place with repetition.
    '''
    # feature dataframe for query
    features_query = SIRLU.load_state_attrs(who)
    
    data_dir = UserDataPart + SIRLU.toWhoString(who) + '/'
    full_traj_path = data_dir + 'all_traj.json'
    chains_dict = SIRLU.loadJsonFile(full_traj_path)
    if date is not None:
        chains_dict = filter(lambda x: x['date'] <= date, chains_dict)
    all_chains = SIRLU.loadTravelDataFromDicts(chains_dict)

    total_array_list = []
    visit_id_list = []
    # unstack the visits: for chain on each day...
    for chain in all_chains:
        feature_array = []
        for fnid, iden in zip(chain.fnid_chain, chain.id_chain):
            # search the feature vector.
            feature_vector = SIRLU.getStateRow(features_query, fnid)
            # ref tuple 单元素解包
            # feature_target = features_query.loc[features_query['fnid'] == fnid, :]
            # fidx, = np.where(feature_target.columns != 'fnid')
            # feature_vector = feature_target.iloc[0, fidx].to_numpy()
            feature_array.append(feature_vector)
            visit_id_list.append(iden)
        feature_array = np.array(feature_array)

        # calculate pe code vector 
        coords = np.array(chain.travel_chain)  # (lon, lat)

        one_chain_array = np.concatenate((feature_array, coords), axis=1)
        total_array_list.append(one_chain_array)

    total_array = np.vstack(total_array_list)
    return total_array, visit_id_list


def grouped_shap(shap_vals, features, groups):
    '''
    add the feature by grouping the variables.
    '''
    # revert the dictionary
    revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))    
    groupmap = revert_dict(groups)
    shap_Tdf = pd.DataFrame(shap_vals, columns=pd.Index(features, name='features')).T
    shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
    shap_grouped = shap_Tdf.groupby('group').sum().T
    return shap_grouped


def sparseBackground(dataset: np.array, visited_id: list = None):
    '''
    Unique the background dataset with its frequency.  
    '''
    column_name = ['C{:02d}'.format(i) for i in range(dataset.shape[1])]
    background_df = pd.DataFrame(dataset, columns=column_name)
    assert len(visited_id) == dataset.shape[0], "The visited id list does not match the dataset."
    background_df['iden'] = visited_id
    # count the unique values with its frequency
    background_df_unique = background_df.drop_duplicates().set_index('iden')
    background_val_freq = background_df.groupby(['iden']).size().reset_index(name='freq').set_index('iden')
    background_df_unique = pd.merge(background_df_unique, background_val_freq, on='iden', how='left').reset_index()
    
    background_uni = background_df_unique.drop(columns=['freq', 'iden']).to_numpy()
    background_weight = background_df_unique['freq'].to_numpy()
    background_iden = background_df_unique['iden'].to_numpy()
    return background_uni, background_weight, background_iden
    

def modelRewardExplain(date: int, who: int, binary_be_vs_loc = True, blank = True):
    '''
        Give the SHAP value by grouping the type.
    '''
    print('Explaining person: {who:9d}, date: {date}'.format(who=who, date=date))
    model = SIRLU.loadModel(who=who, date=date)

    dataset, visited_id = backgroundData(who=who, date=date)
    dataset_uni, dataset_freq, dataset_iden = sparseBackground(dataset, visited_id)
    print('Data with {k} rows'.format(k=dataset_uni.shape[0]))

    if blank:
        built_bench = np.zeros(model.s_dim).reshape(1, -1)
        locat_bench = np.mean(dataset[:, model.s_dim:], axis=0).reshape(1, -1)
        # zero_bench = np.zeros(dataset.shape[1]).reshape(1, -1)
        # 基线意味着：建成环境取最小值，位置环境取平均值
        home_bench = np.hstack((built_bench, locat_bench))
    else:
        # 全部取平均值
        home_bench = np.mean(dataset, axis=0).reshape(1, -1)

    reward_vector = modelPredict(X=dataset_uni, model=model, attribute_type='reward')
    mu = np.average(reward_vector, weights=dataset_freq)
    sigma = np.sqrt(np.average((reward_vector - mu) ** 2, weights=dataset_freq))

    def modelPredWrapper(X):
        return modelPredict(X, model=model, standardize=True,
                            attribute_type='reward', mu=mu, sigma=sigma)
    # modelPredWrapper = partial(modelPredict, weight=dataset_freq, model=model, standardize=True, attribute_type='reward')
    
    explainer = shap.PermutationExplainer(modelPredWrapper, home_bench)
    shap_values = explainer(dataset_uni)
    
    # below: group the shape var names
    varchr = 'LU_Business,LU_Green,LU_Industry,LU_Public,LU_Residence,subway,density,intersections,road_density,rent'
    varname_BE = varchr.split(',')
    varname_PE = ['lon', 'lat']
    varname = varname_BE + varname_PE
    if binary_be_vs_loc:
        groupmap = {
            'BuiltAttr': varname[:len(varname_BE)],
            'Location': varname[len(varname_BE):]
        }
    else:
        groupmap = {v: [v] for v in varname_BE}
        groupmap['Location'] = varname_PE
    shap_grouped_by_classes = grouped_shap(shap_vals=shap_values.values, features=varname, groups=groupmap)
    return shap_grouped_by_classes, dataset_freq, dataset_iden

def modelUserDateCombination(by_week=True):
    '''
    Extract the user and date combination from the model directory.
    Each target for every seven days.
    '''
    model_dir = './model/'
    # list all users with folder name consisting of all digits.
    # note: change here
    user_list = [name for name in os.listdir(model_dir) if name.isdigit()]
    
    combination = []
    for user in user_list:
        evolution_model_path = model_dir + user + '/' + 'evolution_model/'
        date_list = [int(params.rstrip('.pickle')[-8:]) for params in os.listdir(evolution_model_path)]
        date_list = SIRLU.extract_week_ends(date_list)[0] if by_week else date_list[::7]
        for date in date_list:
            combination.append((int(user), date))
    return combination


def modelDateOfUser(user, by_week = True):
    '''
    Extract the date list of the user.
    Each target for every seven days.
    '''
    model_dir = './model/'
    user = SIRLU.toWhoString(user)
    evolution_model_path = model_dir + user + '/' + 'evolution_model/'
    date_list = [int(params.rstrip('.pickle')[-8:]) for params in os.listdir(evolution_model_path)]
    if by_week:
        date_list, _ = SIRLU.extract_week_ends(date_list)
    else:        
        date_list = date_list[::7]
    return date_list

def explainOneUser(user, parallel=False, binary_be_vs_loc=True, blank=True):
    # parallel version of SHAP explain for one user.
    date_list = modelDateOfUser(user)
    if not parallel:
        shap_dict = dict()
        # add reverse to mitigate the load balancing problem.
        for date in reversed(date_list):
            shap_dict[date] = modelRewardExplain(date, who=user, binary_be_vs_loc=binary_be_vs_loc, blank=blank)
    else:
        # parallel version
        MAX_CPU_COUNT = mp.cpu_count() - 2
        combination = [(date, user, binary_be_vs_loc, blank) for date in reversed(date_list)]
        with mp.Pool(MAX_CPU_COUNT) as pool:
            shap_dict_values = pool.starmap(modelRewardExplain, combination)
        shap_dict = dict(zip(reversed(date_list), shap_dict_values))
    return shap_dict


def explainAllRewards(parallel = False, binary_be_vs_loc=True, blank=True):
    # parallel version of SHAP explain for all users.
    combination = modelUserDateCombination()
    shap_dict = dict()
    
    if not parallel:
        for user, date in combination:
            shap_dict[(user, date)] = modelRewardExplain(date, who=user, binary_be_vs_loc=binary_be_vs_loc,
                                                         blank=blank)
    else:
        # parallel version
        combination_switch = [(date, user, binary_be_vs_loc, blank) for user, date in combination]
        with mp.Pool(MAX_CPU_COUNT) as pool:
            shap_dict_values = pool.starmap(modelRewardExplain, combination_switch)
        for idx, (user, date) in enumerate(combination):
            shap_dict[(user, date)] = shap_dict_values[idx]
    return shap_dict

def modelRewardBaselineCalculation(date: int, who: int):
    '''
        Give the SHAP value by grouping the type.
    '''
    print('Explaining person: {who:8d}, date: {date}'.format(who=who, date=date))
    model = SIRLU.loadModel(who=who, date=date)
    # modelPredWrapper = partial(modelPredict, model=model, attribute_type='reward')

    dataset, visited_id = backgroundData(who=who, date = date)
    dataset_uni, dataset_freq, dataset_iden = sparseBackground(dataset, visited_id)
    
    reward_vector = modelPredict(X=dataset_uni, model=model, attribute_type='reward')
    mu = np.average(reward_vector, weights=dataset_freq)
    sigma = np.sqrt(np.average((reward_vector - mu) ** 2, weights=dataset_freq))

    modelPredWrapper = partial(modelPredict, model=model, standardize=True, attribute_type='reward',
                               mu=mu, sigma=sigma)

    reward_comp = modelPredWrapper(dataset_uni)
    # test for the standardization validity
    average_reward = np.average(reward_comp, weights=dataset_freq)
    print('The average reward is: {reward}'.format(reward=average_reward))
    
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
    # model_dir = './model/'
    # user_list = [int(name) for name in os.listdir(model_dir) if name.isdigit()]
    # user_list.sort()
    # for user in user_list:
    #     # note: remember to change back
    #     res = explainOneUser(user, parallel=True, binary_be_vs_loc=False)
    #     with open('./product/shap_res_{:09d}.pkl'.format(user), 'wb') as f:
    #         pickle.dump(res, f)
    '''
    By Hand
    '''
    # model_dir = './model/'
    # user_list = [int(name) for name in os.listdir(model_dir) if name.isdigit()]
    # user_list.sort()

    # user = 1102234
    # res = explainOneUser(user, parallel=True, binary_be_vs_loc=False)
    # with open('./product/shap_res_{:09d}.pkl'.format(user), 'wb') as f:
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
    #         reward_dict[(user, date)] = modelRewardBaselineCalculation(date, who=user)
    #         with open('./product/reward_res.pkl', 'wb') as f:
    #                 pickle.dump(reward_dict, f)
    '''
    Test area
    '''
    shap_dict = dict()
    date = 20230507
    shap_dict[date] = modelRewardExplain(date, who=1102234)