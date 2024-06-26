# import module from the parent directory
import sys
import os
import numpy as np
working_directory = os.path.abspath('.')
sys.path.append(working_directory)

import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import SCBIRL_Global_PE.utils as SIRLU
from SCBIRL_Global_PE.migrationProcess import *
import packFuncForShap as pack4shap
from SCBIRL_Global_PE.utils import TravelData, iter_start_date
from TRAJ_PROCESS.prepareChain import Traveler
from geopy.distance import geodesic
import pickle


def trajectoryCompute(model, tcs: list[TravelData], state_attribute: pd.DataFrame):
    '''
        Compute the reward and value of the trajectory using the model.
        Based on Transformer-based model.
    '''
    stateNextState, _, peNextpe = processTrajectoryData(tcs, state_attribute, model.s_dim)
    states = stateNextState[:, :, 0, np.newaxis, :] # [num_traj, max_steps, 1, state_dim]
    pe_codes = peNextpe[:, :, 0, np.newaxis, :] # [num_traj, max_steps, 1, pecode_dim]
    pad_mask = np.where(states == -999, True, False).all(axis = (-1, -2))
    pad_mask_reverse = ~pad_mask
    selection_mask = np.where(pad_mask_reverse)

    # 根据transformer的结构，这里的关键在于：拟合出来的Q函数都是自动向后掩蔽的
    # 也就是说，计算reward的时候，会考虑全体轨迹。Encoding阶段相当于个体在做出行计划时，每一个地方的奖励值。
    # 计算Q函数的时候，只考虑已经发生的轨迹，Decoding阶段相当于个体出行时的具体规划。
    reward_tensor = model.reward(states, pe_codes) # [num_traj, max_steps, 2], 最后一个维度是均值和对数方差
    q_value_tensor = model.QValue(states, pe_codes) # [num_traj, max_steps, action_dim]
    return reward_tensor, q_value_tensor, selection_mask

def kl_divergence_between_normal(mu1, logvar1, mu2, logvar2):
    term_1 = logvar2 - logvar1
    term_2 = (np.exp(logvar1) + (mu1 - mu2) ** 2) / np.exp(logvar2)
    term_3 = -1
    return (term_1 + term_2 + term_3) / 2
    
def predictRewardEvaluation(reward_tensor: np.array, selection_mask: np.array):
    reward_array = reward_tensor[selection_mask]
    data = {'idx_traj': selection_mask [0], 
        'idx_step': selection_mask [1], 
        'mean_reward': reward_array[:, 0], 
        'log_variance_reward': reward_array[:, 1]}
    # generate the pandas dataframe
    reward_df = pd.DataFrame(data)
    # calculate the mean and variance of the reward
    return reward_df
    
def predictQValueEvalutaion(tcs: list[TravelData], q_value_tensor: np.array, selection_mask: np.array, coords_id: dict):

    #  求出每一步最大Q值的动作索引
    max_q_action_idx = np.argmax(q_value_tensor, axis = -1)
    max_q_action_idx = jax.device_get(max_q_action_idx)
    predicted_action = max_q_action_idx[selection_mask].tolist()
    # 找到他们对应的坐标位置
    predicted_coords = list(map(lambda k: coords_id.get(k, (np.nan, np.nan)), predicted_action))
    # 把对应位置的坐标map出来
    predicted_coords = np.array(predicted_coords)
    # 计算实际的到访坐标
    actual_coords = []
    for tc in tcs:  
        actual_coords.extend(tc.travel_chain)
    actual_coords = np.array(actual_coords)

    predicted_coords = predicted_coords[:, ::-1]
    actual_coords = actual_coords[:, ::-1]
    # 计算距离
    assert len(predicted_coords) == len(actual_coords), "The number of predicted coordinates and actual coordinates do not match."
    geodesic_distances = list(map(lambda p1, p2: geodesic(p1, p2).km, predicted_coords, actual_coords))    
    # 计算空间误差
    average_prediction_distance_error = np.mean(np.array(geodesic_distances))
    # 计算预测精度，500m作为阈值
    prediction_accuracy = np.mean(np.array(geodesic_distances) < 0.5)
    return average_prediction_distance_error, prediction_accuracy


def modelEvaluation(model, who: int, start_date: int, end_date: int, mode: str, coords_id: dict = None):
    
    assert mode in ['reward', 'action'], "The mode should be either 'reward' or 'action'."
    # 读入建成环境数据  
    state_attribute = SIRLU.load_state_attrs(who)
    
    # 读入所有的轨迹数据，这里的I/O函数都解耦了
    loaded_namedtuples_all = SIRLU.load_all_traj(who)
    # 按照时间筛选出符合条件的数据
    tcs = list(filter(lambda tc: start_date <= tc.date <= end_date, loaded_namedtuples_all))
    
    # 计算reward和Q值
    reward_tensor, q_value_tensor, selection_mask = trajectoryCompute(model, tcs, state_attribute)

    if mode == 'action':
        # 计算预测精度
        average_prediction_distance_error, prediction_accuracy = predictQValueEvalutaion(tcs, q_value_tensor, selection_mask, coords_id)
        return average_prediction_distance_error, prediction_accuracy
    else:
        # 计算奖励值
        reward_df = predictRewardEvaluation(reward_tensor, selection_mask)
        return reward_df
    

def visitedDate(all_traj_path='./data/all_traj.json'):
    '''
    Only used for test user. User documented in structured directories owns personal information namedtuple.
    '''
    with open(all_traj_path, 'r') as file:
        loaded_dicts_all = json.load(file)
    visited_dates = [d.get('date') for d in loaded_dicts_all]
    return visited_dates

def personPredictEvaluation(who: int):
    id_coords_mapping = load_id_coords_mapping(who)
    traveler = load_traveler(who)
    
    migrtdate = traveler.migrt
    visitdate = traveler.visit_date
    # create a list of dates for evolution evaluation
    evoludate = list(filter(lambda d: d >= migrtdate, visitdate))
    # create buffer range: 10 future days and the date itself
    evolution_buffer = 10 + 1

    pr_contrib_list = []
    exp_contrib_list = []
    pr_log_contrib_list = []
    exp_log_contrib_list = []
    
    for one_evolution_date in evoludate:
        # find the position of the evolution date
        evolution_pos = evoludate.index(one_evolution_date)
        # if the buffer range is out of range, break the loop
        if evolution_pos + evolution_buffer > len(evoludate):
            break
        print("The evolution date is: ", one_evolution_date)
        
        # create the date range for evaluation
        evolution_date_range = evoludate[evolution_pos : evolution_pos + evolution_buffer]
        # know the start and end date for evaluation func
        start_date = evolution_date_range[0]
        end_date = evolution_date_range[-1]
        
        if evolution_pos < 10:
            prior_model = pack4shap.loadModel(who=who, )
        else:
            prior_evolution_date = evoludate[evolution_pos - 10]
            prior_model = pack4shap.loadModel(who=who, date=prior_evolution_date, prior=True)

        experience_model = pack4shap.loadModel(who=who, date=one_evolution_date, prior=False)

        complete_model = pack4shap.loadModel(who=who, date=one_evolution_date, prior=True)

        tabula_rasa_model = pack4shap.loadModel(who=who, tabular=True)

        state_attribute = SIRLU.load_state_attrs(who, before=False)
        
        mse_pr, acc_pr = modelEvaluation(prior_model, who, start_date, end_date, mode='action', coords_id=id_coords_mapping)
        mse_exp, acc_exp = modelEvaluation(experience_model, who, start_date, end_date, mode='action', coords_id=id_coords_mapping)
        mse_both, acc_both = modelEvaluation(complete_model, who, start_date, end_date, mode='action', coords_id=id_coords_mapping)
        mse_non, acc_non = modelEvaluation(tabula_rasa_model, who, start_date, end_date, mode='action', coords_id=id_coords_mapping)
        
        # compute the contribution based on mse, according to nexus paper
        pr_log_contrib = 0.5 * (np.log10(mse_exp/mse_both) + np.log10(mse_non/mse_pr))
        pr_log_contrib_list.append(pr_log_contrib)
        exp_log_contrib = 0.5 * (np.log10(mse_pr/mse_both) + np.log10(mse_non/mse_exp))
        exp_log_contrib_list.append(exp_log_contrib)

        # compute the contribution based on accuracy
        pr_contrib = 0.5 * (acc_both - acc_exp + acc_pr - acc_non)
        pr_contrib_list.append(pr_contrib)
        exp_contrib = 0.5 * (acc_both - acc_pr + acc_exp - acc_non)
        exp_contrib_list.append(exp_contrib)
        
    # result in the form of a dataframe
    result_df = pd.DataFrame({'date': evoludate[:len(pr_contrib_list)], 
                              'pr_acc_contrib': pr_contrib_list, 
                              'exp_acc_contrib': exp_contrib_list, 
                              'pr_mse_contrib': pr_log_contrib_list, 
                              'exp_mse_contrib': exp_log_contrib_list})
    
    return result_df


def stepwise_kl_div_compute(df_1: pd.DataFrame, df_2: pd.DataFrame):
    df_1 = df_1.set_index(['idx_traj', 'idx_step'])
    df_2 = df_2.set_index(['idx_traj', 'idx_step'])
    # inner join the two df by index
    df = df_1.join(df_2, how='inner', lsuffix='_1', rsuffix='_2')
    # apply the kl divergence function by row
    kl_series = df.apply(lambda row: kl_divergence_between_normal(row['mean_reward_1'], row['log_variance_reward_1'], 
                                                      row['mean_reward_2'], row['log_variance_reward_2']), axis=1)
    return kl_series.mean()


def personInterpretEvaluation(who: int):
    traveler = load_traveler(who)
    
    migrtdate = iter_start_date
    visitdate = traveler.visit_date
    # create a list of dates for evolution evaluation
    evoludate = list(filter(lambda d: d > migrtdate, visitdate))
    # create buffer range: 10 future days and the date itself
    evolution_buffer = 10 + 1

    pr_contrib_list = []
    exp_contrib_list = []
    for one_evolution_date in evoludate:
        # find the position of the evolution date
        evolution_pos = evoludate.index(one_evolution_date)
        # if the buffer range is out of range, break the loop
        if evolution_pos + evolution_buffer > len(evoludate):
            break
        print("The evolution date is: ", one_evolution_date)
        
        # create the date range for evaluation
        evolution_date_range = evoludate[evolution_pos : evolution_pos + evolution_buffer]
        # know the start and end date for evaluation func
        start_date = evolution_date_range[0]
        end_date = evolution_date_range[-1]
        
        if evolution_pos < 10:
            prior_model = pack4shap.loadModel(who=who, )
        else:
            prior_evolution_date = evoludate[evolution_pos - 10]
            prior_model = pack4shap.loadModel(who=who, date=prior_evolution_date, prior=True)

        experience_model = pack4shap.loadModel(who=who, date=one_evolution_date, prior=False)

        complete_model = pack4shap.loadModel(who=who, date=one_evolution_date, prior=True)

        tabula_rasa_model = pack4shap.loadModel(who=who, tabular=True)

        df_prior = modelEvaluation(prior_model, who, start_date, end_date, mode='reward')
        df_experience = modelEvaluation(experience_model, who, start_date, end_date, mode='reward')
        df_complete = modelEvaluation(complete_model, who, start_date, end_date, mode='reward')
        df_tabular = modelEvaluation(tabula_rasa_model, who, start_date, end_date, mode='reward')
        
        prior_contrib = 0.5 * stepwise_kl_div_compute(df_tabular, df_prior) + 0.5 * stepwise_kl_div_compute(df_experience, df_complete)
        exper_contrib = 0.5 * stepwise_kl_div_compute(df_tabular, df_experience) + 0.5 * stepwise_kl_div_compute(df_prior, df_complete)
        pr_contrib_list.append(prior_contrib)
        exp_contrib_list.append(exper_contrib)
    
    # result in the form of a dataframe
    result_df = pd.DataFrame({'date': evoludate[:len(pr_contrib_list)], 
                              'pr_contrib': pr_contrib_list, 
                              'exp_contrib': exp_contrib_list})
    return result_df

if __name__ == "__main__":

    
    wholist = [int(f) for f in os.listdir('./model/')]
    wholist.sort()
    '''
    Parallel Verison
    '''
    # import multiprocessing as mp
    # CPU_COUNT = len(wholist)
    # with mp.Pool(CPU_COUNT) as pool:
    #     iterDfs = pool.map(personInterpretEvaluation, wholist)

    # with open('./product/iterationEvo.pkl', 'wb') as file:
    #     pickle.dump(iterDfs, file)
    
    '''
    Hand Version
    '''
    i = 0
    who = wholist[i]
    res = dict()
    res[who] = personInterpretEvaluation(who)    
    with open('./product/interpretEvo_{:09d}.pkl'.format(i), 'wb') as file:
        pickle.dump(res, file)    