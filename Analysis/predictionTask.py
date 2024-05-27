# import module from the parent directory
import sys
import os
import numpy as np
working_directory = os.path.abspath('.')
sys.path.append(working_directory)

import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
from SCBIRL_Global_PE.migrationProcess import *
import packFuncForShap as pack4shap
from SCBIRL_Global_PE.utils import TravelData
from geopy.distance import geodesic
import pickle

def trajectoryCompute(model, tcs: list[TravelData], ):
    '''
        Compute the reward and value of the trajectory using the model.
        Based on Transformer-based model.
    '''
    state_attribute, _ = preprocessStateAttributes('./data/before_migrt.json')

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


def predictQValueEvalutaion(tcs: list[TravelData], q_value_tensor: np.array, selection_mask: np.array, 
                            id_coord_mapping_path="./data/id_coords_mapping.pkl"):

    with open(id_coord_mapping_path, "rb") as f:
        coords_id = pickle.load(f)

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


def modelEvaluation(model, start_date: int, end_date: int, id_coord_mapping_path="./data/id_coords_mapping.pkl"):
    with open('./data/all_traj.json', 'r') as file:
        loaded_dicts_all = json.load(file)
    loaded_namedtuples_all = [TravelData(**d) for d in loaded_dicts_all]
    # 按照时间筛选出符合条件的数据
    tcs = list(filter(lambda tc: start_date <= tc.date <= end_date, loaded_namedtuples_all))
    # 计算reward和Q值
    reward_tensor, q_value_tensor, selection_mask = trajectoryCompute(model, tcs)
    # 计算预测精度
    average_prediction_distance_error, prediction_accuracy = predictQValueEvalutaion(tcs, q_value_tensor, selection_mask, id_coord_mapping_path)
    return average_prediction_distance_error, prediction_accuracy

def visitedDate(all_traj_path='./data/all_traj.json'):
    '''
    Only used for test user. User documented in structured directories owns personal information namedtuple.
    '''
    with open(all_traj_path, 'r') as file:
        loaded_dicts_all = json.load(file)
    visited_dates = [d.get('date') for d in loaded_dicts_all]
    return visited_dates


if __name__ == "__main__":
    # read in the model
    # use the load model function
    # todo: recheck the function, and move them into correct modules, # for example, the loadModel function should be a utils function.

    migrtdate = 20190707
    visitdate = visitedDate()
    # create a list of dates for evolution evaluation
    evoludate = list(filter(lambda d: d >= migrtdate, visitdate))
    # create buffer range: 10 future days and the date itself
    evolution_buffer = 10 + 1

    pr_contrib_list = []
    exp_contrib_list = []
    pr_log_contrib_list = []
    exp_log_contrib_list = []
    # create a dataframe of evolution
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

        # import the models
        model_path = './model/'

        if evolution_pos < 10:
            prior_model = pack4shap.loadModel()
        else:
            prior_evolution_date = evoludate[evolution_pos - 10]
            prior_model = pack4shap.loadModel(model_path + 'no_prior_model/{}.pickle'.format(prior_evolution_date))

        experience_model = pack4shap.loadModel(model_path + 'no_prior_model/{}.pickle'.format(one_evolution_date))

        complete_model = pack4shap.loadModel(model_path + 'complete_model/{}.pickle'.format(one_evolution_date))

        tabula_rasa_model = pack4shap.loadModel(tabular=True)

        # conduct the evaluation
        mse_pr, acc_pr = modelEvaluation(prior_model, start_date, end_date)
        mse_exp, acc_exp = modelEvaluation(experience_model, start_date, end_date)
        mse_both, acc_both = modelEvaluation(complete_model, start_date, end_date)
        mse_non, acc_non = modelEvaluation(tabula_rasa_model, start_date, end_date)

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
    
    result_df.to_csv('./product/evolution_evaluation.csv', index=False)