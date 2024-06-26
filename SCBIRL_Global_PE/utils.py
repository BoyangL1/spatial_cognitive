import jax.numpy as np
import numpy as onp
import pandas as pd
from scipy.linalg import qr
import json
import pickle
import os

from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler

TravelData = namedtuple('TravelChain', ['date', 'travel_chain','id_chain','fnid_chain'])
Traveler = namedtuple('Traveler', ['who', 'visit_date'])
iter_start_date = 20230501


def loadJsonFile(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def loadTravelDataFromDicts(data_dicts): # data_dicts: list[dict]
    return [TravelData(**d) for d in data_dicts]

def getStateRow(state_attribute, state):
    row = state_attribute[state_attribute['fnid'] == state]
    return np.array(row.values[0][1:])

def getActionDim(all_chains):
    """ 
    Calculate the dimension of the action space for a travel chain. 
    This includes the count of distinct actions plus an additional dimension for the 'no action' (-1) case.
    """
    # 去所有出行链中id 最大的一个编号, 加1为长度, 再加1为no action
    return max({id for tc in all_chains for id in tc.id_chain}) + 2 # id_chain is a sequence, so the length = max +1 +1

def preprocessStateAttributes(all_feature_path):
    state_attribute = pd.read_csv(all_feature_path)

    # Calculate the dimension of state attributes (excluding 'fnid')
    # 特征的维度，不是状态数
    s_dim = state_attribute.shape[1] - 1

    # Separate 'fnid' column and other columns
    fnid_col = state_attribute[['fnid']]
    other_cols = state_attribute.drop(columns=['fnid'])

    # Scale the features using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_cols = scaler.fit_transform(other_cols)

    # Create a DataFrame from the scaled features
    scaled_df = pd.DataFrame(scaled_cols, columns=other_cols.columns)

    # Return the combined DataFrame and the dimension of state attributes
    return pd.concat([fnid_col, scaled_df], axis=1), s_dim

def padSequences(data_list, element_shape, padding_value=-999):
    """
    Pad lists of variable lengths containing elements of a specific shape.

    Args:
        data_list (list of lists): List of lists containing elements of varying lengths.
        element_shape (tuple): The shape of the elements in the inner lists.
        padding_value (int, optional): The value to use for padding.

    Returns:
        numpy array: Padded data_list.
    """
    # Find the maximum length of the inner lists
    max_list_length = max(len(inner_list) for inner_list in data_list)

    # Pad each inner list
    padded_data_list = []
    for inner_list in data_list:
        # Calculate the number of padding elements needed
        num_padding_elements = max_list_length - len(inner_list)
        
        # Create padding elements with the specified shape and value
        padding_elements = [onp.full(element_shape, padding_value) for _ in range(num_padding_elements)]
        
        # Extend the original list with padding elements
        padded_list = inner_list + padding_elements

        # Append the padded list to the result list
        padded_data_list.append(padded_list)

    # Convert the list of lists of numpy arrays to a higher-dimensional numpy array
    return onp.array(padded_data_list)

def processTrajectoryData(traj_chains, state_attribute, s_dim):
    """
    Process trajectory data and return the processed data in the form of arrays.

    Args:
        traj_chains (list): List of trajectory chains.
        state_attribute (str): Attribute to consider for state representation.
        s_dim (int): Dimension of the state representation.

    Returns:
        tuple: A tuple containing three arrays:
            - state_next_state (ndarray): Array of shape (num_chains, max_traj_len, 2, s_dim) representing the current and next states.
            - action_next_action (ndarray): Array of shape (num_chains, max_traj_len, 2, 1) representing the current and next actions.
            - pe_next_pe (ndarray): Array of shape (num_chains, max_traj_len, 2, nlevel*3) representing the current and next grid codes.

    """
    state_next_state = []
    action_next_action = []
    pe_next_pe = []

    for tc in traj_chains:
        sns_chain, ana_chain, pnp_chain = [], [], []
        for t in range(len(tc.travel_chain)):
            # Get the final destination in the travel chain
            # destination = tc.travel_chain[-1]
            # s_n_s: [2, s_dim]的数组，第一行是当前状态的特征，第二行是下一个状态的特征
            # a_n_a: [2, 1]的数组，第一行是当前动作，第二行是下一个动作，编号都是状态码
            # p_n_p: [2, nlevel*3]的数组，第一行是当前状态的grid code，第二行是下一个状态的grid code
            s_n_s, a_n_a, p_n_p = processSingleTrajectory(tc, t, state_attribute, s_dim)

            # Append the results to respective lists
            sns_chain.append(s_n_s)
            ana_chain.append(a_n_a)
            pnp_chain.append(p_n_p)
        state_next_state.append(sns_chain)
        action_next_action.append(ana_chain)
        pe_next_pe.append(pnp_chain)
    # pad sequence to the same length
    # 把traj_len填充到最大的长度，变为max_traj_len, 其余值默认为-999填充
    # todo: 考虑是否要长度对齐
    state_next_state = padSequences(state_next_state,s_n_s.shape) 
    action_next_action = padSequences(action_next_action,a_n_a.shape,padding_value=-1)
    pe_next_pe = padSequences(pe_next_pe,p_n_p.shape)

    return np.array(state_next_state), np.array(action_next_action), np.array(pe_next_pe)

def globalPE(coords,dimension):
    x,y = coords
    Q = np.load('./data_pe/Q_matrix.npy')
    with open('./data_pe/random_angle_list.pkl', 'rb') as file:
        angle_list = pickle.load(file)

    for k in range(1,dimension+1):
        theta = 2 * onp.pi / 3  
        R = onp.array([[onp.cos(theta), -onp.sin(theta)], [onp.sin(theta), onp.cos(theta)]])
        scale_factor = (200**(k/dimension))
        angle = angle_list[k-1]
        omega_n0 = onp.array([onp.cos(angle), onp.sin(angle)]) * scale_factor
        omega_n1 = R.dot(omega_n0)
        omega_n2 = R.dot(omega_n1)

        coords = onp.vstack((x, y))
        eiw0x = onp.exp(1j * onp.dot(omega_n0,coords))
        eiw1x = onp.exp(1j * onp.dot(omega_n1,coords))
        eiw2x = onp.exp(1j * onp.dot(omega_n2,coords))

        g_n = Q.dot(onp.array([eiw0x, eiw1x, eiw2x]))
        if k == 1:
            g = g_n
        else:
            g = onp.concatenate((g, g_n), axis=0)
    return g

def processSingleTrajectory(tc, t, state_attribute, s_dim):
    if t < len(tc.travel_chain)-1:
        this_state, next_state = tc.travel_chain[t], tc.travel_chain[t + 1]
        this_fnid, next_fnid = tc.fnid_chain[t], tc.fnid_chain[t+1]
        # state attribute
        s_n_s = onp.zeros((2, s_dim))
        s_n_s[0, :] = getStateRow(state_attribute, this_fnid)
        s_n_s[1, :] = getStateRow(state_attribute, next_fnid)

        # note: 这一块和之前cann版本代码不同
        # get global positional encoding of state
        this_pe = globalPE(this_state,s_dim)
        next_pe = globalPE(next_state,s_dim)
        # save to s_grid_s
        s_pe_s = onp.empty((2, s_dim*3), dtype=onp.complex_) # Each dimensional grid encoding has three component
        s_pe_s[0, :] = this_pe.flatten()
        s_pe_s[1, :] = next_pe.flatten()

        # action
        a_n_a = onp.zeros((2, 1))   
        a_n_a[0] = tc.id_chain[t + 1]
        a_n_a[1] = tc.id_chain[t + 2] if t + 2 < len(tc.id_chain) else -1
    else:
        this_state, next_state = tc.travel_chain[t], None
        this_fnid, next_fnid = tc.fnid_chain[t], None
        s_n_s = onp.zeros((2, s_dim))
        s_n_s[0, :] = getStateRow(state_attribute, this_fnid)

        # get grid code of state and destination,dim(8,128,128)
        this_pe = globalPE(this_state,s_dim)
        next_pe = onp.zeros_like(this_pe)
        # save to s_grid_s
        s_pe_s = onp.empty((2, s_dim*3), dtype=onp.complex_)
        s_pe_s[0, :] = this_pe.flatten()
        s_pe_s[1, :] = next_pe.flatten()

        a_n_a = onp.zeros((2, 1))
        a_n_a[0] = -1
        a_n_a[1] = -1

    return s_n_s, a_n_a, s_pe_s

def loadTrajChain(user_path, type: str, start_date=None):
    if type not in {'before', 'after', 'all'}:
        raise ValueError("Invalid type. Must be one of 'before', 'after', or 'all'.")
    
    full_traj_path = user_path + 'all_traj.json'
    all_trajs = loadJsonFile(full_traj_path)
    if type == 'before':
        trajs = [chain for chain in all_trajs if chain['date'] < start_date]
    elif type == 'after':
        trajs = [chain for chain in all_trajs if chain['date'] >= start_date]
    else:
        trajs = all_trajs.copy()
    chains_loaded = loadTravelDataFromDicts(trajs)
    a_dim = getActionDim(loadTravelDataFromDicts(all_trajs))
    
    full_feature_path = user_path + 'all_traj_feature.csv'
    state_attribute, s_dim = preprocessStateAttributes(full_feature_path)
    state_next_state, action_next_action, grid_next_grid= processTrajectoryData(chains_loaded, state_attribute, s_dim)
    # 这里的state_next_state是一个四维数组，第一维是轨迹条数，第二维是轨迹最大长度（即每条轨迹pair数），第三维是状态数（2），第四维是特征数
    # action_next_action是一个四维数组，第一维是轨迹条数，第二维是轨迹最大长度（即每条轨迹pair数），第三维是状态数（2），第四维是虚假轴
    # 第三个输出grid_next_grid是四维数组, dim(num_traj, max_traj_len, 2, nlevel)
    return state_next_state, action_next_action, grid_next_grid, a_dim, s_dim
    
def plugInDataPair(tc, stateAttribute, model, visitedState):
    # Preprocess trajectory data and update visited states
    # 每次迭代，高维度数组的轨迹长度都是不一样的，都是本批次（10天内）最长的长度。
    stateNextState, actionNextAction, peNextpe = processTrajectoryData(tc, stateAttribute, model.s_dim)
    # 这里会更新去过的state
    for t in tc:
        visitedState.update(tuple(item) if isinstance(item, list) else item for item in t.travel_chain)

    # Set model inputs for training or evaluation
    model.inputs = stateNextState
    model.targets = actionNextAction
    model.pe_code = peNextpe


def toWhoString(who: int, digits=9):
    return '{:0{digits}d}'.format(who, digits=digits)


def migrationDate(who: int = 36384703):
    # using the os path to get the after traj path
    data_dir = './data/user_data/' + toWhoString(who) + '/'
    after_traj_path = data_dir + 'after_migrt.json'

    after_traj = loadJsonFile(after_traj_path)
    migration_date = after_traj[0]['date']
    return migration_date

def load_traveler(who: int):
    with open(f'./data/user_data/{toWhoString(who)}/traveler_info.pkl', 'rb') as file:
        return pickle.load(file)

def load_id_coords_mapping(who: int):
    data_dir = f'./data/user_data/'
    id_coord_mapping_path = data_dir + toWhoString(who) + '/id_coords_mapping.pkl'
    with open(id_coord_mapping_path, "rb") as f:
        coords_id = pickle.load(f)
    return coords_id

def load_fnid_coords_mapping(who: int):
    data_dir = f'./data/user_data/'
    id_coord_mapping_path = data_dir + toWhoString(who) + '/coords_fnid_mapping.pkl'
    with open(id_coord_mapping_path, "rb") as f:
        coords_fnid = pickle.load(f)
    return coords_fnid

def load_all_traj(who: int):
    data_dir = f'./data/user_data/'
    all_traj_path = data_dir + toWhoString(who) + '/all_traj.json'
    with open(all_traj_path, 'r') as file:
        loaded_dicts_all = json.load(file)
    loaded_namedtuples_all = [TravelData(**d) for d in loaded_dicts_all]
    return loaded_namedtuples_all

def load_state_attrs(who: int, before=True):
    data_dir = f'./data/user_data/'
    filename = 'all.traj.json'
    
    traj_path = data_dir + toWhoString(who) + '/' + filename
    state_attribute, _ = preprocessStateAttributes(traj_path)
    return state_attribute

def visited_date(who: int):
    traveler = load_traveler(who)
    return traveler.visit_date

if __name__ == "__main__":
    path = f'./data/before_migrt.json'
    full_traj_path = f'./data/all_traj.json'

    inputs, targets_action, pe_code, a_dim, s_dim = loadTrajChain(path,full_traj_path)
    print(inputs.shape,targets_action.shape, pe_code.shape)
