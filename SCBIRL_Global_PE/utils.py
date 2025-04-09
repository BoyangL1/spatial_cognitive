import jax.numpy as np
import numpy as onp
import pandas as pd
import json
import pickle
import os
import pyproj


from scipy.linalg import qr
from collections import namedtuple
from datetime import date, timedelta, datetime
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import List
from pyproj import Transformer
from SCBIRL_Global_PE import SCBIRLTransformer as SIRLT

TravelData = namedtuple('TravelChain', ['date', 'travel_chain','id_chain','fnid_chain'])
Traveler = namedtuple('Traveler', ['who', 'visit_date', 'iter_start_date'])
training_baseline_count = 50

UserDataPart = './data/user_data_survey/'
# UserDataPart = './data/user_data_migrt/'
# UserDataPart = './data/user_data_test/'

Padding = -999

def loadJsonFile(file_path):
    '''
    load travel json data as dict
    '''
    with open(file_path, 'r') as file:
        return json.load(file)

def loadTravelDataFromDicts(data_dicts):
    '''
    params: data_dicts: list[dict]
    return: list[TravelData]
    '''
    return [TravelData(**d) for d in data_dicts]

def loadTravelChainAll(who: int):
    '''
    Combination of the above two functions
    '''
    full_traj_path = UserDataPart + toWhoString(who) + '/all_traj.json'
    all_trajs = loadJsonFile(full_traj_path)
    chains_loaded = loadTravelDataFromDicts(all_trajs)
    return chains_loaded
    

def loadModel(who, date = None, prior = True, accumulate = False, tabular = False):
    '''
        Load the model from the model directory.
        Must correctly set the directory at first.
    '''
    data_dir = UserDataPart + toWhoString(who) + '/'
    model_dir = './model/' + toWhoString(who) + '/'
    
    iter_start_date = load_traveler(who).iter_start_date
    inputs, targets_action, pe_code, action_dim, state_dim = loadTrajChain(data_dir, type='before', start_date=iter_start_date)
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


def getStateRow(state_attribute, state_fnid):
    '''
    get the feature vector from a given fnid
    '''
    row = state_attribute[state_attribute['fnid'] == state_fnid]
    return np.array(row.values[0][1:])

def getActionDim(all_chains):
    """ 
    Calculate the dimension of the action space for a travel chain. 
    This includes the count of distinct actions plus an additional dimension for the 'no action' (-1) case.
    """
    # 去所有出行链中id 最大的一个编号, 加1为长度, 再加1为no action
    return max({id for tc in all_chains for id in tc.id_chain}) + 2 # id_chain is a sequence, so the length = max +1 +1

def coords2UTMmeters(coords: np.ndarray):
    """
    Convert geographic coordinates (lon, lat) to UTM coordinates (x, y) in EPSG:32650 system.
    
    Parameters:
    -----------
    id_coords_mapping : dict
        Dictionary mapping IDs to [lon, lat] coordinates
    
    Returns:
    --------
    np.array
        Array of shape with the final dimension as 2 containing [x, y] UTM coordinates
    """
    
    assert coords.shape[-1] == 2, "The final dimension of the input coordinates must be two."
    crs_src = pyproj.CRS('EPSG:4326')
    crs_tgt = pyproj.CRS('EPSG:32650')
    transformer = pyproj.Transformer.from_crs(crs_src, crs_tgt, always_xy=True)
    
    coords_reshape = coords.reshape(-1, 2)
    # Convert coordinates
    utm_reshape = []
    for coord in coords_reshape:
        lon, lat = coord
        if abs(lon) > 180 or abs(lat) > 90:
            # strange value, keep the same
            x, y = lon, lat
        else:
            x, y = transformer.transform(lon, lat)
            # turn to km.
            x, y = x / 1e3, y / 1e3
        utm_reshape.append([x, y])
    utm_reshape = np.array(utm_reshape)
    utm = utm_reshape.reshape(coords.shape)
    return utm



def preprocessStateAttributes(all_feature_path):
    '''
    normalize the state attributes matrix
    '''
    state_attribute = pd.read_csv(all_feature_path)

    # Calculate the dimension of state attributes (excluding 'fnid')
    # 特征的维度，不是状态数
    s_dim = state_attribute.shape[1] - 1

    # Separate 'fnid' column and other columns
    fnid_col = state_attribute[['fnid']]
    other_cols = state_attribute.drop(columns=['fnid'])

    # Scale the features using MinMaxScaler
    scaler = RobustScaler()
    scaled_cols = scaler.fit_transform(other_cols)

    # Create a DataFrame from the scaled features
    scaled_df = pd.DataFrame(scaled_cols, columns=other_cols.columns)

    # Return the combined DataFrame and the dimension of state attributes
    return pd.concat([fnid_col, scaled_df], axis=1), s_dim

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
            - positions_next_positions (ndarray): Array of shape (num_chains, max_traj_len, 2, 2) representing the current and next coordinates.

    """
    state_next_state = []
    action_next_action = []
    positions_next_positions = [] 

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
        positions_next_positions.append(pnp_chain)
    # pad sequence to the same length
    # 把traj_len填充到最大的长度，变为max_traj_len, 其余值默认为-999填充
    # todo: 考虑是否要长度对齐
    state_next_state = padSequences(state_next_state,s_n_s.shape) 
    action_next_action = padSequences(action_next_action,a_n_a.shape,padding_value=-1)
    positions_next_positions = padSequences(positions_next_positions,p_n_p.shape)

    return np.array(state_next_state), np.array(action_next_action), np.array(positions_next_positions)

def processSingleTrajectory(tc, t, state_attribute, s_dim):
    '''
    Given a travel chain and a time step: return a state feature pair or action pair, 
    in order to form the TD training array.
    '''
    if t < len(tc.travel_chain)-1:
        this_coord, next_coord = tc.travel_chain[t], tc.travel_chain[t + 1]
        this_fnid, next_fnid = tc.fnid_chain[t], tc.fnid_chain[t+1]
        # state attribute
        s_n_s = onp.zeros((2, s_dim))
        s_n_s[0, :] = getStateRow(state_attribute, this_fnid)
        s_n_s[1, :] = getStateRow(state_attribute, next_fnid)

        # 位置信息：直接使用经纬度
        p_n_p = onp.zeros((2, 2))  # [2, 2] 表示 [当前/下一个, [lat, lon]]
        p_n_p[0] = this_coord  # this_state应该是[lat, lon]格式
        p_n_p[1] = next_coord

        # action
        a_n_a = onp.zeros((2, 1))   
        a_n_a[0] = tc.id_chain[t + 1]
        a_n_a[1] = tc.id_chain[t + 2] if t + 2 < len(tc.id_chain) else -1
    else:
        # 处理序列末尾
        this_fnid = tc.fnid_chain[t]
        this_coord = tc.travel_chain[t]
        s_n_s = onp.zeros((2, s_dim))
        s_n_s[0, :] = getStateRow(state_attribute, this_fnid)
        # the latter position is filled with padding value
        s_n_s[1, :] = Padding

        p_n_p = onp.zeros((2, 2))
        p_n_p[0] = this_coord
        p_n_p[1] = Padding

        a_n_a = onp.zeros((2, 1))
        a_n_a[0] = -1
        a_n_a[1] = -1

    return s_n_s, a_n_a, p_n_p

def padSequences(data_list, element_shape, padding_value=Padding):
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

def loadTrajChain(user_path, type: str, start_date=None):
    '''
    return the training data.
    '''
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
    # 注意，这里建成环境做了归一化，但是位置编码是没有的
    state_next_state, action_next_action, positions_next_positions = processTrajectoryData(chains_loaded, state_attribute, s_dim)
    # 这里的state_next_state是一个四维数组，第一维是轨迹条数，第二维是轨迹最大长度（即每条轨迹pair数），第三维是状态数（2），第四维是特征数
    # action_next_action是一个四维数组，第一维是轨迹条数，第二维是轨迹最大长度（即每条轨迹pair数），第三维是状态数（2），第四维是虚假轴
    # 第三个输出grid_next_grid是四维数组, dim(num_traj, max_traj_len, 2, nlevel)
    positions_next_positions = coords2UTMmeters(positions_next_positions)
    return state_next_state, action_next_action, positions_next_positions, a_dim, s_dim
    
def plugInDataPair(tc, stateAttribute, model, visitedState):
    ''' 
    process and feed the travel chain data into the model for subsequent training.
    '''
    # Preprocess trajectory data and update visited states
    # 每次迭代，高维度数组的轨迹长度都是不一样的，都是本批次（10天内）最长的长度。
    stateNextState, actionNextAction, peNextpe = processTrajectoryData(tc, stateAttribute, model.s_dim)
    # 这里会更新去过的state
    for t in tc:
        visitedState.update(tuple(item) if isinstance(item, list) else item for item in t.travel_chain)

    # Set model inputs for training or evaluation
    model.inputs = stateNextState
    model.targets = actionNextAction
    model.positions = peNextpe


def toWhoString(who: int, digits=9):
    return '{:0{digits}d}'.format(who, digits=digits)


# def migrationDate(who: int = 36384703):
#     # using the os path to get the after traj path
#     data_dir = UserDataPart + toWhoString(who) + '/'
#     after_traj_path = data_dir + 'after_migrt.json'

#     after_traj = loadJsonFile(after_traj_path)
#     migration_date = after_traj[0]['date']
#     return migration_date

def load_pickle_binary(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def load_traveler(who: int):
    with open(UserDataPart + f'{toWhoString(who)}/traveler_info.pkl', 'rb') as file:
        return pickle.load(file)

def load_id_coords_mapping(who: int):
    '''
    load id-coord mapper
    '''
    data_dir = UserDataPart
    id_coord_mapping_path = data_dir + toWhoString(who) + '/id_coords_mapping.pkl'
    with open(id_coord_mapping_path, "rb") as f:
        coords_id = pickle.load(f)
    return coords_id

def load_id_tempo_mapping(who: int):
    """
    Load the temporal distribution mapping for a user.
    Args:
        who (int): User ID.
    Returns:
        dict: A dictionary mapping location IDs to their temporal distributions.
    """
    data_dir = UserDataPart
    id_tempo_mapping_path = data_dir + f'{toWhoString(who)}/id_tempo_mapping.pkl'
    with open(id_tempo_mapping_path, "rb") as f:
        id_tempo_mapping = pickle.load(f)
    return id_tempo_mapping

def load_fnid_coords_mapping(who: int):
    '''
    load coord-fnid mapper
    '''
    data_dir = UserDataPart
    id_coord_mapping_path = data_dir + toWhoString(who) + '/coords_fnid_mapping.pkl'
    with open(id_coord_mapping_path, "rb") as f:
        coords_fnid = pickle.load(f)
    return coords_fnid

def load_id_month_mapping(who: int):
    """
    获取每个id对应的月份
    """
    # 读取轨迹数据
    data_dir = UserDataPart
    json_path = data_dir + toWhoString(who) + f'/all_traj.json'
    with open(json_path, 'r') as f:
        traj_data = json.load(f)
    
    # 创建id到月份的映射
    id_month = {}
    for tc in traj_data:
        ids = tc['id_chain']
        date = str(tc['date'])
        month = date[:6]
        for idx in ids:
            if idx not in id_month:
                id_month[int(idx)] = month
    return id_month

def fniidMapper(who: int, id: int):    
    '''
    map the id to fnid
    '''
    # Load the mappings
    id_coords_mapping = load_id_coords_mapping(who)
    fnid_coords_mapping = load_fnid_coords_mapping(who)
    
    # Convert id to fnid
    coords = id_coords_mapping.get(id)
    if coords is None:
        raise ValueError(f"ID {id} not found in id_coords_mapping.")
    return fnid_coords_mapping.get(coords, None)

def load_all_traj(who: int):
    '''
    Seemed the same as loadTravelChainAll
    '''
    data_dir = UserDataPart
    all_traj_path = data_dir + toWhoString(who) + '/all_traj.json'
    with open(all_traj_path, 'r') as file:
        loaded_dicts_all = json.load(file)
    loaded_namedtuples_all = [TravelData(**d) for d in loaded_dicts_all]
    return loaded_namedtuples_all

def load_state_attrs(who: int):
    '''
    load the matrix of processed feature matrix.
    '''
    data_dir = UserDataPart
    filename = 'all_traj_feature.csv'
    
    traj_path = data_dir + toWhoString(who) + '/' + filename
    state_attribute, _ = preprocessStateAttributes(traj_path)
    return state_attribute

def visited_date(who: int):
    '''
    load all visited date of a traveler
    '''
    traveler = load_traveler(who)
    return traveler.visit_date

def extract_week_ends(date_seq: List[int]):
    '''
    given a list of dates, return the week end dates and week code.
    '''
    date_seq = sorted(set(date_seq))
    assert all(date_seq[i] < date_seq[i + 1] for i in range(len(date_seq) - 1)), \
        "The date sequence must be strictly increasing and unique."
        
    def fromisoformat(eight_digits_str):
        h = eight_digits_str
        date_str = f"{h[:4]}-{h[4:6]}-{h[6:]}"
        return date.fromisoformat(date_str)
        
    startdate = fromisoformat(str(date_seq[0]))
    date_objects = [fromisoformat(str(d)) for d in date_seq]
    # Date origin: the Monday of the week containing the starting date
    origin_date = startdate - timedelta(startdate.weekday())
    # 按周分组日期，week_seq的键为周号，值为对应周的日期列表
    week_seq = dict()
    for dt in date_objects:
        week_num = (dt - origin_date).days // 7
        if week_num not in week_seq:
            week_seq[week_num] = []
        week_seq[week_num].append(dt)
    
    # 提取每周的最后一日
    last_days_of_weeks = [max(days_in_week) for days_in_week in week_seq.values()]
    week_end_dates = [int(d.strftime('%Y%m%d')) for d in last_days_of_weeks]
    week_code = [(dt - origin_date).days // 7 for dt in date_objects]
    # 将最后一日转换为整数格式返回
    return week_end_dates, week_code

# turn integer date code to date
def intDate2Date(intDate):
    strDate = str(intDate)
    return datetime.strptime(strDate, '%Y%m%d').date()

# turn date to integer coordinates on the timeline
def intDate2TimeX(intDate):
    intDate = intDate2Date(intDate) - date(2020, 1, 1)
    return intDate.days

# copied from DeepMaxEntIRL
def normalize(vals):
    """
    normalize to (0, max_val)
    input:
      vals: 1d array
    """
    min_val = np.min(vals)
    max_val = np.max(vals)
    return (vals - min_val) / (max_val - min_val)

if __name__ == "__main__":
    path = f'./data/before_migrt.json'
    full_traj_path = f'./data/all_traj.json'

    inputs, targets_action, pe_code, a_dim, s_dim = loadTrajChain(path,full_traj_path)
    print(inputs.shape,targets_action.shape, pe_code.shape)