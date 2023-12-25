import jax.numpy as np
import numpy as onp
import pandas as pd
import json
import pickle

from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler

TravelData = namedtuple('TravelChain', ['date', 'travel_chain','id_chain'])

def loadJsonFile(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def loadTravelDataFromDicts(data_dicts):
    return [TravelData(**d) for d in data_dicts]

def getActionDim(all_chains):
    """ 
    Calculate the dimension of the action space for a travel chain. 
    This includes the count of distinct actions plus an additional dimension for the 'no action' (-1) case.
    """
    return max({id for tc in all_chains for id in tc.id_chain}) + 2 # id_chain is a sequence, so the length = max +1 +1

def preprocessStateAttributes(traj_file,all_feature_path = './data/all_traj_feature.csv'):
    state_attribute = pd.read_csv(all_feature_path)
    if traj_file == './data/before_migrt.json':
        state_attribute.drop(columns=['post_home_distance'], inplace=True)
        state_attribute.rename(columns={'pre_home_distance': 'home_distance'}, inplace=True)
    else:
        state_attribute.drop(columns=['pre_home_distance'], inplace=True)
        state_attribute.rename(columns={'post_home_distance': 'home_distance'}, inplace=True)

    s_dim = state_attribute.shape[1] - 1
    fnid_col = state_attribute[['fnid']]
    other_cols = state_attribute.drop(columns=['fnid'])

    scaler = MinMaxScaler()
    scaled_cols = scaler.fit_transform(other_cols)
    scaled_df = pd.DataFrame(scaled_cols, columns=other_cols.columns)
    return pd.concat([fnid_col, scaled_df], axis=1), s_dim

def processTrajectoryData(traj_chains, state_attribute, s_dim, place_grid_data):
    state_next_state = []
    action_next_action = []
    grid_next_grid = []

    for tc in traj_chains:
        for t in range(len(tc.travel_chain)):
            destination = tc.travel_chain[-1]
            s_n_s, a_n_a,s_grid_s = processSingleTrajectory(tc, t, state_attribute, s_dim, destination,place_grid_data)
            state_next_state.append(s_n_s)
            action_next_action.append(a_n_a)
            grid_next_grid.append(s_grid_s)

    return onp.array(state_next_state), onp.array(action_next_action), onp.array(grid_next_grid)

def processSingleTrajectory(tc, t, state_attribute, s_dim, destination, place_grid_data):
    if t < len(tc.travel_chain)-1:
        this_state, next_state = tc.travel_chain[t], tc.travel_chain[t + 1]
        # state attribute
        s_n_s = onp.zeros((2, s_dim))
        s_n_s[0, :] = getStateRow(state_attribute, this_state)
        s_n_s[1, :] = getStateRow(state_attribute, next_state)

        # get grid code of state and destination,dim(8,128,128)
        this_grid = place_grid_data[this_state]
        destination_grid = place_grid_data[destination]
        next_grid = place_grid_data[next_state]
        # combined destination grid code with current grid code
        this_destination_grid = onp.concatenate((this_grid, destination_grid), axis=0)
        next_destination_grid = onp.concatenate((next_grid, destination_grid), axis=0)
        # save to s_grid_s
        grid_shape = this_destination_grid.shape
        s_gird_s = onp.zeros((2,*grid_shape))
        s_gird_s[0, :] = this_destination_grid
        s_gird_s[1, :] = next_destination_grid

        # action
        a_n_a = onp.zeros((2, 1))   
        a_n_a[0] = tc.id_chain[t + 1]
        a_n_a[1] = tc.id_chain[t + 2] if t + 2 < len(tc.id_chain) else -1
    else:
        this_state, next_state = tc.travel_chain[t], None
        s_n_s = onp.zeros((2, s_dim))
        s_n_s[0, :] = getStateRow(state_attribute, this_state)

        # get grid code of state and destination,dim(8,128,128)
        this_grid = place_grid_data[this_state]
        destination_grid = place_grid_data[destination]
        # combined destination grid code with current grid code
        this_destination_grid = onp.concatenate((this_grid, destination_grid), axis=0)
        next_destination_grid = onp.zeros_like(this_destination_grid)
        # save to s_grid_s
        grid_shape = this_destination_grid.shape
        s_gird_s = onp.zeros((2,*grid_shape))
        s_gird_s[0, :] = this_destination_grid
        s_gird_s[1, :] = next_destination_grid

        a_n_a = onp.zeros((2, 1))
        a_n_a[0] = -1
        a_n_a[1] = -1

    return s_n_s, a_n_a,s_gird_s

def getStateRow(state_attribute, state):
    row = state_attribute[state_attribute['fnid'] == state]
    return np.array(row.values[0][1:])

def loadTrajChain(traj_file, full_traj_path, place_grid_data,num_trajs=None):
    loaded_dicts_list1 = loadJsonFile(traj_file)
    traj_chains = loadTravelDataFromDicts(loaded_dicts_list1)

    all_chains = loadTravelDataFromDicts(loadJsonFile(full_traj_path))
    a_dim = getActionDim(all_chains)

    if num_trajs is not None:
        traj_chains = traj_chains[:num_trajs]
    
    state_attribute, s_dim = preprocessStateAttributes(traj_file)
    state_next_state, action_next_action, grid_next_grid= processTrajectoryData(traj_chains, state_attribute, s_dim,place_grid_data)

    return state_next_state, action_next_action, grid_next_grid, a_dim, s_dim
    
if __name__ == "__main__":
    path = f'./data/before_migrt.json'
    full_traj_path = f'./data/all_traj.json'
    file_path = './data/place_grid_data.pkl'
    with open(file_path, 'rb') as file:
        place_grid_data = pickle.load(file)

    inputs, targets_action, grid_code, a_dim, s_dim =loadTrajChain(path,full_traj_path,place_grid_data)
    print(inputs.shape,targets_action.shape,grid_code.shape)
