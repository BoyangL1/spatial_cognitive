import jax.numpy as np
import numpy as onp
import pandas as pd
import json
import pickle

from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler

TravelData = namedtuple('TravelChain', ['date', 'travel_chain','id_chain','fnid_chain'])

def loadJsonFile(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def loadTravelDataFromDicts(data_dicts):
    return [TravelData(**d) for d in data_dicts]

def getStateRow(state_attribute, state):
    row = state_attribute[state_attribute['fnid'] == state]
    return np.array(row.values[0][1:])

def getActionDim(all_chains):
    """ 
    Calculate the dimension of the action space for a travel chain. 
    This includes the count of distinct actions plus an additional dimension for the 'no action' (-1) case.
    """
    return max({id for tc in all_chains for id in tc.id_chain}) + 2 # id_chain is a sequence, so the length = max +1 +1

def preprocessStateAttributes(traj_file, all_feature_path='./data/all_traj_feature.csv'):
    """
    Preprocess state attributes from a trajectory file.

    This function reads state attributes from a CSV file, adjusts the columns based on the trajectory file provided,
    and scales the features using MinMaxScaler.

    Args:
        traj_file (str): Path to the trajectory file.
        all_feature_path (str, optional): Path to the CSV file containing all trajectory features. 

    Returns:
        tuple: A tuple containing the preprocessed DataFrame and the dimension of state attributes (excluding 'fnid').
    """
    state_attribute = pd.read_csv(all_feature_path)

    # Adjust columns based on the trajectory file
    if traj_file == './data/before_migrt.json':
        state_attribute.drop(columns=['post_home_distance'], inplace=True)
        state_attribute.rename(columns={'pre_home_distance': 'home_distance'}, inplace=True)
    else:
        state_attribute.drop(columns=['pre_home_distance'], inplace=True)
        state_attribute.rename(columns={'post_home_distance': 'home_distance'}, inplace=True)

    # Calculate the dimension of state attributes (excluding 'fnid')
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

def processTrajectoryData(traj_chains, state_attribute, s_dim, place_grid_data):
    """
    Process trajectory data to generate sequences of states, actions, and grids.This function iterates through trajectory chains and processes each trajectory to generate sequences of
    state-next_state, action-next_action, and grid-next_grid pairs.

    Args:
        traj_chains (list): List of trajectory chain objects.
        state_attribute (DataFrame): DataFrame containing state attributes.
        s_dim (int): The dimension of state attributes.
        place_grid_data (Dict): Dict containing place grid data.

    Returns:
        tuple: A tuple containing arrays of state-next_state pairs, action-next_action pairs, and grid-next_grid pairs.
    """
    state_next_state = []
    action_next_action = []
    grid_next_grid = []

    for tc in traj_chains:
        sns_chain,ana_chain,gng_chain = [],[],[]
        for t in range(len(tc.travel_chain)):
            # Get the final destination in the travel chain
            destination = tc.travel_chain[-1]
            s_n_s, a_n_a, g_n_g = processSingleTrajectory(tc, t, state_attribute, s_dim, destination, place_grid_data)
            # Append the results to respective lists
            sns_chain.append(s_n_s)
            ana_chain.append(a_n_a)
            gng_chain.append(g_n_g)
        state_next_state.append(sns_chain)
        action_next_action.append(ana_chain)
        grid_next_grid.append(gng_chain)
    # pad sequence to the same length
    state_next_state = padSequences(state_next_state,s_n_s.shape)
    action_next_action = padSequences(action_next_action,a_n_a.shape,padding_value=-1)
    grid_next_grid = padSequences(grid_next_grid,g_n_g.shape)

    return np.array(state_next_state), np.array(action_next_action), np.array(grid_next_grid)


def processSingleTrajectory(tc, t, state_attribute, s_dim, destination, place_grid_data):
    if t < len(tc.travel_chain)-1:
        this_state, next_state = tc.travel_chain[t], tc.travel_chain[t + 1]
        this_fnid, next_fnid = tc.fnid_chain[t], tc.fnid_chain[t+1]
        # state attribute
        s_n_s = onp.zeros((2, s_dim))
        s_n_s[0, :] = getStateRow(state_attribute, this_fnid)
        s_n_s[1, :] = getStateRow(state_attribute, next_fnid)

        # get grid code of state and destination,dim(8,128,128)
        this_grid = place_grid_data[tuple(this_state)]
        next_grid = place_grid_data[tuple(next_state)]
        # save to s_grid_s
        grid_shape = this_grid.shape
        s_gird_s = onp.zeros((2,*grid_shape))
        s_gird_s[0, :] = this_grid
        s_gird_s[1, :] = next_grid

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
        this_grid = place_grid_data[tuple(this_state)]
        destination_grid = place_grid_data[tuple(destination)]
        # save to s_grid_s
        grid_shape = this_grid.shape
        s_gird_s = onp.zeros((2,*grid_shape))
        s_gird_s[0, :] = this_grid
        s_gird_s[1, :] = np.zeros_like(this_grid)

        a_n_a = onp.zeros((2, 1))
        a_n_a[0] = -1
        a_n_a[1] = -1

    return s_n_s, a_n_a,s_gird_s

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
    file_path = './data/coords_grid_data.pkl'
    with open(file_path, 'rb') as file:
        place_grid_data = pickle.load(file)

    inputs, targets_action, grid_code, a_dim, s_dim = loadTrajChain(path,full_traj_path,place_grid_data)
    print(inputs.shape,targets_action.shape,grid_code.shape)