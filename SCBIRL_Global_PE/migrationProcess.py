import jax.numpy as np

import numpy as onp
import pickle
import os
import pandas as pd
from tqdm import tqdm

from .transformer import *
from .utils import *
from .EnDecoder import *
from scipy.special import softmax

def getComputeFunction(model, attribute_type):
    """
    Return the appropriate function for model computation.
    attribute_type should be either 'value', 'reward', or 'transition_prob'.
    
    The fed grid code should be complex array.
    """
    if attribute_type == 'value':
        return lambda state,grid_code: np.max(model.QValue(state, grid_code))
    elif attribute_type == 'transition_prob':
        return lambda state,grid_code: softmax(model.QValue(state, grid_code)[0][0])
    elif attribute_type == 'reward':
        return lambda state,grid_code: model.reward(state, grid_code)[0][0][0]
    else:
        raise ValueError("attribute_type should be either 'value', 'reward', or 'transition_prob'.")


def readAndPrepareData(user_data_path, start_date):
    """
    Reads and prepares data. 
    Return visited coordinates, divide the data into two parts, and preprocess state attributes.

    Args:
        user_data_path (str): The path to the user data.
        start_date (str): The start date for filtering travel chains.

    Returns:
        - visitedState (set): A set of visited coordinates before start date
        - trajInitChains/trajIterChains (list): A list of travel chains before/after migration.
        - stateAttribute (pd.DataFrame): Dataset of preprocessed state attributes.
    """
    all_traj_path = user_data_path + 'all_traj.json'
    all_traj_feature_path = user_data_path + 'all_traj_feature.csv'
    
    # Load and process the data of travel chains before migration.
    chains = loadJsonFile(all_traj_path)
    beforeChains = [chain for chain in chains if chain['date'] < start_date]
    visitedState = {tuple(state) for chain in beforeChains for state in chain['travel_chain']}
    trajInitChains = loadTravelDataFromDicts(beforeChains)

    # Load and process the data of travel chains after migration.
    afterChains = [chain for chain in chains if chain['date'] >= start_date]
    trajIterChains = loadTravelDataFromDicts(afterChains)

    # Preprocess state attributes based on the after migration data.
    stateAttribute, _ = preprocessStateAttributes(all_traj_feature_path)

    return visitedState, trajInitChains, trajIterChains, stateAttribute

def afterMigrt(model, dataPath, outputPath, start_date, iter_type):
    '''
    Iteratively train the model with real traj data.
    '''
    assert iter_type in ['recent', 'prior'], "Argument `iter_type` should be either 'recent' or 'prior'."
    if iter_type == 'recent':
        model_tag = 'iterated'
        folder_name = "evolution_model/"
    else:
        model_tag = 'increased'
        folder_name = 'empirical_model/'
    
    full_traj_path = dataPath + "all_traj.json"

    # Load the mapping between IDs and their corresponding fnid.
    with open(dataPath + "id_coords_mapping.pkl", "rb") as f:
        id_coords = pickle.load(f)
    with open(dataPath + "coords_fnid_mapping.pkl", "rb") as f:
        coords_fnid = pickle.load(f)

    all_chains = loadTravelDataFromDicts(loadJsonFile(full_traj_path))
    actionDim = getActionDim(all_chains)

    # Read and preprocess data for analysis.
    visitedState, trajInitChains, trajIterChains, stateAttribute = readAndPrepareData(dataPath, start_date)

    # Initialize an empty DataFrame with predefined columns
    resultsDf = pd.DataFrame(columns=['coords', 'fnid'])
    # Iterate over the coords_fnid dictionary and append each key-value pair to resultsDf
    for key, value in coords_fnid.items():
        # Append the key-value pair as a new row to resultsDf
        resultsDf = resultsDf._append({'coords': key, 'fnid': value}, ignore_index=True)


    modelDir = outputPath + folder_name
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    memory_buffer = 10 # days

    for i in range(len(trajIterChains)):

        if i < memory_buffer:
            iter_training_set = trajInitChains[-(memory_buffer-i):] + trajIterChains[:i]
        else:
            iter_training_set = trajIterChains[i-memory_buffer:i]
        iter_training_set = iter_training_set + [trajIterChains[i]]

        # Process and calculate reward values after migration.
        plugInDataPair(iter_training_set, stateAttribute, model, visitedState)

        # Train the model.
        # change
        # weights = [1 / 2 ** (memory_buffer - i) for i in range(memory_buffer)]
        weights = None
        model.train(iters=1000,loss_threshold=0.01, weights=weights)

        # Save the current model state.
        modelSavePath = modelDir + model_tag + '_model_' + str(iter_training_set[-1].date) + ".pickle"
        model.modelSave(modelSavePath)