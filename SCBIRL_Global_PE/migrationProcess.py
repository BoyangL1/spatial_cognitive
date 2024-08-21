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

def computeRewardOrValue(model, input_path, output_path, attribute_type='value'):
    """
    Compute rewards or state values for each state using a given model and save to a CSV file.

    Parameters:
    - model: The trained model. Should have a method `rewardValue` for computing rewards and `QValue` for computing Q values.
    - input_path (str): Path to the input CSV file containing states.
    - output_path (str): Path to save the output CSV file with computed attributes (rewards or state values).
    - attribute_type (str): Either 'value' to compute state values or 'reward' to compute rewards.

    Returns:
    None. Writes results to the specified output CSV file.
    """
    
    # Using preprocessing function from utils
    state_attribute, _ = preprocessStateAttributes('./data/before_migrt.json', input_path)

    computeFunc = getComputeFunction(model, attribute_type)

    with open("./data/coords_fnid_mapping.pkl", "rb") as f:
        coords_fnid = pickle.load(f)

    rewardValues = []
    for coords, fnid in tqdm(coords_fnid.items(), total=len(coords_fnid),desc="compute "+ attribute_type):
        if fnid not in state_attribute.fnid.values:
            continue
        # get state attribute of this fnid
        state = getStateRow(state_attribute, fnid)
        pe_code = globalPE(coords,len(state)).flatten()
        # add three dimension
        pe_code = np.expand_dims(np.expand_dims(np.expand_dims(pe_code, axis=0), axis=0), axis = 0)
        state = np.expand_dims(np.expand_dims(np.expand_dims(state, axis=0), axis=0), axis = 0)

        # get reward value
        r = float(computeFunc(state,pe_code))
        rewardValues.append((coords, r))

    reward_df = pd.DataFrame(rewardValues, columns=['coords', 'reward'])
    if output_path is not None:
        reward_df.to_csv(output_path, index=False)
    return

def computeTransitionProb(model, input_path, output_path):
    """
    Compute transition probabilities for each state using a given model and save to a CSV file.

    Parameters:
    - model: The trained model. Should have a method `transitionProb` for computing transition probabilities.
    - input_path (str): Path to the input CSV file containing states.
    - output_path (str): Path to save the output CSV file with computed transition probabilities.

    Returns:
    None. Writes results to the specified output CSV file.
    """
    state_attribute, _ = preprocessStateAttributes('./data/before_migrt.json',input_path)
    computeFunc = getComputeFunction(model, 'transition_prob')
    size = model.a_dim

    with open("./data/id_coords_mapping.pkl", "rb") as f:
        coords_id = pickle.load(f)
    with open("./data/coords_fnid_mapping.pkl", "rb") as f:
        coords_fnid = pickle.load(f)

    # ref sort the dictionary by value
    coords_id = dict(sorted(coords_id.items()))
    transitionProbs = []
    coordsIdx = []
    for id, coords in tqdm(coords_id.items(), total=len(coords_id),desc="compute transition probability"):
        fnid = coords_fnid[coords]
        if fnid not in state_attribute.fnid.values:
            transProbVec = np.full(size, np.nan)
        else:
        # get state attribute of this fnid
            state = getStateRow(state_attribute, fnid)
            pe_code = globalPE(coords,len(state)).flatten()
            # add three dimension
            pe_code = np.expand_dims(np.expand_dims(np.expand_dims(pe_code, axis=0), axis=0), axis = 0)
            state = np.expand_dims(np.expand_dims(np.expand_dims(state, axis=0), axis=0), axis = 0)
            # get transition probability
            transProbVec = computeFunc(state,pe_code)
        transitionProbs.append(transProbVec)
        coordsIdx.append(coords)
    
    res = (np.array(transitionProbs), coordsIdx)
    with open(output_path, 'wb') as f:
        pickle.dump(res, f)
    return     

def getComputeFunction(model, attribute_type):
    """Return the appropriate function to compute either 'value' or 'reward'."""
    if attribute_type == 'value':
        return lambda state,grid_code: np.max(model.QValue(state,grid_code))
    elif attribute_type == 'transition_prob':
        return lambda state,grid_code: softmax(model.QValue(state,grid_code)[0][0])
    elif attribute_type == 'reward':
        return lambda state,grid_code: model.reward(state,grid_code)[0][0][0]
    else:
        raise ValueError("attribute_type should be either 'value' or 'reward'.")


def readAndPrepareData(user_data_path, start_date):
    """
    Reads and prepares data for spatial cognition analysis.

    Args:
        user_data_path (str): The path to the user data.
        start_date (str): The start date for filtering travel chains.

    Returns:
        tuple: A tuple containing the following:
            - visitedState (set): A set of visited states before migration.
            - trajChains (list): A list of travel chains after migration.
            - stateAttribute (dict): A dictionary of preprocessed state attributes.
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


def processBeforeMigrationData(state_attribute, visitedState, computeFunc, id_coords, coords_fnid, actionDim):
    """
    Process and calculate transition probabilities for states before migration. 
    It updates the state attributes with computed action probabilities.

    Args:
        stateAttribute (DataFrame): A DataFrame containing the state attributes.
        visitedState (set): A set of states that have been visited.
        computeFunc (function): A function to compute the action probabilities given a state and grid code.
        idFnid (dict): A dictionary mapping from id to their fnid.
        actionDim (int): The dimension of the action space.

    Returns:
        None: The function writes the results to a CSV file.
    """
    # Initialize a list to store transformed data.
    beforeMigrtTrans = []

    # Iterate over each row in the state attributes DataFrame.
    for coords, fnid in tqdm(coords_fnid.items(), desc="ProcessBeforeMigration"):
        # Extract fnid and state information from the row.
        state = getStateRow(state_attribute, fnid)
        if coords not in visitedState:
            actionProb = [0 for _ in range(actionDim)]
            continue
        pe_code = globalPE(coords,len(state)).flatten()
        # add twree dimension
        pe_code = np.expand_dims(np.expand_dims(np.expand_dims(pe_code, axis=0), axis=0), axis=0)
        state = np.expand_dims(np.expand_dims(np.expand_dims(state, axis=0), axis=0),axis=0)

        # Compute action probabilities based on whether the state has been visited.
        actionProb = []
        for i, prob in enumerate(computeFunc(state, pe_code)[0][0]):
            if i == len(id_coords):# the last one is the "no action"
                actionProb.append(prob)
                break
            if id_coords[i] not in visitedState:
                actionProb.append(0)
            else:
                actionProb.append(prob)

        # Prepare the row for the results DataFrame.
        resultRow = [coords] + list(actionProb)
        beforeMigrtTrans.append(resultRow)

    # Define columns for the results DataFrame, add the "no action" column.
    columns = ['coords'] + [id_coords[i] for i in range(actionDim-1)] +['no action']

    # Create and save the results DataFrame.
    dfResults = pd.DataFrame(beforeMigrtTrans, columns=columns)
    dfResults.to_csv(f"./data_pe/before_migrt_transProb.csv", index=False)


def processAfterMigrationData(tc, stateAttribute, model, visitedState, id_coords, coords_fnid, actionDim, outputPath="./data_pe/"):
    """
    Process data after migration, including calculating rewards and transition probabilities.

    Args:
        tc (TrajectoryChain): The trajectory chain object containing migration data.
        stateAttribute (DataFrame): DataFrame containing state attributes.
        coords_grid_data (dict): Dictionary mapping coords to their grid data.
        model (Model): The model used for calculations.
        visitedState (set): A set of states that have been visited.
        id_coords (dict): Dictionary mapping action indices to coordinate.
        actionDim (int): The dimension of the action space.

    Returns:
        list: List of reward values for each state.
    """
    # 模型在此输入每一天的数据
    plugInDataPair(tc, stateAttribute, model, visitedState)

    # Define functions for reward calculation and Q-value computation
    rewardFunction = getComputeFunction(model, 'reward')
    computeFunc = lambda state, grid_code: model.QValue(state, grid_code)

    # Initialize lists for storing results
    rewardValues = []
    results = []

    # Iterate over each row in state attributes to compute rewards and action probabilities
    for coords, fnid in tqdm(coords_fnid.items(), desc="ProcessAfterMigration"):
        if coords not in visitedState:
            rewardValues.append(0)
            actionProb = [0 for _ in range(actionDim)]
            resultRow = [coords] + list(actionProb)
            results.append(resultRow)
            continue
        # Extract fnid and state information from the row.
        state = getStateRow(stateAttribute, fnid)
        pe_code = globalPE(coords,len(state)).flatten()
        # add twree dimension
        pe_code = np.expand_dims(np.expand_dims(np.expand_dims(pe_code, axis=0), axis=0), axis=0)
        state = np.expand_dims(np.expand_dims(np.expand_dims(state, axis=0), axis=0),axis=0)

        # Calculate reward for visited coords
        r = float(rewardFunction(state, pe_code)) if coords in visitedState else 0
        rewardValues.append(r)

        actionProb = []
        for i, prob in enumerate(computeFunc(state, pe_code)[0][0]):
            if i == len(id_coords):# the last one is the "no action"
                actionProb.append(prob)
                break
            if id_coords[i] not in visitedState:
                actionProb.append(0)
            else:
                actionProb.append(prob)
        # Append results for each state
        resultRow = [coords] + list(actionProb)
        results.append(resultRow)

    # Create and save results DataFrame
    columns = ['coords'] + [id_coords[i] for i in range(actionDim-1)] + ['no action']
    dfResults = pd.DataFrame(results, columns=columns)

    output_dir = outputPath + f"after_migrt/transProb/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dfResults.to_csv(outputPath + f"after_migrt/transProb/{tc[-1].date}.csv", index=False)

    return rewardValues



def afterMigrt(model, dataPath, outputPath, start_date, iter_type):
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
        # weights = [1 / 2 ** (memory_buffer - i) for i in range(memory_buffer)]
        weights = None
        model.train(iters=1000,loss_threshold=0.01, weights=weights)

        # Save the current model state.
        modelSavePath = modelDir + model_tag + '_model_' + str(iter_training_set[-1].date) + ".pickle"
        model.modelSave(modelSavePath)
