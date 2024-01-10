import jax.numpy as np

import numpy as onp
import pickle
import os
import pandas as pd
from tqdm import tqdm

from transformer import *
from BasicCNN import *
from utils import *
from EnDecoder import *

def computeRewardOrValue(model, input_path, output_path, coords_grid_data, attribute_type='value'):
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
    state_attribute, _ = preprocessStateAttributes('./data/before_migrt.json',input_path)

    computeFunc = getComputeFunction(model, attribute_type)

    with open("./data/coords_fnid_mapping.pkl", "rb") as f:
        coords_fnid = pickle.load(f)
    
    rewardValues = []
    for coords, fnid in tqdm(coords_fnid.items(), total=len(coords_fnid)):
        if fnid not in state_attribute.fnid.values:
            continue

        this_coords_grid = coords_grid_data[coords]
        destination_grid = np.zeros_like(this_coords_grid) # if has specific destination, change this line to real destination grid code
        grid_code = onp.concatenate((this_coords_grid, destination_grid), axis=0)
        # get state attribute of this fnid
        state = getStateRow(state_attribute,fnid)
        # add three dimension
        grid_code = np.expand_dims(np.expand_dims(np.expand_dims(grid_code, axis=0), axis=0), axis = 0)
        state = np.expand_dims(np.expand_dims(np.expand_dims(state, axis=0), axis=0), axis = 0)

        # get reward value
        r = float(computeFunc(state,grid_code))
        rewardValues.append((coords, r))

    reward_df = pd.DataFrame(rewardValues, columns=['coords', 'reward'])
    if output_path is not None:
        reward_df.to_csv(output_path, index=False)
    return

def getComputeFunction(model, attribute_type):
    """Return the appropriate function to compute either 'value' or 'reward'."""
    if attribute_type == 'value':
        return lambda state,grid_code: np.max(model.QValue(state,grid_code))
    elif attribute_type == 'reward':
        return lambda state,grid_code: model.reward(state,grid_code)[0][0][0] # (1,1,2)
    else:
        raise ValueError("attribute_type should be either 'value' or 'reward'.")

def readAndPrepareData(afterMigrtFile, beforeMigrtFile, inputPath):
    """
    Read and prepare data for processing. This involves loading and preprocessing
    data related to migration and travel chains.

    Args:
        afterMigrtFile (str): The file path for data after migration.
        beforeMigrtFile (str): The file path for data before migration.
        inputPath (str): The directory path where input feature data is stored.

    Returns:
        set: A set of visited states extracted from before migration data.
        list: A list of trajectory chains loaded from the after migration data.
        DataFrame: A DataFrame containing state attributes.
    """
    # Load and process the data of travel chains before migration.
    beforeChains = loadJsonFile(beforeMigrtFile)
    visitedState = {tuple(state) for chain in beforeChains for state in chain['travel_chain']}

    # Load and process the data of travel chains after migration.
    loadedDictsList1 = loadJsonFile(afterMigrtFile)
    trajChains = loadTravelDataFromDicts(loadedDictsList1)

    # Preprocess state attributes based on the after migration data.
    stateAttribute, _ = preprocessStateAttributes(afterMigrtFile, inputPath)

    return visitedState, trajChains, stateAttribute

def processBeforeMigrationData(state_attribute, visitedState, coords_grid_data, computeFunc, id_coords, coords_fnid, actionDim):
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
        this_coords_grid = coords_grid_data[coords]
        destination_grid = this_coords_grid # if has specific destination, change this line to real destination grid code
        grid_code = onp.concatenate((this_coords_grid, destination_grid), axis=0)
        # add twree dimension
        grid_code = np.expand_dims(np.expand_dims(np.expand_dims(grid_code, axis=0), axis=0), axis=0)
        state = np.expand_dims(np.expand_dims(np.expand_dims(state, axis=0), axis=0),axis=0)

        # Compute action probabilities based on whether the state has been visited.
        if coords in visitedState:
            actionProb = []
            for i, prob in enumerate(computeFunc(state, grid_code)[0][0]):
                if i == len(id_coords):# the last one is the "no action"
                    actionProb.append(prob)
                    break
                if id_coords[i] not in visitedState:
                    actionProb.append(0)
                else:
                    actionProb.append(prob)
        else:
            actionProb = [0 for _ in range(actionDim)]

        # Prepare the row for the results DataFrame.
        resultRow = [coords] + list(actionProb)
        beforeMigrtTrans.append(resultRow)

    # Define columns for the results DataFrame, add the "no action" column.
    columns = ['coords'] + [id_coords[i] for i in range(actionDim-1)] +['no action']

    # Create and save the results DataFrame.
    dfResults = pd.DataFrame(beforeMigrtTrans, columns=columns)
    dfResults.to_csv(f"./data/before_migrt_transProb.csv", index=False)


def processAfterMigrationData(tc, stateAttribute, coords_grid_data, model, visitedState, id_coords, coords_fnid, actionDim):
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
    # Preprocess trajectory data and update visited states
    stateNextState, actionNextAction, grid_next_grid = processTrajectoryData(tc, stateAttribute, model.s_dim, coords_grid_data)
    for t in tc:
        visitedState.update(tuple(item) if isinstance(item, list) else item for item in t.travel_chain)

    # Set model inputs for training or evaluation
    model.inputs = stateNextState
    model.targets = actionNextAction
    model.grid_code = grid_next_grid

    # Define functions for reward calculation and Q-value computation
    rewardFunction = getComputeFunction(model, 'reward')
    computeFunc = lambda state, grid_code: model.QValue(state, grid_code)

    # Initialize lists for storing results
    rewardValues = []
    results = []

    # Iterate over each row in state attributes to compute rewards and action probabilities
    for coords, fnid in tqdm(coords_fnid.items(), desc="ProcessAfterMigration"):
        # Extract fnid and state information from the row.
        state = getStateRow(stateAttribute, fnid)
        this_coords_grid = coords_grid_data[coords]
        destination_grid = this_coords_grid # if has specific destination, change this line to real destination grid code
        grid_code = onp.concatenate((this_coords_grid, destination_grid), axis=0)
        # add twree dimension
        grid_code = np.expand_dims(np.expand_dims(np.expand_dims(grid_code, axis=0), axis=0), axis=0)
        state = np.expand_dims(np.expand_dims(np.expand_dims(state, axis=0), axis=0),axis=0)

        # Calculate reward for visited coords
        r = float(rewardFunction(state, grid_code)) if coords in visitedState else 0
        rewardValues.append(r)

        if coords in visitedState:
            actionProb = []
            for i, prob in enumerate(computeFunc(state, grid_code)[0][0]):
                if i == len(id_coords):# the last one is the "no action"
                    actionProb.append(prob)
                    break
                if id_coords[i] not in visitedState:
                    actionProb.append(0)
                else:
                    actionProb.append(prob)
        else:
            actionProb = [0 for _ in range(actionDim)]

        # Append results for each state
        resultRow = [coords] + list(actionProb)
        results.append(resultRow)

    # Create and save results DataFrame
    columns = ['coords'] + [id_coords[i] for i in range(actionDim-1)] + ['no action']
    dfResults = pd.DataFrame(results, columns=columns)

    output_dir = f"./data/after_migrt/transProb/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dfResults.to_csv(f"./data/after_migrt/transProb/{tc[-1].date}.csv", index=False)

    return rewardValues

def afterMigrt(afterMigrtFile, beforeMigrtFile, full_trajectory_path, coords_grid_data, inputPath, outputPath, model):
    # Load model parameters from a saved state.
    model.loadParams('./model/params_transformer.pickle')

    # Load the mapping between IDs and their corresponding fnid.
    with open("./data/id_coords_mapping.pkl", "rb") as f:
        id_coords = pickle.load(f)
    with open("./data/coords_fnid_mapping.pkl", "rb") as f:
        coords_fnid = pickle.load(f)

    # Define a function to compute Q-values given a state and corresponding grid_code.
    computeFunc = lambda state, grid_code: model.QValue(state,grid_code)

    all_chains = loadTravelDataFromDicts(loadJsonFile(full_trajectory_path))
    before_migrt_chains = loadTravelDataFromDicts(loadJsonFile(beforeMigrtFile))
    actionDim = getActionDim(all_chains)

    # Read and preprocess data for analysis.
    visitedState, trajChains, stateAttribute = readAndPrepareData(afterMigrtFile, beforeMigrtFile, inputPath)

    # Process and update state attributes before migration.
    processBeforeMigrationData(stateAttribute, visitedState, coords_grid_data, computeFunc, id_coords, coords_fnid, actionDim)

    # Initialize an empty DataFrame with predefined columns
    resultsDf = pd.DataFrame(columns=['coords', 'fnid'])
    # Iterate over the coords_fnid dictionary and append each key-value pair to resultsDf
    for key, value in coords_fnid.items():
        # Append the key-value pair as a new row to resultsDf
        resultsDf = resultsDf._append({'coords': key, 'fnid': value}, ignore_index=True)

    modelDir = "./data/after_migrt/model"
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    preDate = 0 # load preDate model parameters
    memory_buffer = 10 # days

    for i in range(len(trajChains)):
        if preDate:
            modelPath = os.path.join(modelDir, f"{preDate}.pickle")
            model.loadParams(modelPath)

        if i < memory_buffer:
            before_chain = before_migrt_chains[-(memory_buffer-i):]
        else:
            before_chain = trajChains[i-memory_buffer:i]
        train_chain = before_chain + [trajChains[i]]

        # Process and calculate reward values after migration.
        rewardValues = processAfterMigrationData(train_chain, stateAttribute, coords_grid_data, model, visitedState, id_coords, coords_fnid, actionDim)

        # Train the model.
        model.train(iters=1000)

        # Save the current model state.
        modelSavePath = "./data/after_migrt/model/" + str(train_chain[-1].date) + ".pickle"
        model.modelSave(modelSavePath)

        # Store the calculated reward values in the results DataFrame.
        resultsDf[str(train_chain[-1].date)] = rewardValues

        preDate = train_chain[-1].date

    # Save the results to a CSV file.
    resultsDf.to_csv(outputPath, index=False)