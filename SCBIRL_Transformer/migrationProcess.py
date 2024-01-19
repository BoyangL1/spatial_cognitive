import jax.numpy as np

import numpy as onp
import pickle
import os
import pandas as pd
from tqdm import tqdm

from .transformer import *
from .BasicCNN import *
from .utils import *
from .EnDecoder import *

from CANN import cann_setup_network as SN
from CANN import real_trajeccotry as TRAJ
from CANN import cann_update_network as UN

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
        if coords not in coords_grid_data.keys():
            rewardValues.append((coords,0))
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
        if coords not in visitedState:
            actionProb = [0 for _ in range(actionDim)]
            continue
        this_coords_grid = coords_grid_data[coords]
        destination_grid = np.zeros_like(this_coords_grid) # if has specific destination, change this line to real destination grid code
        grid_code = onp.concatenate((this_coords_grid, destination_grid), axis=0)
        # add twree dimension
        grid_code = np.expand_dims(np.expand_dims(np.expand_dims(grid_code, axis=0), axis=0), axis=0)
        state = np.expand_dims(np.expand_dims(np.expand_dims(state, axis=0), axis=0),axis=0)

        # Compute action probabilities based on whether the state has been visited.
        actionProb = []
        for i, prob in enumerate(computeFunc(state, grid_code)[0][0]):
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
    dfResults.to_csv(f"./data/before_migrt_transProb.csv", index=False)

def updateCoordsGrid(last_grid_code,coords_grid_data,this_date, GN, w_r, w_l, w_u, w_d, a, r_r, r_l, r_d, r_u, r_masks):

    r = last_grid_code
    for k in range(0, GN.h, 1):
        r_l[k, :, :] = r[k, :, :]*r_masks[0, :, :]
        r_r[k, :, :] = r[k, :, :]*r_masks[1, :, :]
        r_u[k, :, :] = r[k, :, :]*r_masks[2, :, :]
        r_d[k, :, :] = r[k, :, :]*r_masks[3, :, :]
    
    file_name = './data/one_travel_chain.csv'
    df = pd.read_csv(file_name)
    df['is_matching_date'] = df.date == this_date # Mark rows that match the specific date
    df['previous_is_matching'] = df['is_matching_date'].shift(-1) 
    df = df[df['is_matching_date'] | df['previous_is_matching']] # Select rows that match the specific date and their preceding rows
    df.drop(columns=['is_matching_date', 'previous_is_matching'], inplace=True) # Drop the marker columns

    # Get Trajectory Data
    [origin_grid,dest_grid,origin_x,origin_y,dest_x,dest_y,x,y,vx,vy] = TRAJ.get_trajectory(df)
    anchor_x = origin_x + dest_x
    anchor_y = origin_y + dest_y
    grid_list = origin_grid + dest_grid
    # de-duplicate
    unique_dict = {}
    for ax, ay, grid in zip(anchor_x, anchor_y, grid_list):
        unique_dict[(ax, ay)] = grid
    anchor_list = list(unique_dict.keys())
    grid_list = list(unique_dict.values())

    singleneuronrec = False
    t_index = len(vx)
    print(f'Update Grid Cell Model with {this_date} Trajectory!')
    [r, r_field, r_r, r_l, r_d, r_u, sna_eachlayer, coords_grid_dic, _ ] = UN.flow_full_model(
        GN, anchor_list,grid_list,x,y,vx, vy, t_index, a, r, r_r, r_l, r_d, r_u, r_masks,singleneuronrec, w_r, w_l, w_u, w_d)
    coords_grid_data.update(coords_grid_dic)
    return coords_grid_data

def initializeGridModel():
    print("Initialize grid model")
    h = 8  
    n = 128  
    dt = 1.0
    tau = 10.0 

    # Recurrent Inhibition Parameters
    wmag = 2.4  
    lmin = 7  
    lmax =  41 
    lexp = -1 
    wshift = 1

    # Coupling Parameters
    umag = -0.001  # for spike model, mean(w), connection stength from other layer
    urad = 4.0  # for rate model
    u_dv = 0  # 1 corresponds to dorsal to ventral
    u_vd = 0  # 1 corresponds to ventral to dorsal

    # Initial value for rates for grid units
    rinit = 1e-3
    r_field_base = 0.6

    # Hippocampal Parameters
    amag = .6  # now trying . 6 from Alex's paper, was 1;#trying .7 instead of .5 due to particularly low activity in network when checked #1 in rate model
    falloff = 4.0  # 4.0 in rate model
    falloff_low = 2.0
    falloff_high = -999  

    # system noise
    rnoise = 0.0
    vgain = .4

    GN = UN.GridNeuronNetwork(h, n, dt, tau, wmag, lmin, lmax, wshift, umag, urad, u_dv, u_vd, rinit, r_field_base, amag, falloff, falloff_low, falloff_high, n, rnoise, vgain)
    # inhibition length scales
    l = SN.set_inhib_length(h, lexp, lmin, lmax)
    # inhibitory kernel overall and for each directional subpopulation
    [w, w_r, w_l, w_u, w_d] = SN.setup_recurrent(h, n, l, wmag, wshift)
    a = SN.setup_input(amag, n, falloff_high, falloff_low, falloff)

    r_l, r_r, r_u, r_d = onp.zeros((4, h, n, n))
    r_masks = onp.zeros((4, n, n))
    r_masks[0, 0:n:2, 0:n:2] = 1
    r_masks[1, 1:n:2, 1:n:2] = 1
    r_masks[2, 0:n:2, 1:n:2] = 1
    r_masks[3, 1:n:2, 0:n:2] = 1

    return GN, w_r, w_l, w_u, w_d, a, r_r, r_l, r_d, r_u, r_masks

def processAfterMigrationData(tc, stateAttribute, coords_grid_data, model, visitedState, id_coords, coords_fnid, actionDim, GN, w_r, w_l, w_u, w_d, a, r, r_r, r_l, r_d, r_masks):
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
    last_coords = tc[-2].travel_chain[-1]
    last_grid_code = coords_grid_data[tuple(last_coords)]
    coords_grid_data = updateCoordsGrid(last_grid_code,coords_grid_data,tc[-1].date,GN, w_r, w_l, w_u, w_d, a, r, r_r, r_l, r_d, r_masks)

    # Preprocess trajectory data and update visited states
    stateNextState, actionNextAction, gridNextgrid = processTrajectoryData(tc, stateAttribute, model.s_dim, coords_grid_data)
    for t in tc:
        visitedState.update(tuple(item) if isinstance(item, list) else item for item in t.travel_chain)

    # Set model inputs for training or evaluation
    model.inputs = stateNextState
    model.targets = actionNextAction
    model.grid_code = gridNextgrid

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
        this_coords_grid = coords_grid_data[coords]
        destination_grid = np.zeros_like(this_coords_grid) # if has specific destination, change this line to real destination grid code
        grid_code = onp.concatenate((this_coords_grid, destination_grid), axis=0)
        # add twree dimension
        grid_code = np.expand_dims(np.expand_dims(np.expand_dims(grid_code, axis=0), axis=0), axis=0)
        state = np.expand_dims(np.expand_dims(np.expand_dims(state, axis=0), axis=0),axis=0)

        # Calculate reward for visited coords
        r = float(rewardFunction(state, grid_code)) if coords in visitedState else 0
        rewardValues.append(r)

        actionProb = []
        for i, prob in enumerate(computeFunc(state, grid_code)[0][0]):
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

    output_dir = f"./data/after_migrt/transProb/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dfResults.to_csv(f"./data/after_migrt/transProb/{tc[-1].date}.csv", index=False)

    return rewardValues,coords_grid_data

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

    GN, w_r, w_l, w_u, w_d, a, r, r_r, r_l, r_d, r_masks = initializeGridModel()
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
        rewardValues,coords_grid_data = processAfterMigrationData(train_chain, stateAttribute, coords_grid_data, model, visitedState, id_coords, coords_fnid, actionDim, GN, w_r, w_l, w_u, w_d, a, r, r_r, r_l, r_d, r_masks)

        # Train the model.
        model.train(iters=1000,loss_threshold=0.01)

        # Save the current model state.
        modelSavePath = "./data/after_migrt/model/" + str(train_chain[-1].date) + ".pickle"
        model.modelSave(modelSavePath)

        # Store the calculated reward values in the results DataFrame.
        resultsDf[str(train_chain[-1].date)] = rewardValues

        preDate = train_chain[-1].date

    # Save results.
    resultsDf.to_csv(outputPath, index=False)
    with open('./data/coords_grid_data_all.pkl', 'wb') as file:
        pickle.dump(coords_grid_data, file)