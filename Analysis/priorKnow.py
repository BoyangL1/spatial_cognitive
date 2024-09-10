# import module from the parent directory
import sys
import os
import numpy as np
import copy 

working_directory = os.getcwd()
working_directory = os.path.abspath('.')
sys.path.append(working_directory)

from SCBIRL_Global_PE.migrationProcess import *
from SCBIRL_Global_PE.utils import plugInDataPair

def experienceModel(model_no_prior, dataPath, outputPath, start_date):
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

    modelDir = outputPath + "no_prior_model/"
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    memory_buffer = 10 # days

    for i in range(len(trajIterChains)):
        model = copy.deepcopy(model_no_prior)
        if i < memory_buffer:
            iter_training_set = trajInitChains[-(memory_buffer-i):] + trajIterChains[:i]
        else:
            iter_training_set = trajIterChains[i-memory_buffer:i]
        iter_training_set = iter_training_set + [trajIterChains[i]]

        # Process and calculate reward values after migration.
        # rewardValues = processAfterMigrationData(train_chain, stateAttribute, model, visitedState, id_coords, coords_fnid, actionDim, outputPath)
        plugInDataPair(iter_training_set, stateAttribute, model, visitedState)
        # Train the model. We get the model after the past of the date.
        # change
        # weights = [1 / 2 ** (memory_buffer + 1 - i) for i in range(memory_buffer + 1)]
        weights = None
        model.train(iters=1000, loss_threshold=0.01, weights=weights)

        # Save the current model state.
        modelSavePath = modelDir + 'ignorant_model_' + str(iter_training_set[-1].date) + ".pickle"
        model.modelSave(modelSavePath)