# import module from the parent directory
import sys
import os
import numpy as np
import copy 

working_directory = os.getcwd()
working_directory = os.path.abspath('.')
sys.path.append(working_directory)

from SCBIRL_Global_PE.migrationProcess import *

def experienceModel(afterMigrtFile, beforeMigrtFile, full_trajectory_path, inputPath, outputPath, model_no_prior):

    # Load the mapping between IDs and their corresponding fnid.
    with open("./data/id_coords_mapping.pkl", "rb") as f:
        id_coords = pickle.load(f)
    with open("./data/coords_fnid_mapping.pkl", "rb") as f:
        coords_fnid = pickle.load(f)

    all_chains = loadTravelDataFromDicts(loadJsonFile(full_trajectory_path))
    before_migrt_chains = loadTravelDataFromDicts(loadJsonFile(beforeMigrtFile))
    actionDim = getActionDim(all_chains)

    # Read and preprocess data for analysis.
    visitedState, trajChains, stateAttribute = readAndPrepareData(afterMigrtFile, beforeMigrtFile, inputPath)

    # Initialize an empty DataFrame with predefined columns
    resultsDf = pd.DataFrame(columns=['coords', 'fnid'])
    # Iterate over the coords_fnid dictionary and append each key-value pair to resultsDf
    for key, value in coords_fnid.items():
        # Append the key-value pair as a new row to resultsDf
        resultsDf = resultsDf._append({'coords': key, 'fnid': value}, ignore_index=True)

    modelDir = "./data_pe/after_migrt/no_prior_model"
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    memory_buffer = 10 # days

    for i in range(len(trajChains)):
        model = copy.deepcopy(model_no_prior)   
        if i < memory_buffer:
            before_chain = before_migrt_chains[-(memory_buffer-i):] + trajChains[:i]
        else:
            before_chain = trajChains[i-memory_buffer:i]
        train_chain = before_chain + [trajChains[i]]

        # Process and calculate reward values after migration.
        rewardValues = processAfterMigrationData(train_chain, stateAttribute, model, visitedState, id_coords, coords_fnid, actionDim)

        # Train the model.
        model.train(iters=1000,loss_threshold=0.01)

        # Save the current model state.
        modelSavePath = modelDir + "/" + str(train_chain[-1].date) + ".pickle"
        model.modelSave(modelSavePath)