import haiku as hk

from jax import grad, jit, value_and_grad
from jax import random
from jax.example_libraries import optimizers
import jax
import jax.numpy as np
from jax import random

import numpy as onp
import pickle
import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from transformer import *
from BasicCNN import *
from utils import *

def minmax_scale(x):
    min_val = np.min(x)
    max_val = np.max(x)
    return (x - min_val) / (max_val - min_val)

def encoder_model(inputs, grid_code, num_layers, num_heads, dff, rate, output_dim, rng, cnn_output_size = 256):
    # Process grid_code using CNN
    cnn_processed_grid_code = process_grid_code(grid_code, cnn_output_size)
    # Normalization
    cnn_reshaped = cnn_processed_grid_code.reshape(-1, cnn_processed_grid_code.shape[-1])
    cnn_scaled = minmax_scale(cnn_reshaped)
    cnn_processed_grid_code = cnn_scaled.reshape(cnn_processed_grid_code.shape)
    # Combine inputs and flattened grid_code
    inputs = np.concatenate([inputs, cnn_processed_grid_code], axis=-1)

    # Initialize transformer layer
    transformer_layers = [TransformerLayer(inputs.shape[-1], num_heads, dff, rate) for _ in range(num_layers)]
    # forward 
    for layer in transformer_layers:
        inputs = layer(inputs, rng)

    final_layer = hk.Linear(output_dim)
    return final_layer(inputs)

def create_look_ahead_mask(size):
    mask = np.triu(np.ones((size, size)), k=1)
    mask = mask[np.newaxis, np.newaxis, ...]
    return mask

def q_network_model(inputs, enc_output, grid_code, num_layers, num_heads, dff, rate, output_dim, rng, cnn_output_size=256):
    # Process grid_code using CNN
    cnn_processed_grid_code = process_grid_code(grid_code, cnn_output_size)
    # Combine inputs and flattened grid_code
    inputs = np.concatenate([inputs, cnn_processed_grid_code], axis=-1)

    # Initialize transformer decoder layer
    transformer_decoder_layers = [TransformerDecoderLayer(inputs.shape[-1], num_heads, dff, rate) for _ in range(num_layers)]
    
    lood_ahead_mask = create_look_ahead_mask(inputs.shape[1]*inputs.shape[2])
    # forward function
    for layer in transformer_decoder_layers:
        inputs = layer(inputs, enc_output, lood_ahead_mask, None, rng)

    final_layer = hk.Linear(output_dim)
    return final_layer(inputs)

def klGaussianStandard(mean, var):
    return 0.5 * (-np.log(var) - 1.0 + var + mean ** 2)

def kl_divergence(mean1, stddev1, mean2, stddev2):
    """
    Calculate the Kullback-Leibler (KL) divergence between two one-dimensional Gaussian distributions.

    Args:
        mean1 (float): Mean of the first distribution.
        stddev1 (float): Standard deviation of the first distribution.
        mean2 (float): Mean of the second distribution.
        stddev2 (float): Standard deviation of the second distribution.

    Returns:
        float: The KL divergence value.
    """
    # Calculate the KL divergence
    kl = np.log(stddev2 / stddev1) + ((stddev1 ** 2 + (mean1 - mean2) ** 2) / (2 * stddev2 ** 2)) - 0.5
    return kl

class avril:
    """
    Class for implementing the AVRIL algorithm of Chan and van der Schaar (2021).
    This model is designed to be instantiated before calling the .train() method
    to fit to data.
    """

    def __init__(
        self,
        inputs: np.array,
        targets: np.array,
        grid_code: np.array,
        state_dim: int,
        action_dim: int,
        state_only: bool = True,

        cnn_output: int = 256,
        num_layers: int = 6,
        num_heads: int = 2,
        dff = 28,
        rate = 0.1,
        seed: int = 41310,
    ):
        """
        Parameters
        ----------

        inputs: np.array
            State training data of size [num_pairs x 2 x state_dimension]
        targets: np.array
            Action training data of size [num_pairs x 2 x 1]
        state_dim: int
            Dimension of state space
        action_dim: int
            Size of action space
        state_only: bool, True
            Whether learnt reward is state-only (as opposed to state-action)
        seed: int, 41310
            Random seed - required for JAX PRNG to work
        """

        self.key = random.PRNGKey(seed)

        self.encoder = hk.transform(encoder_model)
        self.q_network = hk.transform(q_network_model)

        self.inputs = inputs
        self.targets = targets
        self.grid_code = grid_code
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.state_only = state_only
        self.encoder_o_dim = 2
        self.cnn_output = cnn_output

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff 
        self.rate = rate

        self.e_params = self.encoder.init(
            self.key, inputs, grid_code, num_layers, num_heads,dff, rate, self.encoder_o_dim, self.key, cnn_output
        )

        enc_output = random.normal(self.key, inputs.shape[:-1] + (2,))
        self.q_params = self.q_network.init(
            self.key, inputs, enc_output, grid_code, num_layers, num_heads, dff, rate, action_dim, self.key, cnn_output
        )

        self.params = (self.e_params, self.q_params)

        self.load_params = False
        self.pre_params = None
        return
    
    def modelSave(self,model_save_path):
        with open(model_save_path,'wb') as f:
            print("save params to {}!".format(model_save_path))
            pickle.dump(self.params, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def loadParams(self,model_path):
        print("load params from {}!".format(model_path))
        with open(model_path, 'rb') as f:
            self.params = pickle.load(f)     
            self.load_params = True
            self.pre_params = self.params

    def reward(self,state,grid_code):
        #  Returns reward function parameters for a given state
        r_par = self.encoder.apply(
                self.e_params,
                self.key,
                state,
                grid_code,
                self.num_layers,
                self.num_heads,
                self.dff,
                self.rate,
                self.encoder_o_dim,
                self.key,
                self.cnn_output
            )
        r_par = np.squeeze(r_par,axis = 2)
        return r_par
    
    def QValue(self,state,grid_code):
        enc_output = self.encoder.apply(
                self.e_params,
                self.key,
                state,
                grid_code,
                self.num_layers,
                self.num_heads,
                self.dff,
                self.rate,
                self.encoder_o_dim,
                self.key,
                self.cnn_output
            )
        
        q_values = self.q_network.apply(
            self.q_params,
            self.key,
            state,
            enc_output,
            grid_code,
            self.num_layers,
            self.num_heads,
            self.dff,
            self.rate,
            self.a_dim,
            self.key,
            self.cnn_output
        )
        q_values = np.squeeze(q_values,axis=2)
        return q_values

    def elbo(self, params, key, inputs, targets, grid_code):
        """
        Method for calculating ELBO

        Parameters
        ----------

        params: tuple
            JAX object containing parameters of the model
        key:
            JAX PRNG key
        inputs: np.array
            State training data of size [num_pairs x 2 x state_dimension]
        targets: np.array
            Action training data of size [num_pairs x 2 x 1]

        Returns
        -------

        elbo: float
            Value of the ELBO
        """

        def getRewardParameters(encoder_params,state_dim):
            r_par0 = self.encoder.apply(
                encoder_params,
                key,
                inputs[:, :, state_dim, np.newaxis, :],
                grid_code[:, :, state_dim, np.newaxis, :, :, :],
                self.num_layers,
                self.num_heads,
                self.dff,
                self.rate,
                self.encoder_o_dim,
                self.key,
                self.cnn_output
            )
            r_par = np.squeeze(r_par0,axis = 2)

            if self.state_only:
                means = r_par[:, :, 0].reshape(-1)  # mean
                log_sds = r_par[:,:, 1].reshape(-1)  # log std var
            else:
                means = np.take_along_axis(r_par, (targets[:,:, 0, :]).astype(int), axis=1).reshape((len(inputs),))
                log_sds = np.take_along_axis(r_par, (a_dim + targets[:,:, 0, :]).astype(int), axis=1).reshape((len(inputs),))
            return means, log_sds,r_par0
        
        # get neural network's parameters
        e_params, q_params = params
        
        # calculate the kl difference between current reward and pre reward 
        means, log_sds, enc_output = getRewardParameters(e_params, 0)
        if self.load_params:
            e_params_pre, _ = self.pre_params
            means_pre, log_sds_pre , _ = getRewardParameters(e_params_pre, 0)
            
            kl = kl_divergence(means, np.exp(log_sds), means_pre, np.exp(log_sds_pre)).mean()
        else:
            kl = klGaussianStandard(means, np.exp(log_sds) ** 2).mean()

        # calculate td error
        # calculate Q-values for current state
        q_values = self.q_network.apply(
            q_params,
            key,
            inputs[:, :, 0, np.newaxis, :],
            enc_output,
            grid_code[:, :, 0, np.newaxis, :, :, :],
            self.num_layers,
            self.num_heads,
            self.dff,
            self.rate,
            self.a_dim,
            self.key,
            self.cnn_output
        )
        q_values = np.squeeze(q_values,axis=2)
        q_values_a = np.take_along_axis(
            q_values, targets[:,:, 0, :].astype(np.int32), axis=2
        ).reshape(inputs.shape[0]*inputs.shape[1])

        # Calculate Q-values for next state
        _, _, enc_output1 = getRewardParameters(e_params, 1)
        q_values_next = self.q_network.apply(
            q_params,
            key,
            inputs[:, :, 1, np.newaxis, :],
            enc_output1,
            grid_code[:, :, 1, np.newaxis, :, :, :],
            self.num_layers,
            self.num_heads,
            self.dff,
            self.rate,
            self.a_dim,
            self.key,
            self.cnn_output
        )
        q_values_next = np.squeeze(q_values_next,axis=2)
        q_values_next_a = np.take_along_axis(
            q_values, targets[:,:, 1, :].astype(np.int32), axis=2
        ).reshape(inputs.shape[0]*inputs.shape[1])
        # calculate TD error
        td = q_values_a - q_values_next_a
        
        # Selecting unpadded value corresopnding to the real travel chain, delete nan value
        valid_indices = ~np.isnan(td)
        td = td[valid_indices]
        means = means[valid_indices]
        log_sds = log_sds[valid_indices]

        # Add a negative sign in front of each formula to solve for the minimum value
        # Calculate log-likelihood of TD error given reward parameterisation
        lambda_value = 1
        irl_loss = -jax.scipy.stats.norm.logpdf(td, means, np.exp(log_sds)).mean()

        # Calculate log-likelihood of actions
        pred = jax.nn.log_softmax(q_values)
        neg_log_lik = -np.take_along_axis(
            pred, targets[:,:, 0, :].astype(np.int32), axis=1
        )
        neg_log_lik = np.nanmean(neg_log_lik)
        
        return neg_log_lik + kl + lambda_value*irl_loss
    
    def train(self, iters: int = 1000, batch_size: int = 64, l_rate: float = 1e-4, loss_threshold: float = 0.001):
        """
        Training function for the model.

        Parameters
        ----------
        iters: int, 1000
            Number of training update steps (NOTE: Not epochs)
        batch_size: int, 64
            Batch size for stochastic optimisation
        l_rate: float, 1e-4
            Main learning rate for Adam
        """

        inputs = self.inputs
        targets = self.targets
        grid_code = self.grid_code

        init_fun, update_fun, get_params = optimizers.adam(l_rate)
        update_fun = jit(update_fun)
        get_params = jit(get_params)

        params = self.params

        param_state = init_fun(params)

        # loss_grad = jit(value_and_grad(self.elbo))
        loss_grad = value_and_grad(self.elbo)

        len_x = len(self.inputs[:,:, 0, :])
        num_batches = np.ceil(len_x / batch_size)

        indx_list = np.array(range(len_x))

        key = self.key

        lik_pre = 0
        for itr in tqdm(range(iters)):

            if itr % num_batches == 0:
                indx_list_shuffle = jax.random.permutation(key, indx_list)

            indx = int((itr % num_batches) * batch_size)
            indexs = indx_list_shuffle[indx : (batch_size + indx)]

            key, subkey = random.split(key)

            lik, g_params = loss_grad(params, key, inputs[indexs], targets[indexs], grid_code[indexs])

            loss_diff = abs(lik-lik_pre)
            print(loss_diff,lik)
            if loss_diff < loss_threshold:
                print(f"Training stopped at iteration {itr} as loss {loss_diff} is below the threshold {loss_threshold}")
                break
            lik_pre = lik

            param_state = update_fun(itr, g_params, param_state)

            params = get_params(param_state)

        self.e_params = params[0]
        self.q_params = params[1]
        self.params = params

def computeRewardOrValue(model, input_path, output_path, place_grid_data, attribute_type='value'):
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

    # Add a column for attribute_type
    state_attribute[attribute_type] = 0.0

    computeFunc = getComputeFunction(model, attribute_type)
    
    for index, row in tqdm(state_attribute.iterrows(), total=len(state_attribute)):
        # get grid code of this fnid
        fnid = row.fnid
        this_fnid_grid = place_grid_data[fnid]
        destination_grid = np.zeros_like(this_fnid_grid) # if has specific destination, change this line to real destination grid code
        grid_code = onp.concatenate((this_fnid_grid, destination_grid), axis=0)
        # get state attribute of this fnid
        state = np.array(row.values[1:-1])

        # add three dimension
        grid_code = np.expand_dims(np.expand_dims(np.expand_dims(grid_code, axis=0), axis=0), axis = 0)
        state = np.expand_dims(np.expand_dims(np.expand_dims(state, axis=0), axis=0), axis = 0)
        # change state attribute and gird code to 
        a = computeFunc(state,grid_code)
        state_attribute.iloc[index, -1] = float(computeFunc(state,grid_code))
        
    if output_path is not None:
        state_attribute.to_csv(output_path, index=False)
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
        r = float(rewardFunction(state, grid_code)) if fnid in visitedState else 0
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
            before_chain = all_chains[-(memory_buffer-i):]
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

if __name__=="__main__":
    data_dir = './data/'
    model_dir = './model/'
    coords_file_path = './data/coords_grid_data.pkl'
    place_file_path = './data/place_grid_data.pkl'
    
    with open(coords_file_path, 'rb') as file:
        coords_grid_data = pickle.load(file)

    with open(place_file_path, 'rb') as file:
        place_grid_data = pickle.load(file)

    # Paths for data files
    before_migration_path = data_dir + 'before_migrt.json'
    after_migration_path = data_dir + 'after_migrt.json'
    full_trajectory_path = data_dir + 'all_traj.json'

    inputs, targets_action, grid_code, action_dim, state_dim = loadTrajChain(before_migration_path, full_trajectory_path, coords_grid_data)
    print(inputs.shape,targets_action.shape,grid_code.shape)
    model = avril(inputs, targets_action, grid_code, state_dim, action_dim, state_only=True)

    # NOTE: train the model
    model.train(iters=1000)
    model_save_path = model_dir + 'params_transformer.pickle'
    model.modelSave(model_save_path)

    # NOTE: compute rewards and values before migration
    feature_file = data_dir + 'before_migrt_feature.csv'
    computeRewardOrValue(model, feature_file, data_dir + 'before_migrt_reward.csv', place_grid_data, attribute_type='reward')
    computeRewardOrValue(model, feature_file, data_dir + 'before_migrt_value.csv', place_grid_data, attribute_type='value')

    # NOTE: Compute rewards after migration
    feature_file_all = data_dir + 'all_traj_feature.csv'
    output_reward_path = data_dir + 'after_migrt_reward.csv'
    afterMigrt(after_migration_path, before_migration_path, full_trajectory_path, coords_grid_data, feature_file_all, output_reward_path, model)