import jax.numpy as np
from jax import grad, jit, value_and_grad
from jax import random
from jax.example_libraries import optimizers
import jax
import haiku as hk
import os
import pandas as pd

from tqdm import tqdm
import numpy as onp
import pickle

from utils import *


def hidden_layers(layers=1, units=64):
    hidden = []
    for i in range(layers):
        hidden += [hk.Linear(units), jax.nn.elu]
    return hidden

class BasicCNN(hk.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=5, stride=2, padding='SAME')
        self.conv2 = hk.Conv2D(output_channels=64, kernel_shape=3, stride=2, padding='SAME')
        self.flatten = hk.Flatten()
        self.dense = hk.Linear(output_size)

    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.dense(x)
        return x

def process_grid_code(grid_code, output_size):
    batch_size, state_steps = grid_code.shape[:2]

    grid_code_reshaped = np.reshape(grid_code, [-1, *grid_code.shape[2:]]) 

    cnn = BasicCNN(output_size)
    cnn_output = cnn(grid_code_reshaped) 

    cnn_output_reshaped = np.reshape(cnn_output, [batch_size, state_steps, -1]) # [n_state,2,output_size]
    return cnn_output_reshaped

def encoder_model(inputs, grid_code, cnn_output_size = 16, layers=2, units=64, state_only=True, a_dim=None):
    """
    Create an encoder model that incorporates grid code data.

    Args:
        inputs (jax.numpy): The input tensor representing the state.
        grid_code (jax.numpy): The input tensor representing grid code data.
        layers (int, optional): The number of hidden layers in the encoder model. Defaults to 2.
        units (int, optional): The number of units (neurons) in each hidden layer. Defaults to 64.
        state_only (bool, optional): If True, encode only the state information; if False, encode state-action pairs. Defaults to True.
        a_dim (int, optional): The dimension of the action space (only needed if state_only is False). Defaults to None.
        cnn_output_size (int, optional): The output size of the CNN layers processing the grid code. Defaults to 128.

    Returns:
        The output of the encoder model.
    """
    # Process grid_code using CNN
    cnn_processed_grid_code = process_grid_code(grid_code, cnn_output_size)
    # Combine inputs and flattened grid_code
    combined_input = np.concatenate([inputs, cnn_processed_grid_code], axis=-1)

    # MLP Network
    out_dim = 2 if state_only else a_dim * 2
    mlp_layers = hidden_layers(layers, units) + [hk.Linear(out_dim)]
    mlp = hk.Sequential(mlp_layers)

    return mlp(combined_input)

def q_network_model(inputs, grid_code, a_dim, cnn_output_size = 16, layers=2, units=64):
    """
    Create a Q-network model for reinforcement learning that incorporates grid code data.

    Args:
        inputs (jax.numpy): The input tensor representing the state.
        grid_code (jax.numpy): The input tensor representing grid code data.
        a_dim (int): The number of actions in the action space.
        layers (int, optional): The number of hidden layers in the Q-network. Defaults to 2.
        units (int, optional): The number of units (neurons) in each hidden layer. Defaults to 64.
        cnn_output_size (int, optional): The output size of the CNN layers processing the grid code. Defaults to 128.

    Returns:
        A Q-network model that takes the state and grid code as input and outputs Q-values for each action.
    """
    # Process grid_code using CNN
    cnn_processed_grid_code = process_grid_code(grid_code, cnn_output_size)
    # Combine inputs and flattened grid_code
    combined_input = np.concatenate([inputs, cnn_processed_grid_code], axis=-1)

    # MLP Network
    mlp_layers = hidden_layers(layers, units) + [hk.Linear(a_dim)]
    mlp = hk.Sequential(mlp_layers)

    return mlp(combined_input)

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
        cnn_output: int = 16,
        encoder_layers: int = 2,
        encoder_units: int = 64,
        decoder_layers: int = 2,
        decoder_units: int = 64,
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
        encoder_layers: int, 2
            Number of hidden layers in encoder network
        encoder_units: int, 64
            Number of hidden units per layer of encoder network
        decoder_layers: int, 2
            Number of hidden layers in decoder network
        decoder_units: int, 64
            Number of hidden units per layer of decoder network
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
        self.cnn_output = cnn_output
        self.encoder_layers = encoder_layers
        self.encoder_units = encoder_units
        self.decoder_layers = decoder_layers
        self.decoder_units = decoder_units

        self.e_params = self.encoder.init(
            self.key, inputs, grid_code, cnn_output, encoder_layers, encoder_units, self.state_only, action_dim
        )
        self.q_params = self.q_network.init(
            self.key, inputs, grid_code, action_dim, cnn_output, decoder_layers, decoder_units
        )

        self.params = (self.e_params, self.q_params)

        self.load_params = False
        self.pre_params = None
        return

    def reward(self, state, grid_code):
        #  Returns reward function parameters for a given state
        r_par = self.encoder.apply(
            self.e_params,
            self.key,
            state,
            grid_code,
            self.cnn_output,
            self.encoder_layers,
            self.encoder_units,
            self.state_only,
            self.a_dim,
        )
        r_par = np.squeeze(r_par,axis = 1)
        return r_par

    def rewardValue(self,state,grid_code):
        """
            return the given reward of a given state
        Args:
            state (np.array): attribute of the given state

        Returns:
            (int): reward value
        """        
        mean, log_variance = self.reward(state, grid_code) # mean and log variance
        sample_size=1
        sample_reward = onp.random.normal(mean, np.exp(log_variance), sample_size)
        return sample_reward

    def QValue(self, state,grid_code):
        q_values = self.q_network.apply(
            self.q_params,
            self.key,
            state,
            grid_code,
            self.a_dim,
            self.cnn_output,
            self.decoder_layers,
            self.decoder_units,
        )
        q_values = np.squeeze(q_values,axis=1)
        return q_values
    
    def modelSave(self,model_save_path):
        with open(model_save_path,'wb') as f:
            print("save params to {}!".format(model_save_path))
            pickle.dump(self.params, f, protocol=pickle.HIGHEST_PROTOCOL)

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

        # Calculate Q-values for current state
        e_params, q_params = params
        q_values = self.q_network.apply(
            q_params,
            key,
            inputs[:, 0, np.newaxis, :],
            grid_code[:, 0, np.newaxis, :, :, :],
            self.a_dim,
            self.cnn_output,
            self.decoder_layers,
            self.decoder_units,
        )
        q_values = np.squeeze(q_values,axis=1)
        q_values_a = np.take_along_axis(
            q_values, targets[:, 0, :].astype(np.int32), axis=1
        ).reshape(len(inputs))

        # Calculate Q-values for next state
        q_values_next = self.q_network.apply(
            q_params,
            key,
            inputs[:, 1, np.newaxis, :],
            grid_code[:, 1, np.newaxis, :, :, :],
            self.a_dim,
            self.cnn_output,
            self.decoder_layers,
            self.decoder_units,
        )
        q_values_next = np.squeeze(q_values_next,axis=1)
        q_values_next_a = np.take_along_axis(
            q_values_next, targets[:, 1, :].astype(np.int32), axis=1
        ).reshape(len(inputs))

        # Calaculate TD error
        td = q_values_a - q_values_next_a

        def getRewardParameters(encoder_params):
            r_par = self.encoder.apply(
                encoder_params,
                key,
                inputs[:, 0, np.newaxis, :],
                grid_code[:, 0, np.newaxis, :, :, :],
                self.cnn_output,
                self.encoder_layers,
                self.encoder_units,
                self.state_only,
                self.a_dim,
            )
            r_par = np.squeeze(r_par,axis = 1)

            if self.state_only:
                means = r_par[:, 0].reshape(len(inputs))  # mean
                log_sds = r_par[:, 1].reshape(len(inputs))  # log std var
            else:
                means = np.take_along_axis(
                    r_par, (targets[:, 0, :]).astype(int), axis=1
                ).reshape((len(inputs),))
                log_sds = np.take_along_axis(
                    r_par, (a_dim + targets[:, 0, :]).astype(int), axis=1
                ).reshape((len(inputs),))
            return means, log_sds
        
        means, log_sds = getRewardParameters(e_params)
        if self.load_params:
            e_params_pre, _ = self.pre_params
            means_pre, log_sds_pre = getRewardParameters(e_params_pre)
            
            kl = kl_divergence(means, np.exp(log_sds), means_pre, np.exp(log_sds_pre)).mean()
        else:
            kl = klGaussianStandard(means, np.exp(log_sds) ** 2).mean()

        # NOTE: Add a negative sign in front of each formula to solve for the minimum value
        # Calculate log-likelihood of TD error given reward parameterisation
        lambda_value = 1
        irl_loss = -jax.scipy.stats.norm.logpdf(td, means, np.exp(log_sds)).mean()

        # Calculate log-likelihood of actions
        pred = jax.nn.log_softmax(q_values)
        neg_log_lik = -np.take_along_axis(
            pred, targets[:, 0, :].astype(np.int32), axis=1
        ).mean()

        return neg_log_lik + kl + lambda_value*irl_loss

    def loadParams(self,model_path):
        print("load params from {}!".format(model_path))
        with open(model_path, 'rb') as f:
            self.params = pickle.load(f)     
            self.load_params = True
            self.pre_params = self.params

    def train(self, iters: int = 1000, batch_size: int = 64, l_rate: float = 1e-4, loss_threshold: float = 0.01):
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

        loss_grad = jit(value_and_grad(self.elbo))

        len_x = len(self.inputs[:, 0, :])
        num_batches = np.ceil(len_x / batch_size)

        indx_list = np.array(range(len_x))

        key = self.key

        for itr in tqdm(range(iters)):

            if itr % num_batches == 0:
                indx_list_shuffle = jax.random.permutation(key, indx_list)

            indx = int((itr % num_batches) * batch_size)
            indexs = indx_list_shuffle[indx : (batch_size + indx)]

            key, subkey = random.split(key)

            lik, g_params = loss_grad(params, key, inputs[indexs], targets[indexs], grid_code[indexs])

            print(lik)
            if lik < loss_threshold:
                print(f"Training stopped at iteration {itr} as loss {lik} is below the threshold {loss_threshold}")
                break

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

        # add two dimension, [state_num,1|2,*state.shape]
        grid_code = np.expand_dims(np.expand_dims(grid_code, axis=0), axis=0)
        state = np.expand_dims(np.expand_dims(state, axis=0), axis=0)
        # change state attribute and gird code to 
        state_attribute.iloc[index, -1] = float(computeFunc(state,grid_code))
        
    if output_path is not None:
        state_attribute.to_csv(output_path, index=False)
    return

def getComputeFunction(model, attribute_type):
    """Return the appropriate function to compute either 'value' or 'reward'."""
    if attribute_type == 'value':
        return lambda state,grid_code: np.max(model.QValue(state,grid_code))
    elif attribute_type == 'reward':
        return lambda state,grid_code: model.reward(state,grid_code)[0][0] # array(1,2)
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
    visitedState = {state for chain in beforeChains for state in chain['travel_chain']}

    # Load and process the data of travel chains after migration.
    loadedDictsList1 = loadJsonFile(afterMigrtFile)
    trajChains = loadTravelDataFromDicts(loadedDictsList1)

    # Preprocess state attributes based on the after migration data.
    stateAttribute, _ = preprocessStateAttributes(afterMigrtFile, inputPath)

    return visitedState, trajChains, stateAttribute

def processBeforeMigrationData(stateAttribute, visitedState, place_grid_data, computeFunc, idFnid, actionDim):
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
    for _, row in tqdm(stateAttribute.iterrows(), total=len(stateAttribute)):
        # Extract fnid and state information from the row.
        fnid = row.values[0]
        state = np.array(row.values[1:])
        this_fnid_grid = place_grid_data[fnid]
        destination_grid = np.zeros_like(this_fnid_grid) # if has specific destination, change this line to real destination grid code
        grid_code = onp.concatenate((this_fnid_grid, destination_grid), axis=0)
        # add two dimension, [state_num,1|2,*state.shape]
        grid_code = np.expand_dims(np.expand_dims(grid_code, axis=0), axis=0)
        state = np.expand_dims(np.expand_dims(state, axis=0), axis=0)

        # Compute action probabilities based on whether the state has been visited.
        if fnid in visitedState:
            actionProb = []
            a = computeFunc(state, grid_code)
            for i, prob in enumerate(computeFunc(state, grid_code)[0]):
                if i == len(idFnid):# the last one is the "no action"
                    actionProb.append(prob)
                    break
                if idFnid[i] not in visitedState:
                    actionProb.append(0)
                else:
                    actionProb.append(prob)
        else:
            actionProb = [0 for _ in range(actionDim)]

        # Prepare the row for the results DataFrame.
        resultRow = [fnid] + list(actionProb)
        beforeMigrtTrans.append(resultRow)

    # Define columns for the results DataFrame, add the "no action" column.
    columns = ['fnid'] + [idFnid[i] for i in range(actionDim-1)] +['no action']

    # Create and save the results DataFrame.
    dfResults = pd.DataFrame(beforeMigrtTrans, columns=columns)
    dfResults.to_csv(f"./data/before_migrt_transProb.csv", index=False)


def processAfterMigrationData(tc, stateAttribute, place_grid_data, model, visitedState, idFnid, actionDim):
    """
    Process data after migration, including calculating rewards and transition probabilities.

    Args:
        tc (TrajectoryChain): The trajectory chain object containing migration data.
        stateAttribute (DataFrame): DataFrame containing state attributes.
        place_grid_data (dict): Dictionary mapping places to their grid data.
        model (Model): The model used for calculations.
        visitedState (set): A set of states that have been visited.
        idFnid (dict): Dictionary mapping action indices to identifiers.
        actionDim (int): The dimension of the action space.

    Returns:
        list: List of reward values for each state.
    """
    # Preprocess trajectory data and update visited states
    stateNextState, actionNextAction, grid_next_grid = processTrajectoryData([tc], stateAttribute, model.s_dim, place_grid_data)
    visitedState.update(tc.travel_chain)

    # Set model inputs for training or evaluation
    model.inputs = onp.array(stateNextState)
    model.targets = onp.array(actionNextAction)
    model.grid_code = onp.array(grid_next_grid)

    # Define functions for reward calculation and Q-value computation
    rewardFunction = getComputeFunction(model, 'reward')
    computeFunc = lambda state, grid_code: model.QValue(state, grid_code)

    # Initialize lists for storing results
    rewardValues = []
    results = []

    # Iterate over each row in state attributes to compute rewards and action probabilities
    for _, row in tqdm(stateAttribute.iterrows(), total=len(stateAttribute)):
        fnid = row.values[0]
        state = np.array(row.values[1:])
        this_fnid_grid = place_grid_data[fnid]
        destination_grid = np.zeros_like(this_fnid_grid)  # Update this line for specific destinations
        grid_code = onp.concatenate((this_fnid_grid, destination_grid), axis=0)
        grid_code = np.expand_dims(np.expand_dims(grid_code, axis=0), axis=0)
        state = np.expand_dims(np.expand_dims(state, axis=0), axis=0)

        # Calculate reward for visited states
        r = float(rewardFunction(state, grid_code)) if fnid in visitedState else 0
        rewardValues.append(r)

        # Compute action probabilities
        if fnid in visitedState:
            actionProb = []
            for i, prob in enumerate(computeFunc(state, grid_code)[0]):
                if i == len(idFnid):  # Handle "no action"
                    actionProb.append(prob)
                    break
                actionProb.append(0 if idFnid[i] not in visitedState else prob)
        else:
            actionProb = [0] * actionDim

        # Append results for each state
        resultRow = [fnid] + list(actionProb)
        results.append(resultRow)

    # Create and save results DataFrame
    columns = ['fnid'] + [idFnid[i] for i in range(actionDim-1)] + ['no action']
    dfResults = pd.DataFrame(results, columns=columns)

    output_dir = f"./data/after_migrt/transProb/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dfResults.to_csv(f"./data/after_migrt/transProb/{tc.date}.csv", index=False)

    return rewardValues

def afterMigrt(afterMigrtFile, beforeMigrtFile, full_trajectory_path, place_grid_data, inputPath, outputPath, model):
    # Load model parameters from a saved state.
    model.loadParams('./model/params.pickle')

    # Load the mapping between IDs and their corresponding fnid.
    with open("./data/id_fnid_mapping.pkl", "rb") as f:
        idFnid = pickle.load(f)

    # Define a function to compute Q-values given a state and corresponding grid_code.
    computeFunc = lambda state, grid_code: model.QValue(state,grid_code)

    all_chains = loadTravelDataFromDicts(loadJsonFile(full_trajectory_path))
    actionDim = getActionDim(all_chains)

    # Read and preprocess data for analysis.
    visitedState, trajChains, stateAttribute = readAndPrepareData(afterMigrtFile, beforeMigrtFile, inputPath)

    # Process and update state attributes before migration.
    processBeforeMigrationData(stateAttribute, visitedState, place_grid_data, computeFunc, idFnid, actionDim)

    # Initialize a DataFrame to store results.
    resultsDf = pd.DataFrame()
    resultsDf['fnid'] = stateAttribute['fnid']

    
    modelDir = "./data/after_migrt/model"
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    preDate = 0 # load preDate model parameters
    for tc in trajChains:
        # Update model parameters based on the date of trajectory chain.
        if preDate:
            modelPath = os.path.join(modelDir, f"{preDate}.pickle")
            model.loadParams(modelPath)

        # Process and calculate reward values after migration.
        rewardValues = processAfterMigrationData(tc, stateAttribute, place_grid_data, model, visitedState, idFnid, actionDim)

        # Train the model.
        model.train(iters=5000)

        # Save the current model state.
        modelSavePath = "./data/after_migrt/model/" + str(tc.date) + ".pickle"
        model.modelSave(modelSavePath)

        # Store the calculated reward values in the results DataFrame.
        resultsDf[str(tc.date)] = rewardValues

        preDate = tc.date

    # Save the results to a CSV file.
    resultsDf.to_csv(outputPath, index=False)



if __name__ == "__main__":
    data_dir = './data/'
    model_dir = './model/'
    file_path = './data/place_grid_data.pkl'
    
    with open(file_path, 'rb') as file:
        place_grid_data = pickle.load(file)

    # Paths for data files
    before_migration_path = data_dir + 'before_migrt.json'
    after_migration_path = data_dir + 'after_migrt.json'
    full_trajectory_path = data_dir + 'all_traj.json'
    
    # Initialize the model
    inputs, targets_action, grid_code, action_dim, state_dim= loadTrajChain(before_migration_path, full_trajectory_path,place_grid_data)
    print(inputs.shape, targets_action.shape, grid_code.shape,action_dim, state_dim)
    model = avril(inputs, targets_action, grid_code, state_dim, action_dim, state_only=True)

    # NOTE: train the model
    model.train(iters=50000)
    model_save_path = model_dir + 'params.pickle'
    model.modelSave(model_save_path)
    
    # NOTE: compute rewards and values before migration
    # feature_file = data_dir + 'before_migrt_feature.csv'
    # computeRewardOrValue(model, feature_file, data_dir + 'before_migrt_value.csv', place_grid_data, attribute_type='value')
    # computeRewardOrValue(model, feature_file, data_dir + 'before_migrt_reward.csv', place_grid_data, attribute_type='reward')
    
    # NOTE: Compute rewards after migration
    # feature_file_all = data_dir + 'all_traj_feature.csv'
    # output_reward_path = data_dir + 'after_migrt_reward.csv'
    # afterMigrt(after_migration_path, before_migration_path, full_trajectory_path, place_grid_data, feature_file_all, output_reward_path, model)