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

from transformer import *
from BasicCNN import *
from utils import *

def encoder_model(inputs, grid_code, num_layers, num_heads, dff, rate, output_dim, rng, cnn_output_size = 256):
    # Process grid_code using CNN
    cnn_processed_grid_code = process_grid_code(grid_code, cnn_output_size)
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
        num_layers: int = 2,
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
        
        # Add a negative sign in front of each formula to solve for the minimum value
        # Calculate log-likelihood of TD error given reward parameterisation
        lambda_value = 1
        irl_loss = -jax.scipy.stats.norm.logpdf(td, means, np.exp(log_sds)).mean()

        # Calculate log-likelihood of actions
        pred = jax.nn.log_softmax(q_values)
        neg_log_lik = -np.take_along_axis(
            pred, targets[:,:, 0, :].astype(np.int32), axis=1
        ).mean()

        return neg_log_lik + kl + lambda_value*irl_loss
    
    def train(self, iters: int = 1000, batch_size: int = 64, l_rate: float = 1e-4):
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

        len_x = len(self.inputs[:,:, 0, :])
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
    
if __name__=="__main__":
    data_dir = './data/'
    model_dir = './model/'
    file_path = './data/place_grid_data.pkl'
    
    with open(file_path, 'rb') as file:
        place_grid_data = pickle.load(file)

    # Paths for data files
    before_migration_path = data_dir + 'before_migrt.json'
    after_migration_path = data_dir + 'after_migrt.json'
    full_trajectory_path = data_dir + 'all_traj.json'

    inputs, targets_action, grid_code, action_dim, state_dim = loadTrajChain(before_migration_path, full_trajectory_path, place_grid_data)
    print(inputs.shape,targets_action.shape,grid_code.shape)
    model = avril(inputs, targets_action, grid_code, state_dim, action_dim, state_only=True)

    # NOTE: train the model
    # model.train(iters=500)
    # model_save_path = model_dir + 'params_transformer.pickle'
    # model.modelSave(model_save_path)

    # NOTE: compute rewards and values before migration
    feature_file = data_dir + 'before_migrt_feature.csv'
    computeRewardOrValue(model, feature_file, data_dir + 'before_migrt_reward.csv', place_grid_data, attribute_type='reward')
    computeRewardOrValue(model, feature_file, data_dir + 'before_migrt_value.csv', place_grid_data, attribute_type='value')