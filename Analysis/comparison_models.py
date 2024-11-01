import haiku as hk

from jax import grad, jit, value_and_grad
from jax import random
from jax.example_libraries import optimizers
import jax
import jax.numpy as np
from jax import random
import optax

import numpy as onp
import pickle
import os
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod
import time, datetime


from sklearn.preprocessing import MinMaxScaler

from SCBIRL_Global_PE.EnDecoder import kl_divergence, klGaussianStandard
from SCBIRL_Global_PE.EnDecoderCompare import *
from SCBIRL_Global_PE.SCBIRLTransformer import avril
from SCBIRL_Global_PE.utils import Padding, normalize


class avril_without_pe(avril):
    
    """
    Class for implementing the AVRIL algorithm of Chan and van der Schaar (2021).
    This model is designed to be instantiated before calling the .train() method
    to fit to data.
    """

    def __init__(
        self,
        inputs: np.array,
        targets: np.array,
        state_dim: int,
        action_dim: int,
        state_only: bool = True,
        num_layers: int = 2,
        num_heads: int = 1,
        dff = 28,
        rate = 0.1,
        seed: int = 41310,
    ):
        """
        Parameters
        ----------

        inputs: np.array
            State training data of size [num_traj x npair_per_traj x 2 x state_dimension]
        targets: np.array
            Action training data of size [num_traj x npair_per_traj x 2 x 1]
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

        self.encoder = hk.transform(encoder_sin_model)
        self.q_network = hk.transform(q_network_sin_model)

        self.inputs = inputs
        self.targets = targets
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.state_only = state_only
        self.encoder_o_dim = 2

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff 
        self.rate = rate

        self.e_params = self.encoder.init(
            self.key, inputs, num_layers, num_heads,dff, rate, self.encoder_o_dim, self.key
        )

        enc_output = random.normal(self.key, inputs.shape[:-1] + (2,))
        self.q_params = self.q_network.init(
            self.key, inputs, enc_output, num_layers, num_heads, dff, rate, action_dim, self.key
        )

        self.params = (self.e_params, self.q_params)

        self.load_params = False
        self.pre_params = None
        return
    
    def loadParams(self,model_path):
        print("load params from {}!".format(model_path))
        with open(model_path, 'rb') as f:    
            self.load_params = True
            self.params = pickle.load(f) 
            self.pre_params = self.params
            self.e_params = self.params[0]
            self.q_params = self.params[1]

    def reward(self,state):
        #  Returns reward function parameters for a given state
        r_par = self.encoder.apply(
                self.e_params,
                self.key,
                state,
                self.num_layers,
                self.num_heads,
                self.dff,
                self.rate,
                self.encoder_o_dim,
                self.key
            )
        r_par = np.squeeze(r_par,axis = 2)
        return r_par
    
    def QValue(self,state):
        enc_output = self.encoder.apply(
                self.e_params,
                self.key,
                state,
                self.num_layers,
                self.num_heads,
                self.dff,
                self.rate,
                self.encoder_o_dim,
                self.key
            )
        
        q_values = self.q_network.apply(
            self.q_params,
            self.key,
            state,
            enc_output,
            self.num_layers,
            self.num_heads,
            self.dff,
            self.rate,
            self.a_dim,
            self.key
        )
        q_values = np.squeeze(q_values,axis=2)
        return q_values

    def elbo(self, params, key, inputs, targets, weights = None):
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

        def getRewardParameters(encoder_params, state_dim):
            # here, state_dim is eihter 0 or 1
            # the newaxis is to set the dimension same as before.
            r_par0 = self.encoder.apply(
                encoder_params,
                key,
                inputs[:, :, state_dim, np.newaxis, :],
                self.num_layers,
                self.num_heads,
                self.dff,
                self.rate,
                self.encoder_o_dim,
                self.key
            )
            r_par = np.squeeze(r_par0,axis = 2)

            if self.state_only:
                means = r_par[:, :, 0].reshape(-1)  # mean vector
                log_sds = r_par[:,:, 1].reshape(-1)  # log std var vector
            else:
                means = np.take_along_axis(r_par, (targets[:,:, 0, :]).astype(int), axis=1).reshape((len(inputs),))
                log_sds = np.take_along_axis(r_par, (self.a_dim + targets[:,:, 0, :]).astype(int), axis=1).reshape((len(inputs),))
            return means, log_sds, r_par0
        
        # get neural network's parameters
        e_params, q_params, _ = params
        
        # calculate the kl difference between current reward and pre reward 
        means, log_sds, enc_output = getRewardParameters(e_params, 0)
        # calculate td error
        # calculate Q-values for current state
        q_values = self.q_network.apply(
            q_params,
            key,
            inputs[:, :, 0, np.newaxis, :],
            enc_output,
            self.num_layers,
            self.num_heads,
            self.dff,
            self.rate,
            self.a_dim,
            self.key
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
            self.num_layers,
            self.num_heads,
            self.dff,
            self.rate,
            self.a_dim,
            self.key
        )
        q_values_next = np.squeeze(q_values_next,axis=2)
        q_values_next_a = np.take_along_axis(
            q_values_next, targets[:,:, 1, :].astype(np.int32), axis=2
        ).reshape(inputs.shape[0]*inputs.shape[1])
        # calculate TD error
        td = q_values_a - q_values_next_a
        
        # Selecting unpadded value corresopnding to the real travel chain, delete nan value
        # valid_indices = ~np.isnan(td)
        valid_multi_index = np.any(inputs[:, :, 0, :] != Padding, axis=2)
        valid_indices, = np.where(valid_multi_index.flatten())
        td = td[valid_indices]
        means = means[valid_indices]
        log_sds = log_sds[valid_indices]

        if self.load_params:
            # 有先验迭代
            e_params_pre, _, _ = self.pre_params
            means_pre, log_sds_pre , _ = getRewardParameters(e_params_pre, 0)
            kl = kl_divergence(means, np.exp(log_sds), means_pre, np.exp(log_sds_pre))
        else:
            # 无先验迭代，标准正态分布
            kl = klGaussianStandard(means, np.exp(log_sds) ** 2)
        
        # Add a negative sign in front of each formula to solve for the minimum value
        # Calculate log-likelihood of TD error given reward parameterisation
        lambda_value = 1
        irl_loss = -jax.scipy.stats.norm.logpdf(td, means, np.exp(log_sds))

        # Calculate log-likelihood of actions
        pred = jax.nn.log_softmax(q_values)
        neg_log_lik = - np.take_along_axis(
            pred, targets[:, :, 0, :].astype(np.int32), axis=2
        ).squeeze(axis=2).flatten()
        neg_log_lik = neg_log_lik[valid_indices]
        
        if weights is not None:
            assert len(inputs) == len(weights), 'The length of valid states and weights should be the same.'
            weights = weights.repeat(inputs.shape[1])
            weights = weights[valid_indices]

        neg_log_lik = np.average(neg_log_lik, weights=weights)
        kl = np.average(kl, weights=weights)
        irl_loss = np.average(irl_loss, weights=weights)

        return neg_log_lik + kl + lambda_value * irl_loss
    
    def train(self, iters: int = 1000, batch_size: int = 64, l_rate: float = 1e-4, 
              loss_threshold: float = 0.01, weights = None):
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
        if weights is not None:
            weights_array = np.array(weights)
        
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
            
            if weights is not None:
                weights = weights_array[indexs]

            lik, g_params = loss_grad(params, key, inputs[indexs], targets[indexs], weights = weights)

            loss_diff = abs(lik-lik_pre)
            print(lik-lik_pre, lik)
            if loss_diff < loss_threshold:
                print(f"Training stopped at iteration {itr} as loss {loss_diff} is below the threshold {loss_threshold}")
                break
            lik_pre = lik

            param_state = update_fun(itr, g_params, param_state)

            params = get_params(param_state)

        self.e_params = params[0]
        self.q_params = params[1]
        self.params = params

class avril_deep:
    """
    Class for implementing the AVRIL algorithm with deep neural networks of both encoder and decoder.
    This model is designed to be instantiated before calling the .train() method
    to fit to data.
    """

    def __init__(
        self,
        inputs: np.array,
        targets: np.array,
        state_dim: int,
        action_dim: int,
        state_only: bool = True,
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

        self.encoder = hk.transform(encoder_net_model)
        self.q_network = hk.transform(q_network_model)

        self.inputs = inputs
        self.targets = targets
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.state_only = state_only
        self.encoder_layers = encoder_layers
        self.encoder_units = encoder_units
        self.decoder_layers = decoder_layers
        self.decoder_units = decoder_units

        self.e_params = self.encoder.init(
            self.key, inputs, encoder_layers, encoder_units, self.state_only, action_dim
        )
        self.q_params = self.q_network.init(
            self.key, inputs, action_dim, decoder_layers, decoder_units
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

    def reward(self, state):
        #  Returns reward function parameters for a given state
        r_par = self.encoder.apply(
            self.e_params,
            self.key,
            state,
            self.encoder_layers,
            self.encoder_units,
            self.state_only,
            self.a_dim,
        )
        return r_par

    def rewardSampleValue(self,state):
        """
            return the given reward of a given state
        Args:
            state (np.array): attribute of the given state

        Returns:
            (int): reward value
        """        
        mean, log_variance = self.reward(state) # mean and log variance
        sample_size=1
        sample_reward = onp.random.normal(mean, np.exp(log_variance), sample_size)
        return sample_reward

    def QValue(self, state):
        #  Returns predicted action logits for a given state
        q_values = self.q_network.apply(
            self.q_params,
            self.key,
            state,
            self.a_dim,
            self.decoder_layers,
            self.decoder_units,
        )
        return q_values

    def elbo(self, params, key, inputs, targets):
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

        def getRewardParameters(encoder_params, state_dim):
            r_par = self.encoder.apply(
                encoder_params,
                key,
                inputs[:, state_dim, :],
                self.encoder_layers,
                self.encoder_units,
                self.state_only,
                self.a_dim,
            )
            if self.state_only:
                means = r_par[:, 0].reshape(len(inputs))  # mean
                log_sds = r_par[:, 1].reshape(len(inputs))  # log std var
            else:
                means = np.take_along_axis(
                    r_par, (targets[:, 0, :]).astype(int), axis=1
                ).reshape((len(inputs),))
                log_sds = np.take_along_axis(
                    r_par, (self.a_dim + targets[:, 0, :]).astype(int), axis=1
                ).reshape((len(inputs),))
            return means, log_sds

        # Calculate Q-values for current state
        e_params, q_params = params
        
        means, log_sds = getRewardParameters(e_params, 0)
        q_values = self.q_network.apply(
            q_params,
            key,
            inputs[:, 0, :],
            self.a_dim,
            self.decoder_layers,
            self.decoder_units,
        )
        q_values_a = np.take_along_axis(
            q_values, targets[:, 0, :].astype(np.int32), axis=1
        ).reshape(len(inputs))

        # Calculate Q-values for next state
        q_values_next = self.q_network.apply(
            q_params,
            key,
            inputs[:, 1, :],
            self.a_dim,
            self.decoder_layers,
            self.decoder_units,
        )
        q_values_next_a = np.take_along_axis(
            q_values_next, targets[:, 1, :].astype(np.int32), axis=1
        ).reshape(len(inputs))

        # Calaculate TD error
        td = q_values_a - q_values_next_a

        if self.load_params:
            e_params_pre, _ = self.pre_params
            means_pre, log_sds_pre = getRewardParameters(e_params_pre, 0)
            kl = kl_divergence(means, np.exp(log_sds), means_pre, np.exp(log_sds_pre))
        else:
            kl = klGaussianStandard(means, np.exp(log_sds) ** 2)

        # Note: Add a negative sign in front of each formula to solve for the minimum value
        # Calculate log-likelihood of TD error given reward parameterisation
        lambda_value = 1
        irl_loss = -jax.scipy.stats.norm.logpdf(td, means, np.exp(log_sds))

        # Calculate log-likelihood of actions
        pred = jax.nn.log_softmax(q_values)
        neg_log_lik = -np.take_along_axis(
            pred, targets[:, 0, :].astype(np.int32), axis=1
        ).squeeze(axis=1)
        
        # if weights is not None:
            # assert len(inputs) == len(weights), 'The length of valid states and weights should be the same.'
            # weights = weights[:, np.newaxis]
            # weights = weights[valid_indices]

        neg_log_lik = np.average(neg_log_lik)
        kl = np.average(kl)
        irl_loss = np.average(irl_loss)

        # 这里全部取负数
        return neg_log_lik + kl + lambda_value * irl_loss

    def train(self, iters: int = 1000, batch_size: int = 64, l_rate: float = 1e-4,
              loss_threshold: float = 0.01,):
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
        lik_pre = 0
        
        for itr in tqdm(range(iters)):

            if itr % num_batches == 0:
                indx_list_shuffle = jax.random.permutation(key, indx_list)

            indx = int((itr % num_batches) * batch_size)
            indxes = indx_list_shuffle[indx : (batch_size + indx)]

            key, subkey = random.split(key)

            lik, g_params = loss_grad(params, key, inputs[indxes], targets[indxes])

            loss_diff = abs(lik-lik_pre)
            print(lik-lik_pre, lik)
            if loss_diff < loss_threshold:
                print(f"Training stopped at iteration {itr} as loss {loss_diff} is below the threshold {loss_threshold}")
                break
            lik_pre = lik

            param_state = update_fun(itr, g_params, param_state)

            params = get_params(param_state)

        self.e_params = params[0]
        self.q_params = params[1]
        self.params = params

class avril_linear(avril_deep):
    """
    Class for implementing the AVRIL algorithm with deep neural networks of both encoder and decoder.
    This model is designed to be instantiated before calling the .train() method
    to fit to data.
    """

    def __init__(
        self,
        inputs: np.array,
        targets: np.array,
        state_dim: int,
        action_dim: int,
        state_only: bool = True,
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

        self.encoder = hk.transform(encoder_linear_model)
        self.q_network = hk.transform(q_linear_model)

        self.inputs = inputs
        self.targets = targets
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.state_only = state_only

        self.e_params = self.encoder.init(
            self.key, inputs,self.state_only, action_dim
        )
        self.q_params = self.q_network.init(
            self.key, inputs, action_dim, 
        )

        self.params = (self.e_params, self.q_params)

        self.load_params = False
        self.pre_params = None
        return


    def reward(self, state):
        #  Returns reward function parameters for a given state
        r_par = self.encoder.apply(
            self.e_params,
            self.key,
            state,
            self.state_only,
            self.a_dim,
        )
        return r_par


    def QValue(self, state):
        #  Returns predicted action logits for a given state
        q_values = self.q_network.apply(
            self.q_params,
            self.key,
            state,
            self.a_dim,
        )
        return q_values

    def elbo(self, params, key, inputs, targets):
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

        def getRewardParameters(encoder_params, state_dim):
            r_par = self.encoder.apply(
                encoder_params,
                key,
                inputs[:, state_dim, :],
                self.state_only,
                self.a_dim,
            )
            if self.state_only:
                means = r_par[:, 0].reshape(len(inputs))  # mean
                log_sds = r_par[:, 1].reshape(len(inputs))  # log std var
            else:
                means = np.take_along_axis(
                    r_par, (targets[:, 0, :]).astype(int), axis=1
                ).reshape((len(inputs),))
                log_sds = np.take_along_axis(
                    r_par, (self.a_dim + targets[:, 0, :]).astype(int), axis=1
                ).reshape((len(inputs),))
            return means, log_sds

        # Calculate Q-values for current state
        e_params, q_params = params
        
        means, log_sds = getRewardParameters(e_params, 0)
        q_values = self.q_network.apply(
            q_params,
            key,
            inputs[:, 0, :],
            self.a_dim,
        )
        q_values_a = np.take_along_axis(
            q_values, targets[:, 0, :].astype(np.int32), axis=1
        ).reshape(len(inputs))

        # Calculate Q-values for next state
        q_values_next = self.q_network.apply(
            q_params,
            key,
            inputs[:, 1, :],
            self.a_dim,
        )
        q_values_next_a = np.take_along_axis(
            q_values_next, targets[:, 1, :].astype(np.int32), axis=1
        ).reshape(len(inputs))

        # Calaculate TD error
        td = q_values_a - q_values_next_a

        if self.load_params:
            e_params_pre, _ = self.pre_params
            means_pre, log_sds_pre = getRewardParameters(e_params_pre, 0)
            kl = kl_divergence(means, np.exp(log_sds), means_pre, np.exp(log_sds_pre))
        else:
            kl = klGaussianStandard(means, np.exp(log_sds) ** 2)

        # Note: Add a negative sign in front of each formula to solve for the minimum value
        # Calculate log-likelihood of TD error given reward parameterisation
        lambda_value = 1
        irl_loss = -jax.scipy.stats.norm.logpdf(td, means, np.exp(log_sds))

        # Calculate log-likelihood of actions
        pred = jax.nn.log_softmax(q_values)
        neg_log_lik = -np.take_along_axis(
            pred, targets[:, 0, :].astype(np.int32), axis=1
        ).squeeze(axis=1)
        
        # if weights is not None:
            # assert len(inputs) == len(weights), 'The length of valid states and weights should be the same.'
            # weights = weights[:, np.newaxis]
            # weights = weights[valid_indices]

        neg_log_lik = np.average(neg_log_lik)
        kl = np.average(kl)
        irl_loss = np.average(irl_loss)

        # 这里全部取负数
        return neg_log_lik + kl + lambda_value * irl_loss 

class irl_maxent(ABC):
    def __init__(self, 
                inputs: np.array,
                targets: np.array,
                state_dim: int,
                action_dim: int,
                id_feature_mapping: dict,
                state_only: bool = True,
                transition_model: np.array = None,
                seed=41310):
        '''
        inputs should be original state trajectories rather than pairs.
        id_feature_mapping can be created by: mapping from id to coord to fnid to getStateRow function
        '''
        self.inputs = inputs
        self.targets = targets
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.key = jax.random.PRNGKey(seed)
        self.trajs = self.traj_translator(inputs, id_feature_mapping)
        self.feature_map_array = np.array(list(id_feature_mapping.values()))
        if transition_model is None:
            state_number = action_dim - 1
            transition_model = np.zeros((state_number, state_number, action_dim))
            for i in range(state_number):
                transition_model[:, i, i] = 1 # ? check
        self.P_a = transition_model
        self.reward_learner_setup()


    def loadParams(self, model_path):
        self.reward_learner.params = np.load(model_path, 
                                             allow_pickle=True).item()


    def modelSave(self, model_path):
        np.save(model_path, self.reward_learner.params)


    def policySave(self, policy_path):
        np.save(policy_path, self.policy)


    def traj_translator(self, trajs, id_feature_mapping):
        """
        Process trajectories into state-action pairs and state visitation frequencies
        """
        # catch the original trajectories
        inputs = self.inputs[:, :, 0, :]
        reverse_mapping = {v: k for k, v in id_feature_mapping.items()}
        trajs = []
        for b in range(len(inputs)):
            one_traj = inputs[b, :, :]
            nova_traj = [reverse_mapping[step_features] for step_features in one_traj]
            trajs.append(nova_traj)
        return trajs


    def expectStateVisitFreq(P_a, gamma, trajs, policy, deterministic=True):
        """
        compute the expected states visition frequency p(s| theta, T) 
        using dynamic programming

        inputs:
        P_a     NxNxN_ACTIONS matrix - transition dynamics
        gamma   float - discount factor
        trajs   list of Steps - collected from expert 
        policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

        returns:
        p       Nx1 vector - state visitation frequencies
        """
        N_STATES, _, N_ACTIONS = np.shape(P_a)
        T = []
        for traj in trajs:
            T.append(len(traj))

        avg_T = int(np.mean(T))
        # mu[s, t] is the prob of visiting state s at step t,get the
        mu = np.zeros([N_STATES, avg_T])

        for traj in trajs:
            index = traj[0]
            mu[index, 0] += 1
        mu[:, 0] = mu[:, 0]/len(trajs)

        for s in range(N_STATES):
            for t in range(avg_T-1):
                if deterministic:
                    mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])]
                                    for pre_s in range(N_STATES)])
                else:
                    mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)])
                                    for pre_s in range(N_STATES)])
                mu[s, t+1] *= gamma ** (t+1)
        p = np.sum(mu, 1)
        return p
 
    def stateVisitFreq(trajs, n_states):
        """
        compute state visitation frequences from demonstrations

        input:
        trajs   list of list of Steps - collected from expert # note 这个地方放inputs
        fnid_idx {fnid:index}
        n_states  number of states
        returns:
        p       Nx1 vector - state visitation frequences   
        """

        p = np.zeros(n_states)
        for traj in trajs:
            for step in traj:
                p[step] += 1
        p = p/len(trajs)
        return p
    
    
    def value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True):
        """
        static value iteration function. Perhaps the most useful function in this repo

        inputs:
        P_a         NxNxN_ACTIONS transition probabilities matrix - 
                                    P_a[s0, s1, a] is the transition prob of 
                                    landing at state s1 when taking action 
                                    a at state s0
        rewards     Nx1 matrix - rewards for all the states
        gamma       float - RL discount
        error       float - threshold for a stop
        deterministic   bool - to return deterministic policy or stochastic policy

        returns:
        values    Nx1 matrix - estimated values
        policy    Nx1 (NxN_ACTIONS if non-det) matrix - policy
        """
        N_STATES, _, N_ACTIONS = np.shape(P_a)

        values = np.zeros([N_STATES])

        # estimate values
        while True:
            values_tmp = values.copy()

            for s in range(N_STATES):
                v_s = []
                values[s] = max([sum([P_a[s, s1, a]*(rewards[s] + gamma*values_tmp[s1])
                                for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])

            max_diff = np.max(np.abs(values - values_tmp))
            print(max_diff)
            if  max_diff < error:
                break

        if deterministic:
            # generate deterministic policy
            policy = np.zeros([N_STATES])
            for s in range(N_STATES):
                policy[s] = np.argmax([sum([P_a[s, s1, a]*(rewards[s]+gamma*values[s1])
                                            for s1 in range(N_STATES)])
                                    for a in range(N_ACTIONS)])
            return values, policy
        else:
            # generate stochastic policy
            policy = np.zeros([N_STATES, N_ACTIONS])
            for s in range(N_STATES):
                v_s = np.array([sum([P_a[s, s1, a]*(rewards[s] + gamma*values[s1])
                                    for s1 in range(N_STATES)])
                                for a in range(N_ACTIONS)])
                policy[s, :] = np.transpose(v_s/np.sum(v_s))
            return values, policy
    
    
    @abstractmethod
    def reward_learner_setup(self, ):
        self.reward_learner = None
    
    
    def train(self, gamma, lr, n_iters, restore=True):

        N_STATES, _, N_ACTIONS = np.shape(self.P_a)
        assert N_ACTIONS == self.a_dim, "The dimension of the given action space and transitional model does not match."
        
        # init nn model
        self.reward_learner_setup()
        nn_r = self.reward_learner
        trajs = self.trajs
        feature_array = self.feature_map_array
        P_a = self.P_a
        
        # load the model
        if restore:
            print('Restore model from saved file')
            self.loadParams()

        # find state visitation frequencies using demonstrations
        mu_D = self.stateVisitFreq(trajs, N_STATES)

        # set pre-reward
        pre_reward = np.zeros(len(feature_array))
        

        T0 = time.time()
        now_time = datetime.datetime.now()
        print('this loop start at {}'.format(now_time))
        # training
        for iteration in range(n_iters):
            T1 = time.time()
            if iteration % (n_iters/10) == 0:
                print('iteration: {}'.format(iteration))
                # self.modelSave()
            # compute the reward matrix
            rewards = nn_r.get_rewards(feature_array)
            reward_difference = np.mean(normalize(rewards) - pre_reward)
            print("the current reward difference is {}".format(reward_difference))
            if abs(reward_difference) <= 0.001:
                print('the difference of reward is less than 0.001, then break the loop')
                break

            # compute policy
            values, policy = self.value_iteration(
                P_a, rewards, gamma, error=0.1, deterministic=True)
            # self.policySave()
            self.policy = policy
            print("The calculation of value and policy is finished!")
            # compute expected svf
            mu_exp = self.expectStateVisitFreq(
                P_a, gamma, trajs, policy, deterministic=True)
            # compute gradients on rewards:
            grad_r = mu_D - mu_exp
            print("visit frequency difference is {}".format(np.mean(grad_r)))
            # apply gradients to the neural network
            nn_r.apply_grads(feature_array, grad_r)
            # calculate time pass
            T2 = time.time()
            print("this iteration lasts {:.2f},the loop lasts {:.2f}".format(T2-T1, T2-T0))
            # set pre reward
            pre_reward = normalize(rewards)
        # self.modelSave()
        # self.policySave()
        rewards = nn_r.get_rewards(feature_array)
        return normalize(rewards)

class InverseRewardDeepLearning:
    def __init__(self, n_input, lr, n_layer=2, n_units=64, l2=0.5):
        """initialize DeepIRl, construct function between feature and reward

        Args:
            n_input (_type_): number of features
            lr : learning rate
            n_h1 (int, optional): output size of fc1.
            n_h2 (int, optional): output size of fc2.
            l2 (int, optional): l2 loss gradient. Defaults to 0.1.
            name (str, optional): variable scope. Defaults to 'deep_irl_fc'.
        """
        self.n_input = n_input
        self.lr = lr
        self.n_layer = n_layer
        self.n_units = n_units
        self.l2 = l2

        self.network = hk.transform(self._build_network)
        self.params = self.network.init(jax.random.PRNGKey(42), np.zeros((1, n_input)))
        self.opt = optax.adam(lr)
        self.opt_state = self.opt.init(self.params)

    def _build_network(self, x):
        """build forward network with 3 fully connected layers"""
        mlp = hk.Sequential(
            [hk.Linear(self.n_units), jax.nn.elu] * self.n_layer + [hk.Linear(1)]
        )
        return mlp(x)

    def get_rewards(self, states):
        return self.network.apply(self.params, states)

    def apply_grads(self, feat_map, grad_r):
        def loss_fn(params, feat_map, grad_r):
            rewards = self.network.apply(params, feat_map)
            l2_loss = sum(np.sum(np.square(p)) for p in jax.tree_leaves(params))
            reward_loss = np.sum(rewards * grad_r)
            return reward_loss + self.l2 * l2_loss

        grads = jax.grad(loss_fn)(self.params, feat_map, grad_r)
        updates, self.opt_state = self.opt.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)

class InverseRewardLinearLearning(InverseRewardDeepLearning):
    def __init__(self, n_input, lr, l2=0.5):
        """initialize DeepIRl, construct function between feature and reward

        Args:
            n_input (_type_): number of features
            lr : learning rate
            l2 (int, optional): l2 loss gradient. Defaults to 0.1.
            name (str, optional): variable scope. Defaults to 'deep_irl_fc'.
        """
        self.n_input = n_input
        self.lr = lr
        self.l2 = l2

        self.network = hk.transform(self._build_network)
        self.params = self.network.init(jax.random.PRNGKey(42), np.zeros((1, n_input)))
        self.opt = optax.adam(lr)
        self.opt_state = self.opt.init(self.params)

    def _build_network(self, x):
        """build forward network with 3 fully connected layers"""
        return hk.Linear(1)(x)

class irl_linear(irl_maxent):        
    def reward_learner_setup(self, ):
        self.reward_learner = InverseRewardLinearLearning(
            n_input=self.s_dim, lr=1e-4)

class irl_deep(irl_maxent):   
    def __init__(self, 
                inputs: np.array,
                targets: np.array,
                state_dim: int,
                action_dim: int,
                id_feature_mapping: dict,
                state_only: bool = True,
                transition_model: np.array = None,
                seed=41310,
                deep_layers: int = 2,
                deep_units: int = 64,):
        super().__init__(inputs, targets, state_dim, action_dim, 
                         id_feature_mapping, state_only, transition_model, seed)
        self.deep_layers = deep_layers
        self.deep_units = deep_units

    def reward_learner_setup(self, ):
        self.reward_learner = InverseRewardDeepLearning(
            n_input=self.s_dim, lr=1e-4, n_layer=self.deep_layers, 
            n_units=self.deep_units)
        