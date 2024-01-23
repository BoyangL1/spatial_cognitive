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

from .transformer import *
from .EnDecoder import *
from .utils import *
from .migrationProcess import *

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
        pe_code: np.array,
        state_dim: int,
        action_dim: int,
        state_only: bool = True,
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
        self.pe_code = pe_code
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.state_only = state_only
        self.encoder_o_dim = 2

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff 
        self.rate = rate

        self.e_params = self.encoder.init(
            self.key, inputs, pe_code, num_layers, num_heads,dff, rate, self.encoder_o_dim, self.key
        )

        enc_output = random.normal(self.key, inputs.shape[:-1] + (2,))
        self.q_params = self.q_network.init(
            self.key, inputs, enc_output, pe_code, num_layers, num_heads, dff, rate, action_dim, self.key
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
            self.load_params = True
            self.params = pickle.load(f) 
            self.pre_params = self.params
            self.e_params = self.params[0]
            self.q_params = self.params[1]

    def reward(self,state,pe_code):
        #  Returns reward function parameters for a given state
        r_par = self.encoder.apply(
                self.e_params,
                self.key,
                state,
                pe_code,
                self.num_layers,
                self.num_heads,
                self.dff,
                self.rate,
                self.encoder_o_dim,
                self.key
            )
        r_par = np.squeeze(r_par,axis = 2)
        return r_par
    
    def QValue(self,state,pe_code):
        enc_output = self.encoder.apply(
                self.e_params,
                self.key,
                state,
                pe_code,
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
            pe_code,
            self.num_layers,
            self.num_heads,
            self.dff,
            self.rate,
            self.a_dim,
            self.key
        )
        q_values = np.squeeze(q_values,axis=2)
        return q_values

    def elbo(self, params, key, inputs, targets, pe_code):
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
                pe_code[:, :, state_dim, np.newaxis, :],
                self.num_layers,
                self.num_heads,
                self.dff,
                self.rate,
                self.encoder_o_dim,
                self.key
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
            pe_code[:, :, 0, np.newaxis, :],
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
            pe_code[:, :, 1, np.newaxis, :],
            self.num_layers,
            self.num_heads,
            self.dff,
            self.rate,
            self.a_dim,
            self.key
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
        pe_code = self.pe_code

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

            lik, g_params = loss_grad(params, key, inputs[indexs], targets[indexs], pe_code[indexs])

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