if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "SCBIRL_Global_PE"
    
import pickle
import numpy as onp

import haiku as hk
import jax
import jax.numpy as np
from jax import jit, value_and_grad, random
from jax.example_libraries import optimizers
from tqdm import tqdm

from .EnDecoder import *
from .transformer import *
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
        positions: np.array,
        state_dim: int,
        action_dim: int,
        state_only: bool = True,
        num_layers: int = 2,
        num_heads: int = 2,
        num_scale: int = 32,
        dff_ratio: int = 2,
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
        positions: np.array
            Grid code data of size [num_traj x npair_per_traj x 2 x position_dim]
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
        self.positions = positions
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.state_only = state_only
        self.encoder_o_dim = 2

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_scale = num_scale
        self.dff_ratio = dff_ratio
        self.rate = rate

        self.e_params = self.encoder.init(
            self.key, 
            inputs, positions, num_layers, num_heads, num_scale, dff_ratio, rate, self.encoder_o_dim, self.key
        )

        enc_output = random.normal(self.key, inputs.shape[:-1] + (2,))
        self.q_params = self.q_network.init(
            self.key, 
            inputs, positions, enc_output, num_layers, num_heads, num_scale, dff_ratio, rate, action_dim, self.key
        )

        self.params = (self.e_params, self.q_params)

        self.load_params = False
        self.pre_params = None
        return
    
    def modelSave(self, model_save_path):
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

    def reward(self,state,positions):
        #  Returns reward function parameters for a given state
        r_par = self.encoder.apply(
                self.e_params,
                self.key,
                state,
                positions,
                self.num_layers,
                self.num_heads, 
                self.num_scale,
                self.dff_ratio,
                self.rate,
                self.encoder_o_dim,
                self.key
            )
        r_par = np.squeeze(r_par,axis = 2)
        return r_par
    
    def QValue(self,state, positions):
        enc_output = self.encoder.apply(
                self.e_params,
                self.key,
                state,
                positions,
                self.num_layers,
                self.num_heads,
                self.num_scale,
                self.dff_ratio,
                self.rate,
                self.encoder_o_dim,
                self.key
            )
        
        q_values = self.q_network.apply(
            self.q_params,
            self.key,
            state,
            positions,
            enc_output,
            self.num_layers,
            self.num_heads,
            self.num_scale,
            self.dff_ratio,
            self.rate,
            self.a_dim,
            self.key
        )
        q_values = np.squeeze(q_values,axis=2)
        return q_values

    def elbo(self, params, key, inputs, targets, positions, weights = None):
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
                positions[:, :, state_dim, np.newaxis, :],
                self.num_layers,
                self.num_heads,
                self.num_scale,
                self.dff_ratio,
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
        e_params, q_params = params
        
        # calculate the kl difference between current reward and pre reward 
        means, log_sds, enc_output = getRewardParameters(e_params, 0)
        # calculate td error
        # calculate Q-values for current state
        q_values = self.q_network.apply(
            q_params,
            key,
            inputs[:, :, 0, np.newaxis, :],
            enc_output,
            positions[:, :, 0, np.newaxis, :],
            self.num_layers,
            self.num_heads,
            self.num_scale,
            self.dff_ratio,
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
            positions[:, :, 1, np.newaxis, :],
            self.num_layers,
            self.num_heads,
            self.num_scale,
            self.dff_ratio,
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
            e_params_pre, _ = self.pre_params
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
        positions = self.positions
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

            lik, g_params = loss_grad(params, key, inputs[indexs], targets[indexs], positions[indexs], weights = weights)

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

if __name__ == "__main__":
    # 模拟超参数
    num_traj = 10
    npair_per_traj = 5
    state_dim = 10
    action_dim = 3
    position_dim = 2

    # 构造假数据（随机）
    inputs = onp.random.randn(num_traj, npair_per_traj, 2, state_dim).astype(onp.float32)
    targets = onp.random.randint(0, action_dim, size=(num_traj, npair_per_traj, 2, 1)).astype(onp.float32)
    positions = onp.random.randn(num_traj, npair_per_traj, 2, position_dim).astype(onp.float32)

    # 转为 JAX 数组
    inputs = np.array(inputs)
    targets = np.array(targets)
    positions = np.array(positions)

    # 初始化模型
    model = avril(inputs, targets, positions, state_dim, action_dim)

    # 训练模型
    print("Training model on synthetic data...")
    model.train(iters=200, batch_size=4)

    # 测试 reward 输出
    test_state = inputs[:1, :, 0, :]  # 一个轨迹的第一个状态序列
    test_pos = positions[:1, :, 0, :]
    r = model.reward(test_state, test_pos)
    print("\nSample reward output shape:", r.shape)
    print("Sample reward output:", r)

    # 测试 Q 值输出
    q = model.QValue(test_state, test_pos)
    print("\nSample Q-value output shape:", q.shape)
    print("Sample Q-value output:", q)