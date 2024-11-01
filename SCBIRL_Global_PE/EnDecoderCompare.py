import haiku as hk
from jax import grad, jit, value_and_grad
from jax import random
from jax.example_libraries import optimizers
import jax
import jax.numpy as np
from jax import random
import numpy as onp

from .utils import Padding
from .EnDecoder import create_look_ahead_mask
from .transformer import *

def positional_encoding(inputs):
    
    batch_size, seq_len, _, d_model = inputs.shape
        
    def get_angles(pos, i, d_model):
        angle_rates = 1 / onp.power(10000, (2 * (i//2)) / onp.float32(d_model))
        return pos * angle_rates

    # create the positional encodings: 2D array (seq_len, d_model)
    angle_rads = get_angles(onp.arange(seq_len)[:, np.newaxis],
                            onp.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = onp.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = onp.cos(angle_rads[:, 1::2])

    pos_encoding = np.expand_dims(angle_rads, axis=(0))
    pos_encoding = np.tile(pos_encoding, [batch_size, 1, 1])

    # mask the input with blank with -999
    mask_index = np.any(inputs[:, :, 0, :] == Padding, axis=2)

    # mask the pos_encoding with -999 at the invalid position
    pos_encoding_first = np.where(mask_index[..., np.newaxis], -999, pos_encoding_first)
    # roll the pos_encoding to leftward by 1 at the second axis, the padding position will be filled with -999
    pos_encoding_second = onp.roll(pos_encoding_first, shift=-1, axis=1)
    pos_encoding_second[:, -1, :] = -999
    pos_encoding = np.stack([pos_encoding_first, pos_encoding_second], axis=2)
    return pos_encoding

def encoder_sin_model(inputs, num_layers, num_heads, dff, rate, output_dim, rng):
    # Combine inputs and pe code
    pos_encoding = positional_encoding(inputs)
    inputs = inputs + pos_encoding

    # Initialize transformer layer
    transformer_layers = [TransformerLayer(inputs.shape[-1], num_heads, dff, rate) for _ in range(num_layers)]
    # forward 
    for layer in transformer_layers:
        inputs = layer(inputs, rng)

    final_layer = hk.Linear(output_dim)
    final_output = final_layer(inputs)
    return final_output


def q_network_sin_model(inputs, enc_output, num_layers, num_heads, dff, rate, output_dim, rng):
    # combine inputs and pe code
    pos_encoding = positional_encoding(inputs)
    inputs = inputs + pos_encoding

    # Initialize transformer decoder layer
    transformer_decoder_layers = [TransformerDecoderLayer(inputs.shape[-1], num_heads, dff, rate) for _ in range(num_layers)]
    
    lood_ahead_mask = create_look_ahead_mask(inputs.shape[1]*inputs.shape[2])
    # forward function
    for layer in transformer_decoder_layers:
        inputs = layer(inputs, enc_output, lood_ahead_mask, None, rng)

    final_layer = hk.Linear(output_dim)
    return final_layer(inputs)



def hidden_layers(layers=1, units=64):
    hidden = []
    for i in range(layers):
        hidden += [hk.Linear(units), jax.nn.elu]
    return hidden

def encoder_net_model(inputs, layers=2, units=64, state_only=True, a_dim=None):
    """
    Create an encoder model for fitting posterior probabilities of a reward function.

    Args:
        inputs (jax.numpy):The input tensor representing the state.
        layers (int, optional): The number of hidden layers in the encoder model. Defaults to 2.
        units (int, optional): The number of units (neurons) in each hidden layer. Defaults to 64.
        state_only (bool, optional): If True, encode only the state information; if False, encode state-action pairs. Defaults to True.
        a_dim (int, optional): The dimension of the action space (only needed if state_only is False). Defaults to None.

    Returns:
        outputs posterior probabilities for a reward function. mean and variance
    """    
    out_dim = 2
    if not state_only:
        out_dim = a_dim * 2
    # 线性投射 + 非线性 + 线性投射
    mlp = hk.Sequential(hidden_layers(layers, units) + [hk.Linear(out_dim)])
    
    return mlp(inputs)


def q_network_model(inputs, a_dim, layers=2, units=64):
    """
    Create a Q-network model for reinforcement learning.

    Args:
        inputs (tf.Tensor): The input tensor representing the state.
        a_dim (int): The number of actions in the action space.
        layers (int, optional): The number of hidden layers in the Q-network. Defaults to 2.
        units (int, optional): The number of units (neurons) in each hidden layer. Defaults to 64.

    Returns:
        A Q-network model that takes the state as input and outputs Q-values for each action.
    """
    mlp = hk.Sequential(hidden_layers(layers, units) + [hk.Linear(a_dim)])
    return mlp(inputs)


def encoder_linear_model(inputs, state_only=True, a_dim=None):
    return encoder_net_model(inputs, layers=0, state_only=state_only, a_dim=a_dim)


def q_linear_model(inputs, a_dim):
    return q_network_model(inputs, a_dim)
