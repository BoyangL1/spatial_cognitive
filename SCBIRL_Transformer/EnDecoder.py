import haiku as hk

import jax.numpy as np

from .transformer import *
from .BasicCNN import *

def encoder_model(inputs, grid_code, num_layers, num_heads, dff, rate, output_dim, rng, cnn_output_size = 16):
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

def q_network_model(inputs, enc_output, grid_code, num_layers, num_heads, dff, rate, output_dim, rng, cnn_output_size = 16):
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