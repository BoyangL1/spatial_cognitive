import haiku as hk

import jax.numpy as np

from .transformer import *

def compress_pe_code_complex(pe_code, target_dim):
    pe_code_real = np.real(pe_code)
    pe_code_imag = np.imag(pe_code)

    # comprass real and imag
    linear_pe = hk.Linear(output_size=target_dim)
    
    # reshape concatenated_pe_code
    original_shape = pe_code_real.shape[:-1]
    depth = pe_code_real.shape[-1]

    pe_real_compressed = linear_pe(pe_code_real.reshape(-1, depth))
    pe_imag_compressed = linear_pe(pe_code_imag.reshape(-1,depth))
    
    new_shape = original_shape + (target_dim,)
    pe_real_compressed = pe_real_compressed.reshape(*new_shape)
    pe_imag_compressed = pe_imag_compressed.reshape(*new_shape)

    return pe_real_compressed,pe_imag_compressed


def encoder_model(inputs, positions, num_layers, num_heads, dff, rate, output_dim, rng):
    """
    inputs: 输入特征
    positions: 经纬度坐标 [batch_size, seq_len, 2, 2]
    """
    # 删除原来的位置编码压缩步骤
    # 直接使用transformer层
    transformer_layers = [TransformerLayer(inputs.shape[-1], num_heads, dff, rate) 
                        for _ in range(num_layers)]
    
    x = inputs
    for layer in transformer_layers:
        x = layer(x, positions, rng)  # transformer层会使用GridCellPositionalEncoding

    final_layer = hk.Linear(output_dim)
    return final_layer(x)

def create_look_ahead_mask(size):
    mask = np.triu(np.ones((size, size)), k=1)
    mask = mask[np.newaxis, np.newaxis, ...]
    return mask

def q_network_model(inputs, positions, enc_output, num_layers, num_heads, dff, rate, output_dim, rng):
    """
    inputs: 输入特征
    positions: 经纬度坐标 [batch_size, seq_len, 2]
    """
    # 初始化transformer decoder层
    transformer_decoder_layers = [TransformerDecoderLayer(inputs.shape[-1], num_heads, dff, rate) 
                                for _ in range(num_layers)]
    
    look_ahead_mask = create_look_ahead_mask(inputs.shape[1]*inputs.shape[2])
    
    # forward pass
    x = inputs
    for layer in transformer_decoder_layers:
        x = layer(x, positions, enc_output, look_ahead_mask, None, rng)

    final_layer = hk.Linear(output_dim)
    return final_layer(x)

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
