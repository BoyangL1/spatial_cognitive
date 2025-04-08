import haiku as hk

import jax.numpy as np
import jax

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


def encoder_model(inputs, positions, num_layers, num_heads, num_scale, dff_ratio, rate, output_dim, rng):
    """
    inputs: 输入特征 [batch_size, seq_len, 2, 10]
    positions: 经纬度坐标 [batch_size, seq_len, 2, 2]
    """
    # 升维层：将10维特征升维到144维
    # 提取position最后一个维度的维数
    position_dim = positions.shape[-1]
    embedding_dim = 2 * (position_dim + 1) * num_scale * num_heads
    feature_embedding_layer = hk.Linear(embedding_dim)
    x = feature_embedding_layer(inputs)
    
    transformer_layers = [TransformerLayer(embedding_dim, num_heads, dff_ratio, use_rotation=True, rate=rate) 
                        for _ in range(num_layers)]
    
    for layer in transformer_layers:
        x = layer(x, positions, rng)

    final_layer = hk.Linear(output_dim)
    return final_layer(x)

def create_look_ahead_mask(size):
    mask = np.triu(np.ones((size, size)), k=1)
    mask = mask[np.newaxis, np.newaxis, ...]  # [1, 1, size, size]
    return mask

def q_network_model(inputs, positions, enc_output, num_layers, num_heads, num_scale, dff_ratio, rate, output_dim, rng):
    """
    inputs: 输入特征 [batch_size, seq_len, 2, 10]
    positions: 经纬度坐标 [batch_size, seq_len, 2, 2]
    """
    # 升维层：将10维特征升维到144维
    # 提取position最后一个维度的维数
    position_dim = positions.shape[-1]
    embedding_dim = 2 * (position_dim + 1) * num_scale * num_heads
    feature_embedding_layer = hk.Linear(embedding_dim)
    x = feature_embedding_layer(inputs)
    
    transformer_decoder_layers = [TransformerDecoderLayer(embedding_dim, num_heads, dff_ratio, use_rotation=True, rate=rate) 
                                for _ in range(num_layers)]
    
    look_ahead_mask = create_look_ahead_mask(inputs.shape[1]*inputs.shape[2]) 
    
    for layer in transformer_decoder_layers:
        x = layer(x, enc_output, look_ahead_mask, None, positions, rng)

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
