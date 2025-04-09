import haiku as hk
import jax
import jax.numpy as np
from .gridAttn import GridCellPositionalEncoding

class MultiHeadSelfGridAttention(hk.Module):
    def __init__(self, d_model, num_heads, use_rotation=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // self.num_heads
        self.use_rotation = use_rotation

        self.wq = hk.Linear(self.d_model)
        self.wk = hk.Linear(self.d_model)
        self.wv = hk.Linear(self.d_model)
        self.dense = hk.Linear(self.d_model)
        
        if use_rotation:
            self.grid_pe = GridCellPositionalEncoding(
                dimension=2,
                qk_dim=d_model,
                num_heads=num_heads
            )

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)

    def attention(self, query, key, value):        
        matmul_qk = np.matmul(query, key.transpose(0, 1, 3, 2))
        dk = np.float32(self.depth)
        scaled_attention_logits = matmul_qk / np.sqrt(dk)
        attention_weights = jax.nn.softmax(scaled_attention_logits, axis=-1)
        output = np.matmul(attention_weights, value)
        return output, attention_weights

    def __call__(self, x, positions, rng):
        batch_size, seq_len, _, _ = x.shape

        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        # here the shape is (batch_size, num_heads, 2 * seq_len, depth)
        query = self.split_heads(query, batch_size) 
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        if self.use_rotation:
            # 应用网格细胞位置编码
            position_dim = positions.shape[-1]
            # here the shape turns to (batch_size, 2 * seq_len, 2)
            positions = positions.reshape(batch_size, -1, position_dim) 
            query_rot, key_rot = self.grid_pe(positions, query, key, rng)
            attention, _ = self.attention(query_rot, key_rot, value)
        else:
            attention, _ = self.attention(query, key, value)

        attention = attention.transpose(0, 2, 1, 3)
        concat_attention = attention.reshape(batch_size, seq_len, -1, self.d_model)
        output = self.dense(concat_attention)
        return output

class PointWiseFeedForwardNetwork(hk.Module):
    def __init__(self, d_model, dff):
        super().__init__()
        self.dense1 = hk.Linear(dff)
        self.dense2 = hk.Linear(d_model)

    def __call__(self, x):
        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        return x

class TransformerLayer(hk.Module):
    def __init__(self, d_model, num_heads, dff_ratio, use_rotation=True, rate=0.1):
        super().__init__()
        dff = d_model * dff_ratio
        self.mha = MultiHeadSelfGridAttention(d_model, num_heads, use_rotation=use_rotation)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)
        self.layernorm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.layernorm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.dropout = hk.dropout
        self.rate = rate

    def __call__(self, x, positions, rng):
        attn_rng, ffn_rng = jax.random.split(rng)
        
        attn_output = self.mha(x, positions, attn_rng)
        attn_output = self.dropout(attn_rng, self.rate, attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_rng, self.rate, ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2



"********************Decoder Layer******************"

class MultiHeadGridAttention(hk.Module):
    def __init__(self, d_model, num_heads, use_rotation=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // self.num_heads
        self.use_rotation = use_rotation

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.wq = hk.Linear(self.d_model)
        self.wk = hk.Linear(self.d_model)
        self.wv = hk.Linear(self.d_model)
        self.dense = hk.Linear(self.d_model)

        if use_rotation:
            self.grid_pe = GridCellPositionalEncoding(
                dimension=2,
                qk_dim=d_model,
                num_heads=num_heads
            )

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return np.transpose(x, (0, 2, 1, 3))

    def scaled_dot_product_attention(self, query, key, value, mask):
        """Calculate the attention weights."""
        matmul_qk = np.matmul(query, key.transpose(0, 1, 3, 2))

        # Scale matmul_qk
        dk = np.float32(key.shape[-1])
        scaled_attention_logits = matmul_qk / np.sqrt(dk)

        # Add the mask to the scaled tensor.
        if mask is not None:
            # 确保掩码可以广播到所有批次和注意力头
            scaled_attention_logits = scaled_attention_logits + (mask * -1e9)

        # Softmax is normalized on the last axis (seq_len_k)
        attention_weights = jax.nn.softmax(scaled_attention_logits, axis=-1)

        output = np.matmul(attention_weights, value)
        return output, attention_weights

    def __call__(self, query, key, value, mask=None, positions=None, rng=None):
        batch_size, seq_len, _, _ = query.shape

        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        if self.use_rotation and positions is not None and rng is not None:
            # 应用网格细胞位置编码
            position_dim = positions.shape[-1]
            positions = positions.reshape(batch_size, -1, position_dim)
            query_rot, key_rot = self.grid_pe(positions, query, key, rng)
            attention, _ = self.scaled_dot_product_attention(query_rot, key_rot, value, mask)
        else:
            attention, _ = self.scaled_dot_product_attention(query, key, value, mask)

        attention = np.transpose(attention, (0, 2, 1, 3))
        concat_attention = attention.reshape(batch_size, seq_len, -1, self.d_model)

        output = self.dense(concat_attention)
        return output


class TransformerDecoderLayer(hk.Module):
    def __init__(self, d_model, num_heads, dff_ratio, use_rotation=True, rate=0.1):
        super().__init__()
        dff = d_model * dff_ratio
        self.mha1 = MultiHeadGridAttention(d_model, num_heads, use_rotation=use_rotation)
        self.mha2 = MultiHeadGridAttention(d_model, num_heads, use_rotation=False)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)
        self.layernorm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.layernorm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.layernorm3 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.dropout = hk.dropout
        self.rate = rate

    def __call__(self, x, enc_output, look_ahead_mask, padding_mask, positions, rng):
        attn_rng, ffn_rng = jax.random.split(rng)
        
        # 第一个注意力层：自注意力
        attn1 = self.mha1(x, x, x, look_ahead_mask, positions, attn_rng)
        attn1 = self.dropout(attn_rng, self.rate, attn1)
        out1 = self.layernorm1(x + attn1)

        # 第二个注意力层：编码器-解码器注意力
        attn2 = self.mha2(out1, enc_output, enc_output, padding_mask, positions, attn_rng)
        attn2 = self.dropout(attn_rng, self.rate, attn2)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout(ffn_rng, self.rate, ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3

