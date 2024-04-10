import haiku as hk
import jax
import jax.numpy as np

class MultiHeadSelfAttention(hk.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // self.num_heads

        self.wq = hk.Linear(self.d_model)
        self.wk = hk.Linear(self.d_model)
        self.wv = hk.Linear(self.d_model)

        self.dense = hk.Linear(self.d_model)

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

    def __call__(self, x):
        batch_size, seq_len, _ , _ = x.shape

        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

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
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)

        self.layernorm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.layernorm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        self.dropout = hk.dropout

        self.rate = rate

    def __call__(self, x, rng):
        attn_rng, ffn_rng = jax.random.split(rng)  

        attn_output = self.mha(x)
        attn_output = hk.dropout(attn_rng, self.rate, attn_output)  
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = hk.dropout(ffn_rng, self.rate, ffn_output)  
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

"********************Decoder Layer******************"

class MultiHeadAttention(hk.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.wq = hk.Linear(self.d_model)  # Linear transformation for queries
        self.wk = hk.Linear(self.d_model)  # Linear transformation for keys
        self.wv = hk.Linear(self.d_model)  # Linear transformation for values

        self.dense = hk.Linear(self.d_model)  # Final linear layer

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
            scaled_attention_logits += (mask * -1e9)

        # Softmax is normalized on the last axis (seq_len_k)
        attention_weights = jax.nn.softmax(scaled_attention_logits, axis=-1)

        output = np.matmul(attention_weights, value)
        return output, attention_weights

    def __call__(self, query, key, value, mask=None):
        batch_size, seq_len, _ , _ = query.shape

        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        attention, _ = self.scaled_dot_product_attention(query, key, value, mask)

        attention = np.transpose(attention, (0, 2, 1, 3))
        concat_attention = attention.reshape(batch_size, seq_len, -1, self.d_model)

        output = self.dense(concat_attention)
        return output


class TransformerDecoderLayer(hk.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads) 
        self.mha2 = MultiHeadAttention(d_model, num_heads) 
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)

        self.layernorm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.layernorm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.layernorm3 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        self.dropout = hk.dropout
        self.rate = rate

    def __call__(self, x, enc_output, look_ahead_mask, padding_mask, rng):
        attn1 = self.mha1(x, x, x, look_ahead_mask) # decoder attention
        attn1 = self.dropout(rng, self.rate, attn1)
        out1 = self.layernorm1(x + attn1)

        attn2 = self.mha2(out1, enc_output, enc_output, padding_mask) # en-decoder self attention
        attn2 = self.dropout(rng, self.rate, attn2)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout(rng, self.rate,ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3
