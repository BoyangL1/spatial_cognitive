import haiku as hk
import jax
import jax.numpy as jnp

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
        matmul_qk = jnp.matmul(query, key.transpose(0, 1, 3, 2))

        dk = jnp.float32(self.depth)
        scaled_attention_logits = matmul_qk / jnp.sqrt(dk)

        attention_weights = jax.nn.softmax(scaled_attention_logits, axis=-1)

        output = jnp.matmul(attention_weights, value)
        return output, attention_weights

    def __call__(self, x):
        batch_size = x.shape[0]

        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        attention, _ = self.attention(query, key, value)

        attention = attention.transpose(0, 2, 1, 3)
        concat_attention = attention.reshape(batch_size, -1, self.d_model)

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

    def __call__(self, x, is_training):
        attn_output = self.mha(x)
        attn_output = self.dropout(attn_output, self.rate, is_training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, self.rate, is_training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
