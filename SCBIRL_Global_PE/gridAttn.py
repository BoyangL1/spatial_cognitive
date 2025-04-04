import jax
import jax.numpy as np
from jax import random
import haiku as hk

# === Grid Cell Positional Encoding ===
class GridCellPositionalEncoding(hk.Module):
    def __init__(self, dimension, qk_dim, num_heads):
        """
        Initialize the Grid Cell Positional Encoding using JAX.

        Parameters:
        - dimension: int, spatial dimensionality (e.g., 2 for 2D space)
        - embedding_dim: int, the total embedding dimension (must be divisible by 2*(dimension+1))
        """
        super().__init__()
        self.dimension = dimension
        self.embedding_dim = qk_dim
        self.num_heads = num_heads
        self.num_head_dim = self.embedding_dim // num_heads

        assert self.num_head_dim % (2 * (self.dimension + 1)) == 0, "per num head embedding_dim must be divisible by 2*(dimension+1)"

        self.num_scales = self.num_head_dim // (2 * (dimension + 1)) # number of scales
        self.n = self.dimension + 1 if self.dimension > 1 else 1

    def _generate_simplex_vectors_with_projection(self, dimension, key, apply_random_rotation=True):
        """
        Generate n wave vectors (omega) in an n-dimensional space using a regular simplex projection.
        """
        if dimension == 1:
            return np.array([[1.0]], dtype=np.float32)  # 1D case

        points = np.eye(dimension + 1, dtype=np.float32)
        points -= points.mean(axis=0)
        U, _, _ = np.linalg.svd(points.T, full_matrices=False)
        reduced_vectors = U[:, :-1]
        reduced_vectors /= np.linalg.norm(reduced_vectors, axis=1, keepdims=True)
        if apply_random_rotation:
            Q, _ = np.linalg.qr(random.normal(key, (dimension, dimension)))
            reduced_vectors = reduced_vectors @ Q.T
        return reduced_vectors

    def _generate_batch_encoding(self, positions, key):
        """
        Generate multi-scale positional encoding for batch input positions.

        Parameters:
        - positions: (batch_size, seq_length, d) tensor, input position vectors.

        Returns:
        - theta: (batch_size, num_head, seq_length, n, num_scales) real-valued tensor.
        """
        theta_heads = []
        mag = 1 / (10000 ** (2 * self.n * np.arange(self.num_scales, dtype=np.float32)[:, None] / self.num_head_dim))
        for i in range(self.num_heads):
            subkey = random.fold_in(key, i)
            omega = self._generate_simplex_vectors_with_projection(self.dimension, subkey)
            theta = np.einsum('bsd,nd->bsn', positions, omega)[..., None] * mag.T # (B, N, n, 1) * (1, S)
            theta_heads.append(theta)
        theta_heads = np.stack(theta_heads, axis=0)
        theta_heads = np.transpose(theta_heads, (1, 0, 2, 3, 4))
        return theta_heads

    def _compute_rotation_vectors(self, theta):
        """
        Convert positional encoding angles directly into cosine and sine components.

        Parameters:
        - theta: (batch_size, num_head, seq_length, n, S) tensor of angles.

        Returns:
        - cos_vec: (batch_size, num_head, seq_length, n, S) cosine components.
        - sin_vec: (batch_size, num_head, seq_length, n, S) sine components.
        """
        return np.cos(theta), np.sin(theta)

    def _apply_rotation(self, q, cos_vec, sin_vec):
        """
        Apply the rotation transformation similar to RoPE in a memory-efficient manner.

        Parameters:
        - q: (batch_size, num_heads, seq_length, 2 * n * S) Input tensor representing token embeddings.
        - cos_vec: (batch_size, num_heads, seq_length, n, S) Cosine components.
        - sin_vec: (batch_size, num_heads, seq_length, n, S) Sine components.

        Returns:
        - q_rotated: (batch_size, num_heads, seq_length, 2 * n * S) Output tensor after transformation.
        """
        B, H, N, D = q.shape
        n, S = cos_vec.shape[-2:]

        assert D == 2 * n * S, f"Expected last dim {D} == 2 * n * S = {2 * n * S}"

        # Reshape q to (B, H, N, n, S, 2)
        q = q.reshape(B, H, N, n, S, 2)
        q_even = q[..., 0]  # (B, H, N, n, S)
        q_odd = q[..., 1]   # (B, H, N, n, S)

        # Apply complex rotation (cosθ + i·sinθ)
        q_rotated_even = q_even * cos_vec - q_odd * sin_vec
        q_rotated_odd = q_even * sin_vec + q_odd * cos_vec

        q_rot = np.stack([q_rotated_even, q_rotated_odd], axis=-1)  # (B, H, N, n, S, 2)
        q_rot = q_rot.reshape(B, H, N, 2 * n * S)  # Flatten back
        return q_rot

    def __call__(self, positions, q, k, key):
        """
        Compute the full grid cell positional encoding and apply rotation transformation.

        Parameters:
        - positions: (batch_size, seq_length, d) tensor, input position vectors.
        - q: (batch_size, num_heads, seq_length, 2nS) tensor, input token embeddings.

        Returns:
        - q(k)_rotated: (batch_size, num_heads, seq_length, 2nS) tensor, output after transformation.
        """
        # 确保位置编码计算正确使用经纬度信息
        theta = self._generate_batch_encoding(positions, key)
        cos_vec, sin_vec = self._compute_rotation_vectors(theta)
        q_rotated = self._apply_rotation(q, cos_vec, sin_vec)
        k_rotated = self._apply_rotation(k, cos_vec, sin_vec)
        return q_rotated, k_rotated

# === Attention Layers ===
class Attention(hk.Module):
    """
    Multi-head Attention block in general.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = hk.Linear(dim * 3, with_bias=qkv_bias)
        self.attn_drop = hk.dropout
        self.proj = hk.Linear(dim)
        self.proj_drop = hk.dropout

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = np.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = (q @ np.transpose(k, (0, 1, 3, 2)))
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(0.1, attn)

        x = (attn @ v)
        x = np.transpose(x, (0, 2, 1, 3))
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(0.1, x)
        return x

class GridPEAttention(hk.Module):
    """Multi-head Attention block with GridCell Positional Encoding (RoPE style)."""
    def __init__(self, qk_dim, dimension=2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = qk_dim
        self.num_heads = num_heads
        self.head_dim = qk_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = hk.Linear(qk_dim * 3, with_bias=qkv_bias)
        self.attn_drop = hk.dropout
        self.proj = hk.Linear(qk_dim)
        self.proj_drop = hk.dropout

        self.grid_pe = GridCellPositionalEncoding(
            dimension=dimension,
            qk_dim=qk_dim,
            num_heads=num_heads
        )

    def __call__(self, x, positions, key):
        """
        Parameters:
        - x: (B, N, C) input tokens
        - positions: (B, N, d) spatial/temporal positions
        """
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = np.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ==== Rotary Positional Encoding ====
        q_rot, k_rot = self.grid_pe(positions, q, k, key)
  
        # ==== Attention ====
        q_rot = q_rot * self.scale  # scaled dot-product
        attn = (q_rot @ np.transpose(k_rot, (0, 1, 3, 2)))  # (B, H, N, N)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(key, 0.1, attn)

        out = (attn @ v)
        out = np.transpose(out, (0, 2, 1, 3))
        out = out.reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(key, 0.1, out)
        return out

# === Test Script ===
if __name__ == '__main__':
    import jax.numpy as np
    import haiku as hk

    # Parameters
    dimension = 2
    num_heads = 4
    qk_dim = 96  # per head 96 / 4 = 24
                 # 24 = 2 * (2+1) * S => S = 4（4 num scales）

    batch_size = 2
    seq_len = 5
    x = np.random.randn(batch_size, seq_len, qk_dim)
    positions = np.random.randn(batch_size, seq_len, dimension)

    attn = GridPEAttention(qk_dim=qk_dim, dimension=dimension, num_heads=num_heads)
    out = attn(x, positions)
    print("Output shape:", out.shape)  # (B, N, C)