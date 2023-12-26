import jax.numpy as np
import jax
import haiku as hk

import numpy as onp

class BasicCNN(hk.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=5, stride=2, padding='SAME')
        self.conv2 = hk.Conv2D(output_channels=64, kernel_shape=3, stride=2, padding='SAME')
        self.flatten = hk.Flatten()
        self.dense = hk.Linear(output_size)

    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.dense(x)
        return x

def process_grid_code(grid_code, output_size):
    batch_size, sequence_size, state_steps = grid_code.shape[:3]
    grid_code_reshaped = np.reshape(grid_code, [-1, *grid_code.shape[3:]]) 
    cnn = BasicCNN(output_size)
    cnn_output = cnn(grid_code_reshaped)
    cnn_output_reshaped = np.reshape(cnn_output, [batch_size, sequence_size, state_steps, -1])
    return cnn_output_reshaped