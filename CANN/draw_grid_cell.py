import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

file_path = './data/coords_grid_data.pkl'

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Load the contents from the file
    coords_data = pickle.load(file)

def visualize_and_save_grid_cells(data, directory):
    """
    Visualize each grid cell in the data and save the plots to a specified directory.
    :param data: A dictionary where each key is a coordinate and value is a grid cell (2D array).
    :param directory: Directory where the images will be saved.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, (key, value) in enumerate(tqdm(data.items())):
        if isinstance(value, (list, np.ndarray)) and len(np.array(value).shape) == 3:
            plt.figure()
            plt.imshow(value[3, :, :], cmap='hot', interpolation='nearest')
            plt.title(f"Grid Cell for Coordinate {key}")
            plt.colorbar()
            file_name = os.path.join(directory, f'grid_cell_{i}.png')
            plt.savefig(file_name)
            plt.close()
        else:
            print(f"The value for key {key} is not a 2D array.")

visualize_and_save_grid_cells(coords_data, './img/grid_visulization')
