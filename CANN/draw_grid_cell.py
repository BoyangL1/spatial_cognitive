import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

file_path = './data/place_grid_data.pkl'

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Load the contents from the file
    data = pickle.load(file)

def visualize_and_save_grid_cells(data, directory):
    """
    Visualize each grid cell in the data and save the plots to a specified directory.
    :param data: A dictionary where each key is a coordinate and value is a grid cell (2D array).
    :param directory: Directory where the images will be saved.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, (key, value) in enumerate(data.items()):
        if isinstance(value, (list, np.ndarray)) and len(np.array(value).shape) == 3:
            plt.figure()
            plt.imshow(value[0, :, :], cmap='hot', interpolation='nearest')
            plt.title(f"Grid Cell for Coordinate {key}")
            plt.colorbar()
            file_name = os.path.join(directory, f'grid_cell_first_layer_{i}.png')
            plt.savefig(file_name)
            plt.close()
        else:
            print(f"The value for key {key} is not a 2D array.")

visualize_and_save_grid_cells(data, './img/grid_visulization')
