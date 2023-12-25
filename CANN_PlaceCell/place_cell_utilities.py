import numpy as np
from place_cell import PlaceCell
import matplotlib.pyplot as plt
np.random.seed(1)

def create_place_cells(n_place_cells, anchor_x,anchor_y,sigma):
    """ 
    create place cells
    """
    print('Generating place cells...')
    p_cells = []
    for i in range(0, n_place_cells):
        # randomly create place centers
        x_center, y_center = anchor_x[i],anchor_y[i]
        # create place cell and add to list of other cells
        p_cells.append(PlaceCell(x_center, y_center, sigma))
    return p_cells

def plot_centers(place_cells, save_file):
    """ 
    plot gaussian place cell centers
    """
    def gaussian(x, y, x0, y0, sigma):
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    print('Plotting place cells...')
    x_coords = [cell.x_center for cell in place_cells]
    y_coords = [cell.y_center for cell in place_cells]
    margin = 0 
    x_min, x_max = min(x_coords) - margin, max(x_coords) + margin
    y_min, y_max = min(y_coords) - margin, max(y_coords) + margin
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))
    heatmap = np.zeros_like(grid_x)

    for cell in place_cells:
        heatmap += gaussian(grid_x, grid_y, cell.x_center, cell.y_center, cell.sigma)

    plt.figure(figsize=(8, 8))
    plt.contourf(grid_x, grid_y, heatmap, levels=50, cmap='hot')

    plt.title("Gaussian Heatmaps for Place Cell Centers")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    plt.savefig(save_file)

def evaluate_spiking(place_cell_spiking, place_activity, place_cells, x, y, t_idx):
    """ 
    updates place cell spiking and place activity matrices given spiking at current position x and y at current time index
    """
    for i in range(0, len(place_cells)):
        curr_place_cell = place_cells[i]
        place_cell_spiking[i, t_idx], place_activity[i,t_idx] = curr_place_cell.evaluate_spiking(x[t_idx], y[t_idx])
    return place_cell_spiking, place_activity