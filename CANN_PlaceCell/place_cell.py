from scipy.stats import uniform
import numpy as np


class PlaceCell:
    # initializes cell with fields and gaussian parameter sigma
    def __init__(self, x_center, y_center, sigma):
        self.x_center = x_center
        self.y_center = y_center
        self.sigma = sigma

    # getters
    def get_centers(self):
        return self.x_center, self.y_center

    # evalutes spiking state based on current position
    def evaluate_spiking(self, x, y):
        rand_val = uniform.rvs()  # random float for simulated spiking noise
        curr_place_val = (1/(2*np.pi*self.sigma**2)) * np.exp(-((x -self.x_center)**2 + (y - self.y_center)**2) / (2*self.sigma**2))
        max_val = 1/(2*np.pi*self.sigma**2)
        # normalize to between 0 and 1 - based on sigma=5.0
        curr_place_val = (curr_place_val / max_val)
        # normalize to between 0 and 0.1 
        curr_place_val = 0.1 * curr_place_val
        # normalize to 500 * dt/tau * activ_level, output is Hz
        curr_place_val = curr_place_val * 500 * (1.0/10.0)
        # check if cell does spike
        if (curr_place_val >= rand_val):
            return 1, curr_place_val
        else:
            return 0, curr_place_val
