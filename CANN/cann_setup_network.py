import pyfftw
import numpy as np
from scipy import stats, io, signal, interpolate
import time
from . import quadflip as qp
import os

UNDEF = -999

def set_inhib_length(h_grid, lexp, lmin, lmax):
    """ 
    Sets up inhibition radius across the layers
    """

    if h_grid == 1:

        l_inhibition = lmin

    else:
        l_inhibition = np.zeros(h_grid)  # initialize array for number of layers

        for k in range(0, h_grid):
            # 6/14/18 Update: have to change both to float since both start as int
            zz = float(k) / float(h_grid-1)

            if lexp == 0.:
                l_inhibition[k] = lmin + (lmax - lmin)*zz

            else:
                l_inhibition[k] = (lmin**lexp + (lmax**lexp - lmin**lexp)*zz)**(1/lexp)

    return l_inhibition

def w_value(x, y, l_value, wmag):
    """ 
    General Equation for inhibition
    """
    r = np.sqrt(x*x+y*y)

    if r < 2 * l_value:
        w = -wmag/(2*l_value*l_value) * (1 - np.cos(np.pi*r/l_value))
    else:
        w = 0.

    return w

def setup_recurrent(h, npad, l_inhibition, wmag, wshift):
    """ 
    Set up recurrent continuous attractor network for gird cell
    """
    x = np.arange(-npad//2, npad//2+1)
    y = np.arange(-npad//2, npad//2+1)

    a_row = np.zeros((h, 1, npad))
    a_column = np.zeros((h, npad, 1))

    w = np.zeros((h, npad+1, npad+1))

    for k in range(0, h, 1):
        for i in range(0, npad+1, 1):
            for j in range(0, npad+1, 1):
                w[k, i, j] = w_value(x[i], y[j], l_inhibition[k], wmag) if h != 1 else w_value(x[i], y[j], l_inhibition, wmag)

    if npad % 2 == 0:
        mid = npad // 2
        w = np.delete(w, mid, axis=1)
        w = np.delete(w, mid, axis=2) 

    if wshift > 0:
        w_ltemp = np.delete(w, 0, 2)
        w_ltemp = np.append(w_ltemp, a_column, 2)
        w_rtemp = np.delete(w, npad-1, 2)
        w_rtemp = np.append(a_column, w_rtemp, 2)
        w_dtemp = np.delete(w, npad-1, 1)
        w_dtemp = np.append(a_row, w_dtemp, 1)
        w_utemp = np.delete(w, 0, 1)
        w_utemp = np.append(w_utemp, a_row, 1)

        for k in range(0, h, 1):
            w_rtemp[k, :, :] = qp.quadflip(w_rtemp[k, :, :])
            w_ltemp[k, :, :] = qp.quadflip(w_ltemp[k, :, :])
            w_utemp[k, :, :] = qp.quadflip(w_utemp[k, :, :])
            w_dtemp[k, :, :] = qp.quadflip(w_dtemp[k, :, :])

    else:
        print("Assuming no shift on inhibitory kernel!")
        w_rtemp = w_ltemp = w_utemp = w_dtemp = w

        for k in range(0, h, 1):

            w_rtemp[k, :, :] = qp.quadflip(w_rtemp[k, :, :])
            w_ltemp[k, :, :] = qp.quadflip(w_ltemp[k, :, :])
            w_utemp[k, :, :] = qp.quadflip(w_utemp[k, :, :])
            w_dtemp[k, :, :] = qp.quadflip(w_dtemp[k, :, :])

    shapein = [npad, npad]
    shapeout = [npad, npad//2+1]
    w_fft_arrayin = pyfftw.empty_aligned(shapein, dtype='float64')
    w_fft_arrayout = pyfftw.empty_aligned(shapeout, dtype='complex128')
    w_fft_plan = pyfftw.FFTW(w_fft_arrayin, w_fft_arrayout, axes=(0, 1))

    w_r = np.zeros((h, npad, npad//2+1), dtype='complex128')
    w_l = np.zeros((h, npad, npad//2+1), dtype='complex128')
    w_u = np.zeros((h, npad, npad//2+1), dtype='complex128')
    w_d = np.zeros((h, npad, npad//2+1), dtype='complex128')

    for k in range(0, h, 1):

        w_r[k, :, :] = w_fft_plan(w_rtemp[k, :, :])
        w_l[k, :, :] = w_fft_plan(w_ltemp[k, :, :])
        w_u[k, :, :] = w_fft_plan(w_utemp[k, :, :])
        w_d[k, :, :] = w_fft_plan(w_dtemp[k, :, :])

    return w, w_r, w_l, w_u, w_d

def setup_input(amag, n_grid, falloff_high, falloff_low, falloff):
    """
    Set up input distribution

    Args:
        amag : the strenght of input
        n_grid : the length of grid cells
        falloff_high : the upper bounds of stimulus attenuation.
        falloff_low : the lower bounds of stimulus attenuation.
        falloff : controls the rate at which the stimulus decays with distance.

    Returns:
        array: stimulus attenuation per grid
    """
    a = np.zeros((n_grid, n_grid))  # initialize a to be size of neural sheet
    scaled = np.zeros(n_grid)

    for i in range(0, n_grid, 1):  # iterate over the number of neurons
        # /(n/2.); # sets up scaled axis with 0 at center
        scaled[i] = (i-n_grid/2.+0.5)

    for i in range(0, n_grid, 1):  # iterate over full sheet which is nxn neurons
        for j in range(0, n_grid, 1):

            # altered to align with paper, however they are equivalent
            r = np.sqrt(scaled[i]*scaled[i]+scaled[j]*scaled[j])/(n_grid/2.)

            if falloff_high != UNDEF and r >= falloff_high:
                # If there is an upperlimit to spread and the spot on the sheet exceeds this, no hippocampal input is received
                a[i][j] = 0

            elif falloff_low == UNDEF:  # this is case in code currently
                if np.abs(r) < 1:
                    a[i][j] = amag * np.exp(-falloff * r*r)
                else:
                    a[i][j] = 0.0

            elif r <= falloff_low:

                a[i][j] = amag
            else:
                rshifted = r - falloff_low
                a[i][j] = amag * np.exp(-falloff * rshifted*rshifted)

    return a

def setup_population(h_grid, n_grid, npad, rinit):
    r = np.zeros((h_grid, npad, npad))  # just to multiply
    r_l = np.zeros((h_grid, npad, npad))
    r_r = np.zeros((h_grid, npad, npad))
    r_u = np.zeros((h_grid, npad, npad))
    r_d = np.zeros((h_grid, npad, npad))

    # masks that will be used to get directional subpopulations out
    r_masks = np.zeros((4, npad, npad))
    # order l, r, u, d
    r_masks[0, 0:n_grid:2, 0:n_grid:2] = 1
    r_masks[1, 1:n_grid:2, 1:n_grid:2] = 1
    r_masks[2, 0:n_grid:2, 1:n_grid:2] = 1
    r_masks[3, 1:n_grid:2, 0:n_grid:2] = 1

    for k in range(0, h_grid, 1):
        for i in range(0, n_grid, 2):
            for j in range(0, n_grid, 2):
                r[k][i][j] = rinit * stats.uniform.rvs()
        for i in range(0, n_grid, 2):
            for j in range(1, n_grid, 2):
                r[k][i][j] = rinit * stats.uniform.rvs()
        for i in range(1, n_grid, 2):
            for j in range(0, n_grid, 2):
                r[k][i][j] = rinit * stats.uniform.rvs()
        for i in range(1, n_grid, 2):
            for j in range(1, n_grid, 2):
                r[k][i][j] = rinit * stats.uniform.rvs()

    for k in range(0, h_grid, 1):
        # elementwise multiplication
        r_l[k, :, :] = r[k, :, :]*r_masks[0, :, :]
        r_r[k, :, :] = r[k, :, :]*r_masks[1, :, :]
        r_u[k, :, :] = r[k, :, :]*r_masks[2, :, :]
        r_d[k, :, :] = r[k, :, :]*r_masks[3, :, :]

    # remove zeros to get back to correct size for actual rate matrix
    r = r[:, 0:n_grid, 0:n_grid]

    # return r_masks so don't have to recalculate each time
    return r, r_r, r_l, r_d, r_u, r_masks

###############################################################################
def setup_pg(h_grid, n_grid, n_place, rand_weights_max):
    """
    Initializes place to grid weights
    """    
    w_pg = np.reshape(stats.uniform.rvs(0, 0.00222, h_grid*n_grid*n_grid*n_place), (n_place, h_grid,n_grid,n_grid))
    return w_pg

if __name__ == "__main__":
    print(set_inhib_length(5, -1, 12.5, 15))