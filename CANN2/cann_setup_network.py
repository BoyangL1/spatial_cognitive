import pyfftw
import numpy as np
from scipy import stats, io, signal, interpolate
import time
import quadflip as qp
import os

UNDEF = -999

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def get_trajectory(traj_filename, dt, num_minutes):
    """
    Loads trajectory data
    Args:
        traj_filename: The filename of the trajectory data file
        dt: Time step for the model (in milliseconds)
        num_minutes: Duration of the trajectory data to be loaded (in minutes)

    Returns:
        x, y: Original x and y coordinates (in centimeters) from the trajectory data
        x_ind, y_ind: Indices of the resampled x and y coordinates
        x_of_bins, y_of_bins: Arrays representing the spatial bins for x and y
        vx, vy: Velocity in x and y directions (converted to meters per second)
        spatial_scale: Scaling factor used for the trajectory data
        boundaries: The spatial boundaries of the environment
        time_ind: The number of time steps for the model to run
        traj_filename: The filename of the trajectory data file
    """    
    print('getting trajectory data from Hass lab')

    unpack = io.loadmat(traj_filename)
    times = num_minutes * 60  # convert to seconds
    traj_fs = unpack['traj_fs']

    # currently hard coding in the 5 since we only pulled 5 minutes of data from each session*********
    # time stamps (in seconds) for positions with original sampling rate
    sesh_ts_og = np.arange(0, times, 1/traj_fs)
    # new time stamps (in seconds) for dt of model sampling rate
    sesh_ts_interp = np.arange(0, (times), (dt/1000))

    spatial_scale = unpack['traj_spatial_scale']
    # position in cm, below is in indecies, resample up to dt samples
    x = unpack['traj_x']*spatial_scale
    # position in cm, below is in indecies, resample up to dt samples
    y = unpack['traj_y']*spatial_scale

    # dumb issue with grids, at least here not matching like with like aka loading full x and y even though simulation is for 5 minutes only
    # only take x's for time grabbed, this hopefully will fix issues with x_ind being full session when it should be 5 minutes
    x = x[0:np.size(sesh_ts_og)+1]
    y = y[0:np.size(sesh_ts_og)+1]

    # just to make sure we are all equal
    sesh_ts_og = sesh_ts_og[0:np.size(x)]

    # dumb fix to adjust for interpolation issues with some values of sesh_ts_interp being outside the range of sesh_ts_og
    sesh_ts_interp[sesh_ts_interp > np.max(sesh_ts_og)] = np.max(sesh_ts_og)

    if any(np.isnan(x)) or any(np.isnan(y)):

        nans1, t1 = nan_helper(x)
        nans2, t2 = nan_helper(y)

        x[nans1] = np.interp(t1(nans1), t1(~nans1), x[~nans1])
        y[nans2] = np.interp(t2(nans2), t2(~nans2), y[~nans2])

    # note hardcode issue
    # resample up to dt samples
    x_ind = np.round(signal.resample(x/spatial_scale, int(times/(dt/1000))))
    x_ind = x_ind.astype('int')
    # resample up to dt samples
    y_ind = np.round(signal.resample(y/spatial_scale, int(times/(dt/1000))))
    y_ind = y_ind.astype('int')

    # flipping around to make sure get all velocities we can (i.e. since pairing off need to count from back since first ind will be one without pair)
    x_temp = x[::-1]
    y_temp = y[::-1]

    vx_new = x_temp[0:-2:1]-x_temp[1:-1:1]
    # get first subtraction back in
    vx_new = np.append(vx_new, x_temp[1] - x_temp[0])
    vx_new = vx_new[::-1]  # flip back around
    vx_new = np.append(0, vx_new,)*traj_fs  # append first zero

    vy_new = y_temp[0:-2:1]-y_temp[1:-1:1]
    # get first subtraction back in
    vy_new = np.append(vy_new, y_temp[1] - y_temp[0])
    vy_new = vy_new[::-1]  # flip back around
    vy_new = np.append(0, vy_new,)*traj_fs  # append first zero

    fvx = interpolate.interp1d(sesh_ts_og, vx_new[0, 0:np.size(sesh_ts_og)])

    fvy = interpolate.interp1d(sesh_ts_og, vy_new[0, 0:np.size(sesh_ts_og)])

    vx_new = fvx(sesh_ts_interp)
    vy_new = fvy(sesh_ts_interp)

    vx = vx_new/100.0  # to convert to M/S to align with other models
    vy = vy_new/100.0

    dim_box = np.array([1.0, 1.0])*np.max([np.max(x), np.max(y)])
    boundaries = ([0, 0], [dim_box[0], dim_box[1]])

    x_of_bins = np.arange(np.min(boundaries, 0)[0], np.max(
        boundaries, 0)[0]+spatial_scale, spatial_scale)
    y_of_bins = np.arange(np.min(boundaries, 0)[1], np.max(
        boundaries, 0)[1]+spatial_scale, spatial_scale)

    if np.any(x_ind >= np.size(x_of_bins)) or np.any(y_ind >= np.size(y_of_bins)):

        x_ind[x_ind >= np.size(x_of_bins)] = np.size(x_of_bins)-1
        y_ind[x_ind >= np.size(y_of_bins)] = np.size(y_of_bins)-1

    # simply the number of indecies and therefore iterations for the model to run
    time_ind = int(times/(dt/1000))

    return x, y, x_ind, y_ind, x_of_bins, y_of_bins, vx, vy, spatial_scale, boundaries, time_ind, traj_filename

###############################################################################

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
    x = np.arange(-npad/2, npad/2)  # o.g.  #for aperiodic (-npad/2+1,npad/2+1)
    y = np.arange(-npad/2, npad/2)  

    a_row = np.zeros((h, 1, npad))
    a_column = np.zeros((h, npad, 1))

    w = np.zeros((h, npad, npad))

    for k in range(0, h, 1):
        for i in range(0, npad, 1):
            for j in range(0, npad, 1):
                w[k, i, j] = w_value(x[i], y[j], l_inhibition[k], wmag) if h != 1 else w_value(x[i], y[j], l_inhibition, wmag)

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
    w_pg = np.reshape(stats.uniform.rvs(0, 0.00222, h_grid*n_grid*n_grid*n_place), (n_place,h_grid,n_grid,n_grid))
    return w_pg 

if __name__ == "__main__":
    print(set_inhib_length(5, -1, 12.5, 15))