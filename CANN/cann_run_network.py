import numpy as np
from scipy import stats
import time

def convolve_no(r_fft_plan, r_ifft_plan, r_dir, w_dir, npad, h):
    """ 
    Convolution function if no coupling
    """
    rwu_dir = np.zeros((h, npad, npad))

    for k in range(0, h, 1):

        r_dir_fourier = r_fft_plan(r_dir[k, :, :])

        rwu_dir[k, :, :] = r_ifft_plan(r_dir_fourier*w_dir[k, :, :])

    return rwu_dir


def convolve_vd(r_fft_plan, r_ifft_plan, r_dir, w_dir, u, npad, h):
    """ 
    Convolution function if ventral to dorsal coupling
    """
    rwu_dir = np.zeros((h, npad, npad))
    r_dir_fourier = np.zeros((h, npad, npad//2+1), dtype='complex128')

    for k in range(0, h, 1):
        r_dir_fourier[k, :, :] = r_fft_plan(r_dir[k, :, :])

    for k in range(0, h-1, 1):
        rwu_dir[k, :, :] = r_ifft_plan(
            (w_dir[k, :, :]*r_dir_fourier[k, :, :]+u*r_dir_fourier[k+1, :, :]))

    rwu_dir[h-1, :, :] = r_ifft_plan((w_dir[h-1, :, :]*r_dir_fourier[h-1, :, :]))

    return rwu_dir


def convolve_dv(r_fft_plan, r_ifft_plan, r_dir, w_dir, u, npad, h):
    """ 
    Convolution function if dorsal to ventral coupling
    """
    rwu_dir = np.zeros((h, npad, npad))
    r_dir_fourier = np.zeros((h, npad, npad//2+1), dtype='complex128')

    for k in range(0, h, 1):
        r_dir_fourier[k, :, :] = r_fft_plan(r_dir[k, :, :])

    rwu_dir[0, :, :] = r_ifft_plan((w_dir[0, :, :]*r_dir_fourier[0, :, :]))

    for k in range(1, h, 1):
        rwu_dir[k, :, :] = r_ifft_plan(
            (w_dir[k, :, :]*r_dir_fourier[k, :, :]+u*r_dir_fourier[k-1, :, :]))

    return rwu_dir


def convolve_both(r_fft_plan, r_ifft_plan, r_dir, w_dir, u, npad, h):
    """ 
    Convolution function for both dorsal to ventral  and ventral to dorsal coupling
    """
    rwu_dir = np.zeros((h, npad, npad))
    r_dir_fourier = np.zeros((h, npad, npad//2+1), dtype='complex128')

    for k in range(0, h, 1):
        r_dir_fourier[k, :, :] = r_fft_plan(r_dir[k, :, :])

    rwu_dir[0, :, :] = r_ifft_plan((w_dir[0, :, :]*r_dir_fourier[0, :, :]))

    for k in range(1, h-1, 1):
        rwu_dir[k, :, :] = r_ifft_plan(
            (w_dir[k, :, :]*r_dir_fourier[k, :, :]+u*r_dir_fourier[k-1, :, :]+u*r_dir_fourier[k+1, :, :]))

    rwu_dir[h-1, :, :] = r_ifft_plan((w_dir[h-1, :, :]*r_dir_fourier[h-1, :, :]))

    return rwu_dir

def calculate_field(r, r_dir, rwu_l, rwu_r, rwu_d, rwu_u, r_masks, a, vgain_fac, h, n, npad, itter):
    """ 
    Calculate field with input after all rates are updated from convolution
    """
    r_temp = np.zeros((h, npad, npad))
    r_field_dir = np.zeros((h, n, n))
    r_temp[:, 0:n, 0:n] = r

    r_dir = r_temp * r_masks

    r_field_mask = r_masks[0:n, 0:n]

    r_field_dir[:, :, :] = (rwu_l[:, 0:n, 0:n] + rwu_r[:, 0:n, 0:n] +
                            rwu_u[:, 0:n, 0:n] + rwu_d[:, 0:n, 0:n] + a * vgain_fac)*r_field_mask

    return r_dir, r_field_dir

def update_rate(r, r_field, dt, tau, n, h):
    """Update activity if using a rate system"""
    dt_tau = dt / tau

    activity_mask = (r_field > 0.)*1
    r = r + (-r + r_field*activity_mask)*dt_tau

    return r


def update_activity_spike(r, r_field, spike, h, n, dt, tau, itter):
    "Update activity if using a spiking based system"
    # Originally 500, but scaling according to new dt/tau (which is essentially dt in this case)
    k = 500.0
    # Originally 0.1, but scaled according to new k and dt/tau (which is essentially dt in this case)
    beta_grid = 0.1
    dt_tau = dt/tau  # dt divied by tau or c in Alex's paper
    # 8/23/18 dt is in seconds for Alex's model for determine if spiking, dt_tau is a ratio so maybe works out with this...
    dt_inseconds = dt/1000
    alpha = 0.5  # scale factor from Alex's paper
    # initialize new spike array each time
    spike = np.zeros((h, n, n), dtype='int')
    threshold = np.reshape(stats.uniform.rvs(0, 1, size=h*n*n), (h, n, n))  # grab new thresholds

    activity_mask = (r_field > 0.)*1  # use this to replace if r_field >0 loop

    spike = ((k*dt_inseconds*(r_field*activity_mask - beta_grid)) >
             threshold)*1  # check if spikes on any neurons on all layers

    r = r + (-r + r_field*activity_mask)*dt_tau + alpha * spike

    return r


def update_weights(n_place, h, n, p_activity, r, w_pg, itter):
    """ 
    Updates the weights of the place to grid connections
    """
    # Parameters
    lambda_lr = 0.00001  # learning rate
    epsilon_pg = 0.4
    r_new_shape = np.repeat(r[np.newaxis, :, :, :], n_place, axis=0)
    # r_new_shape = np.repeat(r, n_place, 0)

    w_pg += lambda_lr * ((p_activity * r_new_shape) *
                         (epsilon_pg - w_pg) - r_new_shape*w_pg)

    return w_pg


def get_singleneuron_activity(sna_eachlayer, r_field, spike, spiking, itter, row_record, col_record):
    """
    Retrieves and records the activity of single neurons across layers at different time points.

    Args:
        sna_eachlayer (numpy.ndarray): A 3D array storing the activity data of neurons in each layer, with the time dimension being the last one.
        r_field (numpy.ndarray): An array representing the receptive field of neurons, possibly containing the response strength of neurons to certain stimuli.
        spike : Unused in the function body, may be intended for other purposes.
        spiking : Unused in the function body, may be intended for other purposes.
        itter (int): The index of the current iteration or time point.
        row_record (list of int): A list of row indices indicating the positions of neurons.
        col_record (list of int): A list of column indices indicating the positions of neurons.

    Returns:
        The updated slice of the sna_eachlayer array for the specified time point, containing neuronal activity data for that specific moment.
    """
    sna_eachlayer[:, 0, int(itter)] = r_field[:, row_record[0], col_record[0]]
    sna_eachlayer[:, 1, int(itter)] = r_field[:, row_record[1], col_record[1]]
    sna_eachlayer[:, 2, int(itter)] = r_field[:, row_record[2], col_record[2]]

    return sna_eachlayer[:, :, itter]