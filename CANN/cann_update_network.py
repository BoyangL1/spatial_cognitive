import pyfftw
import numpy as np
from scipy import stats
import cann_run_network as RN
import place_cell_utilities as PCU

#########################################################################
######## All functions used for flowing activity in the network###########
##########################################################################

class GridNeuronNetwork:
    def __init__(self, h,n,dt, tau, n_place_cells, sigma, x_dim, y_dim, dist_thresh, rand_weights_max, wmag, lmin, lmax, wshift, umag, urad, u_dv, u_vd, rinit, amag, falloff, falloff_low, falloff_high, npad, rnoise, vgain):
        # Grid Cell Parameters
        self.h = h 
        self.n = n 
        self.dt = dt
        self.tau = tau

        # Place Cell Parameters
        self.n_place_cells = n_place_cells
        self.sigma = sigma
        # self.norm_const = 1 / (2 * np.pi * sigma**2)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.dist_thresh = dist_thresh
        self.rand_weights_max = rand_weights_max

        # Recurrent Inhibition Parameters
        self.wmag = wmag
        self.lmin = lmin
        self.lmax = lmax
        self.wshift = wshift

        # Coupling Parameters
        self.umag = umag
        self.urad = urad
        self.u_dv = u_dv
        self.u_vd = u_vd

        # Initial value for rates for grid units
        self.rinit = rinit

        # Hippocampal Parameters
        self.amag = amag
        self.falloff = falloff
        self.falloff_low = falloff_low
        self.falloff_high = falloff_high

        self.npad = npad

        self.rnoise = rnoise
        self.vgain = vgain
        
def update_neuron_activity(GN,r, r_r, r_l, r_d, r_u,r_masks,r_fft_plan, r_ifft_plan, vx, vy, r_field, spike, spiking, itter, singleneuronrec, time_ind, sna_eachlayer, row_record, col_record, w_r, w_l, w_u, w_d, a):
    """
    Updating grid cell activity without place cell inputs
    """
    if GN.umag == 0 or GN.u_vd == 0 and GN.u_dv == 0:  # run uncoupled network convolutions
        rwu_l = RN.convolve_no(r_fft_plan, r_ifft_plan, r_l, w_l, GN.npad, GN.h)
        rwu_u = RN.convolve_no(r_fft_plan, r_ifft_plan, r_u, w_u, GN.npad, GN.h)
        rwu_d = RN.convolve_no(r_fft_plan, r_ifft_plan, r_d, w_d, GN.npad, GN.h)
        rwu_r = RN.convolve_no(r_fft_plan, r_ifft_plan, r_r, w_r, GN.npad, GN.h)
    elif GN.umag !=0 and GN.u_vd == 0 and GN.u_dv ==1:
        rwu_l = RN.convolve_dv(r_fft_plan, r_ifft_plan, r_l, w_l, GN.umag, GN.npad, GN.h)
        rwu_u = RN.convolve_dv(r_fft_plan, r_ifft_plan, r_u, w_u, GN.umag, GN.npad, GN.h)
        rwu_d = RN.convolve_dv(r_fft_plan, r_ifft_plan, r_d, w_d, GN.umag, GN.npad, GN.h)
        rwu_r = RN.convolve_dv(r_fft_plan, r_ifft_plan, r_r, w_r, GN.umag, GN.npad, GN.h)
    elif GN.umag !=0 and GN.u_vd == 1 and GN.u_dv ==0:
        rwu_l = RN.convolve_vd(r_fft_plan, r_ifft_plan, r_l, w_l, GN.umag, GN.npad, GN.h)
        rwu_u = RN.convolve_vd(r_fft_plan, r_ifft_plan, r_u, w_u, GN.umag, GN.npad, GN.h)
        rwu_d = RN.convolve_vd(r_fft_plan, r_ifft_plan, r_d, w_d, GN.umag, GN.npad, GN.h)
        rwu_r = RN.convolve_vd(r_fft_plan, r_ifft_plan, r_r, w_r, GN.umag, GN.npad, GN.h)
    elif GN.umag !=0 and GN.u_vd ==1 and GN.u_dv ==1:
        rwu_l = RN.convolve_both(r_fft_plan, r_ifft_plan, r_l, w_l, GN.umag, GN.npad, GN.h)
        rwu_u = RN.convolve_both(r_fft_plan, r_ifft_plan, r_u, w_u, GN.umag, GN.npad, GN.h)
        rwu_d = RN.convolve_both(r_fft_plan, r_ifft_plan, r_d, w_d, GN.umag, GN.npad, GN.h)
        rwu_r = RN.convolve_both(r_fft_plan, r_ifft_plan, r_r, w_r, GN.umag, GN.npad, GN.h)

    # calculate fields
    [r_l, r_field_l] = RN.calculate_field(r, r_l, rwu_l, rwu_r, rwu_d, rwu_u, r_masks[0, :, :], a, 1.0-GN.vgain*vx, GN.h, GN.n, GN.npad, itter, 0)
    [r_r, r_field_r] = RN.calculate_field(r, r_r, rwu_l, rwu_r, rwu_d, rwu_u, r_masks[1, :, :], a, 1.0+GN.vgain*vx, GN.h, GN.n, GN.npad, itter, 0)
    [r_u, r_field_u] = RN.calculate_field(r, r_u, rwu_l, rwu_r, rwu_d, rwu_u, r_masks[2, :, :], a, 1.0+GN.vgain*vy, GN.h, GN.n, GN.npad, itter, 0)
    [r_d, r_field_d] = RN.calculate_field(r, r_d, rwu_l, rwu_r, rwu_d, rwu_u, r_masks[3, :, :], a, 1.0-GN.vgain*vy, GN.h, GN.n, GN.npad, itter, 0)

    # fix error r_field not being updated uniquely for each of the directions correctly
    r_field = r_field_l + r_field_r + r_field_u + r_field_d

    if GN.rnoise > 0.:
        for k in range(0, GN.h, 1):
            for i in range(0, GN.n, 1):
                for j in range(0, GN.n, 1):
                    r_field[k][i][j] = r_field[k][i][j] + \
                        GN.rnoise * (2*stats.uniform.rvs()-1)
    # update fields and weights
    r = RN.update_activity_spike(r, r_field, spike, GN.h, GN.n, GN.dt, GN.tau, itter)

    if singleneuronrec:  # get rate and spiking data for single units on each layer, hate nested if loops, but for now just leave it
        sna_eachlayer[:, :, itter] = RN.get_singleneuron_activity(
            sna_eachlayer, r, spike, spiking, itter, row_record, col_record)
    else:
        sna_eachlayer = -999

    return r, r_field, r_l, r_u, r_d, r_r, sna_eachlayer

def update_neuron_activity_with_place(GN,r, r_r, r_l, r_d, r_u, r_masks,r_fft_plan, r_ifft_plan, vx, vy, r_field, spike, spiking, itter, singleneuronrec, time_ind, sna_eachlayer, row_record, col_record, curr_place_activity, w_pg,w_r, w_l, w_u, w_d, a):
    """
    update grid cell activity with place cell inputs
    """
    if GN.umag == 0 or GN.u_vd == 0 and GN.u_dv == 0:  # run uncoupled network convolutions
        rwu_l = RN.convolve_no(r_fft_plan, r_ifft_plan, r_l, w_l, GN.npad, GN.h)
        rwu_u = RN.convolve_no(r_fft_plan, r_ifft_plan, r_u, w_u, GN.npad, GN.h)
        rwu_d = RN.convolve_no(r_fft_plan, r_ifft_plan, r_d, w_d, GN.npad, GN.h)
        rwu_r = RN.convolve_no(r_fft_plan, r_ifft_plan, r_r, w_r, GN.npad, GN.h)
    elif GN.umag !=0 and GN.u_vd == 0 and GN.u_dv ==1:
        rwu_l = RN.convolve_dv(r_fft_plan, r_ifft_plan, r_l, w_l, GN.umag, GN.npad, GN.h)
        rwu_u = RN.convolve_dv(r_fft_plan, r_ifft_plan, r_u, w_u, GN.umag, GN.npad, GN.h)
        rwu_d = RN.convolve_dv(r_fft_plan, r_ifft_plan, r_d, w_d, GN.umag, GN.npad, GN.h)
        rwu_r = RN.convolve_dv(r_fft_plan, r_ifft_plan, r_r, w_r, GN.umag, GN.npad, GN.h)
    elif GN.umag !=0 and GN.u_vd == 1 and GN.u_dv ==0:
        rwu_l = RN.convolve_vd(r_fft_plan, r_ifft_plan, r_l, w_l, GN.umag, GN.npad, GN.h)
        rwu_u = RN.convolve_vd(r_fft_plan, r_ifft_plan, r_u, w_u, GN.umag, GN.npad, GN.h)
        rwu_d = RN.convolve_vd(r_fft_plan, r_ifft_plan, r_d, w_d, GN.umag, GN.npad, GN.h)
        rwu_r = RN.convolve_vd(r_fft_plan, r_ifft_plan, r_r, w_r, GN.umag, GN.npad, GN.h)
    elif GN.umag !=0 and GN.u_vd ==1 and GN.u_dv ==1:
        rwu_l = RN.convolve_both(r_fft_plan, r_ifft_plan, r_l, w_l, GN.umag, GN.npad, GN.h)
        rwu_u = RN.convolve_both(r_fft_plan, r_ifft_plan, r_u, w_u, GN.umag, GN.npad, GN.h)
        rwu_d = RN.convolve_both(r_fft_plan, r_ifft_plan, r_d, w_d, GN.umag, GN.npad, GN.h)
        rwu_r = RN.convolve_both(r_fft_plan, r_ifft_plan, r_r, w_r, GN.umag, GN.npad, GN.h)

    # get current iteration's place cell activity
    curr_place_activity = curr_place_activity.reshape((GN.n_place_cells, 1, 1, 1))
    p = np.sum((curr_place_activity*w_pg), 0)  # [h,n_grid_cells,n_grid_cells]

    # calculate fields
    [r_l, r_field_l] = RN.calculate_field(
        r, r_l, rwu_l, rwu_r, rwu_d, rwu_u, r_masks[0, :, :], a, 1.0-GN.vgain*vx, GN.h, GN.n, GN.npad, itter, p)
    [r_r, r_field_r] = RN.calculate_field(
        r, r_r, rwu_l, rwu_r, rwu_d, rwu_u, r_masks[1, :, :], a, 1.0+GN.vgain*vx, GN.h, GN.n, GN.npad, itter, p)
    [r_u, r_field_u] = RN.calculate_field(
        r, r_u, rwu_l, rwu_r, rwu_d, rwu_u, r_masks[2, :, :], a, 1.0+GN.vgain*vy, GN.h, GN.n, GN.npad, itter, p)
    [r_d, r_field_d] = RN.calculate_field(
        r, r_d, rwu_l, rwu_r, rwu_d, rwu_u, r_masks[3, :, :], a, 1.0-GN.vgain*vy, GN.h, GN.n, GN.npad, itter, p)

    # fix error r_field not being updated uniquely for each of the directions correctly
    r_field = r_field_l + r_field_r + r_field_u + r_field_d

    if GN.rnoise > 0.:
        for k in range(0, GN.h, 1):
            for i in range(0, GN.n, 1):
                for j in range(0, GN.n, 1):
                    r_field[k][i][j] = r_field[k][i][j] + \
                        GN.rnoise * (2*stats.uniform.rvs()-1)
    # update fields and weights
    r = RN.update_activity_spike(r, r_field, spike, GN.h, GN.n, GN.dt, GN.tau, itter)
    w_pg = RN.update_weights(GN.n_place_cells, GN.h, GN.n, curr_place_activity, r, w_pg, itter)

    if singleneuronrec:  # get rate and spiking data for single units on each layer, hate nested if loops, but for now just leave it
        sna_eachlayer[:, :, itter] = RN.get_singleneuron_activity(
            sna_eachlayer, r, spike, spiking, itter, row_record, col_record)
    else:
        sna_eachlayer = -999

    return r, r_field, r_l, r_u, r_d, r_r, sna_eachlayer, w_pg

def flow_neuron_activity(GN, time_ind, v, theta, nflow, nphase, a, spike, spiking, r, r_r, r_l, r_d, r_u, r_masks,singleneuronrec,w_r, w_l, w_u, w_d):
    """
    Simulating neuronal activity over time without place cell inputs
    """
    print(nphase)
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)
    r_field = np.zeros((GN.h, GN.n, GN.n))

    shapein = [GN.npad, GN.npad]
    shapeout = [GN.npad, GN.npad//2+1]
    r_fft_arrayin = pyfftw.empty_aligned(shapein, dtype='float64')
    r_fft_arrayout = pyfftw.empty_aligned(shapeout, dtype='complex128')
    r_fft_plan = pyfftw.FFTW(r_fft_arrayin, r_fft_arrayout, axes=(0, 1))
    r_ifft_plan = pyfftw.FFTW(r_fft_arrayout, r_fft_arrayin, axes=(
        0, 1), direction='FFTW_BACKWARD')

    if singleneuronrec > 0:
        sna_eachlayer = np.zeros((GN.h, 3, time_ind))
        row_record = np.array([np.floor(np.size(r_field, 1)/2), np.floor(
            np.size(r_field, 1)/2)-10, np.floor(np.size(r_field, 1)/2)+10])
        col_record = np.array([np.floor(np.size(r_field, 2)/2), np.floor(
            np.size(r_field, 2)/2)-10, np.floor(np.size(r_field, 2)/2)+10])
    else:
        row_record = col_record = sna_eachlayer = -999

    for itter in range(1, nflow+1, 1):
        # update neuron activity for grid cells with no place input
        [r, r_field, r_l, r_u, r_d, r_r, sna_eachlayer] = update_neuron_activity(
            GN, r, r_r, r_l, r_d, r_u, r_masks,r_fft_plan, r_ifft_plan, vx, vy, r_field, spike, spiking, itter, singleneuronrec, nflow, sna_eachlayer, row_record, col_record, w_r, w_l, w_u, w_d, a)

    return r, r_field, r_r, r_l, r_d, r_u

def flow_full_model(GN, x, y, vx, vy, time_ind, a, spike, spiking, r, r_r, r_l, r_d, r_u, r_masks,singleneuronrec, place_cell_spiking, place_activity, w_pg, place_cells, n_place_cells,w_r, w_l, w_u, w_d):
    """ 
    The main funciton of the whole model, taking into account the place cell inputs.
    At each time of the simulation, the place cell activity state is updated and the grid cell activity state is updated based on the current position and velocity.
    The neuron activity is updated and the connection weights from place cells to grid cells are adjusted.
    """
    spike = np.zeros((GN.h, GN.n, GN.n), dtype='int')
    shapein = [GN.npad, GN.npad]
    shapeout = [GN.npad, GN.npad//2+1]
    r_fft_arrayin = pyfftw.empty_aligned(shapein, dtype='float64')
    r_fft_arrayout = pyfftw.empty_aligned(shapeout, dtype='complex128')
    r_fft_plan = pyfftw.FFTW(r_fft_arrayin, r_fft_arrayout, axes=(0, 1))
    r_ifft_plan = pyfftw.FFTW(r_fft_arrayout, r_fft_arrayin, axes=(
        0, 1), direction='FFTW_BACKWARD')
    r_field = np.zeros((GN.h, GN.n, GN.n))

    if singleneuronrec > 0:
        sna_eachlayer = np.zeros((GN.h, 3, time_ind))
        row_record = np.array([np.floor(np.size(r_field, 1)/2), np.floor(
            np.size(r_field, 1)/2)-10, np.floor(np.size(r_field, 1)/2)+10], dtype='int')
        col_record = np.array([np.floor(np.size(r_field, 2)/2), np.floor(
            np.size(r_field, 2)/2)-10, np.floor(np.size(r_field, 2)/2)+10], dtype='int')
    else:
        row_record = col_record = sna_eachlayer = -999

    for itter in range(1, time_ind, 1):
        # update place cell activity matrices based on current velocity/position
        place_cell_spiking, place_activity = PCU.evaluate_spiking(
            place_cell_spiking, place_activity, place_cells, x, y, itter)
        curr_place_activity = place_activity[:, itter]
        curr_place_activity = curr_place_activity.reshape((1, n_place_cells))
        if np.mod(itter, 1000) == 0:
            print(itter)
        vx1 = vx[itter]
        vy1 = vy[itter]
        # update neuron activity for grid cells
        [r, r_field, r_l, r_u, r_d, r_r, sna_eachlayer, w_pg] = update_neuron_activity_with_place(
            GN, r, r_r, r_l, r_d, r_u, r_masks, r_fft_plan, r_ifft_plan, vx1, vy1, r_field, spike, spiking, itter, singleneuronrec, time_ind, sna_eachlayer, row_record, col_record, curr_place_activity, w_pg,w_r, w_l, w_u, w_d, a)
        occ = 0

    return r, r_field, r_r, r_l, r_d, r_u, sna_eachlayer, occ, w_pg