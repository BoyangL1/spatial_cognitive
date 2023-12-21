import time
import numpy as np
import os
from scipy import stats, io
import matplotlib.pyplot as plt
import cann_update_network as UN
import cann_setup_network as SN
import cann_run_network as RN
from place_cell import PlaceCell
import place_cell_utilities as PCU
import pickle

np.random.seed(1)
UNDEF = -999

if __name__ == "__main__":
    "****************************************************************************"
    "***********************SETUP THE NETWORK************************************"
    "****************************************************************************"
    ################ PARAMETERS###################

    # Network Basics
    h = 4  # Number of depths, 1
    n = 160  # number of neurons per side in grid tile,  160 in Louis paper, 128 in Alex's new, 90 in Alex OG
    dt = 1.0  # in miliseconds #note code has dividing  by tau throughout so this works out to actually being 1ms , this is a holdover from a mistake reading Louis' model
    tau = 10.0  # neuron time constant in ms
    num_minutes = 0.01  # default as 20

    # Place Cell Parameters
    n_place_cells = 16  # number of place cells
    sigma = 5.0  # standard deviation for Gaussian place fields
    # normalization constant for place cells with sigma
    norm_const = 1/(2*np.pi*sigma**2)
    x_dim = 130  # x dimension for environment - cm
    y_dim = 130  # y dimension for environment - cm
    dist_thresh = 20  # min euclidean distance between place cell centers - cm
    rand_weights_max = 0.00222

    # Recurrent Inhibition Parameters
    wmag = 2.4  # 2.4 for rate model
    lmin = 12.5  # periodic conditions # 7.5 setting to value for medium sized grid fields since only one layer
    lmax = 15.  # 15 in paper
    lexp = -1  # -1 in paper
    wshift = 1

    # Coupling Parameters
    umag = 2.6  # 2.6 for rate model
    urad = 4.0  # 8.0 for rate model
    u_dv = 1  # 1 corresponds to dorsal to ventral
    u_vd = 1  # 1 corresponds to ventral to dorsal

    # Initial value for rates for grid units
    rinit = 1e-3

    # Hippocampal Parameters
    amag = .6  # now trying . 6 from Alex's paper, was 1;#trying .7 instead of .5 due to particularly low activity in network when checked #1 in rate model
    falloff = 4.0  # 4.0 in rate model
    # 1-8-19 quick dialing in check...failed #inside this scaled radius hippocampal input is amg
    falloff_low = 2.0
    falloff_high = UNDEF  # outside this scaled radius hippocampal input is 0

    periodic = True  # to run periodic or aperiodic boundary conditions
    if periodic == False:
        pad_min = np.ceil(2*lmax)+wshift
        npad = int(n+pad_min)+1
        npfft = npad * (npad/2 + 1)
        print('aperiodic boundaries')
    else:
        pad_min = 0
        npad = n
        npfft = n * (n/2 + 1)
        print('periodic boundaries')

    "****************************************************************************"
    "****************************NETWORK DYNAMICS********************************"
    "****************************************************************************"

    nflow0 = 1000  # flow populations activity with no velocity
    nflow1 = 5000  # flow populations activity with constant velocity

    # system noise
    rnoise = 0.0
    vgain = .4

    still = 0  # no motion if true; clears vx and vy from loaded or simulated trajectory

    if still:
        vx = 0
        vy = 0

    vflow = .8
    theta_flow = UNDEF

    # spiking simulation
    spiking = 1  # set whether spiking model or rate model 0 or 1 respectively
    if spiking == 0:
        print('rate coded network')
    else:
        print('spike coded network')
    spike = 0

    GN = UN.GridNeuronNetwork(h,n,dt,tau,n_place_cells, sigma, x_dim, y_dim, dist_thresh, rand_weights_max, wmag, lmin,
                            lmax, wshift, umag, urad, u_dv, u_vd, rinit, amag, falloff, falloff_low, falloff_high,npad, rnoise, vgain)
    
    "*******************************************************************"
    "***************Run Network Setup***********************************"
    "*******************************************************************"

    # Trajectory Parameters/Get Trajectory Data
    [x, y, x_ind, y_ind, x_of_bins, y_of_bins, vx, vy, spatial_scale, boundaries, time_ind,
        traj_filename] = SN.get_trajectory('Trajectory_Data_2_full', dt, num_minutes)

    # update x and y to be interpolated
    x = x_ind * spatial_scale
    y = y_ind * spatial_scale

    # inhibition length scales
    l = SN.set_inhib_length(h, lexp, lmin, lmax)
    # inhibitory kernel overall and for each directional subpopulation
    [w, w_r, w_l, w_u, w_d] = SN.setup_recurrent(h, npad, l, wmag, wshift)
    a = SN.setup_input(amag, n, falloff_high, falloff_low, falloff)
    [r, r_r, r_l, r_d, r_u, r_masks] = SN.setup_population(h, n, npad, rinit)
    r_field = np.zeros((h, n, n))
    # setup place to grid connections
    w_pg_a = SN.setup_pg(h, n, n_place_cells, rand_weights_max)

    ############################ PLACE CELL SETUP ##########################
    # create place cells
    place_cells = PCU.create_place_cells(
        n_place_cells, sigma, x_dim, y_dim, dist_thresh, map_b=True)
    place_cells_a = place_cells[0:n_place_cells]
    # create arrays for keeping track of spiking and overall activity
    place_cell_spiking_a, place_activity_a = np.zeros(
        (n_place_cells, len(x))), np.zeros((n_place_cells, len(x)))

    PCU.plot_centers(place_cells_a, dist_thresh)
    # save place cells for testing simulation later
    pickle_out = open("place_cells_A.pickle", "wb")
    pickle.dump(place_cells_a, pickle_out)
    pickle_out.close()

    "********************************************************************************************"
    "*************************Intialize Grid Activity and Place Connections**********************"
    "********************************************************************************************"
    # no velocity
    status = 'Initializing Grid Activity in 4 Phases - Map A'
    print(status)
    singleneuronrec = False
    t0 = time.time()
    [r, r_field, r_r, r_l, r_d, r_u] = UN.flow_neuron_activity(
        GN,time_ind, 0, 0, nflow0, 1, a, spike, spiking, r, r_r, r_l, r_d, r_u, r_masks, singleneuronrec, w_r, w_l, w_u, w_d)
    # constant velocity
    [r, r_field, r_r, r_l, r_d, r_u] = UN.flow_neuron_activity(
        GN,time_ind,vflow, (np.pi/2 - np.pi/5), nflow1, 2, a, spike, spiking, r, r_r, r_l, r_d, r_u, r_masks, singleneuronrec, w_r, w_l, w_u, w_d)

    [r, r_field, r_r, r_l, r_d, r_u] = UN.flow_neuron_activity(
        GN,time_ind,vflow, (2*np.pi/5), nflow1, 3, a, spike, spiking, r, r_r, r_l, r_d, r_u, r_masks, singleneuronrec, w_r, w_l, w_u, w_d)

    [r, r_field, r_r, r_l, r_d, r_u] = UN.flow_neuron_activity(
        GN,time_ind,vflow, (np.pi/4), nflow1, 4, a, spike, spiking, r, r_r, r_l, r_d, r_u, r_masks, singleneuronrec, w_r, w_l, w_u, w_d)
    t_run1 = time.time()-t0

    singleneuronrec = True
    t0 = time.time()
    status = 'Running Model with Trajectory and Place Cells - Map A'
    print(status)
    [r, r_field, r_r, r_l, r_d, r_u, sna_eachlayer, occ, w_pg_a] = UN.flow_full_model(
        GN, x, y, vx, vy, time_ind, a, spike, spiking, r, r_r, r_l, r_d, r_u, r_masks,singleneuronrec, place_cell_spiking_a, place_activity_a, w_pg_a, place_cells_a, n_place_cells,w_r, w_l, w_u, w_d)
    t_run2 = time.time()-t0

    # calculate single neuron spiking results
    sns_eachlayer = np.zeros((sna_eachlayer.shape))
    print('Calculating single neuron spiking results...')
    for inds in range(0, np.size(sna_eachlayer, 2), 1):
        sns_eachlayer[:, :, inds] = sna_eachlayer[:, :, inds] > stats.uniform.rvs(
            0, 1, size=sna_eachlayer[:, :, inds].shape)

    sns_eachlayer = sns_eachlayer.astype('bool')

    "********************************************************************************************"
    "****************************************Plot Results****************************************"
    "********************************************************************************************"

    # plot place fields
    print('Plotting place centers for map A...')
    PCU.plot_centers(place_cells_a, dist_thresh)

    # plot grid cell results
    print('Plotting grid cell results...')
    for z in range(0, h, 1):
        plt.figure(4)
        plt.imshow(r[z, :, :], cmap='hot')
        plt.savefig(f'./img/grid_cell_result_{z}.png')  # Save the figure
        # plt.show()

    # plot single neuron spiking results
    x_ind = np.reshape(x_ind, (x_ind.size, 1))
    y_ind = np.reshape(y_ind, (y_ind.size, 1))
    print('Plotting grid cell results over trajectory...')
    for z in range(0, 3, 1):
        plt.figure(z, figsize=(5, 5))
        plt.plot(x, y)
        plt.plot(x_ind[sns_eachlayer[0, z, 0:np.size(x_ind)]]*spatial_scale,
                 y_ind[sns_eachlayer[0, z, 0:np.size(x_ind)]]*spatial_scale, 'r.')
        plt.savefig(f'./img/neuron_spiking_{z}.png')  # Save the figure

    # save variables
    var_out = dict()
    for i_vars in ('x', 'y', 'x_ind', 'y_ind', 'sna_eachlayer', 'sns_eachlayer', 'spatial_scale', 'r', 'w_pg_a'):
        var_out[i_vars] = locals()[i_vars]
    cwd = os.getcwd()
    # io.savemat('./data/grid_and_place_training_map_a_20min', var_out)
