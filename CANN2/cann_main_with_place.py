import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import cann_update_network as UN
import cann_setup_network as SN
from place_cell import PlaceCell
import place_cell_utilities as PCU
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import real_trajeccotry as TRAJ
import pickle

np.random.seed(1)
UNDEF = -999

if __name__ == "__main__":
    "****************************************************************************"
    "***********************SETUP THE NETWORK************************************"
    "****************************************************************************"
    ################ PARAMETERS###################

    # Network Basics
    h = 2  # TODO:Number of grid cell network depths 
    n = 128  # TODO:number of neurons per side in grid tile,  160 in Louis paper, 128 in Alex's new, 90 in Alex OG TODO:查找文献，选择一个更合理的数字
    dt = 1.0  # time step, ms
    tau = 10.0  # neuron time constant, ms

    # Recurrent Inhibition Parameters
    wmag = 2.4  
    lmin = 7  # TODO
    lmax =  13 # TODO
    lexp = -1 # default to -1  
    wshift = 1

    # Coupling Parameters
    umag = -0.001  # for spike model, mean(w)
    urad = 4.0  # for rate model
    u_dv = 1  # 1 corresponds to dorsal to ventral
    u_vd = 1  # 1 corresponds to ventral to dorsal

    # Initial value for rates for grid units
    rinit = 1e-3

    # Hippocampal Parameters
    amag = .6  # now trying . 6 from Alex's paper, was 1;#trying .7 instead of .5 due to particularly low activity in network when checked #1 in rate model
    falloff = 4.0  # 4.0 in rate model
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

    file_name = './data/one_travel_chain_time.csv'
    # Get Trajectory Data
    [origin_x,origin_y,dest_x,dest_y,x,y,vx,vy,spatial_scale] = TRAJ.get_trajectory(file_name)
    anchor_x = origin_x + dest_x
    anchor_y = origin_y + dest_y
    # anchor_x, anchor_y = anchor_x[:10], anchor_y[:10]
    anchor_set = set(zip(anchor_x, anchor_y))

    # Place_cell_parameters
    n_place_cells = len(anchor_x)
    sigma = 0.01
    rand_weights_max = 0.00222

    GN = UN.GridNeuronNetwork(h,n,dt,tau,n_place_cells, sigma, 0, 0, 0, rand_weights_max, wmag, lmin,
                            lmax, wshift, umag, urad, u_dv, u_vd, rinit, amag, falloff, falloff_low, falloff_high,npad, rnoise, vgain)
    
    "*******************************************************************"
    "***************Run Network Setup***********************************"
    "*******************************************************************"
    print("Set up grid cell neuron network")
    # inhibition length scales
    l = SN.set_inhib_length(h, lexp, lmin, lmax)
    # inhibitory kernel overall and for each directional subpopulation
    [w, w_r, w_l, w_u, w_d] = SN.setup_recurrent(h, npad, l, wmag, wshift)
    a = SN.setup_input(amag, n, falloff_high, falloff_low, falloff)
    [r, r_r, r_l, r_d, r_u, r_masks] = SN.setup_population(h, n, npad, rinit)
    r_field = np.zeros((h, n, n))
    # setup place to grid connections
    w_pg = SN.setup_pg(h, n, n_place_cells, rand_weights_max) # [n_place,h_grid,n_grid,n_grid]

    ############################ PLACE CELL SETUP ##########################
    place_cells = PCU.create_place_cells(n_place_cells, anchor_x, anchor_y,sigma)
    place_cell_spiking, place_activity = np.zeros((n_place_cells, len(x))), np.zeros((n_place_cells, len(x)))
    # PCU.plot_centers(place_cells,'./img/palce_cell.png')

    "********************************************************************************************"
    "*************************************Intialize Grid Activity *******************************"
    "********************************************************************************************"
    status = 'Initializing Grid Cells Activity in 4 Phases'
    print(status)
    singleneuronrec = False
    t_index = len(x)

    t0 = time.time()
    [r, r_field, r_r, r_l, r_d, r_u] = UN.flow_neuron_activity(
        GN,t_index, 0, 0, nflow0, 1, a, r, r_r, r_l, r_d, r_u, r_masks, singleneuronrec, w_r, w_l, w_u, w_d)    
    t_run1 = time.time()-t0
    print("Initializing grid cells took {} seconds".format(t_run1))

    singleneuronrec = True
    t0 = time.time()
    status = 'Running Model with Trajectory and Place Cells'
    print(status)
    [r, r_field, r_r, r_l, r_d, r_u, sna_eachlayer, w_pg, place_grid_dic] = UN.flow_full_model(
        GN, anchor_set,x, y, vx, vy, t_index, a, r, r_r, r_l, r_d, r_u, r_masks,singleneuronrec, place_cell_spiking, place_activity, w_pg, place_cells, n_place_cells,w_r, w_l, w_u, w_d)
    t_run2 = time.time()-t0
    print("Running grid cells with anchor point and real trajectory took {} seconds".format(t_run2))

     # calculate single neuron spiking results
    sns_eachlayer = np.zeros((sna_eachlayer.shape))
    print('Calculating single neuron spiking results...')
    for inds in range(0, np.size(sna_eachlayer, 2), 1):
        sns_eachlayer[:, :, inds] = sna_eachlayer[:, :, inds] > stats.uniform.rvs(0, 1, size=sna_eachlayer[:, :, inds].shape) 
    sns_eachlayer = sns_eachlayer.astype('bool') # [h_grid, num_cell, t_index]

    "********************************************************************************************"
    "************************************Plot And Save Results***********************************"
    "********************************************************************************************"
    print("Saving places cell corresponding grid cell networks")
    with open('./data/place_grid_data.pkl', 'wb') as file:
        pickle.dump(place_grid_dic, file)

    # plot grid cell results
    print('Plotting grid cell results...')
    for z in range(0, h, 1):
        plt.figure()
        plt.imshow(r[z, :, :], cmap='hot')
        plt.savefig(f'./img/grid_cell_result_{z}.png')  
