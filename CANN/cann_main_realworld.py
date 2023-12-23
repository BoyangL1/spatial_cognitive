import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import cann_update_network as UN
import cann_setup_network as SN
import pandas as pd
# import geopandas as gpd
# from shapely.geometry import LineString
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
    h = 8  # TODO:Number of grid cell network depths 
    n = 128  # TODO:number of neurons per side in grid tile,  160 in Louis paper, 128 in Alex's new, 90 in Alex OG TODO:查找文献，选择一个更合理的数字
    dt = 1.0  # time step, ms
    tau = 10.0  # neuron time constant, ms

    # Recurrent Inhibition Parameters
    wmag = 2.4  
    lmin = 7  # TODO
    lmax =  41 # TODO
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

    GN = UN.GridNeuronNetwork(h,n,dt,tau,0, 0, 0, 0, 0, 0, wmag, lmin,
                            lmax, wshift, umag, urad, u_dv, u_vd, rinit, amag, falloff, falloff_low, falloff_high,npad, rnoise, vgain)
    
    file_name = './data/one_travel_chain_time.csv'
    # Get Trajectory Data
    [origin_grid,dest_grid,origin_x,origin_y,dest_x,dest_y,x,y,vx,vy,spatial_scale] = TRAJ.get_trajectory(file_name)
    anchor_x = origin_x + dest_x
    anchor_y = origin_y + dest_y
    # anchor_x, anchor_y = anchor_x[:10], anchor_y[:10]
    anchor_list = list(zip(anchor_x, anchor_y))
    grid_list = origin_grid+dest_grid

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
    # constant velocity
    [r, r_field, r_r, r_l, r_d, r_u] = UN.flow_neuron_activity(
        GN,t_index,vflow, (np.pi/2-np.pi/5), nflow1, 2, a, r, r_r, r_l, r_d, r_u, r_masks, singleneuronrec, w_r, w_l, w_u, w_d)
    
    [r, r_field, r_r, r_l, r_d, r_u] = UN.flow_neuron_activity(
        GN,t_index,vflow, (np.pi/5), nflow1, 3, a, r, r_r, r_l, r_d, r_u, r_masks, singleneuronrec, w_r, w_l, w_u, w_d)

    [r, r_field, r_r, r_l, r_d, r_u] = UN.flow_neuron_activity(
        GN,t_index,vflow, (np.pi/4), nflow1, 4, a, r, r_r, r_l, r_d, r_u, r_masks, singleneuronrec, w_r, w_l, w_u, w_d)
    t_run1 = time.time()-t0
    print("Initializing grid cells in 4 phases took {} seconds".format(t_run1))

    singleneuronrec = True
    t0 = time.time()
    status = 'Update Grid Cell Model with Real Trajectory！'
    print(status)
    [r, r_field, r_r, r_l, r_d, r_u, sna_eachlayer, place_grid_dic] = UN.flow_full_model(
        GN, anchor_list,grid_list,x,y,vx, vy, t_index, a, r, r_r, r_l, r_d, r_u, r_masks,singleneuronrec, w_r, w_l, w_u, w_d)
    t_run2 = time.time()-t0
    print("Updating grid cell model with real trajectory took {} seconds".format(t_run2))

    # calculate single neuron spiking results
    sns_eachlayer = np.zeros((sna_eachlayer.shape))
    print('Calculating single neuron spiking results...')
    for inds in range(0, np.size(sna_eachlayer, 2), 1):
        sns_eachlayer[:, :, inds] = sna_eachlayer[:, :, inds] > stats.uniform.rvs(0, 1, size=sna_eachlayer[:, :, inds].shape) 
    sns_eachlayer = sns_eachlayer.astype('bool') # [h_grid, num_cell, t_index]

    "********************************************************************************************"
    "****************************************Plot Results****************************************"
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

    # # plot with real traj
    # df = pd.read_csv(file_name)
    # map_file = "./data/shenzhen_grid/shenzhen_grid.shp"
    # map_gdf = gpd.read_file(map_file)
    # gdf = gpd.GeoDataFrame(df, geometry=[LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in zip(df['lambda_o'], df['phi_o'], df['lambda_d'], df['phi_d'])])
    # plot_num = 500
    # print('Plotting grid cell results over trajectory...')
    # for z in range(h):
    #     for cell_index in range(0, 3, 1):
    #         fig, ax = plt.subplots(figsize=(14, 14))

    #         map_gdf.plot(ax=ax, color='lightgrey')  
    #         # gdf.plot(ax=ax, edgecolor='blue')
            
    #         # Method 1
    #         # colors = ['blue' if val else 'red' for val in sns_eachlayer[z, cell_index, :plot_num]]
    #         # ax.scatter(x, y, c=colors)
            
    #         # Method 2
    #         scatter = ax.scatter(x[:plot_num], y[:plot_num], c=sna_eachlayer[z, cell_index, :plot_num], cmap='viridis', s=5, alpha=0.4)

    #         ax.set_xlabel('Longitude')
    #         ax.set_ylabel('Latitude')

    #         plt.savefig(f"./img/cell_{cell_index}_layer_{z}.png",dpi=600)