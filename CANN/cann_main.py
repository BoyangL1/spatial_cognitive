import numpy as np
from periodic_cann import *
from real_trajeccotry import *
import pickle

if __name__ =="__main__":
    n = 2**7 # number of neurons
    h_grid = 8 # layers of grid cells

    dt = 0.5 
    tau = 5 # neuron time constant

    # Envelope and Weight Matrix Parameters
    wmag = 2.4
    wtphase = 2
    lmin = 8 
    lmax =  32
    lexp = -1
    alpha = 1

    # Envelope and Weight Matrix parameters
    x = np.arange(-n/2, n/2)
    envelope_all = True
    if not envelope_all:
        a_envelope = np.exp(-4 * (np.outer(x**2, np.ones(n)) + np.outer(np.ones(n), x**2)) / (n/2)**2) 
    else:
        a_envelope = 0.6*np.ones((n, n))

    # initialize weight matrix, input envelope and r mask
    l_inhibition = setInhibLength(h_grid, lexp, lmin, lmax)
    w, w_u, w_d, w_l, w_r, w_u_small, w_d_small, w_l_small, w_r_small, r_l_mask, r_r_mask, r_u_mask, r_d_mask = initializeWeight(n, h_grid, l_inhibition, wmag, wtphase) 

    # initialize grid pattern
    theta_v = np.pi / 2
    sin_theta_v = np.sin(theta_v)
    cos_theta_v = np.cos(theta_v)
    left, right, up, down = -cos_theta_v, cos_theta_v, sin_theta_v, -sin_theta_v
    vector = 0.5
    nflow = 200
    r = np.zeros((h_grid, n, n))
    r = initializeGrid(n, tau, dt, nflow, vector, left, right, up, down, a_envelope, r, r_l_mask, r_r_mask, r_u_mask, r_d_mask, w_r, w_l, w_d, w_u)

    # real trajectory
    file_name = './data/one_travel_before.csv'
    df = pd.read_csv(file_name)
    [origin_grid,dest_grid,origin_x,origin_y,dest_x,dest_y,x,y,vright,vleft,vup,vdown] = get_trajectory(df)
    anchor_x = origin_x + dest_x
    anchor_y = origin_y + dest_y
    grid_list = origin_grid + dest_grid
    # de-duplicate
    unique_dict = {}
    for ax, ay, grid in zip(anchor_x, anchor_y, grid_list):
        unique_dict[(ax, ay)] = grid

    anchor_list = list(unique_dict.keys())
    grid_list = list(unique_dict.values())

    # A placeholder for a single neuron response
    sNeuronResponse = np.zeros((len(vleft), 1))
    sNeuron = [n // 2, n // 2]

    [sNeuronResponse,r,coords_grid_dic,place_grid_dic] = runCann(dt, tau, n, anchor_list,grid_list,x,y, vleft, vright, vup, vdown, a_envelope, alpha, r, r_r_mask, r_l_mask, r_u_mask, r_d_mask, w_r_small, w_l_small, w_d_small, w_u_small, sNeuron)

    
    print("Saving places coordinate corresponding grid cell networks")
    with open('./data/coords_grid_data.pkl', 'wb') as file:
        pickle.dump(coords_grid_dic, file)
    
    print("Saving place corresponding grid cell networks")
    with open('./data/place_grid_data.pkl', 'wb') as file:
        pickle.dump(place_grid_dic, file)