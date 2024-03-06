import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from tqdm import tqdm

def setInhibLength(h_grid, lexp, lmin, lmax):
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

def wValue(x, y, l_value, wmag):
    """ 
    General Equation for inhibition
    """
    r = np.sqrt(x*x+y*y)

    if r < 2 * l_value:
        w = -wmag/(2*l_value*l_value) * (1 - np.cos(np.pi*r/l_value))
    else:
        w = 0.

    return w

def initializeWeight(n, h_grid, l_inhibition, wmag, wtphase):
    # Envelope and Weight Matrix parameters
    x = np.arange(-n/2, n/2)
    
    w = np.zeros((h_grid, n, n))
    for k in range(0, h_grid, 1):
        for i in range(0, n, 1):
            for j in range(0, n, 1):
                w[k, i, j] = wValue(x[i], x[j], l_inhibition[k], wmag) if h_grid != 1 else wValue(x[i], x[j], l_inhibition, wmag)

    # Create shifted weight matrices for each preferred firing direction
    w_rshift = np.roll(w, shift=wtphase, axis=2)  # Right shift
    w_lshift = np.roll(w, shift=-wtphase, axis=2)  # Left shift
    w_dshift = np.roll(w, shift=wtphase, axis=1)  # Down shift
    w_ushift = np.roll(w, shift=-wtphase, axis=1)  # Up shift

    # Transform them to obtain h(t)
    big = 2 * n
    w_u = fft2(w_ushift, (big, big))
    w_d = fft2(w_dshift, (big, big))
    w_l = fft2(w_lshift, (big, big))
    w_r = fft2(w_rshift, (big, big))

    # Block matrices used for identifying all neurons of one preferred firing direction
    dim = n // 2
    r_l_mask = np.tile(np.array([[1, 0], [0, 0]]), (dim, dim))
    r_r_mask = np.tile(np.array([[0, 0], [0, 1]]), (dim, dim))
    r_u_mask = np.tile(np.array([[0, 1], [0, 0]]), (dim, dim))
    r_d_mask = np.tile(np.array([[0, 0], [1, 0]]), (dim, dim))

    return w, w_u, w_d, w_l, w_r, r_l_mask, r_r_mask, r_u_mask, r_d_mask

def initializeGrid(n, tau, dt, nflow, vector, left, right, up, down, v_envelope, r_origin, r_l, r_r, r_u, r_d, w_r_origin, w_l_origin, w_d_origin, w_u_origin, grid_show=False):
    # Create a function to update r
    def update_r(r, w_r, w_l, w_d, w_u):
        rfield = v_envelope * ((1 + vector * right) * r_r + (1 + vector * left) * r_l +
                               (1 + vector * up) * r_u + (1 + vector * down) * r_d)
        convolution = np.real(ifft2(
            fft2(r * r_r, s=(2*n, 2*n)) * w_r +
            fft2(r * r_l, s=(2*n, 2*n)) * w_l +
            fft2(r * r_d, s=(2*n, 2*n)) * w_d +
            fft2(r * r_u, s=(2*n, 2*n)) * w_u
        ))
        rfield += convolution[n//2:3*n//2, n//2:3*n//2]
        fr = (rfield > 0) * rfield
        return np.minimum(10, (dt / tau) * (5 * fr - r) + r)

    # Iterate over k and iter
    for k in tqdm(range(len(r_origin)), desc='Initialize grid pattern'):
        r = r_origin[k,:,:]
        w_r = w_r_origin[k,:,:]
        w_l = w_l_origin[k,:,:]
        w_d = w_d_origin[k,:,:]
        w_u = w_u_origin[k,:,:]

        for iter in range(nflow):
            r = update_r(r, w_r, w_l, w_d, w_u)

            if grid_show and iter % 20 == 0:
                plt.imshow(r, vmin=0, vmax=2, cmap='hot')
                plt.colorbar()
                plt.title(f'Neural Population Activity {iter}')
                plt.draw()
                plt.pause(0.01)
                plt.clf()

        r_origin[k,:,:] = r

    return r_origin

def runCann(dt, tau, n, anchor_list, x, y, vleft, vright, vup, vdown, v_envelope, alpha, r_origin, r_r, r_l, r_u, r_d, w_r_origin, w_l_origin, w_d_origin, w_u_origin, sNeuron, useSpiking = True, grid_show = True):
    """
    Run the Continuous Attractor Neural Network (CANN) simulation.

    Args:
    dt, tau, n, vleft, vright, vup, vdown, v_envelope, alpha: Simulation parameters.
    r_origin, r_r, r_l, r_u, r_d: Neural activity and directional components.
    w_r_small_origin, w_l_small_origin, w_d_small_origin, w_u_small_origin: Weight matrices.
    sNeuron: Specific neuron for tracking.
    useSpiking: Flag to use spiking neurons.
    grid_show: Flag to show grid plots during simulation.

    Returns:
    sNeuronResponse: Response of the specified neuron.
    r: Final state of the neural grid.
    """
    sampling_length = len(vleft)
    grid_layer = 0
    coords_grid_dic = dict()
    sNeuronResponse = [0.0]*sampling_length

    for iter in tqdm(range(sampling_length), desc='Processing'):
    # for iter in range(sampling_length):
        for k in range(len(r_origin)):
            r = r_origin[k, :, :]
            w_r = w_r_origin[k, :, :]
            w_l = w_l_origin[k, :, :]
            w_d = w_d_origin[k, :, :]
            w_u = w_u_origin[k, :, :]

            rfield = v_envelope * ((1 + alpha * vright[iter]) * r_r +
                                (1 + alpha * vleft[iter]) * r_l +
                                (1 + alpha * vup[iter]) * r_u +
                                (1 + alpha * vdown[iter]) * r_d)

            # Convolution
            convolution = np.real(ifft2(
                fft2(r * r_r, s=(2*n, 2*n)) * w_r +
                fft2(r * r_l, s=(2*n, 2*n)) * w_l +
                fft2(r * r_d, s=(2*n, 2*n)) * w_d +
                fft2(r * r_u, s=(2*n, 2*n)) * w_u
            ))

            rfield += convolution[n//2:3*n//2, n//2:3*n//2]

            # Neural Transfer Function
            fr = np.where(rfield > 0, rfield, 0)

            # Neuron dynamics
            r_old = r
            r_new = np.minimum(10, (dt / tau) * (5 * fr - r_old) + r_old)
            r = r_new

            if useSpiking:
                spike = rfield * dt > np.random.rand(n, n)
                r += (dt / tau) * (-r + (tau / dt) * spike)
                if k == grid_layer:
                    sNeuronResponse[iter] = r[sNeuron[0], sNeuron[1]]

            if grid_show and iter % 20 == 0 and k==grid_layer:
                plt.imshow(r, vmin=0, vmax=2, cmap='hot')
                plt.colorbar()
                plt.title(f'Neural Population Activity - Iteration {iter}, Layer {k+1}')
                plt.draw()
                plt.pause(0.01)
                plt.clf()

            r_origin[k, :, :] = r
        
        if (x[iter],y[iter]) in anchor_list:
            coords_grid_dic[(x[iter], y[iter])] = np.copy(r_origin)

    return sNeuronResponse, r, coords_grid_dic