import numpy as np
from scipy.linalg import cho_solve, cho_factor
from scipy.special import expit  # for calculating 1 / (1 + exp(-x))

from JumpGP_code_py.calculate_gx import calculate_gx

def update_model(model, xt):
    """
    Update the Gaussian Process model by adding a new data point and recalculating the covariance.

    Parameters:
    model (dict): Model parameters, including covariance function, hyperparameters, and current data.
    xt (numpy.ndarray): New data point to add to the model. (-1, D)

    Returns:
    dict: Updated model dictionary with recalculated parameters.
    """
    # Update data points
    D = model['x'].shape[1]
    if not (xt.ndim == 2 and xt.shape[1] == D):
        xt = xt.reshape(-1, D)

    r1 = model['r'].flatten()
    # Recompute covariance matrix with updated data points
    K = model['cv'][0](model['cv'][1], model['logtheta'], model['x'][r1,:])
    _, Kt = model['cv'][0](model['cv'][1],model['logtheta'], model['x'][r1,:], xt)

    model['x'] = np.vstack([model['x'], xt])
    
    # Cholesky factorization of the covariance matrix
    L = cho_factor(K, lower=True)
    
    # Calculate model update for y values
    Ly = cho_solve(L, model['y'][r1,:] - model['ms'])
    LK = cho_solve(L, Kt)
    new_y_value = LK.T @ Ly + model['ms']
    
    # Append the new data point's prediction to model['y']
    model['y'] = np.vstack([model['y'], new_y_value])
    
    # Calculate gx and prior probabilities
    gx, _ = calculate_gx(xt, model['w']) # gx, phi_x #gx shape (N,), phi_x shape (N,d+1)
    prior_z = expit(0.05 * model['nw'] * gx)
    pos_z = prior_z  # This could be updated if likelihoods are involved

    # Update inclusion based on posterior probability
    model['r'] = np.append(model['r'], pos_z >= 0.5)
    model['gamma'] = np.append(model['gamma'], pos_z)
    
    return model
