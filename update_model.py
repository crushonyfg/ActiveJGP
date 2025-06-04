import numpy as np
from scipy.linalg import cholesky
from scipy.special import expit  # for calculating 1 / (1 + exp(-x))

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from JumpGaussianProcess.calculate_gx import calculate_gx

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
    K += 1e-6 * np.eye(K.shape[0])
    L = cholesky(K, lower=True)
    
    # Calculate model update for y values
    Ly = np.linalg.solve(L, model['y'][r1,:] - model['ms'])
    LK = np.linalg.solve(L, Kt)
    new_y_value = LK.T @ Ly + model['ms']
    # print(model['y'][r1,:].shape, Ly.shape, LK.shape, new_y_value.shape)
    
    # Append the new data point's prediction to model['y']
    model['y'] = np.vstack([model['y'], new_y_value])
    # print("model['y'].shape", model['y'].shape)
    
    # Calculate gx and prior probabilities
    gx, _ = calculate_gx(xt, model['w']) # gx, phi_x #gx shape (N,), phi_x shape (N,d+1)
    prior_z = 1 / (1 + np.exp(-0.05 * model['nw'] * gx))
    pos_z = prior_z  # This could be updated if likelihoods are involved
    # print("pos_z.shape", pos_z.shape)

    # Update inclusion based on posterior probability
    model['r'] = np.append(model['r'], pos_z >= 0.5)
    model['gamma'] = np.append(model['gamma'], pos_z)
    
    return model

if __name__ == "__main__":
    from JumpGaussianProcess.JumpGP_LD import JumpGP_LD
    from JumpGaussianProcess.cov.covSum import covSum
    from JumpGaussianProcess.cov.covSEard import covSEard
    from JumpGaussianProcess.cov.covNoise import covNoise

    np.random.seed(0)       
    cv = [covSum, [covSEard, covNoise]]              # 第二项 None，代表 covFunc 不需要额外参数

    x = np.random.randn(20, 2)
    y = np.random.randn(20, 1)
    xt = np.random.randn(1, 2)
    logtheta = np.random.randn(4)

    mu_t, sig2_t, model, h = JumpGP_LD(x, y, xt, 'CEM', False)
    model = update_model(model, xt)
    
