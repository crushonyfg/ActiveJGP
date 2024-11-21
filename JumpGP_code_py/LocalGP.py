import numpy as np
from scipy.optimize import minimize
from .cov.covSum import covSum
from .lik.loglikelihood import loglikelihood
from .cov.covSEard import covSEard
from .cov.covNoise import covNoise

def LocalGP(x, y, xt):
    """
    LocalGP - Implements Local Gaussian Process regression.

    Parameters:
        x : np.ndarray
            Training inputs.
        y : np.ndarray
            Training responses.
        xt : np.ndarray
            Test inputs.

    Returns:
        mu_t : np.ndarray
            Mean prediction at xt.
        sig2_t : np.ndarray
            Variance prediction at xt.
        model : dict
            Fitted Local GP model.
    """
    
    # Define covariance functions
    cv = [covSum, [covSEard, covNoise]]

    # Initialize parameters
    d = x.shape[1]  # Number of input dimensions
    logtheta0 = -np.ones(d + 2)
    logtheta0[-1] = -1.15  # Set last element

    # Optimize log likelihood
    nIter = 100
    res = minimize(loglikelihood, logtheta0, args=(covSum, [covSEard, covNoise], x, y), options={'maxiter': nIter})
    logtheta = res.x

    # Create the model dictionary
    model = {
        'covfunc': cv,
        'logtheta': logtheta,
        'x': x,
        'y': y,
        'xt': xt
    }

    # Compute covariance matrices
    K = covSum([covSEard, covNoise], logtheta, x)  # K = feval(cv{:}, logtheta, x)
    Ktt, Kt = covSum([covSEard, covNoise], logtheta, x, xt)  # Ktt, Kt = feval(cv{:}, logtheta, x, xt)

    # Cholesky decomposition
    L = np.linalg.cholesky(K)  # Lower triangular matrix
    model['L'] = L

    # Compute mean and variance
    Ly = np.linalg.solve(L, y)  # L \ y
    LK = np.linalg.solve(L, Kt)  # L \ Kt
    mu_t = LK.T @ Ly
    sig2_t = Ktt - np.sum(LK.T**2, axis=1)[:,np.newaxis]

    # Compute negative log likelihood
    model['nll'] = loglikelihood(logtheta, covSum, [covSEard, covNoise], x, y) / x.shape[0]

    return mu_t, sig2_t, model
