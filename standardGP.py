import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import minimize

from JumpGP_code_py.cov.covSum import covSum
from JumpGP_code_py.cov.covSEard import covSEard
from JumpGP_code_py.cov.covNoise import covNoise
from JumpGP_code_py.lik.loglikelihood import loglikelihood

def standardGP(x, y, xt, logtheta=None, nIter=50):
    """
    Standard Gaussian Process function.
    """
    y = y.reshape(-1, 1)

    cv = [covSum, [covSEard, covNoise]]
    d = x.shape[1]
    logtheta0 = np.zeros(d + 2)
    logtheta0[-1] = -1  # clean data for ground truth

    # Optimize logtheta if not provided
    if logtheta is None:
        logtheta = minimize(loglikelihood, logtheta0, args=(cv[0], cv[1], x, y), options={'maxiter': nIter}).x
        # logtheta = minimize(logtheta0, loglikelihood, nIter, cv, x, y)

    # Define the GP model parameters
    model = {
        'covfunc': cv,
        'logtheta': logtheta,
        'x': x,
        'y': y,
        'xt': xt
    }

    # Calculate covariance matrices
    K = cv[0](cv[1], logtheta, x)
    Ktt, Kt = cv[0](cv[1], logtheta, x, xt)

    # Perform Cholesky decomposition and predictions
    L = cholesky(K, lower=True)
    model['L'] = L
    Ly = np.linalg.solve(L, y)
    LK = np.linalg.solve(L, Kt)
    mu_t = LK.T @ Ly
    sig2_t = Ktt - np.sum(LK ** 2, axis=0).reshape(-1,1)

    # Negative log likelihood per sample
    model['nll'] = loglikelihood(logtheta, cv[0], cv[1], x, y) / x.shape[0]

    return mu_t, sig2_t, model
