import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import minimize

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from JumpGaussianProcess.cov.covSum import covSum
from JumpGaussianProcess.cov.covSEard import covSEard
from JumpGaussianProcess.cov.covNoise import covNoise
from JumpGaussianProcess.lik.loglikelihood import loglikelihood

def LocalGP(x, y, xt, logtheta=None, nIter=100):
    """
    Standard Gaussian Process function.
    """
    y = y.reshape(-1, 1)

    cv = [covSum, [covSEard, covNoise]]
    d = x.shape[1]
    logtheta0 = np.zeros(d + 2)
    logtheta0[-1] = -1.15  # clean data for ground truth

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
    K += 1e-8 * np.eye(K.shape[0])

    # Perform Cholesky decomposition and predictions
    L = cholesky(K, lower=True)
    model['L'] = L
    Ly = np.linalg.solve(L, y)
    LK = np.linalg.solve(L, Kt)
    mu_t = LK.T @ Ly
    # sig2_t = Ktt - np.sum(LK ** 2, axis=0).reshape(-1,1)
    sig2_t = Ktt - np.sum(LK.T**2, axis=1)
    # Negative log likelihood per sample
    model['nll'] = loglikelihood(logtheta, cv[0], cv[1], x, y) / x.shape[0]

    return mu_t, sig2_t, model

if __name__ == "__main__":
    x = np.random.randn(20, 2)
    y = np.random.randn(20, 1)
    xt = np.random.randn(5, 2)
    logtheta = np.random.randn(4)

    mu_t, sig2_t, model = LocalGP(x, y, xt, logtheta)
    print(mu_t, sig2_t)
    print("mu_t.shape", mu_t.shape, "sig2_t.shape", sig2_t.shape)
