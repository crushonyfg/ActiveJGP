import numpy as np
from scipy.linalg import cholesky

def calcALC(x, xref, cv, logtheta):
    """
    Calculate the Active Learning Cohn (ALC) score for Gaussian Process.
    
    Parameters:
        x (np.array): Training inputs (N x D)
        xref (np.array): Reference points (Nref x D)
        cv (function): Covariance function that takes (logtheta, points1, points2)
        logtheta (np.array): Log hyperparameters for the covariance function

    Returns:
        ALC (np.array): ALC scores for each reference point (Nref, )
    """
    Nref = xref.shape[0]
    ALC = np.zeros(Nref)

    for i in range(Nref):
        # Combine x and xref[i] to get full set of points
        K = cv[0](cv[1], logtheta, np.vstack((x, xref[i, :])))
        
        # Calculate covariance between x and each reference point xref
        _, Kt = cv[0](cv[1], logtheta, np.vstack((x, xref[i, :])), xref)

        # Cholesky decomposition and calculation of LK
        L = cholesky(K, lower=True)
        LK = np.linalg.solve(L, Kt)
        
        # Calculate ALC score for the i-th reference point
        ALC[i] = np.sum(np.sum(LK.T ** 2, axis=1))

    return ALC
