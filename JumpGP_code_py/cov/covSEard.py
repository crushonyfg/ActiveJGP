import numpy as np
from .sq_dist import sq_dist

def covSEard(loghyper=None, x=None, z=None):
    """
    Squared Exponential covariance function with ARD.
    
    Args:
    loghyper : array of hyperparameters [log(ell_1), ..., log(ell_D), log(sqrt(sf2))]
    x : input data matrix (n x D)
    z : optional test set data matrix (m x D), or a scalar to indicate a derivative calculation.
    
    Returns:
    A : covariance matrix or derivative matrix
    B : optional cross-covariance matrix if z is provided
    """
    
    # Step 1: Return number of hyperparameters if no input is provided
    if loghyper is None:
        return '(D+1)'

    n, D = x.shape
    ell = np.exp(loghyper[:D])  # Characteristic length scales
    sf2 = np.exp(2 * loghyper[D])  # Signal variance

    # Step 2: Compute covariance matrix when z is not provided or is an integer
    if z is None:
        K = sf2 * np.exp(-sq_dist(np.diag(1./ell) @ x.T) / 2)
        return K

    # Step 3: If z is an array, compute test set covariances
    elif isinstance(z, np.ndarray):
        A = sf2 * np.ones((z.shape[0], 1))  # Variance for test set
        B = sf2 * np.exp(-sq_dist(np.diag(1./ell) @ x.T, np.diag(1./ell) @ z.T) / 2)
        return A, B

    # Step 4: If z is an integer, compute derivative matrix
    else:
        if z <= D:  # Length scale parameters
            K = sf2 * np.exp(-sq_dist(np.diag(1./ell) @ x.T) / 2)
            A = K * sq_dist(x[:, z-1].T / ell[z-1])
        else:  # Signal variance parameter
            K = sf2 * np.exp(-sq_dist(np.diag(1./ell) @ x.T) / 2)
            A = 2 * K
        return A
