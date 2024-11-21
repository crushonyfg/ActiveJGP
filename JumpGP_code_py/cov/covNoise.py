import numpy as np
# from cov.sq_dist import ensure_2d

def covNoise(logtheta=None, x=None, z=None):
    """
    Independent covariance function (white noise) with specified variance.
    
    Args:
    logtheta : array of hyperparameters [log(sqrt(s2))]
    x : input data matrix (n x D)
    z : optional test set data matrix (m x D), not used in this covariance function.
    
    Returns:
    A : covariance matrix or derivative matrix
    B : optional cross-covariance matrix if z is provided
    """
    
    # Step 1: Return number of hyperparameters if no input is provided
    if x is None:
        return '1'

    s2 = np.exp(2 * logtheta)  # Noise variance
    if s2.shape == (1,):
        s2 = s2.item()

    # Step 2: Compute covariance matrix when z is not provided
    if z is None:
        A = np.dot(s2,np.eye(x.shape[0]))  # Diagonal matrix with noise variance
        return A
    
    # Step 3: If z is provided, compute test set covariances
    elif isinstance(z, np.ndarray):
    # elif n_out==2:
        A = s2  # Noise variance for test set
        B = 0   # Cross covariance is zero (independence)
        return A, B

    # Step 4: Compute derivative matrix if z is not used
    else:
        # if isinstance(s2, float):
        #     A = 2 * s2 * np.eye(x.shape[0])  # Derivative with respect to noise variance
        # else:
        A = 2 * np.dot(s2,np.eye(x.shape[0])) 
        return A
