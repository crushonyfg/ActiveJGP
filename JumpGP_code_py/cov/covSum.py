import numpy as np

def covSum(covfunc, logtheta=None, x=None, z=None):
    """
    covSum - Compose a covariance function as the sum of other covariance functions.
    This function handles bookkeeping and calls other covariance functions to do the actual work.

    Args:
    covfunc : list of covariance functions (each is callable)
    logtheta : list or array of parameters
    x : input data matrix (n x D)
    z : optional test data matrix (m x D)

    Returns:
    A : Covariance matrix or derivative matrix
    B : Optional cross-covariance matrix if z is provided
    """
    param_count = [f() for f in covfunc]  # Assuming each function returns its param count

    # Step 1: Check number of parameters if no data is provided
    if logtheta is None:
        # param_count = [f() for f in covfunc]  # Assuming each function returns its param count
        A = '+'.join([str(c) for c in param_count])
        return A

    # Step 2: Perform actual covariance computation
    n, D = x.shape
    v = []  # Vector indicates which covariance parameters belong to which function
    for i, func in enumerate(covfunc):
        param_count = eval(func())  # Assuming func returns the number of parameters it needs
        v.extend([i] * param_count)

    # Step 3: Depending on number of input arguments, calculate covariances or derivatives
    if z is None:
        # Case when only x is provided, compute covariance matrix
        A = np.zeros((n, n))
        for i, func in enumerate(covfunc):
            params = logtheta[np.array(v) == i]
            A += func(params, x)
        return A

    elif isinstance(z, int):
        # Compute derivative matrix with respect to parameter z
        i = v[z]  # which covariance function
        j = sum(np.array(v[:z]) == i)  # which parameter in that covariance
        func = covfunc[i]
        params = logtheta[np.array(v) == i]
        A = func(params, x, j)  # compute derivative
        return A

    else:
        # Case when both x and z are provided, compute test set covariances
        A = np.zeros((z.shape[0], 1))
        B = np.zeros((x.shape[0], z.shape[0]))
        for i, func in enumerate(covfunc):
            params = logtheta[np.array(v) == i]
            AA, BB = func(params, x, z)
            A += AA
            B += BB
        return A, B
