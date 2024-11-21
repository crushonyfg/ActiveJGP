import numpy as np
from scipy.linalg import cho_solve, cho_factor

def loglikelihood(logtheta, covfunc1, covfunc2, x, y, nargout=1):
    """
    loglikelihood - Computes the negative log-likelihood and its partial derivatives with
    respect to the hyperparameters.

    Parameters:
    logtheta : numpy array
        A vector of log hyperparameters.
    covfunc : function
        Covariance function used to compute the covariance matrix.
    x : numpy array
        Training inputs, an n by D matrix.
    y : numpy array
        Target outputs, a vector of size n.

    Returns:
    loglike : float
        The negative log-likelihood of the data under the GP model.
    dloglike : numpy array
        The partial derivatives of the log-likelihood with respect to hyperparameters.
    """

    # Compute the covariance matrix K
    K = covfunc1(covfunc2, logtheta, x)

    # Cholesky decomposition of the covariance matrix K
    K += 1e-6 * np.eye(K.shape[0])
    L = np.linalg.cholesky(K)
    
    # Solve for alpha
    alpha = cho_solve((L, True), y)

    # Compute the negative log-likelihood
    loglike = (0.5 * y.T @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(y) * np.log(2 * np.pi)).item()

    if nargout == 2:  # If partial derivatives are requested
        dloglike = np.zeros_like(logtheta)
        
        # Precompute for convenience
        W = cho_solve((L, True), np.eye(len(x))) - np.outer(alpha, alpha)

        # Compute partial derivatives
        for i in range(len(logtheta)):
            dK_dtheta = covfunc1(covfunc2, logtheta, x, i)  # Derivative of covariance matrix w.r.t. hyperparameter i
            dloglike[i] = 0.5 * np.sum(W * dK_dtheta)

        return loglike, dloglike
    
    return loglike

# Example usage (assuming you have a valid covariance function):
# logtheta = np.array([log_param_1, log_param_2, ...])
# loglike, dloglike = loglikelihood(logtheta, covfunc, x_train, y_train)
