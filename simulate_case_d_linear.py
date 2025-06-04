import numpy as np
from scipy.stats import multivariate_normal

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from JumpGaussianProcess.cov.covSum import covSum
from JumpGaussianProcess.cov.covSEard import covSEard
from JumpGaussianProcess.cov.covNoise import covNoise

from maxmin_design import maxmin_design

def simulate_case_d_linear(d, sig, N, Nt, Nc):
    """
    Simulate data for a d-dimensional linear case, following the setup of Park (2022).

    Parameters:
    d (int): Input dimension
    sig (float): Noise standard deviation
    N (int): Size of training data
    Nt (int): Size of test data
    Nc (int): Size of candidate locations

    Returns:
    x, y: Training inputs and responses
    xt, yt: Test inputs and (noisy) responses
    xc, yc: Candidate locations and their responses
    func: Decision function
    logtheta: Hyperparameters for the Gaussian Process
    cv: Covariance function settings
    """
    # Define random weights and bounds
    id = np.argsort(np.random.rand(d))  # Equivalent to MATLAB's [~, id] = sort(rand(d,1))
    c = np.random.randint(1, d+1)  # MATLAB's randi(d)
    a = np.ones(d)
    a[id[:c]] = -1

    # Generate test points using maxmin_design
    xt = maxmin_design(Nt, d, None) - 0.5
    D = xt @ a
    xb = xt[np.abs(D) <= 0.1, :]

    # Generate training and candidate points
    x = maxmin_design(N, d, None)
    xc = maxmin_design(Nc, d, x) - 0.5
    x = x - 0.5

    # Concatenate all points and compute labels
    xall = np.vstack((x, xt, xc))
    g = xall @ a
    lbl_all = g <= 0
    Nall = N + Nt + Nc
    func = lambda x: x @ a <= 0

    # Define covariance function and hyperparameters
    cv = [covSum, [covSEard, covNoise]]
    logtheta_d = np.zeros(d + 2)
    logtheta_d[:d] = np.log(0.1 * (d / 2))
    logtheta_d[d] = np.log(np.sqrt(9))
    logtheta_d[d + 1] = -30

    # Generate responses for each class
    yall = np.zeros(Nall)
    unique_labels = np.unique(lbl_all)
    for m, label in enumerate(unique_labels, 1):  # Start enumeration from 1 to match MATLAB
        lbl_m = lbl_all == label
        lv = round(m / 2) * 13
        if m % 2 == 1:
            lv = -lv
        xm = xall[lbl_m, :]
        K = cv[0](cv[1], logtheta_d, xm)
        yall[lbl_m] = multivariate_normal.rvs(mean=lv * np.ones(np.sum(lbl_m)), cov=K)

    # Split data and add noise
    y = yall[:N] + np.random.normal(0, sig, N)
    yt = yall[N:N+Nt]
    yc = yall[N+Nt:] + np.random.normal(0, sig, Nc)

    # Normalize responses
    mean_y = np.mean(y)
    y = y - mean_y
    yt = yt - mean_y
    yc = yc - mean_y

    # Set final logtheta
    logtheta = logtheta_d.copy()
    logtheta[-1] = np.log(sig)

    # Reshape outputs to match MATLAB's column vectors
    y = y.reshape(-1, 1)
    yt = yt.reshape(-1, 1)
    yc = yc.reshape(-1, 1)

    return x, y, xc, yc, xt, yt, func, logtheta, cv

if __name__ == "__main__":
    x, y, xc, yc, xt, yt, func, logtheta, cv = simulate_case_d_linear(2, 0.1, 10, 5, 10)
    print("x.shape", x.shape, "y.shape", y.shape, "xc.shape", xc.shape, "yc.shape", yc.shape, "xt.shape", xt.shape, "yt.shape", yt.shape)
