import numpy as np
from scipy.stats import multivariate_normal

from JumpGP_code_py.cov.covSum import covSum
from JumpGP_code_py.cov.covSEard import covSEard
from JumpGP_code_py.cov.covNoise import covNoise

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
    idx = np.random.permutation(d)
    c = np.random.randint(d)
    a = np.ones(d)
    a[idx[:c]] = -1

    # Generate test points and design matrix
    xt = maxmin_design(Nt, d) - 0.5
    D = xt @ a
    xb = xt[np.abs(D) <= 0.1, :]

    # Training and candidate locations
    x = maxmin_design(N, d) - 0.5
    xc = maxmin_design(Nc, d, x) - 0.5

    # Concatenate all points and define ground truth labels
    xall = np.vstack((x, xt, xc))
    g = xall @ a
    lbl_all = g <= 0
    Nall = N + Nt + Nc
    func = lambda x: x @ a <= 0

    # Define covariance function and hyperparameters for response generation
    cv = [covSum, [covSEard, covNoise]]
    logtheta_d = np.zeros(d + 2)
    logtheta_d[:d] = np.log(0.1 * (d / 2))
    logtheta_d[d] = np.log(np.sqrt(9))
    logtheta_d[d + 1] = -30

    # Generate responses for each class based on GP prior
    yall = np.zeros(Nall)
    unique_labels = np.unique(lbl_all)
    for m, label in enumerate(unique_labels):
        lbl_m = lbl_all == label
        lv = round((m + 1) / 2) * 13
        lv = -lv if m % 2 == 0 else lv
        xm = xall[lbl_m, :]
        K = cv[0](cv[1], logtheta_d, xm)  # Define covSum function separately
        yall[lbl_m] = multivariate_normal.rvs(mean=lv * np.ones(sum(lbl_m)), cov=K)

    # Separate into training, test, and candidate sets
    y = yall[:N] + np.random.normal(0, sig, N)
    yt = yall[N:N+Nt]
    yc = yall[N+Nt:] + np.random.normal(0, sig, Nc)

    # Normalize responses
    mean_y = np.mean(y)
    y -= mean_y
    yt -= mean_y
    yc -= mean_y

    # Adjust logtheta for noisy response
    logtheta_d[-1] = np.log(sig)
    logtheta = logtheta_d

    return x, y.reshape(-1,1), xc, yc.reshape(-1,1), xt, yt.reshape(-1,1), func, logtheta, cv
