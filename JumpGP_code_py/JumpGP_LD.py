# % ***************************************************************************************
# %
# % JumpGP_LD - The function implements Jump GP with a linear decision boundary function
# %             described in the paper,
# % 
# %   Park, C. (2022) Jump Gaussian Process Model for Estimating Piecewise 
# %   Continuous Regression Functions. Journal of Machine Learning Research.
# %   23. 
# % 
# %
# % Inputs:
# %       x - training inputs
# %       y - training responses
# %       xt - test inputs
# %       mode - inference algorithm. It can be either 
# %                        'CEM' : Classification EM Algorithm
# %                        'VEM' : Variational EM Algorithm
# %                        'SEM' : Stochastic EM Algorithm 
# %       bVerbose (Internal Use for Debugging) 
# %                  0: do not visualize output
# %                  1: visualize output 
# % Outputs:
# %       mu_t - mean prediction at xt
# %       sig2_t - variance prediction at xt
# %       model - fitted JGP model
# %       h     - (internal use only) 
# % Copyright ©2022 reserved to Chiwoo Park (cpark5@fsu.edu) 
# % ***************************************************************************************


import numpy as np
import matplotlib.pyplot as plt

from .cov.covSum import covSum
from .cov.covSEard import covSEard
from .cov.covNoise import covNoise
from .lik.loglikelihood import loglikelihood
from .local_linearfit import local_linearfit
from .maximize_PD import maximize_PD
from .calculate_gx import calculate_gx

# Main function for JumpGP
def JumpGP_LD(x, y, xt, mode, bVerbose=None, *args, debug=None):
    """
    x: (N, d)
    y: (N, 1)
    xt: (Nt, d)
    mode: 'CEM', 'VEM', 'SEM'
    bVerbose: bool
    """
    cv = [covSum, [covSEard, covNoise]]
    d = x.shape[1]  # Get the number of features (columns)
    px = x
    pxt = xt

    # Initial estimation of the boundary B(x)
    logtheta = np.zeros(d + 2)
    logtheta[d + 2 - 1] = -1.15
    if len(args)>0:
        logtheta = args[0]
    w, _ = local_linearfit(x, y, xt)
    nw = np.linalg.norm(w)
    w = w / nw

    # Fine-tune the intercept term
    try:
        # print(w)
        b = np.arange(-1 + w[0].item(), 1 + w[0].item() , 0.01)
    except:
        print(f"x is {x}, y is {y}, xt is {xt}")
        print(f"Error with JumpGP_LD, w is {w}")
    fd = []
    for bi in b:
        w_d = w.copy()
        w_d[0] = bi
        gx, _ = calculate_gx(px, w_d)
        r = gx >= 0
        var_r = np.var(y[r], ddof=1) if np.sum(r) > 1 else 0
        var_not_r = np.var(y[~r], ddof=1) if np.sum(~r) > 1 else 0
        fd.append(np.mean(r) * var_r + np.mean(~r) * var_not_r)

    try:
        k = np.nanargmin(fd)
    except:
        print("JumpGP_LD, fd is all nan")
        k = 0
    w[0] = b[k]
    w = nw * w

    # Select algorithm
    if mode == 'CEM':
        model = maximize_PD(x, y, xt, px, pxt, w, logtheta, cv, bVerbose)
    elif mode == 'VEM':
        model = variationalEM(x, y, xt, px, pxt, w, logtheta, cv, bVerbose)
    elif mode == 'SEM':
        model = stochasticEM(x, y, xt, px, pxt, w, logtheta, cv, bVerbose)

    if debug:
        print("===== Variable Values =====")
        print(f"x:\n{x}\n")
        print(f"y:\n{y}\n")
        print(f"xt:\n{xt}\n")
        print(f"px:\n{px}\n")
        print(f"pxt:\n{pxt}\n")
        print(f"w:\n{w}\n")
        print(f"logtheta:\n{logtheta}\n")
        print(f"cv:\n{cv}\n")
        print(f"bVerbose:\n{bVerbose}\n")
        print("===========================")

    
    mu_t = model['mu_t']
    sig2_t = model['sig2_t']

    h = []
    if bVerbose:
        a = np.array([[1, -0.5], [1, 0.5]])
        if debug:
            print(f"model.w is {model['w']}")
        b_plot = -np.dot(a, model['w'][0:2]) / model['w'][2]
        # h1 = plt.plot(a, b_plot, 'r', linewidth=3)
        h1 = []
        gx, _ = calculate_gx(px, model['w'])
        h2 = plt.scatter(x[gx >= 0, 0], x[gx >= 0, 1], color='g', marker='s')
        h = [h2, h1]
        # h = [h2]
        # plt.show()
    
    return mu_t, sig2_t, model, h

# Example usage
if __name__ == "__main__":
    x_train = np.random.rand(100, 2)
    y_train = np.random.rand(100)
    x_test = np.random.rand(20, 2)
    mu_t, sig2_t, model, h = JumpGP_LD(x_train, y_train, x_test, 'CEM', 1)
