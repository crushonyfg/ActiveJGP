import numpy as np
import random

from simulate_case_d_linear import simulate_case_d_linear
from ActiveJGP import ActiveJGP
from plotResults import plotResults
from calcALC import calcALC
from standardGP import standardGP

from scipy.spatial.distance import cdist
import copy


def run_main_simulation(d, N, Nt, Nc, S, outputfile):
    """
    Run main simulation with various active learning or Bayesian optimization methods.
    """
    # Initialize simulation parameters
    sig = 2
    x, y, xc, yc, xt, yt, decfunc, logtheta, cv = simulate_case_d_linear(d, sig, N, Nt, Nc)
    methods = ['MIN_IMSPE', 'MIN_ALC', 'MAX_MSPE', 'MAX_VAR', 'GP_ALC', 'JGP_LHD', 'GP_LHD']

    # keep y's as two-dim
    y = y.reshape(-1, 1)
    yc = yc.reshape(-1, 1)
    yt = yt.reshape(-1, 1)
    
    x_AL = {k: copy.deepcopy(x) for k in range(len(methods))}
    y_AL = {k: copy.deepcopy(y) for k in range(len(methods))}
    step = 0.005

    K = len(methods)
    pred = {}
    pred_var = {}
    mse = np.full((S, K), np.nan)
    rmse = np.full((S, K), np.nan)
    nlpd = np.full((S, K), np.nan)

    # Iterate over sampling steps
    for s in range(S):
        for k, method in enumerate(methods):
            # Sample subset of candidate points
            xc_ind = random.sample(range(xc.shape[0]), round(xc.shape[0] * 0.2))
            xc_s = copy.deepcopy(xc[xc_ind, :])
            yc_s = copy.deepcopy(yc[xc_ind, :])

            # Calculate selection criteria based on method
            if k in [0, 1, 2, 3]:
                if k in [0, 2]:  # MIN_IMSPE or MAX_MSPE
                    _, criteria, _, var_changes, *_ = ActiveJGP(x_AL[k], y_AL[k], xc_s, method, logtheta,
                                                            min(12, max(8, round((N + s) / 4))))
                else:
                    criteria = var_changes
            elif k == 4:  # ALC with GP
                criteria = calcALC(x_AL[k], xc_s, cv, logtheta)
            elif k in [5, 6]:  # LHD
                # dist = pdist2(xc_s, x_AL[k])
                dist = cdist(xc_s, x_AL[k], metric='euclidean')
                criteria = np.min(dist, axis=1)

            # Make predictions at test points
            if k in [0, 1, 2, 3, 5]:
                _, _, _, _, pred_value, _, pred_var_value, _ = ActiveJGP(x_AL[k], y_AL[k], xt, 'MAX_MSPE', logtheta,
                                                                  min(12, max(8, round((N + s) / 4))))
            else:
                pred_value, pred_var_value, _ = standardGP(x_AL[k], y_AL[k], xt, logtheta)
            
            pred[(s, k)] = pred_value
            pred_var[(s, k)] = pred_var_value
            
            # Select the next point to add based on criteria and update sets
            # ia = np.where(np.min(pdist2(xc_s, x_AL[k]), axis=1) > step)[0]
            ia = np.where(np.min(cdist(xc_s, x_AL[k], metric='euclidean'), axis=1) > step)[0]
            ib = np.argmax(criteria[ia])
            x_next = xc_s[ia[ib], :]

            x_AL[k] = np.vstack([x_AL[k], x_next])
            # y_AL[k] = np.vstack([y_AL[k], yc_s[ia[ib], :]])
            y_AL[k] = np.vstack([y_AL[k].reshape(-1,1), yc_s[ia[ib], :].reshape(-1,1)])

            # Calculate errors and metrics
            diff = pred_value - yt
            v = pred_var_value
            mse[s, k] = np.mean(diff ** 2)
            rmse[s, k] = np.mean(np.abs(diff))
            nlpd[s, k] = 0.5 * np.mean(diff ** 2 / v + np.log(2 * np.pi * v))


            # Save results incrementally
            np.savez(outputfile, pred=pred, pred_var=pred_var, mse=mse, rmse=rmse, nlpd=nlpd,
                     x_AL=x_AL, y_AL=y_AL, xt=xt, yt=yt, xc=xc, yc=yc)

        # Plot results (if a function like `plotResults` is defined)
        plotResults(rmse)
        # plt.pause(0.1)

    return pred, pred_var,mse, rmse, nlpd, x_AL, y_AL, xt, yt, xc, yc