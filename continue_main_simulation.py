import numpy as np
import random
from scipy.spatial.distance import cdist

from calcALC import calcALC
from ActiveJGP import ActiveJGP
from JumpGP.LocalGP import standardGP

def continue_main_simulation(pred, pred_var, mse, rmse, nlpd, x_AL, y_AL, xt, yt, xc, yc, S, decfunc, outputfile):
    """
    Simulate the active learning process for different methods in Python.
    """

    sig = 2  # Standard deviation for clean data
    cv = ('covSum', ['covSEard', 'covNoise'])
    d = x_AL[0].shape[1]
    logtheta = np.zeros(d + 2)
    logtheta[:d] = np.log(0.1)
    logtheta[d] = np.log(np.sqrt(9))
    logtheta[d + 1] = np.log(sig)
    
    methods = ['MIN_IMSPE', 'MIN_ALC', 'MAX_MSPE', 'MAX_VAR', 'GP_ALC', 'JGP_LHD', 'GP_LHD']
    Sinit = mse.shape[0]
    step = 0.005
    N = x_AL[0].shape[0] - Sinit

    for s in range(Sinit+1, Sinit + S+1):
        for k, method in enumerate(methods):
            xc_ind = random.sample(range(xc.shape[0]), round(xc.shape[0] * 0.2))
            xc_s = xc[xc_ind, :]
            yc_s = yc[xc_ind, :]

            # Calculate selection criteria
            if k in {0, 1, 2, 3}:
                if k in {0, 2}:  # MIN_IMSPE or MAX_MSPE
                    _, criteria, bias2_changes, var_changes, _, _, _ = ActiveJGP(x_AL[k], y_AL[k], xc_s, method, logtheta, min(12, max(8, round((N + s - 1) / 4))))
                else:
                    criteria = var_changes
            elif k == 4:  # GP_ALC
                criteria = calcALC(x_AL[k], xc_s, cv, logtheta)
            else:  # LHD
                dist = cdist(xc_s, x_AL[k])
                criteria = np.min(dist, axis=1)

            # Make predictions at xt
            if k in {0, 1, 2, 3, 5}:
                _, _, _, _, pred[s, k], _, pred_var[s, k] = ActiveJGP(x_AL[k], y_AL[k], xt, 'MAX_MSPE', logtheta, min(12, max(8, round((N + s - 1) / 4))))
            else:
                pred[s, k], pred_var[s, k] = standardGP(x_AL[k], y_AL[k], xt, logtheta)

            # Update training set with new point
            ia = np.where(np.min(cdist(xc_s, x_AL[k]), axis=1) > step)[0]
            ib = np.argmax(criteria[ia])
            x_next = xc_s[ia[ib], :]

            x_AL[k] = np.vstack((x_AL[k], x_next))
            y_AL[k] = np.vstack((y_AL[k], yc_s[ia[ib], :]))

            # Calculate errors and log probability
            diff = pred[s, k] - yt
            v = pred_var[s, k]
            mse[s, k] = np.mean(diff ** 2)
            rmse[s, k] = np.mean(np.abs(diff))
            nlpd[s, k] = 0.5 * np.mean((diff ** 2 / v) + np.log(2 * np.pi * v))

            # Save simulation results
            np.savez(outputfile, pred=pred, pred_var=pred_var, mse=mse, rmse=rmse, nlpd=nlpd, 
                     x_AL=x_AL, y_AL=y_AL, xt=xt, yt=yt, xc=xc, yc=yc, decfunc=decfunc)

        # Update plotting here if needed
        # plotResults(rmse)
