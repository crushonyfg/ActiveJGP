import numpy as np
from sklearn.neighbors import NearestNeighbors

from calculate_bias_and_variance import calculate_bias_and_variance
from JumpGP_code_py.JumpGP_LD import JumpGP_LD

import warnings
import copy
warnings.filterwarnings("ignore", category=RuntimeWarning)


def ActiveJGP(x, y, xt, mode, *args, debug=None):
    """
    Python translation of the ActiveJGP function from MATLAB.
    
    Parameters:
    x : numpy array, shape (N, D)
        Training inputs.
    y : numpy array, shape (N, 1)
        Training outputs.
    xt : numpy array, shape (M, D)
        Candidate locations.
    mode : str
        Selection mode ('MIN_IMSPE', 'MAX_MSPE', 'MAX_VAR', 'MIN_ALC').
    args : additional arguments
        Optional arguments to control behavior.

    Returns:
    xt_next : numpy array
        Next location to sample.
    criteria : array
        Selection criteria values for candidates.
    bias2_changes, var_changes, pred, pred_xt, pred_var, pred_bias : arrays
        Various predictive and variance values.
    """
    N, D = x.shape
    y = y.reshape(-1,1)
    if y.shape[0] != N:
        raise ValueError("Row sizes of x and y should be the same!")
    if xt.shape[1] != D:
        raise ValueError("Column size of x should match column size of xt!")

    # Handling optional arguments
    k = args[1] if len(args) > 1 else min(15, round(N * 0.2))

    M = xt.shape[0]
    bias2_changes = np.zeros(M)
    var_changes = np.zeros(M)
    pred = np.zeros(M)
    pred_var = np.zeros(M)
    pred_bias = np.zeros(M)
    bias2 = np.zeros(M)
    var = np.zeros(M)

    # Nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(x)
    idx = nbrs.kneighbors(xt, return_distance=False)
    # print(idx)

    models = [None] * M

    for m in range(M):
        xt_j = copy.deepcopy(xt[m, :].reshape(-1, D))
        x_j = copy.deepcopy(x[idx[m, :], :])
        y_j = copy.deepcopy(y[idx[m, :]])
        if debug:
            print(f"check data shape: xj {x_j.shape}, yj {y_j.shape}, xtj shape {xt_j.shape}")

        try:
            if len(args) > 0 and args[0] is not None:
                pred[m], _, tmp_model, _ = JumpGP_LD(x_j, y_j, xt_j, 'CEM', 0, args[0]) # output: mu_t, sig2_t, model, h
                bias2[m], var[m], pred_bias[m], _ = calculate_bias_and_variance(tmp_model, xt_j, args[0]) # bias2.item(), var.item(), bias.item(), parts.reshape(-1,1)
            else:
                pred[m], _, tmp_model, _ = JumpGP_LD(x_j, y_j, xt_j, 'CEM', 0)
                if pred[m]==np.nan:
                    print(f"xj is {x_j}, yj is {y_j}, xtj is {xt_j}")
                bias2[m], var[m], pred_bias[m], _ = calculate_bias_and_variance(tmp_model, xt_j)
            pred_var[m] = var[m]
        except Exception as ex:
            print('ActiveJGP ERROR with pred nan!:',ex)
            # print(f"m is {m}, idx is {idx[m, :]}, y is {y}")
            # print(f"x is {x_j}, y is {y_j}, xt is {xt_j}")
            pred[m] = np.nan

        # print(f"now m is {m}, the prediction value is {pred[m]}")
            

        if mode == 'MAX_MSPE':
            bias2_changes[m] = bias2[m]
            var_changes[m] = var[m]

        if mode in ['MIN_IMSPE', 'MIN_ALC']:
            models[m] = tmp_model

        if mode == 'MAX_VAR':
            bias2_changes[m] = 0
            var_changes[m] = var[m]

    if mode in ['MIN_IMSPE', 'MIN_ALC']:
        print("We start the second phase of ActiveJGP!")
        # print(f"check is there any nan in prediction:{pred}")
        for m in range(M):
            x_candidate = xt[m, :]
            y_candidate = pred[m]
            x_new = np.vstack((x, x_candidate))
            y_new = np.append(y, y_candidate).reshape(-1,1)
            
            nbrs_new = NearestNeighbors(n_neighbors=k).fit(x_new)
            idxm = nbrs_new.kneighbors(xt, return_distance=False)

            bias2_changes[m] = 0
            var_changes[m] = 0

            affected_test_locs = np.where((idxm == N).sum(axis=1) > 0)[0]
            for j in affected_test_locs:
                xt_j = copy.deepcopy(xt[j, :].reshape(-1, D))
                x_j = copy.deepcopy(x_new[idxm[j, :], :])
                y_j = copy.deepcopy(y_new[idxm[j, :], :])

                cur_bias2 = bias2[j]
                cur_var = var[j]

                if len(args) > 0 and args[0] is not None:
                    _, _, tmp_model, _ = JumpGP_LD(x_j, y_j, xt_j, 'CEM', 0, args[0])
                    new_bias2, new_var, _, _ = calculate_bias_and_variance(tmp_model, xt_j, args[0])
                else:
                    _, _, tmp_model, _ = JumpGP_LD(x_j, y_j, xt_j, 'CEM', 0)
                    new_bias2, new_var, _, _ = calculate_bias_and_variance(tmp_model, xt_j)

                if mode == 'MIN_IMSPE':
                    bias2_changes[m] += cur_bias2 - new_bias2
                var_changes[m] += cur_var - new_var

    # Criteria calculation and next location selection
    criteria = bias2_changes + var_changes
    best_all = np.argmax(criteria)
    xt_next = xt[best_all, :]
    pred_xt = pred[best_all]

    return xt_next, criteria, bias2_changes, var_changes, pred, pred_xt, pred_var, pred_bias
