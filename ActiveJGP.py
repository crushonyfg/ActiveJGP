import numpy as np
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed  # For parallel processing

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from calculate_bias_and_variance import calculate_bias_and_variance
from JumpGaussianProcess.JumpGP_LD import JumpGP_LD

import warnings
import copy
warnings.filterwarnings("ignore", category=RuntimeWarning)

def process_candidate(m, x, y, xt, idx, k, mode, args):
    """Helper function for parallel processing"""
    D = x.shape[1]
    xt_j = xt[m, :].reshape(1, D)
    x_j = x[idx[m, :], :]
    y_j = y[idx[m, :]]

    result = {
        'bias2_change': 0,
        'var_change': 0,
        'pred': np.nan,
        'pred_var': 0,
        'bias2': 0,
        'var': 0,
        'pred_bias': 0,
        'model': None
    }

    try:
        if len(args) > 0 and args[0] is not None:
            result['pred'], _, tmp_model, _ = JumpGP_LD(x_j, y_j, xt_j, 'CEM', 0, args[0])
            result['bias2'], result['var'], result['pred_bias'], _ = calculate_bias_and_variance(tmp_model, xt_j, args[0])
        else:
            result['pred'], _, tmp_model, _ = JumpGP_LD(x_j, y_j, xt_j, 'CEM', 0)
            result['bias2'], result['var'], result['pred_bias'], _ = calculate_bias_and_variance(tmp_model, xt_j)
        
        result['pred_var'] = result['var']

        if mode == 'MAX_MSPE':
            result['bias2_change'] = result['bias2']
            result['var_change'] = result['var']
        elif mode in ['MIN_IMSPE', 'MIN_ALC']:
            result['model'] = tmp_model
        elif mode == 'MAX_VAR':
            result['bias2_change'] = 0
            result['var_change'] = result['var']

    except Exception as ex:
        print('ActiveJGP ERROR:', ex)
        
    return result

def process_second_phase(m, x, y, xt, k, N, D, mode, args, bias2, var, pred):
    """Helper function for second phase parallel processing"""
    x_candidate = xt[m, :]
    y_candidate = pred[m]
    x_new = np.vstack((x, x_candidate))
    y_new = np.append(y, y_candidate).reshape(-1,1)
    
    nbrs_new = NearestNeighbors(n_neighbors=k).fit(x_new)
    idxm = nbrs_new.kneighbors(xt, return_distance=False)

    result = {
        'bias2_change': 0,
        'var_change': 0
    }

    affected_test_locs = np.where((idxm == N).sum(axis=1) > 0)[0]
    for j in affected_test_locs:
        xt_j = xt[j, :].reshape(-1, D)
        x_j = x_new[idxm[j, :], :]
        y_j = y_new[idxm[j, :], :]

        cur_bias2 = bias2[j]
        cur_var = var[j]

        try:
            if len(args) > 0 and args[0] is not None:
                _, _, tmp_model, _ = JumpGP_LD(x_j, y_j, xt_j, 'CEM', 0, args[0])
                new_bias2, new_var, _, _ = calculate_bias_and_variance(tmp_model, xt_j, args[0])
            else:
                _, _, tmp_model, _ = JumpGP_LD(x_j, y_j, xt_j, 'CEM', 0)
                new_bias2, new_var, _, _ = calculate_bias_and_variance(tmp_model, xt_j)

            if mode == 'MIN_IMSPE':
                result['bias2_change'] += cur_bias2 - new_bias2
            result['var_change'] += cur_var - new_var
                
        except Exception as ex:
            print('Second phase error:', ex)
            
    return result

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

    # Process candidates in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_candidate)(m, x, y, xt, idx, k, mode, args)
        for m in range(M)
    )

    # Collect results
    for m, result in enumerate(results):
        bias2_changes[m] = result['bias2_change']
        var_changes[m] = result['var_change']
        pred[m] = result['pred']
        pred_var[m] = result['pred_var']
        bias2[m] = result['bias2']
        var[m] = result['var']
        pred_bias[m] = result['pred_bias']
        if result['model'] is not None:
            models[m] = result['model']

    if mode in ['MIN_IMSPE', 'MIN_ALC']:
        # print("We start the second phase of ActiveJGP!")
        
        # Process second phase in parallel
        second_phase_results = Parallel(n_jobs=-1)(
            delayed(process_second_phase)(
                m, x, y, xt, k, N, D, mode, args, bias2, var, pred
            )
            for m in range(M)
        )
        
        # Collect second phase results
        for m, result in enumerate(second_phase_results):
            bias2_changes[m] = result['bias2_change']
            var_changes[m] = result['var_change']

    # Criteria calculation and next location selection
    criteria = bias2_changes + var_changes
    best_all = np.argmax(criteria)
    xt_next = xt[best_all, :]
    pred_xt = pred[best_all]

    return xt_next, criteria, bias2_changes, var_changes, pred, pred_xt, pred_var, pred_bias

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from simulate_case_d_linear import simulate_case_d_linear
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate test data
    print("Generating test data...")
    d = 2  # Input dimension
    sig = 0.1  # Noise standard deviation
    N_train = 20  # Number of training samples
    N_test = 100  # Number of test samples
    N_candidates = 50  # Number of candidate points
    
    # Generate data using simulate_case_d_linear
    x, y, xc, yc, xt, yt, func, logtheta, cv = simulate_case_d_linear(
        d, sig, N_train, N_test, N_candidates
    )
    
    print(f"Training data: x.shape={x.shape}, y.shape={y.shape}")
    print(f"Candidate points: xc.shape={xc.shape}, yc.shape={yc.shape}")
    print(f"Test points: xt.shape={xt.shape}, yt.shape={yt.shape}")
    
    # Test different selection modes
    modes = ['MIN_IMSPE', 'MAX_MSPE', 'MAX_VAR', 'MIN_ALC']
    
    plt.figure(figsize=(15, 5))
    for i, mode in enumerate(modes):
        print(f"\nTesting mode: {mode}")
        
        # Call ActiveJGP
        xt_next, criteria, bias2_changes, var_changes, pred, pred_xt, pred_var, pred_bias = ActiveJGP(
            x, y, xc, mode, logtheta
        )
        
        print(f"Selected next point: {xt_next}")
        print(f"Predicted value: {pred_xt}")
        print(f"Bias change: {np.mean(bias2_changes)}")
        print(f"Variance change: {np.mean(var_changes)}")
        
        # Visualize results
        plt.subplot(1, 4, i+1)
        plt.scatter(x[:, 0], x[:, 1], c='blue', label='Training points')

        # Check if criteria has multiple unique values
        if np.unique(criteria).size > 1:
            scatter = plt.scatter(xc[:, 0], xc[:, 1], c=criteria, cmap='viridis', alpha=0.5, label='Candidate points')
            plt.colorbar(scatter, label='Selection criteria')
        else:
            plt.scatter(xc[:, 0], xc[:, 1], c='orange', alpha=0.5, label='Candidate points')
            print(f"Warning: All criteria values for mode {mode} are identical: {criteria[0]}")

        plt.scatter(xt_next[0], xt_next[1], c='red', s=100, marker='*', label='Selected point')
        
        # Draw decision boundary for 2D data
        if d == 2:
            x_min, x_max = min(np.min(x[:, 0]), np.min(xc[:, 0])), max(np.max(x[:, 0]), np.max(xc[:, 0]))
            y_min, y_max = min(np.min(x[:, 1]), np.min(xc[:, 1])), max(np.max(x[:, 1]), np.max(xc[:, 1]))
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            
            # Calculate labels using func function
            zz = np.array([func(p) for p in grid_points])
            zz = zz.reshape(xx.shape)
            
            plt.contour(xx, yy, zz, levels=[0], colors='green', linestyles='--')
        
        plt.title(f'Mode: {mode}')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('activejgp_test.png')
    plt.show()
    
    print("\nTest completed! Results saved to activejgp_test.png")
