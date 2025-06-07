# active_learning.py

import numpy as np
from scipy.spatial.distance import cdist
import copy
from ActiveJGP import ActiveJGP
from standardGP import standardGP
from calcALC import calcALC
from plotResults import plotResults
from tqdm import tqdm

class ActiveLearner:
    def __init__(self, x, y, xc, yc, xt, yt, method_name, S, use_subsample=True, ratio=0.2, logtheta=None, cv=None):
        """
        initialize the active learner
        
        Parameters:
        x: initial training data features (N x d)
        y: initial training data labels (N x 1)
        xc: candidate points features (Nc x d)
        yc: candidate points labels (Nc x 1)
        xt: test points features (Nt x d)
        yt: test points labels (Nt x 1)
        method_name: active learning method name ('MIN_IMSPE', 'MIN_ALC', 'MAX_MSPE', 'MAX_VAR', 'GP_ALC', 'JGP_LHD', 'GP_LHD')
        S: number of active learning iterations
        logtheta: Gaussian process hyperparameters
        cv: covariance function parameters
        """
        # ensure the input data format is correct
        self.x = x
        self.y = y.reshape(-1, 1)
        self.xc = xc
        self.yc = yc.reshape(-1, 1)
        self.xt = xt
        self.yt = yt.reshape(-1, 1)
        
        self.method_name = method_name
        self.S = S
        self.logtheta = logtheta
        if cv is None:
            from JumpGaussianProcess.cov.covSum import covSum
            from JumpGaussianProcess.cov.covSEard import covSEard
            from JumpGaussianProcess.cov.covNoise import covNoise
            self.cv = [covSum, [covSEard, covNoise]]
        else:   
            self.cv = cv
        
        # initialize the arrays to store the results
        self.x_AL = copy.deepcopy(x)
        self.y_AL = copy.deepcopy(self.y)
        self.mse = np.full(S, np.nan)
        self.rmse = np.full(S, np.nan)
        self.nlpd = np.full(S, np.nan)
        self.pred_history = {}
        self.pred_var_history = {}
        
        # other parameters
        self.step = 0.005  # minimum distance threshold
        self.use_subsample = use_subsample
        self.ratio = ratio
        
    def _select_candidates(self, xc, yc, ratio=0.2):
        """randomly select a subset of candidate points"""
        xc_ind = np.random.choice(xc.shape[0], size=round(xc.shape[0] * ratio), replace=False)
        return xc[xc_ind, :], yc[xc_ind, :]
    
    def _calculate_criteria(self, xc_s, neighbor_num=None):
        """calculate the selection criteria according to the method"""
        if neighbor_num is None:
            neighbor_num = min(12, max(8, round((self.x.shape[0] + len(self.pred_history)) / 4)))
        if self.method_name in ['MIN_IMSPE', 'MAX_MSPE']:
            _, criteria, _, var_changes, *_ = ActiveJGP(
                self.x_AL, 
                self.y_AL, 
                xc_s, 
                self.method_name, 
                self.logtheta,
                neighbor_num
            )
            return criteria
        
        elif self.method_name == 'MIN_ALC':
            # use var_changes of MIN_IMSPE
            _, _, _, var_changes, *_ = ActiveJGP(
                self.x_AL, 
                self.y_AL, 
                xc_s, 
                'MIN_IMSPE',  # change to MIN_IMSPE
                self.logtheta,
                neighbor_num
            )
            return var_changes
        
        elif self.method_name == 'MAX_VAR':
            # use var_changes of MAX_MSPE
            _, _, _, var_changes, *_ = ActiveJGP(
                self.x_AL, 
                self.y_AL, 
                xc_s, 
                'MAX_MSPE',  # keep as MAX_MSPE
                self.logtheta,
                neighbor_num
            )
            return var_changes
        
        elif self.method_name == 'GP_ALC':
            return calcALC(self.x_AL, xc_s, self.cv, self.logtheta)
        
        elif self.method_name in ['JGP_LHD', 'GP_LHD']:
            dist = cdist(xc_s, self.x_AL, metric='euclidean')
            return np.min(dist, axis=1)
    
    def _make_prediction(self):
        """make predictions at the test points"""
        if self.method_name in ['MIN_IMSPE', 'MIN_ALC', 'MAX_MSPE', 'MAX_VAR', 'JGP_LHD']:
            _, _, _, _, pred_value, _, pred_var_value, _ = ActiveJGP(
                self.x_AL, 
                self.y_AL, 
                self.xt, 
                'MAX_MSPE', 
                self.logtheta,
                min(12, max(8, round((self.x.shape[0] + len(self.pred_history)) / 4)))
            )
        else:  # GP_ALC or GP_LHD
            pred_value, pred_var_value, _ = standardGP(self.x_AL, self.y_AL, self.xt, self.logtheta)
            
        return pred_value, pred_var_value
    
    def run(self):
        """run the active learning process"""
        for s in tqdm(range(self.S)):
            # select the candidate points subset
            if self.use_subsample:
                xc_s, yc_s = self._select_candidates(self.xc, self.yc, self.ratio)
            else:
                xc_s = self.xc
                yc_s = self.yc
            
            # calculate the selection criteria
            criteria = self._calculate_criteria(xc_s)
            
            # select the next point
            ia = np.where(np.min(cdist(xc_s, self.x_AL, metric='euclidean'), axis=1) > self.step)[0]
            if len(ia) == 0:  # if no valid points are found
                print(f"Warning: No valid points found at step {s}. Stopping early.")
                break
                
            ib = np.argmax(criteria[ia])
            x_next = xc_s[ia[ib], :]
            
            # update the training set
            self.x_AL = np.vstack([self.x_AL, x_next])
            self.y_AL = np.vstack([self.y_AL, yc_s[ia[ib], :].reshape(-1,1)])
            
            # make predictions and calculate the evaluation metrics
            pred_value, pred_var_value = self._make_prediction()
            self.pred_history[s] = pred_value
            self.pred_var_history[s] = pred_var_value
            
            # calculate the evaluation metrics
            diff = pred_value.ravel() - self.yt.ravel()
            v = pred_var_value.ravel()
            self.mse[s] = np.mean(diff ** 2)
            self.rmse[s] = np.mean(np.abs(diff))
            self.nlpd[s] = 0.5 * np.mean(diff ** 2 / v + np.log(2 * np.pi * v))
            
            # if (s + 1) % 5 == 0:
            #     print(f"Completed iteration {s+1}/{self.S}")
        
        return {
            'x_AL': self.x_AL,
            'y_AL': self.y_AL,
            'mse': self.mse,
            'rmse': self.rmse,
            'nlpd': self.nlpd,
            'predictions': self.pred_history,
            'prediction_variances': self.pred_var_history
        }
    

if __name__ == "__main__":
    import numpy as np
    import random
    from simulate_case_d_linear import simulate_case_d_linear

    # set the random seed
    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    # initialize the parameters
    d = 3
    N = 20 * d
    Nt = 100
    Nc = 100
    S = 20
    
    # generate the simulation data
    sig = 2
    x, y, xc, yc, xt, yt, decfunc, logtheta, cv = simulate_case_d_linear(d, sig, N, Nt, Nc)
    
    # define all methods
    methods = ['MIN_IMSPE', 'MIN_ALC', 'MAX_MSPE', 'MAX_VAR', 'GP_ALC', 'JGP_LHD', 'GP_LHD']
    
    # store the results of all methods
    all_results = {}
    rmse_matrix = np.full((S, len(methods)), np.nan)  # create the RMSE matrix
    
    # run the active learning for each method
    for k, method in enumerate(methods):
        print(f"\nRunning {method}...")
        learner = ActiveLearner(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            method_name=method,
            S=S,
            use_subsample=True,  # use subsampling
            ratio=0.2,           # same sampling ratio as the original code
            logtheta=logtheta,
            cv=cv
        )
        results = learner.run()
        all_results[method] = results
        rmse_matrix[:, k] = results['rmse']  # store the RMSE results into the matrix

    # save the image
    plotResults(rmse_matrix, save=True)

    # save all results
    np.savez('all_results.npz', 
             rmse_matrix=rmse_matrix,
             **{f"{method}_results": results for method, results in all_results.items()})
    
    print("\nResults have been saved:")
    print("1. rmse.png - RMSE comparison plot")
    print("2. all_results.npz - Complete results data")