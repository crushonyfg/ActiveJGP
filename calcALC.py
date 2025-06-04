import numpy as np
from scipy.linalg import cholesky

def calcALC(x, xref, cv, logtheta):
    """
    Calculate the Active Learning Cohn (ALC) score for Gaussian Process.
    
    Parameters:
        x (np.array): Training inputs (N x D)
        xref (np.array): Reference points (Nref x D)
        cv (function): Covariance function that takes (logtheta, points1, points2)
        logtheta (np.array): Log hyperparameters for the covariance function

    Returns:
        ALC (np.array): ALC scores for each reference point (Nref, )
    """
    Nref = xref.shape[0]
    ALC = np.zeros(Nref)

    for i in range(Nref):
        # Combine x and xref[i] to get full set of points
        K = cv[0](cv[1], logtheta, np.vstack((x, xref[i, :])))
        
        # Calculate covariance between x and each reference point xref
        _, Kt = cv[0](cv[1], logtheta, np.vstack((x, xref[i, :])), xref)

        # Cholesky decomposition and calculation of LK
        L = cholesky(K, lower=True)
        LK = np.linalg.solve(L, Kt)
        
        # Calculate ALC score for the i-th reference point
        ALC[i] = np.sum(np.sum(LK.T ** 2, axis=1))

    return ALC

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.linalg import cholesky
from joblib import Parallel, delayed

from JumpGaussianProcess.cov.covSum import covSum
from JumpGaussianProcess.cov.covSEard import covSEard
from JumpGaussianProcess.cov.covNoise import covNoise

def _single_alc(i, x, xref, cv, logtheta):
    Xi = np.vstack((x, xref[i, :]))        # (N+1)×D
    K = cv[0](cv[1], logtheta, Xi)         # (N+1)×(N+1)
    _, Kt = cv[0](cv[1], logtheta, Xi, xref)  # (N+1)×Nref
    L = cholesky(K, lower=True)
    LK = np.linalg.solve(L, Kt)            # (N+1)×Nref
    return np.sum(np.sum(LK.T ** 2, axis=1))

def calcALC_parallel(x, xref, cv, logtheta, n_jobs=-1):
    Nref = xref.shape[0]
    alc_list = Parallel(n_jobs=n_jobs)(
        delayed(_single_alc)(i, x, xref, cv, logtheta)
        for i in range(Nref)
    )
    return np.array(alc_list)

# 示例调用
if __name__ == "__main__":
    # 随便造点测试一下
    np.random.seed(0)
    x = np.random.randn(50, 3)         # 50 个训练点，3 维
    xref = np.random.randn(20, 3)      # 20 个候选点，3 维
    logtheta = np.random.randn(5)        # 例如仅一个长度参数
    cv = [covSum, [covSEard, covNoise]]              # 第二项 None，代表 covFunc 不需要额外参数

    # 并行计算
    alc_values = calcALC_parallel(x, xref, cv, logtheta, n_jobs=4)
    print("ALC:", alc_values)
