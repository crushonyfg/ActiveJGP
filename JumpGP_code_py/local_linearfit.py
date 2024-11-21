# % ***************************************************************************************
# %
# % local_linearfit - The function implements a locally linear fit using local data 
# %                   (x0, y0) around test location xt
# % 
# %
# % Inputs:
# %       x - local training inputs (N x d)
# %       y - local training responses (N x 1)
# %       xt - single test location (1 x d)
# % Outputs:
# %       beta - fitted parameters of a linear model (1+d dimensions)
# %       X - local linear basis matrix (N x (d+1) matrix)
# %
# % Copyright ©2022 reserved to Chiwoo Park (cpark5@fsu.edu) 
# % ***************************************************************************************


import numpy as np

def local_linearfit(x0, y0, xt):
    N = x0.shape[0]
    
    # Calculate distance between x0 and xt
    d = x0 - np.tile(xt, (N, 1))
    d2 = np.sum(d ** 2, axis=1)
    
    # Calculate bandwidth h
    h = np.max(np.sqrt(d2))
    
    # Kernel calculation
    Kh = np.exp(-0.5 * d2 / (h ** 2)) / (2 * np.pi * (h ** 2))
    
    # Create local linear basis matrix X
    X = np.hstack((np.ones((N, 1)), x0))
    
    # Compute X'WX and X'Wy
    XWX = X.T @ np.diag(Kh) @ X
    XWy = X.T @ np.diag(Kh) @ y0
    
    # Solve for beta
    beta = np.linalg.solve(XWX, XWy)
    
    return beta, X

# 测试代码示例
if __name__ == "__main__":
    x0 = np.random.rand(100, 2)  # 100个样本，2个特征
    y0 = np.random.rand(100)  # 100个响应值
    xt = np.random.rand(2)  # 测试点（2个特征）

    beta, X = local_linearfit(x0, y0, xt)
    print("beta:", beta)
