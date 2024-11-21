import numpy as np
import argparse
from dataclasses import dataclass
from scipy.optimize import minimize
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt

from .cov.covSum import covSum
from .cov.covSEard import covSEard
from .cov.covNoise import covNoise
from .lik.loglikelihood import loglikelihood
from .local_linearfit import local_linearfit
from .maximize_PD import maximize_PD
from .calculate_gx import calculate_gx
from .local_qfit import local_qfit

@dataclass
class JumpGPQDModel:
    mu_t: np.ndarray
    sig2_t: np.ndarray
    w: np.ndarray

def JumpGP_QD(x, y, xt, mode='CEM', bVerbose=False):
    cv = [covSum, [covSEard, covNoise]]
    d = x.shape[1]
    g1, g2 = np.meshgrid(np.arange(d), np.arange(d))
    g1, g2 = g1.flatten(order='F'), g2.flatten(order='F')
    id_mask = g1 >= g2

    px = np.hstack([x, (x[:, g1[id_mask]] * x[:, g2[id_mask]])])
    pxt = np.hstack([xt, (xt[:, g1[id_mask]] * xt[:, g2[id_mask]])])

    # Initial estimation of boundary B(x)
    logtheta = np.zeros(d + 2)
    logtheta[-1] = -1.15
    w, _ = local_qfit(x, y, xt)
    nw = np.linalg.norm(w)
    w /= nw

    # Fine-tune intercept term
    b_range = np.arange(-0.2 + w[0], 0.2 + w[0], 0.0005)
    fd = []
    for b_k in b_range:
        w_d = np.copy(w)
        w_d[0] = b_k
        gx, _ = calculate_gx(px, w_d)
        r = gx >= 0
        r1 = r.flatten()
        fd.append(loglikelihood(logtheta, covSum, [covSEard, covNoise], x[r1,:], y[r1]) + loglikelihood(logtheta, covSum, [covSEard, covNoise], x[~r1,:], y[~r1]))
    
    w[0] = b_range[np.argmin(fd)]
    w *= nw

    if mode == 'CEM':
        model = maximize_PD(x, y, xt, px, pxt, w, logtheta, None, bVerbose)
    elif mode == 'VEM':
        model = variationalEM(x, y, xt, px, pxt, w, logtheta, None, bVerbose)
    elif mode == 'SEM':
        model = stochasticEM(x, y, xt, px, pxt, w, logtheta, None, bVerbose)
    else:
        raise ValueError("Invalid mode specified. Choose from 'CEM', 'VEM', or 'SEM'.")

    mu_t = model['mu_t']
    sig2_t = model['sig2_t']

    if bVerbose:
        gx_range = np.linspace(0, 1, int(1 / 0.025)) - 0.5
        ptx, pty = np.meshgrid(gx_range, gx_range)
        allx = np.hstack([ptx.ravel().reshape(-1, 1), pty.ravel().reshape(-1, 1)])
        allx = np.hstack([allx, (allx[:, g1[id_mask]] * allx[:, g2[id_mask]])])
        gx, _ = calculate_gx(allx, model['w'])
        gy = np.sign(gx)
        # Visualization logic can go here, for example:
        print(f"Plotting results for mode: {mode}")
        L = len(gx_range)
        gy_reshaped = np.reshape(gy, (L, L))

        # 绘制等高线
        plt.contour(gx_range, gx_range, gy_reshaped, levels=[1], colors='r', linewidths=3)
        gx,_ = calculate_gx(px, model["w"])  # 调用计算函数

        # 使用布尔索引筛选 x 中的行
        x_filtered = x[gx >= 0]

        # 绘制散点图
        h2 = plt.scatter(x_filtered[:, 0], x_filtered[:, 1], color='g', marker='s')

        # 添加标题和标签
        plt.title('Contour Plot')
        plt.xlabel('gx axis')
        plt.ylabel('gy axis')

        # 显示图形
        # plt.show()
        plt.clf()

    return mu_t, sig2_t, model, h2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Jump GP with Quadratic Decision Boundary")
    parser.add_argument('--mode', type=str, default='CEM', help="Inference algorithm ('CEM', 'VEM', 'SEM')")
    parser.add_argument('--verbose', type=int, default=0, help="Verbose output (0 or 1)")
    
    args = parser.parse_args()

    # Example dummy data (replace with actual)
    x = np.random.rand(100, 2)
    y = np.random.rand(100)
    xt = np.random.rand(10, 2)

    mu_t, sig2_t, model = JumpGP_QD(x, y, xt, mode=args.mode, bVerbose=args.verbose)
    print(f"mu_t: {mu_t}, sig2_t: {sig2_t}")
