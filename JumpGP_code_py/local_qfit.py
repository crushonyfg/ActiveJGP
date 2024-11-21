import numpy as np

def local_qfit(x0, y0, xt):
    N, p = x0.shape

    # Calculate differences between x0 and xt
    d = x0 - xt
    d2 = np.sum(d ** 2, axis=1)
    
    # Bandwidth for Gaussian kernel
    h = np.max(np.sqrt(d2))

    # Gaussian kernel weights
    Kh = np.exp(-0.5 * d2 / (h ** 2)) / (2 * np.pi * (h ** 2))

    # Construct the local linear basis matrix
    g1, g2 = np.meshgrid(np.arange(p), np.arange(p))
    g1, g2 = g1.flatten(order='F'), g2.flatten(order='F')
    # id_mask = g1.ravel() >= g2.ravel()
    id_mask = g1 >= g2
    
    X = np.hstack([np.ones((N, 1)), x0, (x0[:, g1[id_mask]] * x0[:, g2[id_mask]])])

    # Weighted least squares
    XWX = X.T @ np.diag(Kh) @ X
    XWy = X.T @ np.diag(Kh) @ y0

    # Solve for beta
    beta = np.linalg.solve(XWX, XWy)

    return beta, X

# Example usage
if __name__ == "__main__":
    x0 = np.random.rand(100, 2)  # 100 local training inputs with 2 dimensions
    y0 = np.random.rand(100)     # 100 local training responses
    xt = np.random.rand(1, 2)    # Single test location with 2 dimensions

    beta, X = local_qfit(x0, y0, xt[0])
    print(f"Fitted beta: {beta}")
