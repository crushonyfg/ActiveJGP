import numpy as np
def ensure_2d(arr):
    if arr.ndim == 1:  # 检查是否是一维
        arr = arr[np.newaxis, :]  # 在前面添加一个维度
    return arr

def sq_dist(a, b=None, Q=None):
    """
    Computes the matrix of all pairwise squared distances between two sets of vectors.
    
    Parameters:
    a (numpy.ndarray): Matrix of size (D, n), where each column is a vector.
    b (numpy.ndarray, optional): Matrix of size (D, m), where each column is a vector. Defaults to a if not provided.
    Q (numpy.ndarray, optional): Matrix of size (n, m). When provided, returns a vector of traces of the product of Q.T and the coordinate-wise squared distances.
    
    Returns:
    numpy.ndarray: Matrix of squared distances, or a vector when Q is provided.
    """
    
    if a is None or (Q is not None and len(Q.shape) != 2):
        raise ValueError("Wrong number of arguments.")
    
    if b is None:  # if b is not provided, or is empty, it is taken as equal to a
        b = a

    a = ensure_2d(a)
    b = ensure_2d(b)

    D, n = a.shape
    d, m = b.shape
    
    if d != D:
        raise ValueError("Error: column lengths must agree.")
    
    if Q is None:
        # Pairwise squared distances between vectors in a and b
        C = np.zeros((n, m))
        for d in range(D):
            C += (np.tile(b[d, :], (n, 1)) - np.tile(a[d, :].reshape(-1, 1), (1, m))) ** 2
        return C
    else:
        # Compute the trace product with Q for the coordinate-wise squared distances
        if Q.shape != (n, m):
            raise ValueError("Third argument has the wrong size.")
        C = np.zeros(D)
        for d in range(D):
            C[d] = np.sum((np.tile(b[d, :], (n, 1)) - np.tile(a[d, :].reshape(-1, 1), (1, m))) ** 2 * Q)
        return C
