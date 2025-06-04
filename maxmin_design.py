import numpy as np
from scipy.spatial.distance import cdist

def maxmin_design(N, d, X=None):
    """
    Generate a max-min design by iteratively adding points that maximize
    the minimum distance to existing points.

    Parameters:
    N (int): Number of points to generate
    d (int): Dimension of each point
    X (numpy.ndarray or None): Initial set of points (optional)

    Returns:
    numpy.ndarray: Array of generated points with shape (N, d)
    """
    if X is None:
        X = np.array([])
        
    for n in range(N):
        if len(X) == 0:
            X = np.random.rand(1, d)
        else:
            candidates = np.random.rand(N, d)
            D = cdist(candidates, X)
            min_distances = np.min(D, axis=1)
            max_min_distance_idx = np.argmax(min_distances)
            X = np.vstack([X, candidates[max_min_distance_idx]])
    
    # Take the last N points
    return X[-N:]

if __name__ == "__main__":
    X = maxmin_design(10, 2)
    print("X.shape", X.shape)
