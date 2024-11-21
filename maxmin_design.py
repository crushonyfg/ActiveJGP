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
    if X is None or len(X) == 0:
        X = np.random.rand(1, d)
    else:
        X = np.array(X)

    for n in range(N - len(X)):
        # Generate candidate points and calculate distances to existing points
        candidates = np.random.rand(N, d)
        D = cdist(candidates, X)
        
        # Find the candidate with the maximum of the minimum distances
        min_distances = np.min(D, axis=1)
        max_min_distance_idx = np.argmax(min_distances)
        
        # Append the selected candidate to the design
        X = np.vstack([X, candidates[max_min_distance_idx]])

    # Ensure the output has exactly N points
    return X[-N:]
