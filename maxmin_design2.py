import numpy as np
from scipy.spatial.distance import cdist

def maxmin_design2(Xc, N):
    """
    Select N points from candidate set Xc using maxmin distance criterion.
    
    Parameters:
    -----------
    Xc : numpy.ndarray
        Candidate points array of shape (Nc, d) where Nc is number of candidates
        and d is dimension of each point
    N : int
        Number of points to select
    
    Returns:
    --------
    X : numpy.ndarray
        Selected points array of shape (N, d)
    id : numpy.ndarray
        Indices of selected points in original candidate set
    """
    # Get number of candidate points
    Nc = Xc.shape[0]
    
    # Initialize index array
    id = np.zeros(N, dtype=int)
    
    # Randomly select first point
    id[0] = np.random.randint(0, Nc)
    
    # Get initial selected points
    X = Xc[id[0:1], :]
    
    # Select remaining points
    for n in range(1, N):
        # Calculate distances between candidates and selected points
        D = cdist(Xc, X)
        
        # Find point that maximizes minimum distance to selected points
        min_distances = np.min(D, axis=1)
        id[n] = np.argmax(min_distances)
        
        # Update selected points
        X = Xc[id[:n+1], :]
    
    # Return final selection
    X = Xc[id, :]
    return X, id

# Example usage
if __name__ == "__main__":
    # Generate some candidate points
    Xc = np.random.rand(100, 2)
    # Select 10 points
    X, id = maxmin_design2(Xc, 10)
    print("Selected points shape:", X.shape)
    print("Selected indices:", id)
