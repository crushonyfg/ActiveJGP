import matplotlib.pyplot as plt
import numpy as np

def plotResults(rmse):
    """
    Plot the cumulative minimum RMSE for different methods.

    Parameters:
    rmse (numpy.ndarray): 2D array of RMSE values with each column corresponding to a method.
    """
    methods = ['JGP.IMSPE', 'JGP.ALC', 'JGP.MSPE', 'JGP.VAR', 'GP.ALC', 'JGP.LHD', 'GP.LHD']
    line_specs = ['--r', '--g', '--b', '--m', '-b', '--k', '-k']
    
    plt.figure(figsize=(10, 6))
    
    for m, method in enumerate(methods):
        # Calculate cumulative minimum RMSE for each method
        cummin_rmse = np.minimum.accumulate(rmse[:, m])
        plt.plot(cummin_rmse, line_specs[m], label=method)
    
    plt.legend()
    plt.xlabel('Stages')
    plt.ylabel('RMSE')
    plt.title('Cumulative Minimum RMSE Across Stages for Different Methods')
    plt.grid(True)
    plt.show()
