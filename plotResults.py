import matplotlib.pyplot as plt
import numpy as np

def plotResults(rmse, save=False):
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
    if save:
        plt.savefig('rmse.png')
    plt.pause(0.1)
    # plt.close()

if __name__ == "__main__":
    import numpy as np
    
    # Create sample RMSE data for testing
    # Simulating 10 stages with 7 different methods
    np.random.seed(42)  # For reproducibility
    
    # Number of stages and methods
    n_stages = 10
    n_methods = 7  # Matches the number of methods in plotResults
    
    # Generate random RMSE values that generally decrease over time
    # to simulate improving performance with more stages
    base_rmse = np.linspace(1.0, 0.2, n_stages).reshape(-1, 1)  # Decreasing trend
    noise = np.random.normal(0, 0.1, (n_stages, n_methods))     # Random variations
    
    # Different methods have different baseline performance
    method_factors = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    
    # Combine to create final RMSE data
    rmse_data = base_rmse * method_factors + noise
    
    # Ensure all RMSE values are positive
    rmse_data = np.abs(rmse_data)
    
    print("Generated RMSE data shape:", rmse_data.shape)
    print("RMSE data first few rows:")
    print(rmse_data[:3, :])
    
    # Call the plotResults function with our test data
    print("\nPlotting results...")
    plotResults(rmse_data)
    
    print("Plot displayed. Close the plot window to exit.")
