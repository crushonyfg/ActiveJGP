import numpy as np
from scipy.stats import multivariate_normal
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from skimage import io, color, transform

# Helper function to generate the covariance matrix
def covariance_matrix(x, length_scale, variance, noise_level):
    rbf = RBF(length_scale=length_scale)  # CovSEard equivalent
    noise = WhiteKernel(noise_level=noise_level)  # CovNoise equivalent
    kernel = variance * rbf + noise
    K = kernel(x)
    return K

# Main function to simulate the cases
def simulate_case(caseno, sig, percent_train):
    # Generate training input grid
    gx = np.arange(0, 1.02, 0.02) - 0.5  # Grid shifted to range [-0.5, 0.5]
    px, py = np.meshgrid(gx, gx)
    x = np.column_stack((px.ravel(), py.ravel()))
    N = x.shape[0]

    # Define covariance function and parameters
    d = x.shape[1]
    length_scale = 0.1
    variance = 9
    noise_level = 1e-30  # Clean data for ground truth

    # Compute covariance matrix
    K = covariance_matrix(x, length_scale, variance, noise_level)

    # Emulating jumps based on the case number
    if caseno == 1:
        # Linear boundary
        beta = np.array([-2, 1]) / np.linalg.norm([-2, 1])
        beta_0 = 0.1
        idx = np.dot(x, beta) < beta_0
        bw = idx
    elif caseno == 2:
        # Quadratic boundary
        beta = np.array([1, 1, 0.7])
        idx = (beta[0]**2 * (x[:, 0])**2 + beta[1]**2 * (x[:, 1] - 0.7)**2 - beta[2]**2) < 0
        bw = idx
    elif caseno == 3:
        # Sharp jump
        idx = (np.abs(x[:, 0]) + np.abs(x[:, 1] - 0.5)) < 0.5
        bw = idx
    elif caseno == 4 or caseno == 5 or caseno == 6:
        # Phantom boundaries (using images)
        image_file = {4: 'C:/Users/yxu59/autumn2024/codeBase/code_al_jgp_py/JumpGP_code_py/bound3.png', 5: 'C:/Users/yxu59/autumn2024/codeBase/code_al_jgp_py/JumpGP_code_py/bound1.png', 6: 'C:/Users/yxu59/autumn2024/codeBase/code_al_jgp_py/JumpGP_code_py/bound4.png'}[caseno]
        I = io.imread(image_file)
        bw = color.rgb2gray(I) > 0.5
        bw = transform.resize(bw, (len(gx), len(gx)))
        idx = bw.ravel() == 1

    # Generate training responses based on the boundary
    y = np.zeros(N)
    y[idx] = multivariate_normal.rvs(mean=27*np.ones(sum(idx)), cov=K[np.ix_(idx, idx)])
    y[~idx] = multivariate_normal.rvs(mean=np.ones(sum(~idx)), cov=K[np.ix_(~idx, ~idx)])
    y = y.reshape(-1)

    # Add Gaussian noise
    y0 = y.copy()  # Non-noisy responses
    y += np.random.normal(0, sig, size=y.shape)
    y0 -= np.mean(y0)
    y -= np.mean(y)

    # Test data (currently the same as training data)
    yt = y
    xt = x
    mask = idx

    # Randomly select a subset of the data for training
    random_idx = np.random.choice(N, size=int(N * percent_train), replace=False)
    x_train = x[random_idx, :]
    y_train = y[random_idx]
    r = mask[random_idx]

    return x_train, y_train, xt, yt, y0, gx, r, bw

# Example usage:
if __name__ == "__main__":
    caseno = 1  # Choose the case number
    sig = 2  # Standard deviation of noise
    percent_train = 0.5  # Percentage of training data

    x_train, y_train, xt, yt, y0, gx, r, bw = simulate_case(caseno, sig, percent_train)
