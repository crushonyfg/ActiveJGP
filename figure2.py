import numpy as np
import matplotlib.pyplot as plt
from JumpGaussianProcess.simulate_case import simulate_case  # Assuming you have this implemented based on your earlier MATLAB code

# Data generation
percent_train = 0.7
sig = 2

titles = ['(a) Linear', '(b) Round', '(c) Sharp', '(d) Phantom', '', '(e) Star']

# Create subplots layout similar to tiledlayout in MATLAB
fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
axs = axs.ravel()

# Loop over the different cases for plotting
for i, caseno in enumerate([1, 2, 3, 4, 6]):
    x, y, xt, yt, y0, gx, r, bw = simulate_case(caseno, sig, percent_train)

    # Preprocessing
    L = len(gx)
    my = np.mean(yt)
    yt = yt - my

    # Reshape the data to 2D
    yt_reshaped = yt.reshape(L, L)
    
    # Plotting
    ax = axs[i]
    im = ax.imshow(yt_reshaped, extent=[gx.min(), gx.max(), gx.min(), gx.max()], cmap='gray', origin='lower')
    ax.set_title(titles[caseno - 1])
    
    if caseno == 1:
        ax.set_xlabel('first input')
        ax.set_ylabel('second input')

# Show the plot
plt.show()
