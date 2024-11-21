import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from skimage import measure

from JumpGP_code_py.simulate_case import simulate_case
from JumpGP_code_py.JumpGP_LD import JumpGP_LD

# Assuming functions `simulate_case`, `JumpGP_LD` are already implemented
# Add path to functions (if applicable)
# sys.path.append('./cov')
# sys.path.append('./lik')
def check_and_reshape(x, D):
    # 检查 x 的形状是否是 (任意, D)
    if len(x.shape) != 2 or x.shape[1] != D:
        # 如果不是，重塑为 (-1, D)
        x = x.reshape(-1, D)
    return x

# Data generation parameters
percent_train = 0.5
sig = 2

# Choose case number for Figures 3, 4, 5, and 6
caseno = 1  # For Figure 3
# caseno = 2  # For Figure 4
# caseno = 3  # For Figure 5
# caseno = 4  # For Figure 6

# Simulate case
x, y, xt, yt, y0, gx, r, bw = simulate_case(caseno, sig, percent_train)

L = len(gx)
# Example of saving data (if needed):
# np.savetxt('phantom41.csv', np.reshape(y0, (L, L)))
# np.savetxt('phantom41_noisy.csv', np.reshape(yt, (L, L)))
# np.savetxt('phantom41_J.csv', np.reshape(bw, (L, L)))

my = np.mean(yt)
y -= my
yt -= my
y0 -= my

# Boundary and scaling
# B = bwboundaries(bw) --> Matlab自带
# # 使用 find_contours 提取边界
# contours = measure.find_contours(bw, level=0.5)

# # 可视化边界
# for contour in contours:
#     plt.plot(contour[:, 1], contour[:, 0], linewidth=2)

bw = np.reshape(bw, (L, L))
sc = 0.025 / 0.02

# Define test points for different cases
if caseno == 1:
    xs = np.array([[20, 33], [16, 16], [7, 5], [25, 36], [8, 32], [38, 18]])
elif caseno == 2:
    xs = np.array([[7, 22], [17, 22], [35, 22], [5, 28], [22, 35], [31, 7]])
elif caseno == 3:
    xs = np.array([[37, 11], [17, 24], [21, 21], [36, 36], [8, 35], [34, 32]])
elif caseno == 4:
    xs = np.array([[37, 11], [36, 36], [8, 35], [17, 24], [21, 21], [34, 32]])

# Normalize test points
xs = xs / len(np.arange(0, 1.025, 0.025)) - 0.5

# Plotting configuration
Nt = xs.shape[0]
subplot_layout = (2, 4)
sel = [0, 1, 2, 5]  # Index selection for test points
loc = list(range(1, 9))

for j in range(4):
    k = 0 if j < 4 else 1
    i = j if k == 0 else j - 4
    # ax = axs[j // 4, j % 4]
    xt = xs[j, :]  # 每个子图的测试点

    # 查找最近的邻居 (40个最近的)
    nbrs = NearestNeighbors(n_neighbors=40, algorithm='auto').fit(x)
    idx = nbrs.kneighbors([xt], return_distance=False)[0]
    lx, ly = x[idx, :], y[idx]
    ly = ly.reshape(-1, 1)  # 可能需要reshape，确保形状一致
    xt = xt.reshape(1, -1)  # 重新reshape测试点

    if k == 0:
        mu_t, sig2_t, model, h3 = JumpGP_LD(lx, ly, xt, 'CEM', True)
    else:
        mu_t, sig2_t, model, h3 = JumpGP_LD(lx, ly, xt, 'VEM', True)

    plt.imshow(np.reshape(yt, (L, L)), cmap='gray', extent=(gx[0], gx[-1], gx[-1], gx[0]))
    
    # 绘制局部训练点
    plt.scatter(lx[:, 0], lx[:, 1], color='r', marker='+', s=100, label='Local Training Inputs')
    
    # 绘制测试点
    plt.scatter(xt[0,0], xt[0,1], color='c', marker='o', s=100, label='Test Point')
    
    current_ax = plt.gca()
    
    # 函数用于复制 PathCollection
    def copy_path_collection(artist, ax):
        # 获取 PathCollection 的属性
        offsets = artist.get_offsets()  # 获取点的坐标
        sizes = artist.get_sizes()  # 获取点的大小
        facecolors = artist.get_facecolor()  # 获取点的颜色
        edgecolors = artist.get_edgecolor()  # 获取边缘颜色
    
        # 重新绘制散点图
        return ax.scatter(offsets[:, 0], offsets[:, 1], s=sizes, c=facecolors, 
                          edgecolor=edgecolors, alpha=artist.get_alpha(), label=artist.get_label())
    
    # 重新绘制 h3
    new_artist = copy_path_collection(h3[0], current_ax)
    a = np.array([[1, -0.5], [1, 0.5]])
    b_plot = -np.dot(a, model['w'][0:2]) / model['w'][2]
    # current_ax.plot(a, b_plot, 'r', linewidth=2)
    current_ax.plot(np.array([-0.5,0.5]), b_plot, 'r', linewidth=3)
    # line = h3[1][1]
    # current_ax.plot(line.get_xdata(), line.get_ydata(),
    #                 color=line.get_color(), linewidth=line.get_linewidth(),
    #                 label=line.get_label())
    plt.imshow(np.reshape(yt, (L, L)), cmap='gray', extent=(gx[0], gx[-1], gx[-1], gx[0]))
    # 添加图例
    plt.legend()
    plt.savefig(f'figure_{j}.png', dpi=300, bbox_inches='tight')  # 保存为PNG格式，分辨率300dpi，裁剪空白区域

    # 清空当前图形，为下一个图形做准备
    plt.clf()
