from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import label
from datetime import datetime
import os

from JumpGP_code_py.simulate_case import simulate_case
from JumpGP_code_py.JumpGP_QD import JumpGP_QD
from JumpGP_code_py.JumpGP_LD import JumpGP_LD
from JumpGP_code_py.calculate_gx import calculate_gx

@dataclass
class FigureParams:
    percent_train: float = 0.5
    sig: float = 2
    caseno: int = 6  # 使用4为图8，使用6为图9
    n_neighbors: int = 40
    path: str = 'C:/Users/yxu59/autumn2024/codeBase/JumpGP_code_py/results/figure_8_9/'

def generate_figure(params):
    if not os.path.exists(params.path):
        os.makedirs(params.path)
    # 生成数据
    x, y, xt, yt, y0, gx, r, bw = simulate_case(params.caseno, params.sig, params.percent_train)
    L = len(gx)

    # 数据中心化
    my = np.mean(yt)
    y, yt, y0 = y - my, yt - my, y0 - my

    # 获取边界
    bw = bw.reshape(L, L)

    # 定义测试点
    xs_cases = {
        4: np.array([[37, 11], [36, 36], [8, 35], [17, 24], [21, 21], [34, 32]]),
        6: np.array([[27, 21], [10, 27], [37, 31], [9, 19], [21, 22], [27, 15]])
    }
    xs = xs_cases[params.caseno] / len(np.arange(0, 1.025, 0.025)) - 0.5

    # 绘图设置
    # fig, axs = plt.subplots(3, 4, figsize=(15, 15))
    sel = [1, 2, 3, 6]

    for j in range(4):
        # ax = axs[(j - 1) // 4, (j - 1) % 4]
        
        k = 0 if j < 4 else 1
        xt = xs[j, :]

        # 找到k个最近邻
        nbrs = NearestNeighbors(n_neighbors=params.n_neighbors, algorithm='auto').fit(x)
        idx = nbrs.kneighbors([xt], return_distance=False)[0]
        lx, ly = x[idx, :], y[idx]
        ly = ly.reshape(-1, 1)
        xt = xt.reshape(1, -1)

        # 选择JumpGP函数
        if k == 0:
            mu_t, sig2_t, model, h3 = JumpGP_QD(lx, ly, xt, 'CEM', True)
        else:
            mu_t, sig2_t, model, h3 = JumpGP_QD(lx, ly, xt, 'VEM', True)

        plt.imshow(np.reshape(yt, (L, L)), cmap='gray', extent=(gx[0], gx[-1], gx[-1], gx[0]))
        plt.scatter(lx[:, 0], lx[:, 1], color='r', marker='+', s=100, label='Local Training Inputs')
        
        plt.scatter(xt[0, 0], xt[0, 1], color='c', marker='o', s=100, label='test point')
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
        new_artist = copy_path_collection(h3, current_ax)

        d = x.shape[1]
        g1, g2 = np.meshgrid(np.arange(d), np.arange(d))
        g1, g2 = g1.flatten(order='F'), g2.flatten(order='F')
        id_mask = g1 >= g2
        gx_range = np.linspace(0, 1, 400) - 0.5
        ptx, pty = np.meshgrid(gx_range, gx_range)
        allx = np.hstack([ptx.ravel().reshape(-1, 1), pty.ravel().reshape(-1, 1)])
        allx = np.hstack([allx, (allx[:, g1[id_mask]] * allx[:, g2[id_mask]])])
        gx0, _ = calculate_gx(allx, model['w'])
        gy = np.sign(gx0)
        # Visualization logic can go here, for example:
        # print(f"Plotting results for mode: {mode}")
        L1 = len(gx_range)
        gy_reshaped = np.reshape(gy, (L1, L1))

        # 绘制等高线
        plt.contour(gx_range, gx_range, gy_reshaped, levels=[0], colors='r')

        # if i == 1 and k == 1:
        #     ax.set_xlabel('x1')
        #     ax.set_ylabel('x2')

        plt.legend()
        # plt.show()
        current_time = datetime.now().strftime("%Y%m%d_%H")
        plt.savefig(f'{params.path}caseno_{params.caseno}_figure_{j}_{current_time}.png', dpi=300, bbox_inches='tight')
        plt.clf()

    # 添加图例
    # axs[2, 3].axis('off')
    # axs[2, 3].legend([h1, ax.lines[0], h3], 
    #                  ['测试点', '局部训练输入', '选定的局部设计'],
    #                  loc='center', ncol=1, frameon=False)

    # # 调整布局
    # plt.tight_layout()

    # # 为行添加标题
    # titles = ['(a) JPG-CEM', '(b) JPG-VEM']
    # for idx, title in enumerate(titles):
    #     fig.text(0.5, 0.98 - (idx * 0.33), title, ha='center', fontsize=16)

    # # 保存图片
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # plt.savefig(f'{params.path}caseno_{params.caseno}_figure_{current_time}.png', dpi=300, bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    params = FigureParams()
    generate_figure(params)
