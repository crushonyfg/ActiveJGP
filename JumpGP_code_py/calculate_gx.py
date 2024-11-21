# % ***************************************************************************************
# % calculate_gx - This function evaluates the boundary function with basis function values px
# % and basis weights w
# % 
# % Inputs:
# %       px - basis function values of a boundary function evaluated at
# %       training inputs, excluding a constant basis
# %       w - associated parameters of the boundary function
# % Outputs:
# %       gx - boundary function values
# %       phi_x - corresponding basis matrix 
# % 
# % Copyright ©2022 reserved to Chiwoo Park (cpark5@fsu.edu) 
# % ***************************************************************************************


import numpy as np

def calculate_gx(px, w):
    N = px.shape[0]  # 样本数量
    # 构建基矩阵 phi_x，首列为1，表示偏差项
    phi_x = np.column_stack((np.ones(N), px))
    
    # 计算边界函数值 gx
    gx = phi_x @ w
    
    return gx, phi_x #gx shape (N,), phi_x shape (N,d+1)

# 示例使用
if __name__ == "__main__":
    px = np.random.rand(100, 2)  # 100个样本，2维特征
    w = np.random.rand(3)        # 包含偏差项的权重向量
    gx, phi_x = calculate_gx(px, w)
    print("gx:", gx)
