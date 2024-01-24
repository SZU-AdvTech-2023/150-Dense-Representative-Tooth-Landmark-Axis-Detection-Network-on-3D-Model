import mat4py as mp
import numpy as np
import open3d as o3d
# 载入必要库
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

import pandas as pd
import networkx as nx  # 导入 NetworkX 工具包

from sklearn.neighbors import NearestNeighbors

from sklearn.datasets import make_swiss_roll


# def cedi_line(X):
#     if len(X[0]) == 3:
#         x = []
#         y = []
#         z = []
#         for i2 in minWPath:
#             x.append(X[i2, 0])
#             y.append(X[i2, 1])
#             z.append(X[i2, 2])
#     return x, y, z

def get_points(points):
    xi = []
    yi = []
    zi = []
    for i in range(0, len(points)):
        xi = points[i][0]
        yi = points[i][1]
        zi = points[i][2]
    return xi, yi, zi


# # 用make_swiss_roll得到渐变色
# X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)


stl_file_path1 = './data_11.stl'
mesh = o3d.io.read_triangle_mesh(stl_file_path1)
vertices = np.asarray(mesh.vertices)
X = np.array(vertices)
m = len(X)

n_neighbors = 5
# j 计算每个点的k近邻：
nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
dist_matrix = np.zeros((m, m))

for i in range(m):
    for j in range(m):
        if j not in indices[i]:  # 若X[j]点不是X[i]的k近邻，则距离为0
            dist_matrix[i][j] = 0
        else:  # 若X[j]点是X[i]的k近邻
            for index in range(len(indices[i])):  # 求X[j]到X[i]的距离
                if indices[i][index] == j:
                    dist_matrix[i][j] = distances[i][index]
                    break

dfAdj = pd.DataFrame(dist_matrix)
G1 = nx.from_pandas_adjacency(dfAdj)  # 由 pandas 顶点邻接矩阵 创建 NetworkX 图
# 两个指定顶点之间的最短加权路径
source = 1
target = 500
minWPath = nx.bellman_ford_path(G1, source=source, target=target)  # 顶点 10 到 顶点 100 的最短加权路径
print("最短路径为：", minWPath)

# 显示图形
# 绘图
fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig, elev=10, azim=80)
ax.set_xlim(-30, 0)
ax.set_ylim(-30, 30)
ax.set_zlim(40, 25)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
ax.set_title('S Curve', fontsize=20)
fig.add_axes(ax)

# x1, y1, z1 = cedi_line(X)
x1, y1, z1 = get_points(X)
ax.plot(x1, y1, z1, label='parametric curve', color='red')
# ax.set_axis_off()

# 显示图例
ax.legend()
plt.show()
