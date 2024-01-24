import open3d as o3d
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 读取.npy格式的点云数据
point_cloud_data = np.load('../landmark_axis/2.npy')
stl_file_path1 = './data_11.stl'
stl_file_path2 = './data_21.stl'
stl_file_path3 = './data_31.stl'
stl_file_path4 = './data_41.stl'
mesh = o3d.io.read_triangle_mesh(stl_file_path1)
vertices = np.asarray(mesh.vertices)

mesh2 = o3d.io.read_triangle_mesh(stl_file_path2)
mesh3 = o3d.io.read_triangle_mesh(stl_file_path3)
mesh4 = o3d.io.read_triangle_mesh(stl_file_path4)

# 将点云坐标数据转换为NumPy数组
point_cloud_data = np.array(point_cloud_data)


##  由空间3维点拟合出一条直线
def linear_fitting_3D_points(points):
    '''

    注意; 文中的公式推导有误，k1,b1,k2,b2中的系数2， 应该为n，n表示数据点的个数。

    直线方程可以转化成如下形式（具体见上面的文献）：
    x = k1 * z + b1
    y = k2 * z + b2

    '''
    # 表示矩阵中的值

    Sum_X = 0.0
    Sum_Y = 0.0
    Sum_Z = 0.0
    Sum_XZ = 0.0
    Sum_YZ = 0.0
    Sum_Z2 = 0.0

    for i in range(0, len(points)):
        xi = points[i][0]
        yi = points[i][1]
        zi = points[i][2]

        Sum_X = Sum_X + xi
        Sum_Y = Sum_Y + yi
        Sum_Z = Sum_Z + zi
        Sum_XZ = Sum_XZ + xi * zi
        Sum_YZ = Sum_YZ + yi * zi
        Sum_Z2 = Sum_Z2 + zi ** 2

    n = len(points)  # 点数
    den = n * Sum_Z2 - Sum_Z * Sum_Z  # 公式分母
    k1 = (n * Sum_XZ - Sum_X * Sum_Z) / den
    b1 = (Sum_X - k1 * Sum_Z) / n
    k2 = (n * Sum_YZ - Sum_Y * Sum_Z) / den
    b2 = (Sum_Y - k2 * Sum_Z) / n

    return k1, b1, k2, b2


k1, b1, k2, b2 = linear_fitting_3D_points(point_cloud_data)
z_values = np.linspace(22, 39, 100)
x_values = k1 * z_values + b1
y_values = k2 * z_values + b2

# 创建点云
points = np.column_stack((x_values, y_values, z_values))
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
# 创建线段
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
lines = [[i, i + 1] for i in range(len(z_values) - 1)]
line_set.lines = o3d.utility.Vector2iVector(lines)

# 设置线段的颜色（RGB，取值范围为0-1）
line_color = [0.8, 0.2, 0.2]  # 红色
line_set.colors = o3d.utility.Vector3dVector([line_color] * len(lines))

# 可视化点云
original_point_cloud = o3d.geometry.PointCloud()
original_point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)

point_s = o3d.geometry.PointCloud()
point_s.points = o3d.utility.Vector3dVector(vertices)
# 设置颜色和透明度
point_cloud.paint_uniform_color([0.25, 0.41, 0.88])
point_s.paint_uniform_color([0.43, 0.5, 0.56])
# 可视化点云
o3d.visualization.draw_geometries([mesh, line_set])
