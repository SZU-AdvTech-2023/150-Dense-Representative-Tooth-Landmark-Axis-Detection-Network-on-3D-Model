from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import vtkmodules.all as vtk
import pandas as pd

class ToothLandmark(Dataset):

    def __init__(self, data_root, axis_root):
        self.data_root = data_root
        self.patient_dirs = os.listdir(data_root)
        self.axis_root = axis_root
        self.sigma = 0.3

    def read_stl(self, file_path):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_path)
        reader.Update()
        return reader

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, index):
        dir = os.path.join(self.data_root, self.patient_dirs[index])
        axis_dir = os.path.join(self.axis_root, self.patient_dirs[index])
        files = os.listdir(dir)
        # 牙齿数、点云数、坐标三个分量
        incisor_points = np.zeros((4, 1024, 3), np.float64)  # 四颗门牙
        teeth_normals = np.zeros((4, 1024, 3), np.float64)
        heatmap_axis = np.zeros((4, 1024, 3), np.float64)  # 四个牙轴的热点图
        list_axis = np.zeros((4, 3), np.float64)  # 四个牙轴顶点的坐标
        teeth_center = np.zeros((4, 3), np.float64)
        # teeth_center_molar = np.zeros((2, 3), np.float64)
        for file in files:  # 每个病人应该有4颗牙，4个文件
            data = os.path.join(dir, file)
            stl_reader = self.read_stl(data)
            num = int(file.replace(".stl", "").split("_")[-1])  # 从文件名中提取牙齿编号
            if num % 10 == 1:  # 如果是门牙
                polydata = stl_reader.GetOutput()
                points = polydata.GetPoints()

                normal_generator = vtk.vtkPolyDataNormals()
                normal_generator.SetInputData(polydata)
                normal_generator.ComputePointNormalsOn()  # 计算点法向量
                normal_generator.ComputeCellNormalsOff()  # 不计算单元法向量
                normal_generator.Update()
                output_data = normal_generator.GetOutput()
                normals = output_data.GetPointData().GetNormals()
                nors = np.array([normals.GetTuple(i) for i in range(points.GetNumberOfPoints())])

                mesh_points = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
                se_index = np.random.randint(0, mesh_points.shape[0], 1024)  # 采样算法可以修改
                se_points = mesh_points[se_index]
                se_normals = nors[se_index]

                teeth_num = num // 10 - 1
                incisor_points[teeth_num] = se_points
                teeth_normals[teeth_num]= se_normals
                teeth_center[teeth_num] = np.mean(mesh_points, axis=0)  # 门牙的中心点

                with open(os.path.join(axis_dir, file.replace(".stl", ".txt"))) as f:  # 打开储存了关键点坐标的.txt
                    landmarks = f.readlines()  # 返回一个字符串列表
                    if len(landmarks) == 2:  # 上门牙：关键点和牙轴顶点 11 41
                        axis = landmarks[1].split()[0].split(',')  # 此处读取到的是牙轴顶点
                    else:  # 下门牙：牙轴顶点
                        axis = landmarks[0].split()[0].split(',')
                    axis = np.array([float(axis[j]) for j in range(len(axis))])
                    list_axis[teeth_num] = axis

        axis_vector = list_axis - teeth_center  # 牙轴的方向向量（4，3）
        axis_vector_length = np.linalg.norm(axis_vector, axis=1)  # （4，）
        axis_vector = axis_vector / axis_vector_length[:, np.newaxis]  # ;牙轴的单位向量(4,3)
        # 门牙点云每个点相对牙齿中心的偏移
        cen_to_point = incisor_points - teeth_center[:, np.newaxis, :]  # (4,1024,3)
        # 偏移与牙轴方向向量的点积即为投影长度  |     列向量乘以方向得到投影在牙轴方向上的矢量投影  |将上述矢量从原始偏移中减去得到垂直于牙轴方向的矢量
        for j in range(4):
            heatmap_axis[j] = np.dot(cen_to_point[j], axis_vector[j])[:, np.newaxis] * axis_vector[j] - cen_to_point[j]
        # dot_product=np.dot(cen_to_point,axis_vector)#(4,1024)
        # heatmap_axis=cen_to_point-dot_product*axis_vector

        incisor_points = torch.tensor(incisor_points)
        teeth_normals = torch.tensor(teeth_normals)
        teeth_center = torch.tensor(teeth_center)
        heatmap_axis = torch.tensor(heatmap_axis)

        return incisor_points, teeth_normals, teeth_center, heatmap_axis


if __name__ == "__main__":
    dataset = ToothLandmark("./cbct_data", "./label_landmark")
    loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0, drop_last=True)
    print(len(loader))
    data_loader = iter(loader)
    for i in range(1):
        incisor_points, teeth_normals, teeth_center, heatmap_axis = next(data_loader)
        print(incisor_points.shape)
        # print(molar_points.shape)
        print(teeth_center.shape)
        # print(heatmap_landmark.shape)
        print(heatmap_axis.shape)
        # 使用 torch.eq() 函数找出等于特定值的元素的布尔张量
        # equal_to_value = torch.eq(heatmap_landmark, 1.0)
        #
        # # 使用 torch.nonzero() 函数找出布尔张量中为 True 的索引
        # indices = torch.nonzero(equal_to_value).squeeze()
        indices = torch.argmax(heatmap_axis[0, 2])
        print(incisor_points[0, 1][indices])
        # print(label)
