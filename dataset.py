from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import vtkmodules.all as vtk


class ToothLandmark(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.teeth_dir = os.path.join(data_dir, "teeth")
        self.label_dir = os.path.join(data_dir, "label")

        if mode == 'train':
            self.teeth = os.path.join(self.teeth_dir, "Training_data")
        else:
            self.teeth = os.path.join(self.teeth_dir, "Test_data2")
        self.stl_data = os.listdir(self.teeth)
        # print(self.stl_data)
        self.mode = mode
        self.sigma = 0.3


    def read_stl(self, file_path):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_path)
        reader.Update()
        return reader

    def __len__(self):
        return len(self.stl_data)

    def __getitem__(self, index):
        teeth_points = np.zeros((4096, 3), np.float64)
        teeth_normals = np.zeros((4096, 3), np.float64)
        label_distance = np.zeros((4096, 3), np.float64)
        data_path = os.path.join(self.teeth, self.stl_data[index])

        stl_reader = self.read_stl(data_path)
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
        verts = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        centroid = np.mean(verts, axis=0)
        # print(centroid)

        se_index = np.random.randint(0, verts.shape[0], 4096)
        se_points = verts[se_index]
        se_normals = nors[se_index]

        teeth_points = se_points - centroid
        teeth_normals = se_normals


        # 这部分代码用于生成训练标签 label_distance
        name = self.stl_data[index][:-4]
        if self.mode == "train":
            landmark_list = list()
            with open(os.path.join(self.label_dir, name + ".txt")) as f:
                for i in range(3):
                    landmark = f.readline().split()[0].split(',')
                    landmark = [float(landmark[j]) for j in range(len(landmark))]
                    landmark_list.append(landmark)
            landmarks = np.array(landmark_list)
            landmarks = landmarks - centroid
            for i in range(3):
                distances = np.linalg.norm(teeth_points - landmarks[i, :], axis=1)
                label_distance[:, i] = distances
            label_distance = np.exp(-0.5 * label_distance ** 2 / self.sigma ** 2)

        # 将处理后的数据转换为 PyTorch 的 Tensor 类型，并对点的坐标进行归一化。
        teeth_points = torch.tensor(teeth_points)
        teeth_normals = torch.tensor(teeth_normals)
        label_distance = torch.tensor(label_distance)
        tmaxv = torch.max(torch.abs(teeth_points.reshape(4096 * 3)))
        teeth_points = teeth_points / tmaxv
        centroid = torch.tensor(centroid)
        # print(teeth_points)
        # 根据模式返回不同的数据。
        # 在测试模式下返回牙齿点云、法向量、文件名、中心和最大坐标值；
        # 在训练模式下返回牙齿点云、法向量和标签。
        if self.mode == "test":
            return teeth_points, teeth_normals, name, centroid, tmaxv
        else:
            return teeth_points, teeth_normals, label_distance


if __name__ == "__main__":
    dataset = ToothLandmark("./test_data")
    loader = DataLoader(dataset, shuffle=True, batch_size=2, num_workers=0, drop_last=True)
    print(len(loader))
    data_loader = iter(loader)
    for i in range(len(data_loader)):
        points, normals, label = next(data_loader)
        # print(label)
