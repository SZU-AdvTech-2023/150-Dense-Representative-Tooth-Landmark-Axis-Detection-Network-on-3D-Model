import argparse
import torch
from torch.utils.data import DataLoader
# from model_cep import teeth_landmark_axis_model
from models.pointnet2_cep import pointnet2_seg_ssg
from dataset_cep import ToothLandmark
from tqdm import tqdm
import numpy as np
import json
import os
import vtkmodules.all as vtk
from open3d import *


def read_stl(file_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader


def test(args):
    test_dir = "./test_data_cep"
    patient_dirs = os.listdir(test_dir)
    # model = teeth_landmark_axis_model(args)
    model = pointnet2_seg_ssg(6, 3)
    pretrained_model = torch.load(args.checkpoint)
    model.load_state_dict(pretrained_model)
    model.cuda()
    model.eval()
    dict_list = []  # 字典列表
    for patient in patient_dirs:
        name = patient[7:]
        dir = os.path.join(test_dir, patient)
        paths = os.listdir(dir)
        incisor_points = np.zeros((4, 1024, 3), np.float64)  # 四颗门牙
        teeth_normals = np.zeros((4, 1024, 3), np.float64)
        teeth_center = np.zeros((4, 3), np.float64)
        print(patient, patient_dirs)
        for path in paths:
            file = os.path.join(dir, path)
            stl_reader = read_stl(file)
            num = int(path.replace(".stl", "").split("_")[-1])
            print(path, num)
            # if num == 16 or num == 46:  # 上磨牙则跳过   1右上 2 右下
            #     continue
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
                teeth_normals[teeth_num] = se_normals
                teeth_center[teeth_num] = np.mean(mesh_points, axis=0)  # 门牙的中心点
        # teeth_center = np.concatenate([teeth_center, teeth_center_molar], axis=0)
        incisor_points = torch.unsqueeze(torch.tensor(incisor_points), dim=0)
        # molar_points = torch.unsqueeze(torch.tensor(molar_points), dim=0)
        teeth_center = torch.unsqueeze(torch.tensor(teeth_center), dim=0)
        teeth_normals = torch.unsqueeze(torch.tensor(teeth_normals), dim=0)

        teeth_center = teeth_center.cuda().float()
        incisor_points = incisor_points.cuda().float()
        teeth_normals = teeth_normals.cuda().float()
        # molar_points = molar_points.cuda().float()

        # with torch.no_grad():
        #     result_landmark, result_axis = model(incisor_points, molar_points, teeth_center)
        # print(result_landmark.shape)
        with torch.no_grad():
            result_axis = []
            for j in range(4):
                pred = model(incisor_points[:, j, :, :], teeth_normals[:, j, :, :])
                result_axis.append(pred)
            result_axis = torch.stack(result_axis, dim=1)
            result_axis = result_axis.permute(0, 1, 3, 2)  # (1,4,1024,3)
            print(result_axis)
        # result_landmark = result_landmark.squeeze(0).squeeze(2)  # (4,1024)
        # res = result_landmark.detach().cpu().numpy()
        # result_axis = result_axis.squeeze(0).detach().cpu().numpy()
        # original_incisor = incisor_points.squeeze(0).detach().cpu().numpy()  # (4,1024,3)
        # print("original_incisor.shape:",original_incisor.shape,result_axis.shape)
        re = incisor_points + result_axis
        re = re.squeeze(0).detach().cpu().numpy()
        print(patient, re.shape)

        np.save("../landmark_axis/2.npy", re[0])

        # 写文件句柄
        handle = open("out.pcd", 'a')
        for m in range(1):
            point_num = re.shape[1]
            # pcd头部
            handle.write(
                '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
            string = '\nWIDTH ' + str(point_num)
            handle.write(string)
            handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
            string = '\nPOINTS ' + str(point_num)
            handle.write(string)
            handle.write('\nDATA ascii')

            # 依次写入点
            for i in range(point_num):
                string = '\n' + str(re[m, i, 0]) + ' ' + str(re[m, i, 1]) + ' ' + str(re[m, i, 2])
                # print(m,i,string)
                handle.write(string)
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Config')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--checkpoint', type=str, default="./checkpoint_cep/teeth_axis_400.pth")
    args = parser.parse_args()
    test(args)
