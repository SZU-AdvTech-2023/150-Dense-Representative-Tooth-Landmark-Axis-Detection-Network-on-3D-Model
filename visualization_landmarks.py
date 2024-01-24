import vtkmodules.all as vtk
import plyfile
import numpy as np
import json


def save_to_ply(verts, verts_color, faces, ply_filename_out):
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    if verts_color is None:
        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

        for i in range(0, num_verts):
            verts_tuple[i] = tuple(verts[i, :])
    else:
        verts_tuple = np.zeros(
            (num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])

        for i in range(0, num_verts):
            verts_tuple[i] = (verts[i][0], verts[i][1], verts[i][2],
                              verts_color[i][0], verts_color[i][1], verts_color[i][2])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)


if __name__ == "__main__":
    result_path = "./test_result_premolars.json"
    sigma = 0.3
    with open(result_path, "r") as json_file:
        result_data = json.load(json_file)
    for i in range(len(result_data)):
        dic = result_data[i]
        name = dic["name"]
        cor_1 = np.array(dic["landmark_1"])
        cor_2 = np.array(dic["landmark_2"])
        cor_3 = np.array(dic["landmark_3"])
        # cor_4 = np.array(dic["landmark_4"])
        # cor_5 = np.array(dic["landmark_5"])
        landmarks = np.vstack((cor_1, cor_2, cor_3))

        file_path = f"./premolars/teeth/Test_data2/{name}.stl"
        stl_reader = vtk.vtkSTLReader()
        stl_reader.SetFileName(file_path)

        stl_reader.Update()

        verts = stl_reader.GetOutput().GetPoints()
        faces = stl_reader.GetOutput().GetPolys()

        verts_array = np.array([verts.GetPoint(i) for i in range(verts.GetNumberOfPoints())])
        num = verts_array.shape[0]
        gaussian = np.ones((num,))
        for j in range(3):
            landmark = landmarks[j]
            distances = np.linalg.norm(verts_array - landmark, axis=1)
            distances = np.exp(-0.5 * distances ** 2 / sigma ** 2)
            if j == 0:
                gaussian = distances
            else:
                gaussian = np.maximum(gaussian, distances)
        print(verts_array.shape)

        colors = np.zeros((num, 3))
        colors[:, 0] = gaussian * 255
        colors[:, 2] = 255

        # colors=np.array([[0, 255, 0]]).repeat(num,axis=0)
        # colors[selected_vertices]=[255,0,0]
        # print(colors.shape)

        # 将面数据转化为NumPy数组
        faces_array = np.zeros((faces.GetNumberOfCells(), 3), dtype=int)

        # 从面数据中提取顶点ID并将其存储到NumPy数组中
        faces.InitTraversal()
        id_list = vtk.vtkIdList()
        for i in range(faces.GetNumberOfCells()):
            faces.GetNextCell(id_list)
            for j in range(3):  # 假设每个面都有3个顶点
                faces_array[i, j] = id_list.GetId(j)
        ply_path = f"./premolars/testv/{name}.ply"
        save_to_ply(verts_array, colors, faces_array, ply_path)
    # 打印所有面
    print("Faces:")
    for i in range(faces_array.shape[0]):
        print(f"Face {i}: {faces_array[i]}")
