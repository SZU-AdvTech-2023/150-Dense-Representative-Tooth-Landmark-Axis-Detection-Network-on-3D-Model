import trimesh
import open3d as o3d
# import numpy as np
import copy
# from glob import glob
# import vedo
import os


# from os.path import join
# from glob import glob


# 可视化配准结果。给定源点云、目标点云和变换矩阵，函数深拷贝源点云和目标点云，分别涂上颜色，然后根据给定的变换矩阵对源点云进行变换，并使用Open3D进行可视化。
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


# 对点云进行预处理。包括点云的降采样、法线估计和FPFH特征计算。
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 100  # 搜索半径
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal // 10, max_nn=300))
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=300))

    radius_feature = voxel_size * 200
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                                               o3d.geometry.KDTreeSearchParamHybrid(
                                                                   radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


# 用于准备配准所需的数据集。加载源点云和目标点云，然后对它们进行降采样和特征计算。
def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(r'i:/total_seg.ply')
    target = o3d.io.read_point_cloud(r'H:/daiyasi/UpperJawScan_rotated_label.ply')
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


# 执行全局配准。使用RANSAC算法进行基于特征匹配的配准，返回配准结果。
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 15
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 50000))
    return result


# 对配准结果进行细化。使用ICP（迭代最近点）算法进行点到平面的配准，返回细化后的配准结果。
def refine_registration(source, target, source_fpfh, target_fpfh, result_ransac, voxel_size):
    distance_threshold = voxel_size * 4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


# 对单个牙齿的点云进行配准。加载源点云和目标点云，进行全局配准和细化，最后保存配准后的点云。
def r_f(i):
    voxel_size = 0.4
    source_path = r'E:\2022_Teeth\feicai\lc\kousao\seg\lc_l_eL_' + str(i) + '.ply'
    target_path = r'E:\2022_Teeth\feicai\lc\kousao\target_l_rotated.ply'
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, result_ransac,
                                     voxel_size)
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation)
    source.transform(result_icp.transformation)
    print(result_icp.transformation)
    ply = trimesh.load(source_path)
    ply.apply_transform(result_icp.transformation)
    ply.export(r'E:\2022_Teeth\feicai\lc\kousao\regis' + "\\" + 'crown_' + str(i) + '_trans_crown.ply')


# if __name__ == "__main__":
#     # for i in [1,2,3,4,5,6,7,9,10,11,12,13,14,17,18,19,20,21,22,23,25,26,27,28,29,30]:
#     for i in [1,2,3,4,5,6,7,9,10,11,12,13,14]:
#         r_f(i)

if __name__ == "__main__":
    # toothids = [31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47]
    toothids = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]
    voxel_size = 0.4  # means 3cm for the dataset

    # 原点云为口扫模型
    source_path = "zq_vol_trans.ply"
    # source_path = r'../data/cbct/cbct_lower.ply'
    # target_path= r'../data/mesh_ios/upper_tooth_after_cut_tri.ply'
    # 目标点云为CBCT模型
    # target_path = r'../data/mesh_ios/result_upper_tooth.ply'
    target_path = "zqscan.ply"

    # target_path = r'../data/target_l.ply'
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    # source.transform(trans_init)

    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)

    print(result_ransac)
    # draw_registration_result(source_down, target_down,result_ransac.transformation)

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, result_ransac,
                                     voxel_size)
    # print(result_icp)
    # visulization
    draw_registration_result(source, target, result_icp.transformation)
    # source.transform(result_icp.transformation)
    # print(result_icp.transformation)
    # #toothids = [0,31,32,33,34,35,36,37,41,42,43,44,45,46,47]
    # #toothids = [0, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]
    # source_bone = '../data/bone/bone_u.ply'
    # # ply = trimesh.load(source_path[:-4]+'.ply')
    # # ply.apply_transform(result_icp.transformation)
    # # ply.export(source_path[:-4] + '_after_register.ply')
    # ply = trimesh.load(source_bone)
    # ply.apply_transform(result_icp.transformation)
    # ply.export(source_bone[:-4] + '_after_register.ply')

    # source_path = '..\data/bone'
    # for i in toothids:
    #
    #     #filename = str(i) + '.ply'
    #     filename = str(i) + '.ply'
    #     path = os.path.join(source_path,filename)
    #     print(path)
    #     ply = trimesh.load(path)
    #     ply.apply_transform(result_icp.transformation)
    #     #存放源点云经过icp得到变换后的点云
    #     ply.export(path[:-4]+'_register.ply')

    # source_path = r'E:\Desktop\result_teeth\126259135_shell_occlusion_l.ply'
    # ply = trimesh.load(source_path)
    # ply.apply_transform(result_icp.transformation)
    # #存放源点云经过icp得到变换后的点云
    # ply.export(source_path[:-4]+'_0611trans.ply')

    #
    # ply.export(source_path[:-4]+'_trans_crown.ply')
    # ply.export(r'E:\2022\test_data\PD\uerD_u_label_rotate.ply')
"""
# 对每个都应用的 跑这里的循环
    for i in range(1,14,1):
        p= r'E:\Desktop\newdata_9_1\cbct_seg_bao\overall_seg_result\erotion0_LabelMapVolume_'+str(i)+'.ply'
        ply = trimesh.load(p)
        ply.apply_transform(result_icp.transformation)
        ply.export(p[:-4]+'_trans_overall.ply')
"""
dir = "zq_erotion"
paths = os.listdir(dir)
for path in paths:
    name = path[:-4]
    stl_path = os.path.join(dir, path)
    stl = trimesh.load(stl_path)
    stl.apply_transform(result_icp.transformation)
    stl.export("output\{}.stl".format(name))
