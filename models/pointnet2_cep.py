import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import numpy as np


def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1,
                                                                                                                   N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists)  # Very Important for dist = 0.
    return torch.sqrt(dists).float()


def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]


def three_nn(xyz1, xyz2):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :return: dists: shape=(B, N1, 3), inds: shape=(B, N1, 3)
    '''
    dists = get_dists(xyz1, xyz2)
    dists, inds = torch.sort(dists, dim=-1)
    dists, inds = dists[:, :, :3], inds[:, :, :3]
    return dists, inds


def three_interpolate(xyz1, xyz2, points2):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :param points2: shape=(B, N2, C2)
    :return: interpolated_points: shape=(B, N1, C2)
    '''
    _, _, C2 = points2.shape
    dists, inds = three_nn(xyz1, xyz2)
    inversed_dists = 1.0 / (dists + 1e-8)
    weight = inversed_dists / torch.sum(inversed_dists, dim=-1, keepdim=True)  # shape=(B, N1, 3)
    weight = torch.unsqueeze(weight, -1).repeat(1, 1, 1, C2)
    interpolated_points = gather_points(points2, inds)  # shape=(B, N1, 3, C2)
    interpolated_points = torch.sum(weight * interpolated_points, dim=2)
    return interpolated_points


class PointNet_FP_Module(nn.Module):
    def __init__(self, in_channels, mlp, bn=True):
        super(PointNet_FP_Module, self).__init__()
        self.backbone = nn.Sequential()
        bias = False if bn else True
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv_{}'.format(i), nn.Conv2d(in_channels,
                                                                    out_channels,
                                                                    1,
                                                                    stride=1,
                                                                    padding=0,
                                                                    bias=bias))
            if bn:
                self.backbone.add_module('Bn_{}'.format(i), nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu_{}'.format(i), nn.ReLU())
            in_channels = out_channels

    def forward(self, xyz1, xyz2, points1, points2):
        '''

        :param xyz1: shape=(B, N1, 3)
        :param xyz2: shape=(B, N2, 3)   (N1 >= N2)
        :param points1: shape=(B, N1, C1)
        :param points2: shape=(B, N2, C2)
        :return: new_points2: shape = (B, N1, mlp[-1])
        '''
        B, N1, C1 = points1.shape
        _, N2, C2 = points2.shape
        if N2 == 1:
            interpolated_points = points2.repeat(1, N1, 1)
        else:
            interpolated_points = three_interpolate(xyz1, xyz2, points2)
        cat_interpolated_points = torch.cat([interpolated_points, points1], dim=-1).permute(0, 2, 1).contiguous()
        new_points = torch.squeeze(self.backbone(torch.unsqueeze(cat_interpolated_points, -1)), dim=-1)
        return new_points.permute(0, 2, 1).contiguous()


def fps(xyz, M):
    '''
    Sample M points from points according to the farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: centroids: shape=(B, M)
    '''
    # device = xyz.device
    # B, N, C = xyz.shape
    # # 初始化一个centroids矩阵，用于存储Mpoint个采样点的索引位置，大小为B×Mpoint
    # centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    # # distance矩阵(B×N)记录某个batch中所有点到某一个点的距离，初始化的值很大，后面会迭代更新
    # dists = torch.ones(B, N).to(device) * 1e5
    # # farthest表示当前最远的点，也是随机初始化，范围为0~N，初始化B个；每个batch都随机有一个初始最远点
    # farthest = torch.randint(0, N, size=(B,), dtype=torch.long).to(device)
    # batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    # for i in range(M):
    #     centroids[:, i] = farthest
    #     cur_point = xyz[batchlists, farthest, :].view(B, 1, 3)  # (B, 3)
    #     # cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)
    #     # 求出所有点到该centroid点的欧式距离，存在cur_dist矩阵中
    #     cur_dist = torch.sum((xyz - cur_point) ** 2, -1)
    #     dists[cur_dist < dists] = cur_dist[cur_dist < dists]
    #     farthest = torch.max(dists, dim=1)[1]
    #     # print(i,farthest)


    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e5
    inds = torch.randint(0, N, size=(B, ), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :] # (B, 3)
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids




# 根据FPS的结果，在每个中心点的半径为R的范围内，采样N个领域点，输出领域点索引
def ball_query(xyz, new_xyz, radius, K):
    '''
    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = get_dists(new_xyz, xyz)
    grouped_inds[dists > radius] = N
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K)
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    return grouped_inds


def sample_and_group(xyz, points, M, radius, K, use_xyz=True):
    '''
    :param xyz: shape=(B, N, 3)
    :param points: shape=(B, N, C)
    :param M: int
    :param radius:float
    :param K: int
    :param use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    :return: new_xyz, shape=(B, M, 3); new_points, shape=(B, M, K, C+3);
             group_inds, shape=(B, M, K); grouped_xyz, shape=(B, M, K, 3)
    '''
    new_xyz = gather_points(xyz, fps(xyz, M))  # 找到M个最远点
    grouped_inds = ball_query(xyz, new_xyz, radius, K)  # R内领域点索引
    grouped_xyz = gather_points(xyz, grouped_inds)  # 根据索引值，选取对应点
    grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)
    if points is not None:
        grouped_points = gather_points(points, grouped_inds)  # 根据索引值，对应的点附近的Nsample
        if use_xyz:
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, grouped_inds, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''

    :param xyz: shape=(B, M, 3)
    :param points: shape=(B, M, C)
    :param use_xyz:
    :return: new_xyz, shape=(B, 1, 3); new_points, shape=(B, 1, M, C+3);
             group_inds, shape=(B, 1, M); grouped_xyz, shape=(B, 1, M, 3)
    '''
    B, M, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C)
    grouped_inds = torch.arange(0, M).long().view(1, 1, M).repeat(B, 1, 1)
    grouped_xyz = xyz.view(B, 1, M, C)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz.float(), points.float()], dim=2)
        else:
            new_points = points
        new_points = torch.unsqueeze(new_points, dim=1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, grouped_inds, grouped_xyz


class PointNet_SA_Module(nn.Module):
    def __init__(self, M, radius, K, in_channels, mlp, group_all, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_Module, self).__init__()
        self.M = M
        self.radius = radius
        self.K = K
        self.in_channels = in_channels
        self.mlp = mlp
        self.group_all = group_all
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.backbone = nn.Sequential()
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv{}'.format(i),
                                     nn.Conv2d(in_channels, out_channels, 1,
                                               stride=1, padding=0, bias=False))
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            in_channels = out_channels

    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(xyz=xyz,
                                                                              points=points,
                                                                              M=self.M,
                                                                              radius=self.radius,
                                                                              K=self.K,
                                                                              use_xyz=self.use_xyz)
        new_points = self.backbone(new_points.permute(0, 3, 2, 1).contiguous())
        if self.pooling == 'avg':
            new_points = torch.mean(new_points, dim=2)
        else:
            new_points = torch.max(new_points, dim=2)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points


class pointnet2_seg_ssg(nn.Module):
    def __init__(self, in_channels, nclasses):
        super(pointnet2_seg_ssg, self).__init__()
        self.pt_sa1 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=in_channels, mlp=[64, 64, 128],
                                         group_all=False)
        self.pt_sa2 = PointNet_SA_Module(M=128, radius=0.4, K=64, in_channels=131, mlp=[128, 128, 256], group_all=False)
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=259, mlp=[256, 512, 1024],
                                         group_all=True)

        self.pt_fp1 = PointNet_FP_Module(in_channels=1024 + 256, mlp=[256, 256], bn=True)
        self.pt_fp2 = PointNet_FP_Module(in_channels=256 + 128, mlp=[256, 128], bn=True)
        self.pt_fp3 = PointNet_FP_Module(in_channels=128 + 6, mlp=[128, 128, 128], bn=True)

        self.conv1 = nn.Conv1d(128, 128, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.cls = nn.Conv1d(128, nclasses, 1, stride=1)

    def forward(self, l0_xyz, l0_points):
        l1_xyz, l1_points = self.pt_sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.pt_sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.pt_sa3(l2_xyz, l2_points)

        l2_points = self.pt_fp1(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.pt_fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.pt_fp3(l0_xyz, l1_xyz, torch.cat([l0_points, l0_xyz], dim=-1), l1_points)

        net = l0_points.permute(0, 2, 1).contiguous()
        net = self.dropout1(F.relu(self.bn1(self.conv1(net))))
        net = self.cls(net)
        return net



class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, pre_axis, target_axis):
        distances = torch.norm(pre_axis - target_axis, dim=-1)
        loss1_axis = torch.mean(distances[:, 0, :])
        loss2_axis = torch.mean(distances[:, 1, :])
        loss3_axis = torch.mean(distances[:, 2, :])
        loss4_axis = torch.mean(distances[:, 3, :])
        return loss1_axis, loss2_axis, loss3_axis, loss4_axis


if __name__ == "__main__":
    teeth_points = torch.randn(4, 4096, 3).cuda()
    teeth_normals = torch.randn(4, 4096, 3).cuda()
    model = pointnet2_seg_ssg(6, 3)
    model.cuda()
    y = model(teeth_points, teeth_normals)
    print(y.shape)
