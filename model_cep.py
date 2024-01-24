import torch
import torch.nn as nn
from component import DGCNN_semseg_s3dis
from timm.models.vision_transformer import Block
from functools import partial
import argparse
import torch.nn.functional as F


class teeth_axis_model(nn.Module):
    def __init__(self, args, embed_dim=256, num_heads=4, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depth=4):
        super(teeth_axis_model, self).__init__()
        self.incisor_encoder = DGCNN_semseg_s3dis(args)
        self.cplinear = nn.Linear(3, embed_dim)
        self.cp_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.linear_axis = nn.Linear(embed_dim * 2, 3)  # 可以考虑把线性层换成卷积层

    def forward(self, incisor_points, teeth_center):
        incisor_features = []
        for i in range(incisor_points.size(1)):
            x = incisor_points[:, i, :, :].permute(0, 2, 1)
            x = self.incisor_encoder(x)  # (B,256,1024)
            x = x.unsqueeze(1)  # (B,1,256,1024)
            incisor_features.append(x)
        incisor_features = torch.cat(incisor_features, dim=1)  # (B,4,256,1024)

        # molar_features = []
        # for i in range(molar_points.size(1)):
        #     x = molar_points[:, i, :, :].permute(0, 2, 1)
        #     x = self.incisor_encoder(x)
        #     x = x.unsqueeze(1)
        #     molar_features.append(x)
        # molar_features = torch.cat(molar_features, dim=1)  # (B,2,256,1024)

        xc = self.cplinear(teeth_center)  # (B,6,256)
        for cpb in self.cp_blocks:
            xc = cpb(xc)  # (B,6,256)

        incisor_feature1 = incisor_features[:, 0, :, :].permute(0, 2, 1)
        incisor_feature2 = incisor_features[:, 1, :, :].permute(0, 2, 1)
        incisor_feature3 = incisor_features[:, 2, :, :].permute(0, 2, 1)
        incisor_feature4 = incisor_features[:, 3, :, :].permute(0, 2, 1)
        incisor_feature1 = torch.cat([incisor_feature1, xc[:, 0, :].unsqueeze(1).repeat(1, 1024, 1)], dim=2)
        incisor_feature2 = torch.cat([incisor_feature2, xc[:, 1, :].unsqueeze(1).repeat(1, 1024, 1)], dim=2)
        incisor_feature3 = torch.cat([incisor_feature3, xc[:, 2, :].unsqueeze(1).repeat(1, 1024, 1)], dim=2)
        incisor_feature4 = torch.cat([incisor_feature4, xc[:, 3, :].unsqueeze(1).repeat(1, 1024, 1)], dim=2)

        incisor_axis_feature = torch.cat(
            [incisor_feature1.unsqueeze(1), incisor_feature2.unsqueeze(1), incisor_feature3.unsqueeze(1),
             incisor_feature4.unsqueeze(1)], dim=1)
        result_axis = self.linear_axis(incisor_axis_feature)

        return result_axis


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, pre_landmark, target_landmark, pre_axis, target_axis):
        distances = torch.norm(pre_axis - target_axis, dim=-1)
        loss1_axis = torch.mean(distances[:, 0, :])
        loss2_axis = torch.mean(distances[:, 1, :])
        loss3_axis = torch.mean(distances[:, 2, :])
        loss4_axis = torch.mean(distances[:, 3, :])
        return loss1_axis, loss2_axis, loss3_axis, loss4_axis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teeth Arrangement')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = teeth_axis_model(args).to(device)
    incisor_points = torch.randn(4, 4, 4096, 3).to(device)
    teeth_center = torch.randn(4, 6, 3).to(device)
    result_landmark, result_axis = model(incisor_points, teeth_center)
    # print(result_landmark.shape)
    # print(result_axis.shape)
