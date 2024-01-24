import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from dataset_cep import ToothLandmark
from tqdm import tqdm
# from model_cep import teeth_axis_model, Loss
from models.pointnet2_cep import pointnet2_seg_ssg, Loss
import torch.nn.functional as F


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./cbct_data"
    landmark_dir = "./label_landmark"
    train_loader = DataLoader(ToothLandmark(data_dir, landmark_dir), num_workers=0,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = pointnet2_seg_ssg(6, 3)
    loss = Loss()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-6, last_epoch=-1)

    model.cuda()
    model.train()
    iter_nums = len(train_loader)
    pbar = tqdm(range(args.epochs))
    print(iter_nums)

    for i in pbar:
        train_loss = 0.0
        loss_axis1 = 0.0
        loss_axis2 = 0.0
        loss_axis3 = 0.0
        loss_axis4 = 0.0

        data_loader = iter(train_loader)
        for data_num in range(iter_nums):
            incisor_points, teeth_normals, teeth_center, heatmap_axis = next(data_loader)
            incisor_points = incisor_points.cuda().float()
            # print(incisor_points.shape)
            teeth_normals = teeth_normals.cuda().float()
            # print(teeth_normals.shape)
            heatmap_axis = heatmap_axis.cuda().float()

            # bs = incisor_points.shape[0]

            opt.zero_grad()
            result_axis = []
            # result_axis = model(incisor_points, teeth_center)
            for j in range(4):
                pred = model(incisor_points[:, j, :, :], teeth_normals[:, j, :, :])
                result_axis.append(pred)

            result_axis = torch.stack(result_axis, dim=1)
            # print(result_axis.shape)
            result_axis = result_axis.permute(0, 1, 3, 2)

            loss1_axis, loss2_axis, loss3_axis, loss4_axis = loss(result_axis, heatmap_axis)
            total_loss = loss1_axis + loss2_axis + loss3_axis + loss4_axis

            total_loss.backward()
            opt.step()

            train_loss += total_loss.item()
            loss_axis1 += loss1_axis.item()
            loss_axis2 += loss2_axis.item()
            loss_axis3 += loss3_axis.item()
            loss_axis4 += loss4_axis.item()
            state_msg = (
                f's: {train_loss:.3f};a1:{loss_axis1:.3f};a2:{loss_axis2:.3f};a3:{loss_axis3:.3f};a4:{loss_axis4:.3f}'
            )
            pbar.set_description(state_msg)
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)
            train_loss = 0.0
            loss_axis1 = 0.0
            loss_axis2 = 0.0
            loss_axis3 = 0.0
            loss_axis4 = 0.0

        if (i + 1) % 10 == 0 or i == 0:
            print(state_msg)
        if i == 0:
            torch.save(model.state_dict(), 'checkpoint_cep/teeth_axis_' + str(i) + '.pth')
        if (i + 1) % 100 == 0:
            torch.save(model.state_dict(), 'checkpoint_cep/teeth_axis_' + str(i + 1) + '.pth')
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teeth axis detection')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=1.5 * 1e-4, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    args = parser.parse_args()
    train(args)
