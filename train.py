import time
import argparse
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from models.pointnet2 import pointnet2_seg_ssg, Loss
from dataset import ToothLandmark
from tqdm import tqdm


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = "./premolars"
    train_loader = DataLoader(ToothLandmark(train_dir), num_workers=0,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = pointnet2_seg_ssg(6, 3)
    # pretrained_model=torch.load("./checkpoint_canine/teeth_landmark_400.pth")
    # model.load_state_dict(pretrained_model)
    loss = Loss()

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD([{'params': model.local_fea.parameters(), 'lr': args.lr}], lr=args.lr, momentum=args.momentum,
                        weight_decay=1e-4)
        # opt = optim.SGD([
        #     {'params': model.teeth_fea.parameters(), 'lr': args.lr},
        #     {'params': model.global_fea.parameters(), 'lr': args.lr},
        #     {'params': model.output.parameters(), 'lr': args.lr}])

    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-6, last_epoch=-1)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)

    model.cuda()
    model.train()
    inter_nums = len(train_loader)
    pbar = tqdm(range(args.epochs))

    for i in pbar:
        train_loss = 0.0
        cen_loss = 0.0
        cusp_loss1 = 0.0
        cusp_loss2 = 0.0
        # left_loss=0.0
        # right_loss=0.0
        # back_loss=0.0
        # cusp_loss=0.0
        count = 0

        data_loader = iter(train_loader)
        for data_num in range(inter_nums):
            teeth_points, teeth_normals, label = next(data_loader)
            teeth_points = teeth_points.cuda().float()
            teeth_normals = teeth_normals.cuda().float()
            label = label.cuda().float()

            bs = teeth_points.shape[0]

            opt.zero_grad()
            pred = model(teeth_points, teeth_normals)
            loss1, loss2, loss3 = loss(label, pred)
            total_loss = loss1 + loss2 + loss3
            total_loss = total_loss / bs

            total_loss.backward()
            opt.step()

            train_loss += total_loss.item()
            cen_loss += loss1.item()
            cusp_loss1 += loss2.item()
            cusp_loss2 += loss3.item()
            # right_loss+=loss4.item()
            # cusp_loss+=loss5.item()
            state_msg = (
                f'S: {train_loss:.3f}; C: {cen_loss:.3f};pre:{cusp_loss1:.3f};after:{cusp_loss2:.3f}'
            )

            pbar.set_description(state_msg)

            train_loss = 0.0
            cen_loss = 0.0
            cusp_loss1 = 0.0
            cusp_loss2 = 0.0
            # left_loss=0.0
            # right_loss=0.0
            # back_loss=0.0
            # cusp_loss=0.0

        if (i + 1) % 10 == 0 or i == 0:
            print(state_msg)
            #torch.save(model.state_dict(), 'checkpoint_premolars/teeth2_landmark_' + str(i) + '.pth')
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teeth landmark')
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=1.5 * 1e-4, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    args = parser.parse_args()
    train(args)
