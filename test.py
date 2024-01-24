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
import numpy as np
import json


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = "./premolars"
    test_loader = DataLoader(ToothLandmark(train_dir, mode='test'), num_workers=0,
                             batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = pointnet2_seg_ssg(6, 3)
    pretrained_model = torch.load(args.checkpoint)
    model.load_state_dict(pretrained_model)
    model.cuda()
    model.eval()
    # loss=Loss()
    pbar = tqdm(range(len(test_loader)))
    data_loader = iter(test_loader)
    dict_list = []
    for i in pbar:
        teeth_points, teeth_normals, name, centroid, tmaxv = next(data_loader)
        print(teeth_points.shape)
        teeth_points = teeth_points.cuda().float()
        print(teeth_points.shape)
        teeth_normals = teeth_normals.cuda().float()
        centroid = centroid.cuda().float()
        tmaxv = tmaxv.cuda().float()
        # label.cuda().float()
        bs = teeth_points.shape[0]
        with torch.no_grad():
            pred = model(teeth_points, teeth_normals)
            print(teeth_points.shape)
            for j in range(bs):
                origin_points = teeth_points[j] * tmaxv[j] + centroid[j]
                origin_points = origin_points.detach().cpu().numpy()
                patient = name[j]
                res = pred[j].detach().cpu().numpy()
                indices_1 = np.argmax(res[0])
                landmark_1 = origin_points[indices_1]
                indices_2 = np.argmax(res[1])
                landmark_2 = origin_points[indices_2]
                indices_3 = np.argmax(res[2])
                landmark_3 = origin_points[indices_3]

                landmark_1 = landmark_1.tolist()
                landmark_2 = landmark_2.tolist()
                landmark_3 = landmark_3.tolist()


                result = {
                    "name": patient,
                    "landmark_1": landmark_1,
                    "landmark_2": landmark_2,
                    "landmark_3": landmark_3,
                }
                dict_list.append(result)
                print(result)
    with open("test_result_premolars_t.json", 'w') as file:
        json.dump(dict_list, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config')
    parser.add_argument('--batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--checkpoint', type=str, default="./checkpoint/teeth_landmark_1800.pth")
    args = parser.parse_args()
    test(args)
