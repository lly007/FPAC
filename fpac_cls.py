from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import PointNetSetAbstraction
from fpac import FPAC
import torch


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True,framepoint=torch.tensor([[0.0,0.0,0.0]])):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        framepoint = torch.tensor([[1.0,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],[0,0,0]]).cuda()
        self.fpac1 = FPAC(npoint=512, cin=3, cout=128, radius=0.2, nsample=32, m1=[3,9,1], m2=[1,32,128], framepoints=framepoint)
        self.fpac2 = FPAC(npoint=128, cin=128, cout=256, radius=0.4, nsample=64, m1=[3,64,1], m2=[1,64,256],framepoints=framepoint)
        self.fpac3 = FPAC(npoint=None, cin=256, cout=1024, radius=0.8, nsample=32, m1=[3,64,1], m2=[1,128,1024], framepoints=framepoint)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        xyz = xyz.transpose(1, 2)
        norm = norm.transpose(1, 2)
        f1, s1 = self.fpac1(xyz, norm)
        f2, s2 = self.fpac2(s1, f1)
        f3, s3 = self.fpac3(s2, f2)
        x = f3.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)



        return x, s3



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
